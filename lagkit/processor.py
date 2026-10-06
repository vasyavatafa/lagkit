"""Класс ``LagProcessor`` и функция ``build_views`` — программный интерфейс модуля."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
import pandas as pd

from . import core, diagnostics as diag
from .blocks import detect_layouts
from .config import LAYOUTS, VIEWS, LagConfig, LagkitError
from .io import Source, read_data, read_master, write_frame
from .lags import resolve_lags

logger = logging.getLogger("lagkit")

ConfigLike = Union[None, str, Path, Mapping[str, Any], LagConfig]


class LagProcessor:
    """Превращает витрину (значения на отчётную дату) в таблицу значений на дату доступности.

    Три представления (раздел 6 ТЗ):

    * ``raw`` — витрина, отфильтрованная по факторам master;
    * ``journal`` — журнал доступности: у каждого значения его ``available_date``;
    * ``known`` — таблица известных значений для обучения и предикта.

    Входные данные читаются и проверяются в конструкторе; представления вычисляются
    лениво при первом обращении и кэшируются. Таблица публикаций строится один раз,
    все представления берутся из неё.

    Пример::

        lp = LagProcessor("master.xlsx", "vitrina.csv", "lag_config.yaml")
        lp.lags
        lp.known()
        lp.trace("factor_name")
        lp.save(view="known")
    """

    def __init__(self, master: Source, data: Source, config: ConfigLike = None) -> None:
        self.config: LagConfig = LagConfig.load(config)
        cfg = self.config
        self.master: pd.DataFrame = read_master(master, cfg)
        frame, present, issues = read_data(data, cfg, self.master["factor"].tolist())
        self._input_issues: List[Dict[str, Any]] = issues
        diag.log_rows(issues)

        self.factors: List[str] = present
        self.dates = pd.DatetimeIndex(frame[cfg.date_col])
        self._values = frame[present].to_numpy(dtype=float) if present else np.empty((len(frame), 0))
        freq_map = dict(zip(self.master["factor"], self.master["data_frequency"]))
        self.freqs: List[str] = [freq_map[f] for f in present]
        if cfg.snapshot_date is not None:
            self.snapshot_date = pd.Timestamp(cfg.snapshot_date)
        elif len(self.dates):
            self.snapshot_date = self.dates.max()
        else:
            raise LagkitError("Витрина пуста: нет ни одной строки с датой")
        self._cache: Dict[str, Any] = {}

    # --- промежуточные структуры --------------------------------------------
    @property
    def lags(self) -> pd.DataFrame:
        """Итоговые лаги: ``factor``, ``data_frequency``, ``object_lag``, ``lag_source``, ``layout``."""
        if "lags" not in self._cache:
            layouts = detect_layouts(self._values, self.freqs, self.config.diagnostics)
            lags, issues = resolve_lags(self.master, self.config, dict(zip(self.factors, layouts)))
            self._cache["lags"] = lags
            self._cache["lag_issues"] = issues
        return self._cache["lags"].copy()

    @property
    def publications(self) -> pd.DataFrame:
        """Таблица публикаций: ``factor``, ``report_date``, ``block_end``, ``available_date``, ``value``."""
        return self._publications()[["factor", "report_date", "block_end", "available_date", "value"]].copy()

    def _publications(self) -> pd.DataFrame:
        if "pubs" not in self._cache:
            lags = self.lags
            self._cache["pubs"] = core.build_publications(
                self.dates, self._values, self.factors, self.freqs,
                lags["object_lag"].tolist(), self.config.value_tol,
            )
        return self._cache["pubs"]

    def _full(self, view: str) -> pd.DataFrame:
        key = "full_" + view
        if key in self._cache:
            return self._cache[key]
        cfg = self.config
        if view == "raw":
            full = core.raw_full(self.dates, self._values, self.factors, cfg.date_col)
        elif view == "journal":
            full = core.journal_full(self.dates, self._values, self._publications(), self.factors, cfg.date_col)
        elif view == "known":
            calendar = core.make_calendar(self.dates, self.snapshot_date, cfg.calendar)
            full = core.known_full(calendar, self._publications(), self.factors, cfg.known_date_col,
                                   cfg.ffill_limit, cfg.drop_leading_empty)
            leaks = core.check_no_leak(full, cfg.known_date_col)
            if len(leaks):
                row = diag._diag(str(leaks["factor"].iloc[0]), "leak_invariant", "error",
                                 f"Нарушен инвариант available_date <= {cfg.known_date_col} в {len(leaks)} строках: "
                                 "ошибка в коде модуля", leaks.head(10).to_dict("records"))
                diag.log_rows([row])
                raise LagkitError(row["message"])
        else:
            raise LagkitError(f"Неизвестное представление {view!r}; допустимы {list(VIEWS)}")
        self._cache[key] = full
        return full

    # --- представления ------------------------------------------------------
    def get(self, view: str = "known", layout: Optional[str] = None) -> pd.DataFrame:
        """Возвращает представление ``raw``/``journal``/``known`` в формате ``wide``/``long``.

        Широкий формат — ``pivot`` плоского, поэтому форматы не расходятся.

        * ``raw``: широкий — дата и по колонке на фактор; плоский — дата, ``factor``, ``value``.
        * ``journal``: широкий — на каждый фактор ``<factor>`` и ``<factor><date_suffix>``
          (и ``<factor><fill_flag_suffix>`` при ``journal_fill_flag``); плоский — дата,
          ``factor``, ``available_date``, ``value``, ``is_filled``. У протянутой строки
          (``is_filled=True``) ``available_date`` — дата, когда стало известно последнее
          подставляемое значение, а не значение этой строки.
        * ``known``: широкий — дата (``known_date_col``) и по колонке на фактор; плоский —
          дата, ``factor``, ``value``, ``available_date`` (дата доступности подставленного
          значения).

        В плоском формате строки с пустым значением не выдаются, если не ``keep_empty``.
        """
        cfg = self.config
        layout = layout or cfg.layout
        if layout not in LAYOUTS:
            raise LagkitError(f"layout={layout!r}; допустимые значения: {list(LAYOUTS)}")
        if view not in VIEWS:
            raise LagkitError(f"Неизвестное представление {view!r}; допустимы {list(VIEWS)}")
        key = f"{view}_{layout}"
        if key not in self._cache:
            full = self._full(view)
            date_col = cfg.known_date_col if view == "known" else cfg.date_col
            if layout == "long":
                out = core.full_to_long(full, cfg.keep_empty)
            else:
                columns = {"value": ""}
                if view == "journal":
                    columns["available_date"] = cfg.date_suffix
                    if cfg.journal_fill_flag:
                        columns["is_filled"] = cfg.fill_flag_suffix
                out = core.long_to_wide(full, date_col, self.factors, columns)
            self._cache[key] = out
        return self._cache[key].copy()

    def raw(self, layout: Optional[str] = None) -> pd.DataFrame:
        """Синоним ``get('raw', layout)``."""
        return self.get("raw", layout)

    def journal(self, layout: Optional[str] = None) -> pd.DataFrame:
        """Синоним ``get('journal', layout)``."""
        return self.get("journal", layout)

    def known(self, layout: Optional[str] = None) -> pd.DataFrame:
        """Синоним ``get('known', layout)``: таблица для обучения и предикта."""
        return self.get("known", layout)

    def trace(self, factor: str) -> pd.DataFrame:
        """Три представления одного фактора бок о бок для отладки сдвига.

        Колонки: дата, ``raw`` (витрина), ``journal`` (значение блока в журнале),
        ``available_date``, ``is_filled``, ``known`` (значение в таблице известных значений).
        """
        if factor not in self.factors:
            raise LagkitError(f"Фактора {factor!r} нет среди обработанных факторов")
        cfg = self.config
        raw = self._full("raw")
        raw = raw.loc[raw["factor"] == factor, [cfg.date_col, "value"]].rename(columns={"value": "raw"})
        jr = self._full("journal")
        jr = jr.loc[jr["factor"] == factor, [cfg.date_col, "value", "available_date", "is_filled"]]
        jr = jr.rename(columns={"value": "journal"})
        kn = self._full("known")
        kn = kn.loc[kn["factor"] == factor, [cfg.known_date_col, "value"]]
        kn = kn.rename(columns={cfg.known_date_col: cfg.date_col, "value": "known"})
        out = raw.merge(jr, on=cfg.date_col, how="left").merge(kn, on=cfg.date_col, how="outer")
        return out.sort_values(cfg.date_col).reset_index(drop=True)

    # --- диагностика и сохранение ----------------------------------------------
    def diagnostics(self) -> pd.DataFrame:
        """Проверки раздела 9 ТЗ: ``factor``, ``check``, ``severity``, ``message``, ``details``.

        Главный инвариант ``available_date <= DATE`` проверяется при построении ``known``;
        его нарушение — исключение (ошибка в коде модуля, а не в данных).
        """
        if "diagnostics" not in self._cache:
            lags = self.lags
            self._full("known")  # инвариант утечки
            rows = list(self._cache["lag_issues"])
            rows += diag.check_publications(self._publications(), lags, self._values,
                                            self.factors, self.snapshot_date, self.config)
            diag.log_rows(rows)
            self._cache["diagnostics"] = diag.to_frame(self._input_issues + rows)
        return self._cache["diagnostics"].copy()

    def save(
        self,
        view: str = "known",
        layout: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        output_format: Optional[str] = None,
    ) -> Dict[str, Path]:
        """Сохраняет представление (или все при ``view='all'``) вместе с таблицей лагов
        и отчётом диагностики — без них результат нельзя воспроизвести и проверить.

        Файлы: ``<view>_<layout>.<ext>``, ``lags.<ext>``, ``diagnostics.<ext>``.
        Возвращает словарь «имя → путь».
        """
        cfg = self.config
        layout = layout or cfg.layout
        fmt = (output_format or cfg.output_format).lower().lstrip(".")
        out_dir = Path(output_dir or cfg.output_dir)
        views = list(VIEWS) if view == "all" else [view]
        paths: Dict[str, Path] = {}
        for v in views:
            paths[f"{v}_{layout}"] = write_frame(self.get(v, layout), out_dir / f"{v}_{layout}", fmt)
        paths["lags"] = write_frame(self.lags, out_dir / "lags", fmt)
        paths["diagnostics"] = write_frame(self.diagnostics(), out_dir / "diagnostics", fmt)
        logger.info("lagkit: результаты сохранены в %s", out_dir)
        return paths


def build_views(
    master: Source, data: Source, config: ConfigLike = None, layout: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """Строит все три представления без создания класса вручную.

    Возвращает словарь ``{'raw', 'journal', 'known', 'lags', 'diagnostics'}``.
    """
    lp = LagProcessor(master, data, config)
    out = {v: lp.get(v, layout) for v in VIEWS}
    out["lags"] = lp.lags
    out["diagnostics"] = lp.diagnostics()
    return out
