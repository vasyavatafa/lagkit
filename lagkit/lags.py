"""Лаги публикации: разбор ``object_lag``, лаги по умолчанию, проверка по снимку витрины."""
from __future__ import annotations

import numbers
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import LagConfig, LagkitError
from .io import read_table

LAG_COLUMNS = ["factor", "data_frequency", "object_lag", "lag_source", "layout"]


def parse_object_lag(value: Any, factor: str = "") -> Optional[int]:
    """Разбирает значение ``object_lag`` в целое число календарных дней.

    Пусто (None, NaN, пустая строка) — ``None``: лаг будет взят по умолчанию.
    Принимаются целые числа, числа с нулевой дробной частью (``2.0`` из Excel) и строки
    вида ``"2"``/``"2.0"``. Дробное, отрицательное или нечисловое значение — ошибка.
    Нулевой лаг допустим, о нём предупреждает диагностика.
    """
    if value is pd.NA or value is pd.NaT:
        return None
    if value is None or isinstance(value, bool):
        if isinstance(value, bool):
            raise LagkitError(f"object_lag фактора {factor!r}: недопустимое значение {value!r}")
        return None
    if isinstance(value, str):
        text = value.strip().replace(",", ".")
        if text == "":
            return None
        try:
            num = float(text)
        except ValueError:
            raise LagkitError(
                f"object_lag фактора {factor!r}: {value!r} не является числом (ожидается целое число дней)"
            ) from None
    elif isinstance(value, numbers.Number):
        num = float(value)
    else:
        raise LagkitError(f"object_lag фактора {factor!r}: неподдерживаемый тип {type(value).__name__}")
    if np.isnan(num):
        return None
    if not np.isfinite(num) or num != int(num):
        raise LagkitError(f"object_lag фактора {factor!r}: {value!r} не целое число дней")
    if num < 0:
        raise LagkitError(f"object_lag фактора {factor!r}: отрицательный лаг {value!r}")
    return int(num)


def _diag(factor: str, check: str, severity: str, message: str, details: str = "") -> Dict[str, Any]:
    return dict(factor=factor, check=check, severity=severity, message=message, details=details)


def resolve_lags(
    master: pd.DataFrame, cfg: LagConfig, layouts: Dict[str, str]
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Строит таблицу итоговых лагов для факторов, присутствующих в ``layouts``.

    Лаг берётся из ``object_lag`` master (``lag_source='master'``), а если он пуст — равен
    ``default_lags`` по частотности (``lag_source='default'``). При заданном
    ``cfg.lags_override`` лаги и их источник берутся из сохранённой таблицы прошлого запуска.
    """
    issues: List[Dict[str, Any]] = []
    rows = []
    for factor, freq, raw_lag in master[["factor", "data_frequency", "object_lag"]].itertuples(index=False):
        if factor not in layouts:
            continue
        lag = parse_object_lag(raw_lag, factor)
        if lag is None:
            rows.append((factor, freq, cfg.default_lags[freq], "default", layouts[factor]))
        else:
            rows.append((factor, freq, lag, "master", layouts[factor]))
    lags = pd.DataFrame(rows, columns=LAG_COLUMNS)

    if cfg.lags_override:
        lags, override_issues = _apply_override(lags, cfg)
        issues.extend(override_issues)

    for factor, lag in zip(lags["factor"], lags["object_lag"]):
        if lag == 0:
            issues.append(_diag(
                factor, "zero_lag", "warning",
                "Нулевой лаг: значение считается известным в день своего отчёта; "
                "для макроданных это почти всегда ошибка заполнения",
            ))
    lags["object_lag"] = lags["object_lag"].astype("int64")
    return lags, issues


def _apply_override(lags: pd.DataFrame, cfg: LagConfig) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    issues: List[Dict[str, Any]] = []
    frozen = read_table(cfg.lags_override, sep=cfg.csv_sep, encoding=cfg.csv_encoding,
                        dtype=None, what="таблица лагов lags_override")
    frozen.columns = [str(c).strip() for c in frozen.columns]
    if not {"factor", "object_lag"} <= set(frozen.columns):
        raise LagkitError("lags_override: в таблице лагов нужны колонки factor и object_lag")
    frozen["factor"] = frozen["factor"].astype(str).str.strip()
    if frozen["factor"].duplicated().any():
        raise LagkitError("lags_override: дубликаты factor в таблице лагов")
    frozen = frozen.set_index("factor")
    lags = lags.copy()
    for i, factor in enumerate(lags["factor"]):
        if factor not in frozen.index:
            issues.append(_diag(factor, "lag_override_missing", "warning",
                                "Фактора нет в таблице lags_override; лаг взят из master/по умолчанию"))
            continue
        lag = parse_object_lag(frozen.at[factor, "object_lag"], factor)
        if lag is None:
            raise LagkitError(f"lags_override: пустой object_lag у фактора {factor!r}")
        lags.at[i, "object_lag"] = lag
        source = frozen.at[factor, "lag_source"] if "lag_source" in frozen.columns else None
        lags.at[i, "lag_source"] = str(source) if isinstance(source, str) and source else "override"
    return lags, issues


def snapshot_bounds(snapshot: pd.Timestamp, last_report: pd.Timestamp, period: int) -> Tuple[int, int]:
    """Границы лага по снимку: ``T - r_last - P < lag <= T - r_last`` (раздел 5.3 ТЗ).

    Возвращает ``(lower_exclusive, upper_inclusive)`` в днях.
    """
    age = int((snapshot - last_report).days)
    return age - period, age
