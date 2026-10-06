"""Диагностика сдвигов и проверки на утечку (раздел 9 ТЗ).

Результат — DataFrame с колонками ``factor``, ``check``, ``severity``
(``error``/``warning``/``info``), ``message``, ``details``. Сообщения пишутся в лог через
стандартный ``logging`` (логгер ``lagkit``), чтобы уровень подробности регулировался
настройками проекта.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from .config import LagConfig
from .lags import snapshot_bounds

logger = logging.getLogger("lagkit")

DIAG_COLUMNS = ["factor", "check", "severity", "message", "details"]
_LEVELS = {"error": logging.ERROR, "warning": logging.WARNING, "info": logging.INFO}


def _diag(factor: str, check: str, severity: str, message: str, details: Any = "") -> Dict[str, Any]:
    if not isinstance(details, str):
        details = json.dumps(details, ensure_ascii=False, default=str)
    return dict(factor=factor, check=check, severity=severity, message=message, details=details)


def _d(ts: Any) -> str:
    return pd.Timestamp(ts).strftime("%d.%m.%Y")


def to_frame(rows: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """Список проверок в DataFrame с фиксированным набором колонок."""
    return pd.DataFrame(list(rows), columns=DIAG_COLUMNS)


def log_rows(rows: Iterable[Dict[str, Any]]) -> None:
    """Пишет проверки в лог с уровнем, соответствующим ``severity``."""
    for r in rows:
        logger.log(_LEVELS.get(r["severity"], logging.INFO), "[%s] %s: %s", r["check"], r["factor"], r["message"])


def check_publications(
    pubs: pd.DataFrame,
    lags: pd.DataFrame,
    values: np.ndarray,
    factors: Sequence[str],
    snapshot: pd.Timestamp,
    cfg: LagConfig,
) -> List[Dict[str, Any]]:
    """Эвристические проверки по таблице публикаций.

    Все они зависят от допущений о данных (шаг отчётов, полнота загрузки) и поэтому только
    сообщают, но не останавливают построение.
    """
    dcfg = cfg.diagnostics
    rows: List[Dict[str, Any]] = []
    lag_by = lags.set_index("factor")
    groups = {f: g for f, g in pubs.groupby("factor", observed=True, sort=False)}

    for j, factor in enumerate(factors):
        freq = lag_by.at[factor, "data_frequency"]
        lag = int(lag_by.at[factor, "object_lag"])
        layout = lag_by.at[factor, "layout"]
        period = cfg.periods[freq]
        g = groups.get(factor)
        if g is None or g.empty:
            continue
        rdates = pd.DatetimeIndex(g["report_date"])

        # Пропуски отчётов: интервал > gap_factor периодов.
        if len(rdates) > 1:
            gaps_days = np.diff(rdates.values).astype("timedelta64[D]").astype(int)
            gap_idx = np.flatnonzero(gaps_days > dcfg.gap_factor * period)
            if gap_idx.size:
                gaps = [dict(prev_report=_d(rdates[i]), next_report=_d(rdates[i + 1]),
                             interval_days=int(gaps_days[i]),
                             missed=int(round(gaps_days[i] / period)) - 1) for i in gap_idx]
                total = sum(x["missed"] for x in gaps)
                shown = "; ".join(f"{x['prev_report']} → {x['next_report']}: пропущено {x['missed']}"
                                  for x in gaps[:dcfg.max_gaps_in_message])
                more = f" и ещё {len(gaps) - dcfg.max_gaps_in_message}" if len(gaps) > dcfg.max_gaps_in_message else ""
                rows.append(_diag(factor, "missing_reports", "warning",
                                  f"Пропуски отчётов: {len(gaps)} разрыв(ов), всего пропущено периодов {total} "
                                  f"({shown}{more}); значения протянуты", gaps))

            # Частотность: медианный интервал меньше доли периода.
            if freq in ("м", "н"):
                median = float(np.median(gaps_days))
                if median < dcfg.min_interval_share * period:
                    rows.append(_diag(factor, "frequency_too_high", "warning",
                                      f"Медианный интервал между отчётами {median:g} дн. меньше "
                                      f"{dcfg.min_interval_share:g} периода ({period} дн.): неверная частотность "
                                      f"в master или значение меняется внутри периода",
                                      dict(median_interval_days=median, period=period)))

        # Проверка лага по снимку и «устаревание» ряда.
        before = rdates[rdates <= snapshot]
        if len(before):
            r_last = before[-1]
            age = int((snapshot - r_last).days)
            lower, upper = snapshot_bounds(snapshot, r_last, period)
            details = dict(snapshot=_d(snapshot), last_report=_d(r_last), period=period,
                           lag=lag, lower_exclusive=lower, upper_inclusive=upper)
            if age > dcfg.stale_periods * period + lag:
                # Ряд перестал обновляться: допущение проверки по снимку (следующий отчёт
                # не пропущен) заведомо нарушено, поэтому границы лага не проверяем.
                rows.append(_diag(factor, "stale", "warning",
                                  f"Последний отчёт {_d(r_last)} старше снимка на {age} дн. "
                                  f"(> {dcfg.stale_periods:g} периода + лаг): ряд перестал обновляться",
                                  dict(last_report=_d(r_last), age_days=age)))
            elif lag <= lower:
                rows.append(_diag(factor, "lag_below_snapshot_bound", "warning",
                                  f"Лаг {lag} дн. не больше нижней границы {lower} по снимку "
                                  f"(последний отчёт {_d(r_last)}): риск утечки на текущем краю данных",
                                  details))
            elif lag > upper:
                rows.append(_diag(factor, "lag_above_snapshot_bound", "info",
                                  f"Лаг {lag} дн. больше верхней границы {upper} по снимку "
                                  f"(последний отчёт {_d(r_last)}): значение, вероятно, известно раньше, "
                                  f"свежая информация теряется", details))

        # Дневной фактор, который почти не меняется.
        if freq == "д":
            col = values[:, j]
            v = col[~np.isnan(col)]
            if v.size >= dcfg.daily_min_obs:
                changes = np.abs(np.diff(v)) > cfg.value_tol
                share = float(changes.mean())
                if share < dcfg.daily_change_share:
                    rows.append(_diag(factor, "daily_rarely_changes", "info",
                                      f"Дневной фактор меняется в {share:.1%} дней (< {dcfg.daily_change_share:.1%}): "
                                      f"вероятно, частотность «д» указана по ошибке и лаг по умолчанию мал",
                                      dict(change_share=round(share, 4))))

        if layout == "mixed":
            rows.append(_diag(factor, "mixed_layout", "info",
                              "Раскладка mixed: витрина меняла способ хранения фактора "
                              "(плотная/разреженная); на значения не влияет"))
    return rows
