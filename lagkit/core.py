"""Таблица публикаций, журнал доступности и таблица известных значений.

Все три представления строятся из одной структуры — таблицы публикаций (по строке на
отчёт: ``report_date``, ``block_end``, ``available_date``, ``value``), поэтому разойтись не
могут. Каждое представление сначала строится в «полном» плоском виде (все пары
дата × фактор), широкий формат — ``pivot`` этого плоского, а выдаваемый плоский —
он же без пустых строк (если не ``keep_empty``).
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .blocks import report_mask
from .config import LagkitError

DAY = np.timedelta64(1, "D")


def build_publications(
    dates: pd.DatetimeIndex,
    values: np.ndarray,
    factors: Sequence[str],
    freqs: Sequence[str],
    lags: Sequence[int],
    value_tol: float = 0.0,
) -> pd.DataFrame:
    """Таблица публикаций: одна строка на отчёт (шаг 4 алгоритма раздела 4.4 ТЗ).

    ``available_date = report_date + lag`` календарных дней и одинакова для всего блока.
    Колонки ``_fidx``, ``_row``, ``_row_end`` — служебные позиции фактора и строк витрины.
    Строки упорядочены по фактору, затем по дате отчёта.
    """
    values = np.asarray(values, dtype=float)
    is_daily = np.array([f == "д" for f in freqs], dtype=bool)
    rep = report_mask(values, is_daily, value_tol)

    # Непустые ячейки в порядке «фактор, затем строка». Первая непустая ячейка каждого
    # фактора всегда дата отчёта, поэтому блоки не переходят через границу факторов.
    fcol, row = np.nonzero(~np.isnan(values.T))
    starts = np.flatnonzero(rep.T[fcol, row])
    ends = np.r_[starts[1:] - 1, len(row) - 1] if starts.size else starts

    f_idx = fcol[starts]
    r0, r1 = row[starts], row[ends]
    date_values = dates.values.astype("datetime64[ns]")
    report_date = date_values[r0]
    lag_days = np.asarray(lags, dtype="int64")[f_idx]
    return pd.DataFrame({
        "factor": pd.Categorical.from_codes(f_idx, categories=list(factors)),
        "report_date": report_date,
        "block_end": date_values[r1],
        "available_date": report_date + lag_days * DAY,
        "value": values[r0, f_idx],
        "_fidx": f_idx.astype("int64"),
        "_row": r0.astype("int64"),
        "_row_end": r1.astype("int64"),
    })


def _factor_major(n_dates: int, factors: Sequence[str]) -> pd.Categorical:
    codes = np.repeat(np.arange(len(factors)), n_dates)
    return pd.Categorical.from_codes(codes, categories=list(factors))


def raw_full(dates: pd.DatetimeIndex, values: np.ndarray, factors: Sequence[str], date_col: str) -> pd.DataFrame:
    """Полная плоская витрина: ``date_col``, ``factor``, ``value`` для всех пар дата × фактор."""
    n = len(dates)
    return pd.DataFrame({
        date_col: np.tile(dates.values, len(factors)),
        "factor": _factor_major(n, factors),
        "value": np.asarray(values, dtype=float).T.ravel(),
    })


def journal_full(
    dates: pd.DatetimeIndex, values: np.ndarray, pubs: pd.DataFrame, factors: Sequence[str], date_col: str
) -> pd.DataFrame:
    """Полный плоский журнал доступности (раздел 6.2 ТЗ).

    Строки блока получают значение и ``available_date`` своего отчёта. Пустые строки после
    блока — протянутые значение и ``available_date`` последнего отчёта, ``is_filled=True``
    (хвост витрины, сбой загрузки, промежутки разреженной раскладки). До первого отчёта
    оба поля пустые.

    Смысл ``available_date`` различается: у собственной строки (``is_filled=False``) это дата,
    когда стало известно значение этой строки; у протянутой — дата, когда стало известно
    последнее значение, которое можно подставить на эту дату.
    """
    values = np.asarray(values, dtype=float)
    n, nf = values.shape
    # Номер последнего отчёта не позже строки. Номера публикаций внутри фактора растут
    # вместе со строкой, поэтому «протягивание» — это накопленный максимум по строкам.
    pid = np.full((n, nf), -1, dtype="int64")
    pid[pubs["_row"].to_numpy(), pubs["_fidx"].to_numpy()] = np.arange(len(pubs))
    pid = np.maximum.accumulate(pid, axis=0).T.ravel()
    has = pid >= 0
    safe = np.where(has, pid, 0)
    pub_value = pubs["value"].to_numpy(dtype=float)
    pub_avail = pubs["available_date"].to_numpy(dtype="datetime64[ns]")
    if len(pubs):
        value = np.where(has, pub_value[safe], np.nan)
        avail = np.where(has, pub_avail[safe], np.datetime64("NaT"))
    else:
        value = np.full(pid.shape, np.nan)
        avail = np.full(pid.shape, np.datetime64("NaT"), dtype="datetime64[ns]")
    is_filled = has & np.isnan(values.T.ravel())
    return pd.DataFrame({
        date_col: np.tile(dates.values, nf),
        "factor": _factor_major(n, factors),
        "available_date": avail.astype("datetime64[ns]"),
        "value": value,
        "is_filled": is_filled,
    })


def make_calendar(
    dates: pd.DatetimeIndex, snapshot: pd.Timestamp, kind: str
) -> pd.DatetimeIndex:
    """Календарь таблицы известных значений: даты витрины (``vitrine``) или все дни (``daily``)
    от первой даты витрины до ``snapshot`` включительно. Если снимок позже последней даты
    витрины, дата снимка добавляется в календарь, чтобы последняя строка была признаками на снимок.
    """
    if len(dates) == 0:
        return pd.DatetimeIndex([], name=None)
    if kind == "daily":
        return pd.date_range(dates[0], snapshot, freq="D")
    cal = dates[dates <= snapshot]
    if snapshot >= dates[0] and (len(cal) == 0 or cal[-1] < snapshot):
        cal = cal.append(pd.DatetimeIndex([snapshot]))
    return cal


def known_full(
    calendar: pd.DatetimeIndex,
    pubs: pd.DataFrame,
    factors: Sequence[str],
    date_col: str,
    ffill_limit: int = None,
    drop_leading_empty: bool = False,
) -> pd.DataFrame:
    """Полная плоская таблица известных значений (раздел 6.3 ТЗ).

    На дату t берётся значение отчёта с наибольшей ``available_date <= t``; при равных
    датах доступности — с более поздней ``report_date``. Реализовано через
    ``pd.merge_asof(..., direction='backward')``, а не через ``shift``: as-of join корректен,
    даже если ``available_date`` попадает на дату, которой нет в календаре, — значение
    появится в ближайшую следующую дату. Протягивание встроено в as-of join и не ограничено,
    если не задан ``ffill_limit`` (дни от ``available_date``).
    """
    nf, nc = len(factors), len(calendar)
    left = pd.DataFrame({
        "_date": np.repeat(calendar.values.astype("datetime64[ns]"), nf),
        "_fidx": np.tile(np.arange(nf, dtype="int64"), nc),
    })
    right = (
        pubs[["available_date", "report_date", "value", "_fidx"]]
        .sort_values(["available_date", "report_date"], kind="mergesort")
        .reset_index(drop=True)
    )
    if len(right):
        merged = pd.merge_asof(left, right, left_on="_date", right_on="available_date",
                               by="_fidx", direction="backward")
        value = merged["value"].to_numpy(dtype=float)
        avail = merged["available_date"].to_numpy(dtype="datetime64[ns]")
    else:
        value = np.full(len(left), np.nan)
        avail = np.full(len(left), np.datetime64("NaT"), dtype="datetime64[ns]")

    # merge_asof сохраняет порядок left («дата, затем фактор») — переводим в «фактор, затем дата».
    perm = (np.arange(nc)[None, :] * nf + np.arange(nf)[:, None]).ravel()
    value, avail = value[perm], avail[perm]
    date_arr = np.tile(calendar.values.astype("datetime64[ns]"), nf)

    if ffill_limit is not None:
        too_old = ~np.isnat(avail) & ((date_arr - avail) > ffill_limit * DAY)
        value = np.where(too_old, np.nan, value)
        avail = np.where(too_old, np.datetime64("NaT"), avail)

    out = pd.DataFrame({
        date_col: date_arr,
        "factor": _factor_major(nc, factors),
        "value": value,
        "available_date": avail.astype("datetime64[ns]"),
    })
    if drop_leading_empty and nc:
        known_dates = out.loc[out["value"].notna(), date_col]
        first = known_dates.min() if len(known_dates) else None
        out = out[out[date_col] >= first] if first is not None else out.iloc[0:0]
        out = out.reset_index(drop=True)
    return out


def check_no_leak(known: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Главный инвариант: в строке даты t стоит только значение, ставшее доступным не позже t.

    Это прямое определение отсутствия утечки будущего, а не эвристика: все остальные
    проверки зависят от допущений о данных, эта — нет. Поэтому её нарушение означает
    ошибку в коде модуля, а не в данных, и построение останавливается.
    Возвращает нарушающие строки (пустой DataFrame, если утечки нет).
    """
    bad = known["available_date"].notna() & (known["available_date"] > known[date_col])
    bad |= known["value"].notna() & known["available_date"].isna()
    return known[bad]


def long_to_wide(
    full: pd.DataFrame,
    date_col: str,
    factors: Sequence[str],
    columns: Dict[str, str],
) -> pd.DataFrame:
    """Широкий формат как ``pivot`` полного плоского представления.

    ``columns`` — соответствие «колонка плоской таблицы → суффикс имени в широкой».
    Колонки идут группами по фактору в порядке master (пара «значение — дата доступности»
    стоит рядом).
    """
    index = pd.DatetimeIndex(pd.unique(full[date_col]), name=date_col)
    blocks: List[pd.DataFrame] = []
    for col, suffix in columns.items():
        w = full.pivot(index=date_col, columns="factor", values=col)
        w.columns = [str(c) for c in w.columns]
        w = w.reindex(index=index, columns=list(factors))
        w.columns = [f + suffix for f in factors]
        blocks.append(w)
    wide = pd.concat(blocks, axis=1)
    order = [f + suffix for f in factors for suffix in columns.values()]
    wide = wide[order]
    wide.index.name = date_col
    return wide.reset_index()


def full_to_long(full: pd.DataFrame, keep_empty: bool) -> pd.DataFrame:
    """Плоский формат для выдачи: без строк с пустым значением (если не ``keep_empty``),
    имя фактора — обычная строка."""
    out = full if keep_empty else full[full["value"].notna()]
    out = out.reset_index(drop=True)
    out["factor"] = out["factor"].astype(str)
    return out
