"""Поиск дат отчёта и отчётных блоков, определение раскладки фактора.

Правило поиска дат отчёта (раздел 4.3 ТЗ) не зависит от раскладки:

* для ``м`` и ``н`` датой отчёта считается непустая строка, у которой предыдущая строка
  пустая или значение отличается от предыдущего (больше чем на ``value_tol``);
* для ``д`` каждая непустая строка — отдельный отчёт, даже если значение не изменилось.

Ограничение: если в плотной раскладке новый отчёт м/н совпал по значению с предыдущим,
два блока склеиваются и дата второго отчёта теряется. Значения в таблице известных
значений от этого не меняются (число то же), но ``available_date`` в журнале и счётчик
пропущенных отчётов для такого блока будут неточны.

Раскладка (``dense``/``sparse``/``daily``/``mixed``/``empty``) определяется только для отчёта
и диагностики и в расчёте значений не участвует.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .config import DiagnosticsConfig


def report_mask(values: np.ndarray, is_daily: np.ndarray, value_tol: float = 0.0) -> np.ndarray:
    """Матрица (даты × факторы): True там, где строка — дата отчёта.

    Векторизовано по всем колонкам сразу (сравнение со сдвинутой на строку матрицей),
    без цикла по строкам.
    """
    values = np.asarray(values, dtype=float)
    notna = ~np.isnan(values)
    prev = np.empty_like(values)
    prev[:1] = np.nan
    prev[1:] = values[:-1]
    with np.errstate(invalid="ignore"):
        if value_tol > 0:
            changed = ~(np.abs(values - prev) <= value_tol)  # NaN в prev -> True
        else:
            changed = values != prev  # NaN в prev -> True
    return notna & (changed | np.asarray(is_daily, dtype=bool)[None, :])


def detect_layouts(values: np.ndarray, freqs: Sequence[str], cfg: DiagnosticsConfig) -> List[str]:
    """Определяет раскладку каждого фактора.

    На интервале от первой до последней непустой строки считаются серии подряд идущих
    непустых строк. Медианная длина серии 1 при частотности ``м``/``н`` — ``sparse``,
    иначе ``dense``; для ``д`` — ``daily``. Если доля одиночных значений (серий длины 1)
    в первой и второй половине истории различается больше чем на ``mixed_share_diff`` —
    ``mixed`` (строки между сериями относятся к типу предыдущей серии). Пустой фактор — ``empty``.
    """
    values = np.asarray(values, dtype=float)
    notna = ~np.isnan(values)
    result: List[str] = []
    for j, freq in enumerate(freqs):  # цикл по колонкам, не по строкам
        idx = np.flatnonzero(notna[:, j])
        if idx.size == 0:
            result.append("empty")
            continue
        if freq == "д":
            result.append("daily")
            continue
        breaks = np.flatnonzero(np.diff(idx) != 1)
        starts = np.r_[idx[0], idx[breaks + 1]]
        ends = np.r_[idx[breaks], idx[-1]]
        lengths = ends - starts + 1
        layout = "sparse" if np.median(lengths) <= 1 else "dense"

        # Каждая строка интервала получает тип последней начавшейся серии: «разреженный»
        # (серия длины 1) или «плотный». Сравниваются доли разреженных строк в половинах.
        span = np.diff(np.r_[starts, idx[-1] + 1])
        sparse_rows = np.repeat(lengths == 1, span)
        mid = (idx[-1] - idx[0]) / 2
        pos = np.arange(sparse_rows.size)
        first, second = pos <= mid, pos > mid
        cells_first = np.count_nonzero(idx - idx[0] <= mid)
        cells_second = idx.size - cells_first
        if cells_first >= cfg.mixed_min_cells and cells_second >= cfg.mixed_min_cells:
            if abs(sparse_rows[first].mean() - sparse_rows[second].mean()) > cfg.mixed_share_diff:
                layout = "mixed"
        result.append(layout)
    return result
