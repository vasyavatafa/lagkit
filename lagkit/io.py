"""Чтение master-файла и витрины (csv/xlsx), запись результатов."""
from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .config import LagConfig, LagkitError, normalize_frequency

logger = logging.getLogger("lagkit")

Source = Union[str, Path, pd.DataFrame]

MASTER_COLUMNS = ("factor", "data_frequency", "object_lag")
XLSX_MAX_ROWS = 1_048_576
XLSX_MAX_COLS = 16_384


# --- общее чтение таблиц ----------------------------------------------------
def _detect_encoding(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "cp1251"


def _detect_sep(path: Path, encoding: str) -> str:
    with path.open("r", encoding=encoding, errors="replace") as fh:
        header = fh.readline()
    counts = {sep: header.count(sep) for sep in (";", ",", "\t")}
    best = max(counts, key=lambda s: counts[s])
    return best if counts[best] > 0 else ","


def read_table(
    source: Source,
    *,
    sheet: Union[str, int] = 0,
    sep: Optional[str] = None,
    encoding: Optional[str] = None,
    dtype: Any = None,
    what: str = "таблица",
) -> pd.DataFrame:
    """Читает csv/xlsx по расширению или возвращает копию переданного DataFrame.

    Для csv разделитель (``,``/``;``/табуляция) и кодировка (utf-8 с откатом на cp1251)
    определяются автоматически, если не заданы явно.
    """
    if isinstance(source, pd.DataFrame):
        return source.copy()
    path = Path(source)
    if not path.exists():
        raise LagkitError(f"{what}: файл не найден: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        enc = encoding or _detect_encoding(path)
        delimiter = sep or _detect_sep(path, enc)
        try:
            return pd.read_csv(path, sep=delimiter, encoding=enc, dtype=dtype)
        except UnicodeDecodeError as exc:
            raise LagkitError(f"{what}: не удалось прочитать {path} в кодировке {enc}: {exc}") from exc
    if suffix in (".xlsx", ".xlsm", ".xls"):
        if importlib.util.find_spec("openpyxl") is None and suffix != ".xls":
            raise LagkitError("Для чтения .xlsx нужен пакет openpyxl (pip install openpyxl)")
        return pd.read_excel(path, sheet_name=sheet, dtype=dtype)
    if suffix == ".parquet":
        _require_parquet()
        return pd.read_parquet(path)
    raise LagkitError(f"{what}: неподдерживаемый формат {suffix!r} (ожидается .csv или .xlsx)")


# --- master -------------------------------------------------------------------
def read_master(source: Source, cfg: LagConfig) -> pd.DataFrame:
    """Читает и нормализует master-файл.

    Возвращает DataFrame с колонками ``factor``, ``data_frequency`` (``м``/``н``/``д``)
    и ``object_lag`` (сырое значение, разбирается в :mod:`lagkit.lags`). Лишние колонки
    отбрасываются. Неизвестная частотность, пустое имя фактора и дубликаты — ошибка.
    """
    df = read_table(
        source, sheet=cfg.master_sheet, sep=cfg.csv_sep, encoding=cfg.csv_encoding,
        dtype=object, what="master-файл",
    )
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in MASTER_COLUMNS if c not in df.columns]
    if missing:
        raise LagkitError(f"В master-файле нет обязательных колонок: {missing}")
    df = df[list(MASTER_COLUMNS)].copy()
    # Полностью пустые строки (частый хвост Excel) пропускаем.
    df = df[~df.apply(lambda c: c.map(_is_blank)).all(axis=1)]
    rownum = df.index.to_series() + 2  # номер строки в файле с учётом заголовка

    df["factor"] = df["factor"].map(lambda x: None if _is_blank(x) else str(x).strip())
    empty = df["factor"].isna()
    if empty.any():
        raise LagkitError(f"В master-файле пустое имя фактора в строках: {rownum[empty].tolist()}")

    freq = df["data_frequency"].map(normalize_frequency)
    bad = freq.isna()
    if bad.any():
        rows = [f"строка {r}: {f!r} -> {v!r}" for r, f, v in
                zip(rownum[bad], df.loc[bad, "factor"], df.loc[bad, "data_frequency"])]
        raise LagkitError(
            "Недопустимая частотность data_frequency (ожидается м/н/д или m/w/d): " + "; ".join(rows)
        )
    df["data_frequency"] = freq

    dup = df["factor"].duplicated(keep=False)
    if dup.any():
        rows = [f"{f!r} (строка {r})" for r, f in zip(rownum[dup], df.loc[dup, "factor"])]
        raise LagkitError("Дубликаты factor в master-файле: " + ", ".join(rows))
    return df.reset_index(drop=True)


def _is_blank(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    return isinstance(x, str) and x.strip() == ""


# --- витрина ------------------------------------------------------------------
def to_numeric_clean(s: pd.Series) -> Tuple[pd.Series, int]:
    """Приводит колонку к float, понимая ``"1 234,5"``.

    Возвращает ряд и число непустых ячеек, которые не удалось привести к числу.
    """
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float), 0
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float), 0
    txt = s.astype(str).str.strip()
    empty = s.isna() | txt.eq("")
    cleaned = (
        txt.str.replace(r"[\s  ']", "", regex=True)
        .str.replace(",", ".", regex=False)
        .where(~empty)
    )
    num = pd.to_numeric(cleaned, errors="coerce").astype(float)
    failed = int((num.isna() & ~empty).sum())
    return num, failed


def parse_dates(s: pd.Series, date_format: Optional[str]) -> pd.Series:
    """Разбирает колонку дат: явный формат или dayfirst (ДД.ММ.ГГГГ); время отбрасывается."""
    if pd.api.types.is_datetime64_any_dtype(s):
        out = s
    elif date_format:
        out = pd.to_datetime(s, format=date_format, errors="coerce")
    else:
        txt = s.astype(str).str.strip()
        iso = txt.str.match(r"^\d{4}-\d{1,2}-\d{1,2}")
        out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
        if iso.any():
            out[iso] = pd.to_datetime(txt[iso], errors="coerce", format="ISO8601") \
                if _pandas_ge_2() else pd.to_datetime(txt[iso], errors="coerce")
        if (~iso).any():
            out[~iso] = pd.to_datetime(s[~iso], dayfirst=True, errors="coerce")
    out = pd.to_datetime(out)
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_localize(None)
    return out.dt.normalize().astype("datetime64[ns]")


def _pandas_ge_2() -> bool:
    return int(pd.__version__.split(".")[0]) >= 2


def read_data(
    source: Source, cfg: LagConfig, factors: List[str]
) -> Tuple[pd.DataFrame, List[str], List[Dict[str, Any]]]:
    """Читает витрину и оставляет только ``[date_col] + factors`` (в порядке master).

    Возвращает очищенную таблицу (даты отсортированы и уникальны, значения — float),
    список присутствующих факторов и список проблем входных данных для диагностики.
    """
    df = read_table(
        source, sheet=cfg.data_sheet, sep=cfg.csv_sep, encoding=cfg.csv_encoding,
        dtype={cfg.date_col: object} if not isinstance(source, pd.DataFrame) else None,
        what="витрина",
    )
    df.columns = [str(c).strip() for c in df.columns]
    if cfg.date_col not in df.columns:
        raise LagkitError(f"В витрине нет колонки даты {cfg.date_col!r}")
    dup_cols = sorted({c for c in df.columns if list(df.columns).count(c) > 1})
    if dup_cols:
        raise LagkitError(f"В витрине повторяются имена колонок: {dup_cols}")

    issues: List[Dict[str, Any]] = []
    present = [f for f in factors if f in df.columns and f != cfg.date_col]
    absent = [f for f in factors if f not in present]
    for f in absent:
        issues.append(dict(factor=f, check="missing_factor", severity="warning",
                           message="Фактор из master отсутствует в витрине и исключён", details=""))
    if absent and cfg.strict:
        raise LagkitError(f"strict=True: факторов из master нет в витрине: {absent}")

    df = df[[cfg.date_col] + present]
    # Полностью пустые строки (хвост выгрузки) удаляем молча.
    blank_date = df[cfg.date_col].map(_is_blank)
    all_blank = blank_date & df[present].isna().all(axis=1) if present else blank_date
    df = df[~all_blank]

    dates = parse_dates(df[cfg.date_col], cfg.date_format)
    bad = dates.isna()
    if bad.any():
        sample = df.loc[bad, cfg.date_col].head(10).tolist()
        raise LagkitError(
            f"Не удалось разобрать {int(bad.sum())} дат в колонке {cfg.date_col!r} "
            f"(формат {cfg.date_format or 'ДД.ММ.ГГГГ, dayfirst'}), примеры: {sample}"
        )
    dup = dates.duplicated(keep=False)
    if dup.any():
        sample = sorted({d.strftime("%Y-%m-%d") for d in dates[dup]})[:10]
        raise LagkitError(f"В витрине повторяются даты ({int(dup.sum())} строк), например: {sample}")

    out = pd.DataFrame({cfg.date_col: dates.values})
    numeric = {}
    for f in present:
        num, failed = to_numeric_clean(df[f])
        numeric[f] = num.values
        if failed:
            issues.append(dict(factor=f, check="non_numeric", severity="warning",
                               message=f"{failed} непустых ячеек не приводятся к числу и заменены пустыми",
                               details=f"failed={failed}"))
        if np.isnan(numeric[f]).all():
            issues.append(dict(factor=f, check="empty_factor", severity="warning",
                               message="Фактор в витрине пуст (нет ни одного числового значения)", details=""))
    out = pd.concat([out, pd.DataFrame(numeric, columns=present)], axis=1)
    out = out.sort_values(cfg.date_col, kind="mergesort").reset_index(drop=True)

    if cfg.strict:
        fatal = [i for i in issues if i["check"] in ("non_numeric", "empty_factor")]
        if fatal:
            raise LagkitError("strict=True: проблемы входных данных: " +
                              "; ".join(f"{i['factor']}: {i['message']}" for i in fatal))
    return out, present, issues


# --- запись ---------------------------------------------------------------------
def _require_parquet() -> None:
    if importlib.util.find_spec("pyarrow") is None and importlib.util.find_spec("fastparquet") is None:
        raise LagkitError(
            "Для формата parquet нужен пакет pyarrow (pip install pyarrow); "
            "либо используйте output_format='csv'"
        )


def write_frame(df: pd.DataFrame, path: Union[str, Path], fmt: str) -> Path:
    """Пишет таблицу в parquet/csv/xlsx; путь дополняется расширением формата.

    Для xlsx превышение лимита листа — ошибка (данные не обрезаются); csv пишется в utf-8
    с датами в ISO-формате ГГГГ-ММ-ДД.
    """
    path = Path(path).with_suffix("." + fmt)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        _require_parquet()
        df.to_parquet(path, index=False)
    elif fmt == "csv":
        df.to_csv(path, index=False, encoding="utf-8", date_format="%Y-%m-%d")
    elif fmt == "xlsx":
        if len(df) + 1 > XLSX_MAX_ROWS or df.shape[1] > XLSX_MAX_COLS:
            raise LagkitError(
                f"Таблица {path.name} ({len(df)} строк × {df.shape[1]} колонок) не помещается на лист xlsx "
                f"(максимум {XLSX_MAX_ROWS} строк и {XLSX_MAX_COLS} колонок). "
                "Используйте output_format='parquet' или 'csv', либо широкий формат"
            )
        if importlib.util.find_spec("openpyxl") is None:
            raise LagkitError("Для записи .xlsx нужен пакет openpyxl (pip install openpyxl)")
        df.to_excel(path, index=False)
    else:
        raise LagkitError(f"Неподдерживаемый формат вывода {fmt!r}")
    return path
