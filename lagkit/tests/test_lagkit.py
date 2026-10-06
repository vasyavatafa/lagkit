"""Тесты и критерии приёмки (раздел 10 ТЗ). Все данные синтетические, создаются в коде."""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lagkit import LagConfig, LagkitError, LagProcessor, build_views
from lagkit import io as lk_io
from lagkit.lags import parse_object_lag

PKG_DIR = Path(__file__).resolve().parents[1]
HAS_XLSX = importlib.util.find_spec("openpyxl") is not None


def D(s: str) -> pd.Timestamp:
    """Дата в формате ДД.ММ.ГГГГ."""
    return pd.Timestamp(pd.to_datetime(s, dayfirst=True))


def master_df(rows):
    return pd.DataFrame(rows, columns=["factor", "data_frequency", "object_lag"])


def diag_of(lp: LagProcessor, check: str) -> pd.DataFrame:
    d = lp.diagnostics()
    return d[d["check"] == check]


# --- 1. Приложенный пример буквально -------------------------------------------
def example_processor(**cfg) -> LagProcessor:
    data = pd.DataFrame({
        "DATE": ["01.01.2026", "02.01.2026", "03.01.2026", "04.01.2026", "05.01.2026", "06.01.2026"],
        "f": [1, 1, 2, 2, None, None],
    })
    return LagProcessor(master_df([("f", "м", 2)]), data, cfg)


def test_example_literally():
    lp = example_processor(drop_leading_empty=True)
    j = lp.journal("long").set_index("DATE")
    assert j["value"].tolist() == [1, 1, 2, 2, 2, 2]
    assert j["available_date"].tolist() == [D("03.01.2026")] * 2 + [D("05.01.2026")] * 4
    assert j["is_filled"].tolist() == [False, False, False, False, True, True]
    assert j.loc[D("02.01.2026"), "available_date"] == D("03.01.2026")

    jw = lp.journal("wide")
    assert list(jw.columns) == ["DATE", "f", "f__available_date"]

    k = lp.known("wide")
    assert k["DATE"].tolist() == [D("03.01.2026"), D("04.01.2026"), D("05.01.2026"), D("06.01.2026")]
    assert k["f"].tolist() == [1, 1, 2, 2]

    # По умолчанию ведущие пустые строки сохраняются.
    k_all = example_processor().known("wide")
    assert len(k_all) == 6 and k_all["f"].iloc[:2].isna().all()

    t = lp.trace("f")
    assert list(t.columns) == ["DATE", "raw", "journal", "available_date", "is_filled", "known"]


def test_known_date_col_rename():
    k = example_processor(known_date_col="available_date").known("long")
    assert "available_date" in k.columns  # дата строки; available_date значения — отдельно не дублируется
    lp = example_processor(known_date_col="DT")
    assert list(lp.known("long").columns) == ["DT", "factor", "value", "available_date"]


# --- 2. Эквивалентность построчному сдвигу --------------------------------------
def random_dense(rng: np.random.Generator, n: int, freq: str) -> np.ndarray:
    period = {"м": 30, "н": 7, "д": 1}[freq]
    lengths = rng.integers(max(1, period - 3), period + 4, size=n)
    vals = np.repeat(rng.integers(0, 4, size=n).astype(float), lengths)[:n]  # повторы значений намеренно
    start = int(rng.integers(0, 40))
    vals[:start] = np.nan
    return vals


@pytest.mark.parametrize("seed", range(5))
def test_equivalent_to_row_shift(seed):
    rng = np.random.default_rng(seed)
    n = 400
    dates = pd.date_range("2020-01-01", periods=n)
    freqs = ["м", "н", "д", "м", "н", "д"]
    lags = [int(x) for x in rng.integers(0, 45, size=len(freqs))]
    data = pd.DataFrame({"DATE": dates})
    for i, f in enumerate(freqs):
        data[f"x{i}"] = random_dense(rng, n, f)
    master = master_df([(f"x{i}", f, lags[i]) for i, f in enumerate(freqs)])
    known = LagProcessor(master, data).known("wide")
    assert known["DATE"].tolist() == dates.tolist()
    for i in range(len(freqs)):
        expected = data[f"x{i}"].shift(lags[i])
        pd.testing.assert_series_equal(known[f"x{i}"], expected, check_names=False)


# --- 3. Разреженная раскладка ------------------------------------------------------
def test_sparse_layout():
    dates = pd.date_range("2026-01-01", "2026-05-31")
    data = pd.DataFrame({"DATE": dates, "m": np.nan})
    reports = {"30.01.2026": 10.0, "27.02.2026": 11.0, "30.03.2026": 12.0, "29.04.2026": 13.0}
    for d, v in reports.items():
        data.loc[data["DATE"] == D(d), "m"] = v
    lp = LagProcessor(master_df([("m", "м", 30)]), data)
    assert lp.lags.set_index("factor").at["m", "layout"] == "sparse"

    k = lp.known("wide").set_index("DATE")["m"]
    assert k.first_valid_index() == D("01.03.2026")
    assert k[D("01.03.2026")] == 10.0
    assert k[D("29.03.2026")] == 11.0  # 27.02 + 30 = 29.03

    j = lp.journal("long").set_index("DATE")
    between = j.loc[D("31.01.2026"):D("26.02.2026")]
    assert between["is_filled"].all()
    assert (between["value"] == 10.0).all()
    assert (between["available_date"] == D("01.03.2026")).all()
    assert not j.loc[D("30.01.2026"), "is_filled"]


# --- 4. Пропуски отчётов -------------------------------------------------------------
def test_missing_reports():
    dates = pd.date_range("2026-01-05", periods=10 * 7)
    week = np.arange(len(dates)) // 7
    vals = (week + 1).astype(float)
    vals[(week == 4) | (week == 5)] = np.nan  # два пропущенных отчёта подряд
    data = pd.DataFrame({"DATE": dates, "w": vals})
    lp = LagProcessor(master_df([("w", "н", 7)]), data)

    k = lp.known("wide").set_index("DATE")["w"]
    gap_known = k[(dates[4 * 7] + pd.Timedelta(days=7)):(dates[6 * 7] + pd.Timedelta(days=6))]
    assert (gap_known == 4.0).all() and len(gap_known) == 14  # протянуто через весь пропуск

    d = diag_of(lp, "missing_reports")
    assert len(d) == 1 and d["severity"].iloc[0] == "warning"
    details = json.loads(d["details"].iloc[0])
    assert [g["missed"] for g in details] == [2]
    assert "пропущено 2" in d["message"].iloc[0]


# --- 5. Дневной ряд с повторами ----------------------------------------------------------
def test_daily_repeats():
    data = pd.DataFrame({"DATE": pd.date_range("2026-01-01", periods=3), "d": [7.5, 7.5, 7.5]})
    lp = LagProcessor(master_df([("d", "д", 1)]), data)
    j = lp.journal("long").set_index("DATE")
    assert j.loc[D("03.01.2026"), "available_date"] == D("04.01.2026")
    assert j["available_date"].tolist() == [D("02.01.2026"), D("03.01.2026"), D("04.01.2026")]
    assert len(lp.publications) == 3
    assert lp.lags.at[0, "layout"] == "daily"


# --- 6. Дыры в календаре ---------------------------------------------------------------------
def test_calendar_holes():
    dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05", "2026-01-06", "2026-01-09"])
    data = pd.DataFrame({"DATE": dates, "m": [1.0, np.nan, np.nan, 2.0, np.nan]})
    lp = LagProcessor(master_df([("m", "м", 3)]), data)  # 01.01 + 3 = 04.01 — такой даты нет
    k = lp.known("long").set_index("DATE")
    assert k.index[0] == D("05.01.2026")
    assert k.loc[D("05.01.2026"), "value"] == 1.0
    assert k.loc[D("05.01.2026"), "available_date"] == D("04.01.2026")
    assert k.loc[D("09.01.2026"), "value"] == 2.0  # 06.01 + 3 = 09.01

    daily = LagProcessor(master_df([("m", "м", 3)]), data, {"calendar": "daily"}).known("wide")
    assert len(daily) == 9
    assert daily.set_index("DATE").loc[D("04.01.2026"), "m"] == 1.0


# --- 7. Лаги -------------------------------------------------------------------------------------------
def test_default_lags():
    dates = pd.date_range("2026-01-01", periods=60)
    data = pd.DataFrame({"DATE": dates, "a": 1.0, "b": 1.0, "c": np.arange(60.0)})
    lp = LagProcessor(master_df([("a", "м", None), ("b", "н", ""), ("c", "д", np.nan)]), data)
    lags = lp.lags.set_index("factor")
    assert lags["object_lag"].tolist() == [30, 7, 1]
    assert (lags["lag_source"] == "default").all()

    lp2 = LagProcessor(master_df([("a", "m", 5)]), data, {"default_lags": {"m": 31}})
    assert lp2.lags.at[0, "lag_source"] == "master" and lp2.lags.at[0, "object_lag"] == 5
    lp3 = LagProcessor(master_df([("a", "m", None)]), data, {"default_lags": {"m": 31}})
    assert lp3.lags.at[0, "object_lag"] == 31


@pytest.mark.parametrize("value", [2, 2.0, "2", " 2.0 ", np.int64(2)])
def test_lag_accepted(value):
    assert parse_object_lag(value, "f") == 2


@pytest.mark.parametrize("value", [1.5, -1, "abc", "1,5", True])
def test_lag_rejected(value):
    with pytest.raises(LagkitError, match="object_lag"):
        parse_object_lag(value, "f")


def test_lag_rejected_in_processor_names_factor():
    data = pd.DataFrame({"DATE": pd.date_range("2026-01-01", periods=5), "bad_factor": 1.0})
    lp = LagProcessor(master_df([("bad_factor", "м", "abc")]), data)
    with pytest.raises(LagkitError, match="bad_factor"):
        lp.lags


def test_zero_lag_warning():
    data = pd.DataFrame({"DATE": pd.date_range("2026-01-01", periods=5), "z": 1.0})
    lp = LagProcessor(master_df([("z", "д", 0)]), data)
    d = diag_of(lp, "zero_lag")
    assert len(d) == 1 and d["severity"].iloc[0] == "warning"
    assert lp.lags.at[0, "object_lag"] == 0


def test_snapshot_checks():
    dates = pd.date_range("2025-01-01", "2026-03-31")
    # Месячный разреженный ряд с отчётами на 1-е число; последний — 01.03.2026.
    data = pd.DataFrame({"DATE": dates, "small": np.nan, "big": np.nan, "old": np.nan})
    month_starts = dates[dates.day == 1]
    data.loc[data["DATE"].isin(month_starts), "small"] = np.arange(len(month_starts), dtype=float)
    data.loc[data["DATE"].isin(month_starts), "big"] = np.arange(len(month_starts), dtype=float)
    data.loc[data["DATE"].isin(month_starts[:3]), "old"] = 1.0
    lp = LagProcessor(master_df([("small", "м", 1), ("big", "м", 45), ("old", "м", 30)]), data)
    # T=31.03, r_last=01.03: 0 < lag <= 30
    assert diag_of(lp, "lag_below_snapshot_bound")["factor"].tolist() == []
    assert diag_of(lp, "lag_above_snapshot_bound")["factor"].tolist() == ["big"]
    assert diag_of(lp, "stale")["factor"].tolist() == ["old"]

    lp2 = LagProcessor(master_df([("small", "м", 1)]), data[["DATE", "small"]], {"snapshot_date": "2026-04-15"})
    # T=15.04, r_last=01.03: 14 < lag <= 45 -> лаг 1 слишком мал
    assert diag_of(lp2, "lag_below_snapshot_bound")["factor"].tolist() == ["small"]
    assert lp2.known("wide")["DATE"].iloc[-1] == D("15.04.2026")


def test_lags_override(tmp_path):
    data = pd.DataFrame({"DATE": pd.date_range("2026-01-01", periods=40), "a": 1.0, "b": 2.0})
    lp = LagProcessor(master_df([("a", "м", 3), ("b", "н", None)]), data)
    path = lp.save(view="known", output_dir=tmp_path, output_format="csv")["lags"]
    # master изменился, но лаги заморожены
    lp2 = LagProcessor(master_df([("a", "м", 10), ("b", "н", 1)]), data, {"lags_override": str(path)})
    lags = lp2.lags.set_index("factor")
    assert lags["object_lag"].tolist() == [3, 7]
    assert lags["lag_source"].tolist() == ["master", "default"]


# --- 8. Определение раскладки ------------------------------------------------------------------------
def test_layout_detection_and_values_independent():
    rng = np.random.default_rng(1)
    dates = pd.date_range("2024-01-01", "2025-12-31")
    month_start = (dates.day == 1)
    month_vals = rng.normal(size=month_start.sum())
    sparse = np.full(len(dates), np.nan)
    sparse[month_start] = month_vals
    dense = pd.Series(sparse).ffill().to_numpy()
    half = len(dates) // 2
    mixed = np.where(np.arange(len(dates)) < half, dense, sparse)
    daily = rng.normal(size=len(dates))
    data = pd.DataFrame({"DATE": dates, "dense": dense, "sparse": sparse, "mixed": mixed, "daily": daily})
    master = master_df([("dense", "м", 20), ("sparse", "м", 20), ("mixed", "м", 20), ("daily", "д", None)])
    lp = LagProcessor(master, data)
    layouts = lp.lags.set_index("factor")["layout"].to_dict()
    assert layouts == {"dense": "dense", "sparse": "sparse", "mixed": "mixed", "daily": "daily"}
    assert diag_of(lp, "mixed_layout")["factor"].tolist() == ["mixed"]

    k = lp.known("wide")
    pd.testing.assert_series_equal(k["dense"], k["sparse"], check_names=False)
    pd.testing.assert_series_equal(k["dense"], k["mixed"], check_names=False)


def test_frequency_diagnostics():
    dates = pd.date_range("2025-01-01", periods=200)
    rng = np.random.default_rng(2)
    data = pd.DataFrame({
        "DATE": dates,
        "jitter": rng.normal(size=200),                   # указан как месячный, а меняется каждый день
        "slow": np.repeat(np.arange(7.0), 30)[:200],      # указан как дневной, а меняется раз в месяц
    })
    lp = LagProcessor(master_df([("jitter", "м", None), ("slow", "д", None)]), data)
    assert diag_of(lp, "frequency_too_high")["factor"].tolist() == ["jitter"]
    assert diag_of(lp, "daily_rarely_changes")["factor"].tolist() == ["slow"]


def test_value_tol():
    data = pd.DataFrame({"DATE": pd.date_range("2026-01-01", periods=4), "f": [1.0, 1.0 + 1e-12, 2.0, 2.0]})
    assert len(LagProcessor(master_df([("f", "м", 2)]), data).publications) == 3
    assert len(LagProcessor(master_df([("f", "м", 2)]), data, {"value_tol": 1e-9}).publications) == 2


# --- 9. Согласованность форматов -------------------------------------------------------------------------
@pytest.mark.parametrize("keep_empty", [False, True])
def test_wide_is_pivot_of_long(keep_empty):
    rng = np.random.default_rng(3)
    dates = pd.date_range("2025-01-01", periods=120)
    data = pd.DataFrame({"DATE": dates})
    for i, f in enumerate(["м", "н", "д"]):
        data[f"x{i}"] = random_dense(rng, len(dates), f)
    data.loc[50:60, "x1"] = np.nan
    master = master_df([("x0", "м", 10), ("x1", "н", 3), ("x2", "д", None)])
    cfg = {"journal_fill_flag": True, "keep_empty": keep_empty}
    lp = LagProcessor(master, data, cfg)
    factors = ["x0", "x1", "x2"]

    pairs = {
        "raw": {"value": ""},
        "journal": {"value": "", "available_date": "__available_date", "is_filled": "__is_filled"},
        "known": {"value": ""},
    }
    for view, cols in pairs.items():
        wide = lp.get(view, "wide").set_index("DATE")
        long = lp.get(view, "long")
        for col, suffix in cols.items():
            piv = long.pivot(index="DATE", columns="factor", values=col).reindex(index=wide.index, columns=factors)
            got = wide[[f + suffix for f in factors]]
            got.columns = factors
            if col == "is_filled":  # в плоском без keep_empty пустые строки отсутствуют
                piv, got = piv.eq(True), got.eq(True)
            pd.testing.assert_frame_equal(got, piv, check_names=False, check_dtype=False)
    jw = lp.journal("wide")
    assert list(jw.columns[1:4]) == ["x0", "x0__available_date", "x0__is_filled"]


# --- 10. Инвариант ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(20))
def test_no_leak_invariant_random(seed):
    rng = np.random.default_rng(100 + seed)
    all_dates = pd.date_range("2023-01-01", periods=300)
    keep = rng.random(len(all_dates)) > 0.2  # дыры в календаре
    dates = all_dates[keep]
    data = pd.DataFrame({"DATE": dates})
    rows = []
    for i in range(8):
        f = rng.choice(["м", "н", "д"])
        v = random_dense(rng, len(dates), f)
        v[rng.random(len(dates)) < 0.3] = np.nan  # смесь плотных и разреженных участков
        data[f"x{i}"] = v
        rows.append((f"x{i}", f, int(rng.integers(0, 60)) if rng.random() > 0.3 else None))
    cfg = {"calendar": rng.choice(["vitrine", "daily"]), "ffill_limit": None if seed % 2 else 40}
    lp = LagProcessor(master_df(rows), data, cfg)
    k = lp.known("long")
    assert (k["available_date"] <= k["DATE"]).all()
    assert k["available_date"].notna().all()
    assert (lp.diagnostics()["severity"] != "error").all()


# --- 11. Вход ----------------------------------------------------------------------------------------------
def test_csv_semicolon_cp1251(tmp_path):
    master = tmp_path / "master.csv"
    master.write_bytes("factor;data_frequency;object_lag;comment\nКурс;Д;;лишняя\nВВП ;m;45;\n".encode("cp1251"))
    data = tmp_path / "vitrina.csv"
    data.write_bytes(
        "DATE;Курс;ВВП;Лишний\n01.01.2026;1 234,5;100;x\n02.01.2026; 1 235,0;;y\n"
        "03.01.2026;abc;;\n".encode("cp1251", errors="ignore")
    )
    lp = LagProcessor(master, data)
    raw = lp.raw("wide")
    assert list(raw.columns) == ["DATE", "Курс", "ВВП"]
    assert raw["Курс"].iloc[0] == 1234.5
    assert raw["DATE"].iloc[0] == D("01.01.2026")
    assert lp.lags["data_frequency"].tolist() == ["д", "м"]
    nn = diag_of(lp, "non_numeric")
    assert nn["factor"].tolist() == ["Курс"] and "1 непуст" in nn["message"].iloc[0]


def test_numbers_with_comma_and_spaces():
    s = pd.Series(["1 234,5", "1 000", "-3,25", None, "", "12abc"])
    num, failed = lk_io.to_numeric_clean(s)
    assert num.iloc[:3].tolist() == [1234.5, 1000.0, -3.25]
    assert num.iloc[3:].isna().all()
    assert failed == 1


@pytest.mark.skipif(not HAS_XLSX, reason="нет openpyxl")
def test_xlsx_input(tmp_path):
    master = tmp_path / "master.xlsx"
    with pd.ExcelWriter(master) as w:
        pd.DataFrame({"x": [1]}).to_excel(w, sheet_name="other", index=False)
        master_df([("f", "н", 2.0)]).to_excel(w, sheet_name="master", index=False)
    data = tmp_path / "vitrina.xlsx"
    pd.DataFrame({"DATE": pd.date_range("2026-01-01", periods=14), "f": np.repeat([1.0, 2.0], 7)}) \
        .to_excel(data, index=False)
    lp = LagProcessor(master, data, {"master_sheet": "master"})
    assert lp.lags.at[0, "object_lag"] == 2
    assert lp.known("wide").set_index("DATE").loc[D("10.01.2026"), "f"] == 2.0


def test_duplicate_dates_error():
    data = pd.DataFrame({"DATE": ["01.01.2026", "02.01.2026", "01.01.2026"], "f": [1, 2, 3]})
    with pytest.raises(LagkitError, match="повторяются даты"):
        LagProcessor(master_df([("f", "м", 1)]), data)


def test_bad_date_error():
    data = pd.DataFrame({"DATE": ["01.01.2026", "не дата"], "f": [1, 2]})
    with pytest.raises(LagkitError, match="разобрать"):
        LagProcessor(master_df([("f", "м", 1)]), data)


def test_unsorted_dates_are_sorted():
    data = pd.DataFrame({"DATE": ["03.01.2026", "01.01.2026", "02.01.2026"], "f": [3.0, 1.0, 2.0]})
    raw = LagProcessor(master_df([("f", "д", 1)]), data).raw()
    assert raw["f"].tolist() == [1.0, 2.0, 3.0]


def test_duplicate_factor_error():
    data = pd.DataFrame({"DATE": ["01.01.2026"], "f": [1]})
    with pytest.raises(LagkitError, match="Дубликаты factor"):
        LagProcessor(master_df([("f", "м", 1), (" f", "н", 2)]), data)


def test_bad_frequency_error():
    data = pd.DataFrame({"DATE": ["01.01.2026"], "f": [1]})
    with pytest.raises(LagkitError, match="частотность.*'к'"):
        LagProcessor(master_df([("f", "к", 1)]), data)


def test_missing_factor_warning_and_strict():
    data = pd.DataFrame({"DATE": ["01.01.2026", "02.01.2026"], "f": [1, 2]})
    master = master_df([("f", "д", 1), ("absent", "м", None)])
    lp = LagProcessor(master, data)
    assert lp.factors == ["f"]
    d = diag_of(lp, "missing_factor")
    assert d["factor"].tolist() == ["absent"] and d["severity"].iloc[0] == "warning"
    with pytest.raises(LagkitError, match="absent"):
        LagProcessor(master, data, {"strict": True})


def test_empty_factor_warning():
    data = pd.DataFrame({"DATE": ["01.01.2026", "02.01.2026"], "f": [1, 2], "e": [None, None]})
    lp = LagProcessor(master_df([("f", "д", 1), ("e", "м", None)]), data)
    assert diag_of(lp, "empty_factor")["factor"].tolist() == ["e"]
    assert lp.lags.set_index("factor").at["e", "layout"] == "empty"
    assert lp.known("wide")["e"].isna().all()


# --- Конфиг и сохранение -----------------------------------------------------------------------------------
def test_config_unknown_key():
    with pytest.raises(LagkitError, match="ancor"):
        LagConfig.from_dict({"ancor": 1})
    with pytest.raises(LagkitError, match="gap_facter"):
        LagConfig.from_dict({"diagnostics": {"gap_facter": 2}})


def test_config_files(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"layout": "long", "diagnostics": {"gap_factor": 2}}), encoding="utf-8")
    cfg = LagConfig.from_file(p)
    assert cfg.layout == "long" and cfg.diagnostics.gap_factor == 2
    if importlib.util.find_spec("yaml"):
        cfg2 = LagConfig.from_file(PKG_DIR / "example_config.yaml")
        assert cfg2 == LagConfig()


def test_save_formats(tmp_path, monkeypatch):
    lp = example_processor()
    paths = lp.save(view="all", layout="long", output_dir=tmp_path, output_format="csv")
    assert set(paths) == {"raw_long", "journal_long", "known_long", "lags", "diagnostics"}
    text = paths["journal_long"].read_text(encoding="utf-8")
    assert "2026-01-01,f,2026-01-03,1.0,False" in text

    monkeypatch.setattr(lk_io, "XLSX_MAX_ROWS", 3)
    with pytest.raises(LagkitError, match="parquet"):
        lp.save(view="journal", layout="long", output_dir=tmp_path, output_format="xlsx")

    monkeypatch.setattr(lk_io.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(LagkitError, match="pyarrow"):
        lp.save(output_dir=tmp_path, output_format="parquet")


def test_build_views_and_cli(tmp_path):
    data = pd.DataFrame({"DATE": pd.date_range("2026-01-01", periods=10).strftime("%d.%m.%Y"),
                         "f": np.arange(10.0)})
    out = build_views(master_df([("f", "д", 2)]), data, {"layout": "long"})
    assert set(out) == {"raw", "journal", "known", "lags", "diagnostics"}
    assert "factor" in out["known"].columns

    master_path, data_path = tmp_path / "m.csv", tmp_path / "d.csv"
    master_df([("f", "д", 2)]).to_csv(master_path, index=False)
    data.to_csv(data_path, index=False)
    from lagkit.__main__ import main
    assert main(["--master", str(master_path), "--data", str(data_path), "--view", "all",
                 "--output-dir", str(tmp_path / "out"), "--format", "csv"]) == 0
    assert (tmp_path / "out" / "known_wide.csv").exists()
    assert (tmp_path / "out" / "lags.csv").exists()


# --- 12. Производительность -----------------------------------------------------------------------------------
def test_performance_600x5000():
    rng = np.random.default_rng(0)
    n, nf = 5000, 600
    dates = pd.date_range("2010-01-01", periods=n)
    freqs = rng.choice(["м", "н", "д"], nf)
    cols = {}
    for j, f in enumerate(freqs):
        p = {"м": 30, "н": 7, "д": 1}[f]
        v = np.repeat(rng.normal(size=n // p + 2), p)[:n]
        if f == "м" and j % 2:
            v = np.where(np.arange(n) % p == 0, v, np.nan)
        v[-int(rng.integers(0, 60)):] = np.nan
        cols[f"f{j}"] = v
    data = pd.concat([pd.DataFrame({"DATE": dates}), pd.DataFrame(cols)], axis=1)
    master = master_df([(c, f, None) for c, f in zip(cols, freqs)])
    t0 = time.perf_counter()
    lp = LagProcessor(master, data)
    for view in ("raw", "journal", "known"):
        for layout in ("wide", "long"):
            lp.get(view, layout)
    lp.diagnostics()
    elapsed = time.perf_counter() - t0
    assert elapsed < 60, f"{elapsed:.1f} c"


# --- 13. Переносимость -------------------------------------------------------------------------------------------
def test_portable_copy(tmp_path):
    project = tmp_path / "other_project"
    shutil.copytree(PKG_DIR, project / "lagkit", ignore=shutil.ignore_patterns("__pycache__", "tests"))
    data = pd.DataFrame({"DATE": pd.date_range("2026-01-01", periods=60).strftime("%d.%m.%Y"),
                         "cb_rate": np.repeat([16.0, 17.0], 30), "fx": np.arange(60.0)})
    data.to_csv(project / "vitrina.csv", index=False, sep=";")
    master = master_df([("cb_rate", "м", 25), ("fx", "д", None)])
    if HAS_XLSX:
        master_name = "master.xlsx"
        master.to_excel(project / master_name, index=False)
    else:
        master_name = "master.csv"
        master.to_csv(project / master_name, index=False)
    has_yaml = importlib.util.find_spec("yaml") is not None
    cfg_name = "lag_config.yaml" if has_yaml else "lag_config.json"
    (project / cfg_name).write_text(
        "output_format: csv\noutput_dir: ./out\n" if has_yaml else '{"output_format": "csv", "output_dir": "./out"}',
        encoding="utf-8",
    )
    script = textwrap.dedent(f"""
        import lagkit
        from pathlib import Path
        assert Path(lagkit.__file__).resolve().parent == Path.cwd().resolve() / "lagkit", lagkit.__file__
        from lagkit import LagProcessor, LagConfig, build_views
        lp = LagProcessor(master="{master_name}", data="vitrina.csv", config="{cfg_name}")
        print(lp.lags.to_string())
        lp.get("raw"); lp.get("journal", layout="long"); k = lp.get("known")
        lp.raw(); lp.journal(); lp.known()
        t = lp.trace("cb_rate")
        assert list(t.columns) == ["DATE", "raw", "journal", "available_date", "is_filled", "known"]
        lp.diagnostics()
        paths = lp.save(view="known", layout="wide")
        assert all(Path(p).exists() for p in paths.values())
        assert k.set_index("DATE").loc["2026-01-26", "cb_rate"] == 16.0
        print("OK")
    """)
    env = {k: v for k, v in __import__("os").environ.items() if k != "PYTHONPATH"}
    res = subprocess.run([sys.executable, "-c", script], cwd=project, capture_output=True, text=True, env=env)
    assert res.returncode == 0, res.stderr
    assert "OK" in res.stdout
