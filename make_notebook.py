"""Собирает ноутбук lag_demo.ipynb (синтетические данные + проверка сдвигов lagkit)."""
import nbformat as nbf

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md("""# Проверка lagkit на синтетических данных

Витрина: 01.07.2026 – 01.09.2026, три фактора с разной частотностью; у каждого в master задан лаг.

| factor | частотность | лаг | как лежит в витрине |
|---|---|---|---|
| `usd_rub` | д | 1 | значение каждый день; 01.09 ещё не загружено (хвост) |
| `cpi_mom` | м | 30 | плотная раскладка: 01.07–31.07 = 100.0, 01.08–01.09 = 101.5 (сентябрьского отчёта ещё нет) |
| `deposits_w` | н | 3 | плотная раскладка, отчёт по понедельникам (первый — 01.07, среда); неделя 10.08 **не загружена** (сбой); отчёт 31.08 ещё не пришёл |

Ожидания:
* `usd_rub`: на дату t стоит значение витрины за t−1.
* `cpi_mom`: 100.0 известно с 31.07 (01.07+30), 101.5 — с 31.08 (01.08+30); до 31.07 пусто.
* `deposits_w`: отчёт за понедельник d известен с d+3 (четверг); во время сбоя протягивается значение отчёта 03.08, пока 20.08 не станет известен отчёт 17.08."""),
    code("""import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))  # папка lagkit лежит рядом с ноутбуком
from lagkit import LagProcessor

pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 160)
OUT = Path("synthetic")
OUT.mkdir(exist_ok=True)"""),
    md("## 1. Синтетические master и data"),
    code("""master = pd.DataFrame({
    "factor":         ["usd_rub", "cpi_mom", "deposits_w"],
    "data_frequency": ["д",       "м",       "н"],
    "object_lag":     [1,          30,        3],
})

dates = pd.date_range("2026-07-01", "2026-09-01", freq="D")
data = pd.DataFrame({"DATE": dates})

# Дневной курс: своё значение каждый день, 01.09 ещё не загружено.
data["usd_rub"] = np.round(90 + 0.1 * np.arange(len(dates)), 2)
data.loc[data["DATE"] == "2026-09-01", "usd_rub"] = np.nan

# Месячный индекс в плотной раскладке.
data["cpi_mom"] = np.where(data["DATE"] < "2026-08-01", 100.0, 101.5)

# Недельный ряд: новое значение каждый понедельник (+ первый отчёт 01.07), ffill внутри недели.
week_start = data["DATE"].dt.to_period("W-SUN").dt.start_time.clip(lower=dates[0])
week_no = week_start.rank(method="dense").astype(int) - 1
data["deposits_w"] = 7.0 + 0.25 * week_no
data.loc[data["DATE"].between("2026-08-10", "2026-08-16"), "deposits_w"] = np.nan  # сбой загрузки
# Отчёт за 31.08 ещё не загружен: витрина протянула значение отчёта 24.08 до конца.
data.loc[data["DATE"] >= "2026-08-31", "deposits_w"] = 9.0

master.to_csv(OUT / "master.csv", index=False, sep=";", encoding="utf-8")
# Витрина в «российском» виде: ДД.ММ.ГГГГ, разделитель ';', десятичная запятая.
data_csv = data.copy()
data_csv["DATE"] = data_csv["DATE"].dt.strftime("%d.%m.%Y")
data_csv.to_csv(OUT / "data.csv", index=False, sep=";", decimal=",", encoding="utf-8")

print((OUT / "master.csv").read_text())
print("\\n".join((OUT / "data.csv").read_text().splitlines()[:8]), "\\n...")"""),
    md("## 2. Обработка lagkit"),
    code("""lp = LagProcessor(master=OUT / "master.csv", data=OUT / "data.csv",
                  config={"output_format": "csv", "output_dir": str(OUT / "lag_output")})
lp.lags"""),
    code("""lp.publications  # одна строка на отчёт: report_date -> available_date"""),
    code("""lp.diagnostics()"""),
    md("## 3. Трассировка каждого фактора: raw → журнал → known"),
    code("""for f in lp.factors:
    print(f"=== {f} ===")
    display(lp.trace(f))"""),
    md("""## 4. Проверки

**4.1. Независимый расчёт «в лоб».** Даты отчёта ищутся простым циклом по правилу ТЗ: для д —
каждая непустая строка, для м/н — непустая строка после пустой или с новым значением. Затем
на каждую дату t берётся отчёт с наибольшей `report_date + lag <= t`. Этот код не использует lagkit."""),
    code("""def expected_known(raw: pd.Series, lag: int, freq: str, calendar) -> pd.Series:
    reports, prev = [], None
    for d, v in raw.items():
        if pd.isna(v):
            prev = None
            continue
        if freq == "д" or prev is None or v != prev:
            reports.append((d + pd.Timedelta(days=lag), d, v))  # (available, report, value)
        prev = v
    out = {}
    for t in calendar:
        known = [r for r in reports if r[0] <= t]
        out[t] = max(known)[2] if known else np.nan
    return pd.Series(out)

known = lp.known().set_index("DATE")
raw = data.set_index("DATE")
for f, freq, lag in master.itertuples(index=False):
    exp = expected_known(raw[f], lag, freq, known.index)
    pd.testing.assert_series_equal(known[f], exp, check_names=False, check_freq=False)
    print(f"{f:11s} OK: known совпадает с независимым расчётом на всех {len(exp)} датах")"""),
    md("**4.2. Ключевые даты, посчитанные вручную.**"),
    code("""checks = [
    # factor,      дата,         ожидаемое значение, почему
    ("usd_rub",    "2026-07-01", np.nan, "первый курс (01.07) известен только с 02.07"),
    ("usd_rub",    "2026-07-02", 90.0,   "курс за 01.07"),
    ("usd_rub",    "2026-09-01", 96.1,   "курс за 31.08 (01.09 в витрине пуст — не важно)"),
    ("cpi_mom",    "2026-07-30", np.nan, "отчёт 01.07 + 30 = 31.07, ещё не известен"),
    ("cpi_mom",    "2026-07-31", 100.0,  "01.07 + 30"),
    ("cpi_mom",    "2026-08-30", 100.0,  "август (101.5) ещё не опубликован"),
    ("cpi_mom",    "2026-08-31", 101.5,  "01.08 + 30"),
    ("deposits_w", "2026-07-03", np.nan, "отчёт 01.07 + 3 = 04.07"),
    ("deposits_w", "2026-07-04", 7.0,    "01.07 + 3"),
    ("deposits_w", "2026-07-08", 7.0,    "отчёт 06.07 известен с 09.07"),
    ("deposits_w", "2026-07-09", 7.25,   "06.07 + 3"),
    ("deposits_w", "2026-08-06", 8.25,   "отчёт 03.08 + 3"),
    ("deposits_w", "2026-08-13", 8.25,   "сбой: отчёт 10.08 не пришёл, протягиваем 03.08"),
    ("deposits_w", "2026-08-19", 8.25,   "отчёт 17.08 известен только с 20.08"),
    ("deposits_w", "2026-08-20", 8.75,   "17.08 + 3"),
    ("deposits_w", "2026-09-01", 9.0,    "отчёт 24.08 + 3 = 27.08; 31.08–01.09 витрина протянула тот же отчёт"),
]
rows = []
for f, d, exp, why in checks:
    got = known.loc[d, f]
    ok = (np.isnan(exp) and np.isnan(got)) or np.isclose(got, exp)
    rows.append(dict(factor=f, DATE=d, expected=exp, known=got, ok=ok, why=why))
res = pd.DataFrame(rows)
display(res)
assert res["ok"].all()"""),
    md("**4.3. Журнал: середина блока наследует дату доступности блока, хвост и сбой помечены `is_filled`.**"),
    code("""j = lp.journal(layout="long").set_index(["factor", "DATE"])
assert j.loc[("cpi_mom", "2026-07-15"), "available_date"] == pd.Timestamp("2026-07-31")
assert j.loc[("cpi_mom", "2026-09-01"), "available_date"] == pd.Timestamp("2026-08-31")
assert j.loc[("deposits_w", "2026-07-08"), "available_date"] == pd.Timestamp("2026-07-09")
gap = j.loc["deposits_w"].loc["2026-08-10":"2026-08-16"]
assert gap["is_filled"].all() and (gap["value"] == 8.25).all()
assert j.loc[("usd_rub", "2026-09-01"), "is_filled"]
assert j.loc[("usd_rub", "2026-08-31"), "available_date"] == pd.Timestamp("2026-09-01")
print("журнал OK")"""),
    md("**4.4. Инвариант отсутствия утечки и диагностика пропуска.**"),
    code("""kl = lp.known(layout="long")
assert (kl["available_date"] <= kl["DATE"]).all()
print(f"available_date <= DATE во всех {len(kl)} строках known")

d = lp.diagnostics()
gap_diag = d[(d["check"] == "missing_reports") & (d["factor"] == "deposits_w")]
assert len(gap_diag) == 1
print(gap_diag["message"].iloc[0])
assert (d["severity"] != "error").all()"""),
    md("## 5. Сохранение"),
    code("""paths = lp.save(view="all", layout="wide")
for name, p in paths.items():
    print(f"{name:14s} {p}")
pd.read_csv(paths["known_wide"]).tail(5)"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
})
nbf.write(nb, "lag_demo.ipynb")
