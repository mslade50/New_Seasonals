"""Is the 2026 shortfall selection (overfit) or regime?

(1) Yearly OOS-style ratio base rate: for each year Y (2008-2026), avgR in Y
    divided by avgR over the trailing 5 years (same strategy mix as year Y).
    Where does 2026's ratio sit in that distribution? A rule tuned through 2026-08
    should NOT be unusually bad in 2026 unless the tuning did not help.
(2) Midterm conditioning: OVS avgR in prior midterm years (2006/10/14/18/22, Jan-Aug)
    vs non-midterm Jan-Aug; the book the same way.
(3) Dial conditioning: 2026 trades by signal-date dial bucket vs the same buckets
    2016-2025 (current-weights vintage, stated).
(4) Per-strategy 2026 vs trailing 5y with a day-block bootstrap CI.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:/Users/McKinley Slade/dev/New_Seasonals")
OUT = ROOT / "scratch/ultracode_sizing_2026-09-02"
RNG = np.random.default_rng(11)
res: dict = {}

led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
led = led[led["PnL_flat_750k"].notna()].copy()
led["Signal Date"] = pd.to_datetime(led["Signal Date"])
led["year"] = led["Signal Date"].dt.year
led["month"] = led["Signal Date"].dt.month

# (1) yearly ratio to trailing-5y at year mix
rows = []
for y in range(2008, 2027):
    cur = led[led["year"] == y]
    if y == 2026:
        prev = led[(led["year"] >= y - 5) & (led["year"] < y)]
        # compare Jan-Aug only for the trailing window too (seasonal mix)
        prev = prev[prev["month"] <= 8]
    else:
        prev = led[(led["year"] >= y - 5) & (led["year"] < y)]
    mix = cur.groupby("Strategy").size()
    pm = prev.groupby("Strategy")["R_Multiple"].mean().reindex(mix.index)
    ok = pm.notna()
    exp = float((mix[ok] * pm[ok]).sum() / mix[ok].sum())
    act = float(cur[cur["Strategy"].isin(mix.index[ok])]["R_Multiple"].mean())
    rows.append({"year": y, "N": int(mix[ok].sum()), "avgR": act, "trailing5_at_mix": exp, "ratio": act / exp if exp > 0 else None})
yr = pd.DataFrame(rows)
print(yr.round(3).to_string(index=False))
res["yearly_ratio_to_trailing5"] = yr.to_dict("records")
hist = yr[yr["year"] < 2026]["ratio"].dropna()
res["yearly_ratio_summary"] = {"mean": float(hist.mean()), "median": float(hist.median()), "sd": float(hist.std()), "min": float(hist.min()),
                               "share_below_0.5": float((hist < 0.5).mean()), "share_below_0.7": float((hist < 0.7).mean()),
                               "ratio_2026": float(yr.loc[yr["year"] == 2026, "ratio"].iloc[0])}
print("yearly ratio summary:", res["yearly_ratio_summary"])

# (2) midterm conditioning, Jan-Aug windows
ja = led[led["month"] <= 8]
ja["midterm"] = (ja["year"] % 4 == 2)
def cell(df):
    return {"N": int(len(df)), "avgR": float(df["R_Multiple"].mean()) if len(df) else None}
res["midterm"] = {
    "OVS_midterm_JanAug_pre2026": cell(ja[(ja["Strategy"] == "Overbot Vol Spike") & ja["midterm"] & (ja["year"] < 2026)]),
    "OVS_nonmidterm_JanAug": cell(ja[(ja["Strategy"] == "Overbot Vol Spike") & ~ja["midterm"]]),
    "OVS_2026_JanAug": cell(ja[(ja["Strategy"] == "Overbot Vol Spike") & (ja["year"] == 2026)]),
    "book_midterm_JanAug_pre2026": cell(ja[ja["midterm"] & (ja["year"] < 2026)]),
    "book_nonmidterm_JanAug": cell(ja[~ja["midterm"]]),
    "book_2026_JanAug": cell(ja[ja["year"] == 2026]),
    "OVS_by_midterm_year": {int(y): cell(g) for y, g in ja[(ja["Strategy"] == "Overbot Vol Spike") & ja["midterm"]].groupby("year")},
    "book_by_midterm_year": {int(y): cell(g) for y, g in ja[ja["midterm"]].groupby("year")},
}
print("midterm:", json.dumps(res["midterm"], indent=1))

# (3) dial conditioning (current-weights vintage: rows before 2026-07-02 recompute, after PIT)
frag = pd.read_parquet(ROOT / "data/rd2_fragility.parquet")
dial = frag["63d"].rolling(10).mean()
dial.index = pd.to_datetime(dial.index).tz_localize(None) if getattr(dial.index, "tz", None) is not None else pd.to_datetime(dial.index)
led["dial"] = led["Signal Date"].map(dial)
led["dbkt"] = pd.cut(led["dial"], [-1, 30, 50, 65, 200], labels=["<30", "30-50", "50-65", "65+"])
d26 = led[led["year"] == 2026]
dpre = led[(led["year"] >= 2016) & (led["year"] < 2026)]
tab = pd.DataFrame({"pre_N": dpre.groupby("dbkt", observed=False).size(), "pre_avgR": dpre.groupby("dbkt", observed=False)["R_Multiple"].mean(),
                    "2026_N": d26.groupby("dbkt", observed=False).size(), "2026_avgR": d26.groupby("dbkt", observed=False)["R_Multiple"].mean()})
print("\ndial buckets:\n", tab.round(3))
res["dial_buckets"] = {str(k): {c: (None if pd.isna(v[c]) else float(v[c])) for c in tab.columns} for k, v in tab.iterrows()}
# expected 2026 avgR if only the dial mix changed
mix26 = d26.groupby("dbkt", observed=False).size()
exp_dial = float((mix26 * tab["pre_avgR"]).sum() / mix26.sum())
res["dial_mix_expected_2026_avgR"] = exp_dial
res["dial_mix_ratio_2026"] = float(d26["R_Multiple"].mean() / exp_dial)
print("2026 expected at 2026 dial mix:", round(exp_dial, 3), "actual", round(d26["R_Multiple"].mean(), 3))
# strategy x dial mix jointly
mixsd = d26.groupby(["Strategy", "dbkt"], observed=False).size()
pre_sd = dpre.groupby(["Strategy", "dbkt"], observed=False)["R_Multiple"].mean()
j = pd.concat([mixsd.rename("n"), pre_sd.rename("m")], axis=1).dropna()
j = j[j["n"] > 0]
exp_sd = float((j["n"] * j["m"]).sum() / j["n"].sum())
res["strategy_x_dial_mix_expected_2026_avgR"] = exp_sd
res["strategy_x_dial_mix_ratio_2026"] = float(d26["R_Multiple"].mean() / exp_sd)
print("2026 expected at strategy x dial mix:", round(exp_sd, 3), "ratio", round(d26["R_Multiple"].mean() / exp_sd, 2))

# (4) per-strategy 2026 vs trailing 5y with day-block bootstrap
per = {}
for s, g in led.groupby("Strategy"):
    c = g[g["year"] == 2026]
    p = g[(g["year"] >= 2021) & (g["year"] < 2026)]
    if len(c) < 3 or len(p) < 10:
        continue
    days = c.groupby("Signal Date")["R_Multiple"].apply(list)
    bs = []
    for _ in range(3000):
        pick = RNG.choice(len(days), len(days))
        bs.append(np.concatenate([days.iloc[i] for i in pick]).mean())
    bs = np.array(bs)
    per[s] = {"N_2026": int(len(c)), "avgR_2026": float(c["R_Multiple"].mean()), "avgR_2021_25": float(p["R_Multiple"].mean()), "N_2021_25": int(len(p)),
              "ratio": float(c["R_Multiple"].mean() / p["R_Multiple"].mean()) if p["R_Multiple"].mean() > 0 else None,
              "ci95_2026": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
              "P_2026_below_half_trailing": float((bs < 0.5 * p["R_Multiple"].mean()).mean())}
res["per_strategy_2026_vs_trailing5"] = per
print("\n", pd.DataFrame(per).T.round(3).to_string())
(OUT / "estimation_haircut_regime.json").write_text(json.dumps(res, indent=1, default=str))
