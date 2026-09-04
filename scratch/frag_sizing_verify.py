"""Independent re-verification of the fragility-vs-R study.

Different join method from frag_sizing_study.py (exact date reindex with
forward-fill instead of merge_asof), all three dials (5d/21d/63d), several MA
windows, and a threshold x floor grid. Non-OVS trades only (the live mult's
domain).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
trades = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
frag.index = pd.to_datetime(frag.index).normalize()

trades["Signal Date"] = pd.to_datetime(trades["Signal Date"]).dt.normalize()
t0 = trades[~trades["Strategy"].str.contains("Overbot Vol|OVS", case=False, na=False)].copy()

# business-day grid, ffill (limit 5) so any signal date maps to the latest score
grid = pd.date_range(frag.index.min(), frag.index.max(), freq="D")


def dial_series(col: str, ma: int) -> pd.Series:
    s = frag[col].dropna()
    if ma > 1:
        s = s.rolling(ma, min_periods=ma).mean()
    return s.reindex(grid).ffill(limit=5)


def join(col: str, ma: int) -> pd.DataFrame:
    s = dial_series(col, ma)
    d = t0.copy()
    d["frag"] = s.reindex(d["Signal Date"]).values
    d = d.dropna(subset=["frag", "R_Multiple"])
    d["ym"] = d["Signal Date"].dt.to_period("M")
    return d


def hi_lo(d: pd.DataFrame, thr: float):
    hi, lo = d[d.frag >= thr], d[d.frag < thr]
    if len(hi) < 30 or len(lo) < 30:
        return None
    hm = hi.groupby("ym")["R_Multiple"].mean()
    lm = lo.groupby("ym")["R_Multiple"].mean()
    tt = stats.ttest_ind(hm, lm, equal_var=False)
    return dict(N_hi=len(hi), avgR_hi=hi.R_Multiple.mean(),
                N_lo=len(lo), avgR_lo=lo.R_Multiple.mean(),
                t=tt.statistic, p=tt.pvalue)


print("=== discrimination by dial x MA window (threshold 50) ===")
rows = []
for col in ["5d", "21d", "63d"]:
    for ma in [1, 5, 10, 21]:
        d = join(col, ma)
        r = hi_lo(d, 50)
        if r is None:
            rows.append({"dial": col, "MA": ma, "note": "too few hi-frag trades"})
            continue
        rows.append({"dial": col, "MA": ma, "N>=50": r["N_hi"],
                     "avgR>=50": round(r["avgR_hi"], 3),
                     "avgR<50": round(r["avgR_lo"], 3),
                     "t(monthly)": round(r["t"], 2), "p": round(r["p"], 3)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== threshold sweep, 63d MA10 (live basis) ===")
d63 = join("63d", 10)
rows = []
for thr in [30, 40, 50, 60, 70]:
    r = hi_lo(d63, thr)
    if r is None:
        rows.append({"thr": thr, "note": "too few trades above"})
        continue
    rows.append({"thr": thr, "N>=thr": r["N_hi"], "avgR>=thr": round(r["avgR_hi"], 3),
                 "avgR<thr": round(r["avgR_lo"], 3),
                 "t(monthly)": round(r["t"], 2), "p": round(r["p"], 3)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== bucket detail per dial (own MA10) ===")
for col in ["5d", "21d", "63d"]:
    d = join(col, 10)
    d["bucket"] = pd.cut(d.frag, [0, 12.5, 25, 50, 75, 200],
                         labels=["0-12.5", "12.5-25", "25-50", "50-75", "75+"],
                         include_lowest=True)
    g = d.groupby("bucket", observed=False)["R_Multiple"]
    tbl = pd.DataFrame({"N": g.size(), "avgR": g.mean().round(3),
                        "medR": g.median().round(3),
                        "win%": (g.apply(lambda s: (s > 0).mean() * 100)).round(1)})
    print(f"--- {col} ---")
    print(tbl.to_string())

print("\n=== step-schedule grid, 63d MA10: mult=floor when frag>=thr, else 1.0 ===")
print("cells: totR | avgR-per-unit-risk | worst R drawdown")


def replay(d: pd.DataFrame, mult: pd.Series):
    radj = d["R_Multiple"] * mult
    curve = d.assign(x=radj).sort_values("Exit Date").groupby("Exit Date")["x"].sum().cumsum()
    dd = (curve - curve.cummax()).min()
    return radj.sum(), radj.sum() / mult.sum(), dd


base_tot, base_avg, base_dd = replay(d63, pd.Series(1.0, index=d63.index))
print(f"baseline 1.0x: {base_tot:+.1f} | {base_avg:+.4f} | {base_dd:+.1f}")
rows = []
for thr in [30, 40, 50, 60]:
    row = {"thr": thr}
    for floor in [0.0, 0.25, 0.5, 0.75]:
        m = pd.Series(np.where(d63.frag >= thr, floor, 1.0), index=d63.index)
        tot, avg, dd = replay(d63, m)
        row[f"floor={floor}"] = f"{tot:+.0f}|{avg:+.3f}|{dd:+.0f}"
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))

print("\ncurrent live ramp (thr25 boost1.25 floor0.10, linear):")


def ramp(f):
    if f <= 25:
        return 1.25 - (f / 25) * 0.25
    return max(0.10, 1.0 - ((f - 25) / 75) * 0.90)


m = d63.frag.map(ramp)
tot, avg, dd = replay(d63, m)
print(f"  {tot:+.1f} | {avg:+.4f} | {dd:+.1f}  (avg mult {m.mean():.2f})")

# spot-check the join on 3 random trades (reproducible sample)
print("\nspot-check joins (63d MA10):")
chk = d63.sample(3, random_state=7)[["Ticker", "Signal Date", "frag"]]
s = dial_series("63d", 10)
for _, r in chk.iterrows():
    direct = s.loc[r["Signal Date"]]
    print(f"  {r['Ticker']:6s} {r['Signal Date'].date()}  joined={r['frag']:.2f}  direct={direct:.2f}  "
          f"{'OK' if abs(direct - r['frag']) < 1e-9 else 'MISMATCH'}")
