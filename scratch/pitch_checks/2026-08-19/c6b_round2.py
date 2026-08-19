"""C6 round 2, short DX leg: decluster, neighbours, era + rate-regime split,
gate attribution, magnitude gradient at today's reading, cost at each era.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import pandas as pd, numpy as np

px = close_panel(["DX-Y.NYB", "UUP", "^TNX", "TLT", "SPY"]).dropna(
    subset=["DX-Y.NYB", "^TNX"])
dx, tnx = px["DX-Y.NYB"], px["^TNX"]
rk_tnx, rk_dx = pct_rank(tnx, 21), pct_rank(dx, 21)
r21_tnx, r21_dx = tnx.pct_change(21), dx.pct_change(21)
base = ((r21_tnx > 0) & (rk_tnx >= 65) & (rk_dx <= 20)).fillna(False)
LEG = [("DX-Y.NYB", -1.0)]   # SHORT the dollar, the only positive-signed side

def cell(mask, h, gap=21, label="", legs=LEG):
    ret = vehicle_ret(px, legs, h)
    valid = ret.dropna().index
    idx = px.index[mask.reindex(px.index, fill_value=False).values].intersection(valid)
    epi = declusters(idx, gap, valid)
    r = summarize(ret.loc[epi].values, label)
    if r["n"]:
        r["sign_p"] = round(sign_test(int((ret.loc[epi] > 0).sum()), r["n"]), 4)
        r["boot_p_le0"] = round(bootstrap_p_le0(ret.loc[epi].values), 3)
        r["bps"] = round(100*r["mean_pct"], 1)
        r["x_cost"] = round(100*r["mean_pct"]/1.5, 1)   # DX futures 1.5 bps rt
    return r, epi, ret

print("=== 1. decluster sensitivity (short DX) ===")
for h in (3, 5, 10):
    rows = [cell(base, h, g, f"h={h} gap={g}td")[0] for g in (1, 5, 10, 21, 42, 63)]
    show(rows, f"h={h}")
r, epi, ret = cell(base, 5, 21)
v = ret.loc[epi].values
print(" ", cluster_note(epi, v, k=3))
o = np.sort(v)
print("  drop-best-1 %+0.3f%% (%.1f bps) drop-best-3 %+0.3f%% drop-worst-1 %+0.3f%%"
      % (100*o[:-1].mean(), 10000*o[:-1].mean(), 100*o[:-3].mean(), 100*o[1:].mean()))
yrs = pd.DatetimeIndex(epi).year
loyo = [(y, 100*v[yrs != y].mean()) for y in sorted(set(yrs))]
w = min(loyo, key=lambda x: x[1])
print("  LOYO floor: drop %d -> %+0.3f%% (%.1f bps)" % (w[0], w[1], 100*w[1]))

print("\n=== 2. definition neighbours (h=5, gap=21) ===")
rows = []
for ln in (10, 21, 42, 63):
    rt, rd, rr = pct_rank(tnx, ln), pct_rank(dx, ln), tnx.pct_change(ln)
    rows.append(cell(((rr > 0) & (rt >= 65) & (rd <= 20)).fillna(False), 5, 21,
                     f"lookback {ln}d")[0])
for a in (55, 60, 65, 70, 75, 85):
    rows.append(cell(((r21_tnx > 0) & (rk_tnx >= a) & (rk_dx <= 20)).fillna(False),
                     5, 21, f"TNX rank>={a}")[0])
for b in (5, 10, 15, 20, 25, 30, 40):
    rows.append(cell(((r21_tnx > 0) & (rk_tnx >= 65) & (rk_dx <= b)).fillna(False),
                     5, 21, f"DX rank<={b}")[0])
lvl = tnx - tnx.shift(21)
for pt in (0.05, 0.10, 0.20):
    rows.append(cell(((lvl >= pt) & (rk_dx <= 20)).fillna(False), 5, 21,
                     f"TNX level>=+{pt}pt")[0])
show(rows, "neighbour grid, short DX h=5")

print("\n=== 3. era + rate-regime split (h=5, gap=21) ===")
show(era_split(epi, v), "pre-2018 / 2018+")
show(era_split(epi, v, cut="2013-01-01"), "pre-2013 / 2013+")
show(era_split(epi, v, cut="2021-01-01"), "pre-2021 / 2021+")
sec = (tnx - tnx.shift(252)).reindex(epi)
for lbl, m in [("secular yields FALLING", sec < 0), ("secular yields RISING", sec >= 0)]:
    s = summarize(v[m.values], lbl)
    s["bps"] = round(100*s["mean_pct"], 1)
    show([s])
print("  TODAY's 252d TNX change %+0.3f pts -> RISING half"
      % (tnx.iloc[-1]-tnx.iloc[-253]))
by_yr = pd.Series(v).groupby(pd.DatetimeIndex(epi).year.values).agg(["count", "mean"])
by_yr["mean"] *= 100
print("\n  by year:\n", by_yr.round(2).to_string())

print("\n=== 4. gate attribution + magnitude gradient (h=5, gap=21) ===")
tnx_only = ((r21_tnx > 0) & (rk_tnx >= 65)).fillna(False)
dx_only = (rk_dx <= 20).fillna(False)
rows = []
for lbl, m in [("JOINT", base), ("TNX gate alone", tnx_only),
               ("TNX gate, DX rank>20", tnx_only & ~dx_only),
               ("DX gate alone", dx_only),
               ("DX gate, no TNX gate", dx_only & ~tnx_only),
               ("neither", ~tnx_only & ~dx_only)]:
    rows.append(cell(m, 5, 21, lbl)[0])
show(rows, "short-DX by gate cell")
# UUP mirror on the same episodes (vehicle disagreement check)
rowsu = []
for lbl, m in [("JOINT", base), ("DX gate alone", dx_only)]:
    rowsu.append(cell(m, 5, 21, f"UUP short {lbl}", legs=[("UUP", -1.0)])[0])
show(rowsu, "same cells expressed in UUP (short)")

lvl_c = (tnx - tnx.shift(21)).reindex(epi)
rows = []
for lo, hi, lbl in [(-9, 0.10, "TNX 21d chg <=0.10pt (TODAY +0.108)"),
                    (0.10, 0.25, "0.10-0.25pt"), (0.25, 9, ">0.25pt")]:
    m = ((lvl_c > lo) & (lvl_c <= hi)).values
    rows.append(summarize(v[m], lbl))
show(rows, "by yield-rise magnitude")
dxr = r21_dx.reindex(epi)
rows = []
for lo, hi, lbl in [(-9, -0.04, "DX 21d <=-4%"), (-0.04, -0.02, "-4 to -2%"),
                    (-0.02, 9, ">-2% (TODAY -1.33%)")]:
    m = ((dxr > lo) & (dxr <= hi)).values
    rows.append(summarize(v[m], lbl))
show(rows, "by dollar-fall magnitude")

print("\n=== 5. cost at each era (DX futures 1.5 bps round trip, need 5x = 7.5 bps) ===")
for lbl, m in [("full sample", np.ones(len(v), bool)),
               ("pre-2018", (pd.DatetimeIndex(epi) < "2018-01-01")),
               ("2018+", (pd.DatetimeIndex(epi) >= "2018-01-01"))]:
    if m.sum():
        print("  %-12s N=%2d mean %+0.3f%% = %+.1f bps = %.1fx cost, hit %.1f%%"
              % (lbl, m.sum(), 100*v[m].mean(), 10000*v[m].mean(),
                 10000*v[m].mean()/1.5, 100*(v[m] > 0).mean()))
