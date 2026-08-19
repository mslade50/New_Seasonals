"""C7 round 2: decluster, definition neighbours, era + RATE-REGIME split,
gate attribution, and the live-state gradient (gold already strong).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import pandas as pd, numpy as np

px = close_panel(["DX-Y.NYB", "^TNX", "GLD", "SPY"]).dropna(
    subset=["DX-Y.NYB", "^TNX", "GLD"])
dx, tnx, gld = px["DX-Y.NYB"], px["^TNX"], px["GLD"]
rk_tnx, rk_dx = pct_rank(tnx, 21), pct_rank(dx, 21)
r21_tnx = tnx.pct_change(21)
base = ((r21_tnx > 0) & (rk_tnx >= 65) & (rk_dx <= 20)).fillna(False)
LEG = [("GLD", 1.0)]

def cell(mask, h, gap=21, label=""):
    ret = vehicle_ret(px, LEG, h)
    valid = ret.dropna().index
    idx = px.index[mask.reindex(px.index, fill_value=False).values].intersection(valid)
    epi = declusters(idx, gap, valid)
    r = summarize(ret.loc[epi].values, label)
    if r["n"]:
        r["sign_p"] = round(sign_test(int((ret.loc[epi] > 0).sum()), r["n"]), 4)
        r["boot_p_le0"] = round(bootstrap_p_le0(ret.loc[epi].values), 3)
        r["edge_pp"] = round(r["mean_pct"] - 100*ret.loc[valid].mean(), 3)
    return r, epi, ret

# ---------- 1. DECLUSTER + CONCENTRATION ----------
print("=== 1. decluster sensitivity + concentration ===")
for h in (1, 3, 5):
    rows = []
    for gap in (1, 5, 10, 21, 42, 63):
        r, epi, ret = cell(base, h, gap, f"h={h} gap={gap}td")
        rows.append(r)
    show(rows, f"h={h}")
    r, epi, ret = cell(base, h, 21)
    v = ret.loc[epi].values
    print("  ", cluster_note(epi, v, k=3))
    o = np.sort(v)
    print("   drop-best-1 %+0.3f%%  drop-best-3 %+0.3f%%  drop-worst-1 %+0.3f%%"
          % (100*o[:-1].mean(), 100*o[:-3].mean(), 100*o[1:].mean()))
    # leave-one-year-out floor
    yrs = pd.DatetimeIndex(epi).year
    loyo = [(y, 100*v[yrs != y].mean()) for y in sorted(set(yrs))]
    worst = min(loyo, key=lambda x: x[1])
    print("   LOYO floor: drop %d -> %+0.3f%%   (all %d years: min %+0.3f max %+0.3f)"
          % (worst[0], worst[1], len(loyo), worst[1], max(l[1] for l in loyo)))

# ---------- 2. DEFINITION NEIGHBOURS ----------
print("\n=== 2. definition neighbours (h=3, gap=21) ===")
rows = []
for ln in (10, 21, 42, 63):
    rt, rd = pct_rank(tnx, ln), pct_rank(dx, ln)
    rr = tnx.pct_change(ln)
    m = ((rr > 0) & (rt >= 65) & (rd <= 20)).fillna(False)
    r, _, _ = cell(m, 3, 21, f"lookback {ln}d both legs")
    rows.append(r)
for a in (55, 60, 65, 70, 75, 85):
    m = ((r21_tnx > 0) & (rk_tnx >= a) & (rk_dx <= 20)).fillna(False)
    r, _, _ = cell(m, 3, 21, f"TNX rank>={a}, DX<=20")
    rows.append(r)
for b in (5, 10, 15, 20, 25, 30, 40):
    m = ((r21_tnx > 0) & (rk_tnx >= 65) & (rk_dx <= b)).fillna(False)
    r, _, _ = cell(m, 3, 21, f"TNX>=65, DX rank<={b}")
    rows.append(r)
# TNX measured as a LEVEL change instead of a pct change
lvl = tnx - tnx.shift(21)
for pt in (0.05, 0.10, 0.20):
    m = ((lvl >= pt) & (rk_dx <= 20)).fillna(False)
    r, _, _ = cell(m, 3, 21, f"TNX 21d level >=+{pt}pt, DX<=20")
    rows.append(r)
# drop the "rank" leg entirely, keep the sign
m = ((r21_tnx > 0) & (rk_dx <= 20)).fillna(False)
rows.append(cell(m, 3, 21, "TNX up (no rank gate), DX<=20")[0])
show(rows, "neighbour grid")

# ---------- 3. ERA + RATE-REGIME SPLIT ----------
print("\n=== 3. era and RATE-REGIME splits (h=3, gap=21) ===")
r, epi, ret = cell(base, 3, 21)
v = ret.loc[epi].values
show(era_split(epi, v), "era pre-2018 / 2018+")
show(era_split(epi, v, cut="2013-01-01"), "era pre-2013 / 2013+")
show(era_split(epi, v, cut="2021-01-01"), "era pre-2021 / 2021+")
# secular rate regime: 252d change in the 10y yield at the signal date
sec = (tnx - tnx.shift(252)).reindex(epi)
for lbl, m in [("secular yields FALLING (252d chg<0)", sec < 0),
               ("secular yields RISING (252d chg>=0)", sec >= 0)]:
    print("  %-38s %s" % (lbl, summarize(v[m.values], "")))
print("  TODAY's 252d TNX change: %+0.3f pts (%s)"
      % (tnx.iloc[-1]-tnx.iloc[-253],
         "RISING" if tnx.iloc[-1] >= tnx.iloc[-253] else "FALLING"))
# gold's own secular regime (is the cell a gold-bull fossil?)
g252 = gld.pct_change(252).reindex(epi)
for lbl, m in [("GLD 252d ret < 0 (gold bear)", g252 < 0),
               ("GLD 252d ret >= 0 (gold bull)", g252 >= 0)]:
    print("  %-38s %s" % (lbl, summarize(v[m.values], "")))
print("  TODAY's GLD 252d return: %+0.1f%%" % (100*gld.pct_change(252).iloc[-1]))
by_yr = pd.Series(v).groupby(pd.DatetimeIndex(epi).year.values).agg(["count", "mean"])
by_yr["mean"] *= 100
print("\n  by year:\n", by_yr.round(2).to_string())

# ---------- 4. LIVE-STATE GRADIENT ----------
print("\n=== 4. live-state gradient: gold ALREADY strong ===")
rk_gld = pct_rank(gld, 21)
r21_gld = gld.pct_change(21)
for h in (1, 3, 5, 10):
    r, epi, ret = cell(base, h, 21)
    v = ret.loc[epi].values
    rg = rk_gld.reindex(epi)
    hi = (rg >= 75).values
    a = summarize(v[hi], f"h={h} GLD rank21>=75 (TODAY 77.0)")
    b = summarize(v[~hi], f"h={h} GLD rank21<75")
    a["sign_p"] = round(sign_test(int((v[hi] > 0).sum()), int(hi.sum())), 4)
    show([a, b])
# and by prior 21d return, two buckets
print("\n  by prior 21d gold return, h=3:")
r, epi, ret = cell(base, 3, 21)
v = ret.loc[epi].values
pr = r21_gld.reindex(epi)
for lo, hi_, lbl in [(-9, 0.05, "GLD 21d <+5%"), (0.05, 9, "GLD 21d >=+5% (TODAY +8.42%)")]:
    m = ((pr > lo) & (pr <= hi_)).values
    s = summarize(v[m], lbl)
    s["sign_p"] = round(sign_test(int((v[m] > 0).sum()), int(m.sum())), 4)
    show([s])
