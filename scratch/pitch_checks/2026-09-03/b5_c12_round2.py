"""C12 ROUND 2 -- the only candidate of the four with a pulse.

Round 1 (b4) gave: SPY long, entry MOC on the last session before the print,
exit MOC at the print close, gated VIX rel-range pctile <= 15:
    +0.236% over 45 episodes, 62.2% hit, 28-17, sign p 0.0676,
    excess +0.197pp over SPY's drift and +0.218pp over the tdom-matched control,
    gate value +0.183pp, complement -0.015%, 11.8x cost.

Round 2 has to answer one question above all others: TODAY'S READING IS 3.57,
i.e. the extreme bottom of the gate. Is the gate monotone into that corner, or
is the edge an interior spike that today's tape sits OUTSIDE of? The 2026-09-02
registry entry found the compression story monotone in the WRONG direction on a
neighbouring cell, so this is the decisive attack.

  1. MARGINAL bucket ladder (not cumulative) with the live value marked
  2. placebo anchor ladder k=-8..+8 with the gate applied
  3. concentration / drop-best / year histogram
  4. definition neighbours: range window 10/21/42, pctile lookback 126/252/504,
     abs vs rel range form
  5. cluster depth of the STATE at the anchor vs today's depth
  6. multiplicity charge for the grid this search actually ran
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import (load_prices, load_events, anchor_positions, summarize,
                       show, sign_test, bootstrap_p_le0, declusters)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 230)

raw = load_prices(["SPY", "^VIX"])
px = pd.DataFrame({t: raw[t]["Close"] for t in ["SPY", "^VIX"]}).dropna(subset=["SPY"])
cal = px.index
vix = raw["^VIX"]["Close"].dropna()


def range_pctile(win: int, lb: int, form: str) -> pd.Series:
    rng = vix.rolling(win).max() - vix.rolling(win).min()
    x = rng / vix.rolling(win).mean() if form == "rel" else rng
    return (x.rolling(lb).rank(pct=True) * 100).reindex(cal).ffill(limit=3)


REL = range_pctile(21, 252, "rel")
ABS = range_pctile(21, 252, "abs")
LIVE_REL, LIVE_ABS = REL.iloc[-1], ABS.iloc[-1]
print("live 2026-09-02: rel-range pctile %.2f   abs-range pctile %.2f" % (LIVE_REL, LIVE_ABS))

nfp = load_events(["nfp"])["date"]
pos, _ = anchor_positions(cal, nfp, -2)
anchor_pos = [i for i in pos if i + 1 < len(cal)]
entry = pd.DatetimeIndex([cal[i + 1] for i in anchor_pos])
gate_at_entry = pd.Series([REL.iloc[i] for i in anchor_pos], index=entry)
abs_at_entry = pd.Series([ABS.iloc[i] for i in anchor_pos], index=entry)


def ret_from_entry(s, h):
    return s.shift(-h) / s - 1.0


spy = px["SPY"]

# ==========================================================================
print("\n" + "=" * 78)
print("ATTACK 1 -- MARGINAL bucket ladder (today sits at rel 3.57 = bottom bucket)")
print("=" * 78)
EDGES = [0, 5, 10, 15, 25, 50, 75, 100.01]
for h in (1, 2, 3):
    r = ret_from_entry(spy, h)
    rows = []
    drift = r.dropna().mean()
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        sel = gate_at_entry[(gate_at_entry >= lo) & (gate_at_entry < hi)].index
        v = r.reindex(sel).dropna()
        s = summarize(v.values, f"h={h} rel in [{lo},{hi})"
                                + ("   <-- LIVE 3.57" if lo <= LIVE_REL < hi else ""))
        if s["n"]:
            s["edge_pp"] = round(100 * (v.mean() - drift), 3)
            s["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
            s["x_cost"] = round(100 * v.mean() * 100 / 2.0, 1)
        rows.append(s)
    show(rows, f"MARGINAL rel-range buckets, SPY h={h}  (all-days drift {100*drift:+.3f}%)")
    # same on the abs-range form (today 1.98)
    rows = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        sel = abs_at_entry[(abs_at_entry >= lo) & (abs_at_entry < hi)].index
        v = r.reindex(sel).dropna()
        s = summarize(v.values, f"h={h} ABS in [{lo},{hi})"
                                + ("   <-- LIVE 1.98" if lo <= LIVE_ABS < hi else ""))
        if s["n"]:
            s["edge_pp"] = round(100 * (v.mean() - drift), 3)
            s["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
        rows.append(s)
    show(rows, f"MARGINAL abs-range buckets, SPY h={h}")

# ==========================================================================
print("\n" + "=" * 78)
print("ATTACK 2 -- placebo anchor ladder k=-8..+8, gate applied at each anchor")
print("=" * 78)
for h in (1, 3):
    rows = []
    for k in range(-8, 9):
        p2, _ = anchor_positions(cal, nfp, k)
        p2 = [i for i in p2 if i + 1 < len(cal)]
        ent = pd.DatetimeIndex([cal[i + 1] for i in p2])
        g = pd.Series([REL.iloc[i] for i in p2], index=ent)
        sel = g[g <= 15.0].index
        r = ret_from_entry(spy, h)
        v = r.reindex(sel).dropna()
        rows.append({"k": k, "h": h, "n": len(v),
                     "mean_pct": round(100 * v.mean(), 3),
                     "hit": round(100 * (v > 0).mean(), 1),
                     "sign_p": round(sign_test(int((v > 0).sum()), len(v)), 4)})
    df = pd.DataFrame(rows).sort_values("mean_pct", ascending=False)
    print(f"\n  gated anchor ladder h={h}, best first (LIVE config is k=-2):")
    print(df.to_string(index=False))

# ==========================================================================
print("\n" + "=" * 78)
print("ATTACK 3 -- concentration, drop-best, year histogram")
print("=" * 78)
for h in (1,):
    r = ret_from_entry(spy, h)
    sel = gate_at_entry[gate_at_entry <= 15.0].index
    v = r.reindex(sel).dropna()
    order = np.argsort(-v.values)
    print("  full: %+.3f%% n=%d  bootstrap P(mean<=0)=%.3f"
          % (100 * v.mean(), len(v), bootstrap_p_le0(v.values)))
    for k in (1, 2, 3, 5):
        cut = v.drop(index=v.index[order[:k]])
        print("   drop-best-%d: %+.3f%% n=%d hit %.1f%% sign p %.4f  (x cost %.1f)"
              % (k, 100 * cut.mean(), len(cut), 100 * (cut > 0).mean(),
                 sign_test(int((cut > 0).sum()), len(cut)), 100 * cut.mean() * 100 / 2.0))
    by_yr = (100 * v).groupby(v.index.year).agg(["sum", "count", "mean"]).round(2)
    print("\n  by year:\n", by_yr.to_string())
    for drop in (2009, 2023, 2008):
        cut = v[v.index.year != drop]
        print("   ex-%d: %+.3f%% n=%d hit %.1f%%"
              % (drop, 100 * cut.mean(), len(cut), 100 * (cut > 0).mean()))
    tot = v.sum()
    print("   top-3 episodes = %+.2fpp of %+.2fpp total (%.0f%%)"
          % (100 * v.values[order[:3]].sum(), 100 * tot,
             100 * v.values[order[:3]].sum() / tot))

# ==========================================================================
print("\n" + "=" * 78)
print("ATTACK 4 -- definition neighbours: nudge every threshold to its neighbour")
print("=" * 78)
rows = []
for win in (10, 21, 42):
    for lb in (126, 252, 504):
        for form in ("rel", "abs"):
            G = range_pctile(win, lb, form)
            g = pd.Series([G.iloc[i] for i in anchor_pos], index=entry)
            for thr in (10, 15, 20):
                sel = g[g <= thr].index
                r = ret_from_entry(spy, 1)
                v = r.reindex(sel).dropna()
                if len(v) < 8:
                    continue
                rows.append({"win": win, "lb": lb, "form": form, "thr": thr,
                             "n": len(v), "mean_pct": round(100 * v.mean(), 3),
                             "hit": round(100 * (v > 0).mean(), 1),
                             "t": round(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), 2),
                             "live_val": round(G.iloc[-1], 2),
                             "live_on": bool(G.iloc[-1] <= thr)})
df = pd.DataFrame(rows)
print(df.to_string(index=False))
print("\n  neighbour summary: %d of %d specs positive; mean %+.3f%%; min %+.3f%% max %+.3f%%"
      % (int((df.mean_pct > 0).sum()), len(df), df.mean_pct.mean(),
         df.mean_pct.min(), df.mean_pct.max()))
print("  live-ON specs only: n=%d  mean %+.3f%%   live-OFF specs: n=%d mean %+.3f%%"
      % (int(df.live_on.sum()), df[df.live_on].mean_pct.mean(),
         int((~df.live_on).sum()), df[~df.live_on].mean_pct.mean()))

# ==========================================================================
print("\n" + "=" * 78)
print("ATTACK 5 -- cluster depth of the STATE at the anchor vs today's depth")
print("=" * 78)
on = (REL <= 15.0).fillna(False)
depth = on.groupby((~on).cumsum()).cumcount() + 1
depth = depth.where(on, 0)
print("  today's consecutive sessions with rel-range pctile <= 15: %d" % depth.iloc[-1])
r = ret_from_entry(spy, 1)
d_at = pd.Series([depth.iloc[i] for i in anchor_pos], index=entry)
sel = gate_at_entry[gate_at_entry <= 15.0].index
rows = []
for lo, hi in ((1, 5), (5, 15), (15, 40), (40, 10000)):
    s2 = d_at[(d_at >= lo) & (d_at < hi)].index.intersection(sel)
    v = r.reindex(s2).dropna()
    x = summarize(v.values, f"state depth [{lo},{hi}) sessions")
    if x["n"]:
        x["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
    rows.append(x)
show(rows, "h=1 by cluster depth of the dead-range state at the anchor")

# ==========================================================================
print("\n" + "=" * 78)
print("ATTACK 6 -- multiplicity charge for the grid this search ran")
print("=" * 78)
grid_t = []
for h in (1, 2, 3, 5):
    for thr in (5, 10, 15, 25, 50):
        for form, G in (("rel", REL), ("abs", ABS)):
            g = pd.Series([G.iloc[i] for i in anchor_pos], index=entry)
            sel = g[g <= thr].index
            v = ret_from_entry(spy, h).reindex(sel).dropna()
            if len(v) > 3:
                grid_t.append(abs(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))))
best = max(grid_t)
m = len(grid_t)
print("  grid searched: 4 horizons x 5 thresholds x 2 range forms = %d cells" % m)
print("  best |t| in the grid = %.3f" % best)
from math import erfc, sqrt
p_one = 0.5 * erfc(best / sqrt(2))
print("  nominal one-sided p = %.4f   Sidak over %d looks = %.4f"
      % (p_one, m, 1 - (1 - p_one) ** m))
print("  (the cells overlap heavily, so Sidak is conservative; quoted as a bound)")
print("\nDONE.")
