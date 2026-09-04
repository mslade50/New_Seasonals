"""C6 FOLLOW-UP -- teardown of the rescue path b2's attack 4 surfaced.

C6 AS PITCHED IS DEAD (b2): the joint washout-x-miners-bid state crossed with the
NFP anchor has 2 days in 22 years, and today's GLD drawdown bucket (dd <= -10%,
live -18.78%) is the wrong-signed half of the parent -- 3 episodes paying
-1.187%/-1.667%/-2.116% at h=2/3/5 against the shallow half's
+1.062%/+0.745%/+1.190%.

While hunting that kill, b2 attack 4 turned up a DIFFERENT cell that IS live:
drop the miner leg entirely and the plain "GLD 5-day washout at the pre-payrolls
anchor" pays +0.520% on n=33 at a 75.8% hit, sign p 0.0023, against GLD's own
h=1 drift of +0.047%.

Registry 2026-08-07: "post-hoc sign flips recovered from a kill report" carry the
multiple comparisons of the search that found them, and both survivors of that
morning died on re-examination. So this gets the full battery and an explicit
multiplicity charge, not a promotion.

  1. is it the ANCHOR or the washout? placebo anchor ladder k=-8..+8
  2. tdom-matched and month-matched controls
  3. threshold ladder on r5, and the drawdown split
  4. era / midterm splits
  5. concentration and drop-best
  6. the multiplicity charge for the buckets b2 attack 4 actually looked at
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import (load_prices, load_events, anchor_positions, summarize,
                       show, sign_test, bootstrap_p_le0, pct_rank,
                       rolling_on_valid)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 230)

raw = load_prices(["GLD", "SLV", "GDX"])
px = pd.DataFrame({t: raw[t]["Close"] for t in ["GLD", "SLV", "GDX"]}).dropna(subset=["GLD"])
cal = px.index
gld = px["GLD"]
r5 = pct_rank(gld, 5)
hi252 = rolling_on_valid(gld, lambda x: x.rolling(252).max())
dd = gld / hi252 - 1.0
print("live 2026-09-02: GLD r5 %.1f   dd from 52wh %.2f%%" % (r5.iloc[-1], 100 * dd.iloc[-1]))

nfp = load_events(["nfp"])["date"]
pos, _ = anchor_positions(cal, nfp, -2)
anchor_pos = [i for i in pos if i + 1 < len(cal)]
entry = pd.DatetimeIndex([cal[i + 1] for i in anchor_pos])
r5_at = pd.Series([r5.iloc[i] for i in anchor_pos], index=entry)
dd_at = pd.Series([dd.iloc[i] for i in anchor_pos], index=entry)
_one = pd.Series(1, index=cal)
tdom = pd.Series(_one.groupby([cal.year, cal.month]).cumcount().values + 1, index=cal)


def ret_from_entry(s, h):
    return s.shift(-h) / s - 1.0


# ==========================================================================
print("\n" + "=" * 78)
print("ATTACK 1 -- placebo anchor ladder: is it the print, or just the washout?")
print("=" * 78)
for h in (1, 2, 3):
    rows = []
    for k in range(-8, 9):
        p2, _ = anchor_positions(cal, nfp, k)
        p2 = [i for i in p2 if i + 1 < len(cal)]
        ent = pd.DatetimeIndex([cal[i + 1] for i in p2])
        g = pd.Series([r5.iloc[i] for i in p2], index=ent)
        sel = g[g <= 15].index
        v = ret_from_entry(gld, h).reindex(sel).dropna()
        rows.append({"k": k, "h": h, "n": len(v), "mean_pct": round(100 * v.mean(), 3),
                     "hit": round(100 * (v > 0).mean(), 1),
                     "sign_p": round(sign_test(int((v > 0).sum()), len(v)), 4)})
    df = pd.DataFrame(rows).sort_values("mean_pct", ascending=False)
    print(f"\n  h={h} washout-gated anchor ladder, best first (LIVE is k=-2):")
    print(df.to_string(index=False))

# also: the washout on ALL days, no anchor at all (the honest parent)
print("\n  the washout with NO event anchor at all (every session, declustered 5td):")
from pitch_lab import declusters
for h in (1, 2, 3):
    days = cal[(r5 <= 15).fillna(False).values]
    epi = declusters(days, 5, cal)
    v = ret_from_entry(gld, h).reindex(epi).dropna()
    drift = ret_from_entry(gld, h).dropna().mean()
    print("    h=%d  %+.3f%% n=%d hit %.1f%% sign p %.4f   drift %+.3f%% -> edge %+.3fpp"
          % (h, 100 * v.mean(), len(v), 100 * (v > 0).mean(),
             sign_test(int((v > 0).sum()), len(v)), 100 * drift,
             100 * (v.mean() - drift)))

# ==========================================================================
print("\n" + "=" * 78)
print("ATTACK 2 -- tdom-matched and month-matched controls")
print("=" * 78)
tdom_set = sorted(set(tdom.loc[entry].values))
for h in (1, 2, 3):
    r = ret_from_entry(gld, h)
    sel = r5_at[r5_at <= 15].index
    v = r.reindex(sel).dropna()
    ctl_tdom = r[tdom.isin(tdom_set).values].dropna()
    # tdom-matched AND washout-matched: the honest control for this cell
    wash_all = (r5 <= 15).reindex(cal).fillna(False)
    ctl_wash_tdom = r[(wash_all & tdom.isin(tdom_set)).values].dropna()
    print("  h=%d cell %+.3f%% (n=%d) | tdom-matched %+.3f%% (n=%d) -> %+.3fpp | "
          "WASHOUT+tdom-matched %+.3f%% (n=%d) -> %+.3fpp"
          % (h, 100 * v.mean(), len(v), 100 * ctl_tdom.mean(), len(ctl_tdom),
             100 * (v.mean() - ctl_tdom.mean()), 100 * ctl_wash_tdom.mean(),
             len(ctl_wash_tdom), 100 * (v.mean() - ctl_wash_tdom.mean())))
    by_m = r.dropna().groupby(r.dropna().index.month).mean() * 100
    print("     GLD h=%d by month (%%): %s" % (h, {int(k): round(x, 3) for k, x in by_m.items()}))
    sep = pd.DatetimeIndex([d for d in sel if d.month == 9])
    vs = r.reindex(sep).dropna()
    if len(vs):
        print("     SEPTEMBER instances: %+.3f%% n=%d hit %.0f%% sign p %.4f"
              % (100 * vs.mean(), len(vs), 100 * (vs > 0).mean(),
                 sign_test(int((vs > 0).sum()), len(vs))))

# ==========================================================================
print("\n" + "=" * 78)
print("ATTACK 3 -- threshold ladder on r5 (marginal buckets) + drawdown split")
print("=" * 78)
EDG = [0, 5, 10, 15, 25, 50, 75, 100.01]
for h in (1, 3):
    r = ret_from_entry(gld, h)
    drift = r.dropna().mean()
    rows = []
    for lo, hi in zip(EDG[:-1], EDG[1:]):
        sel = r5_at[(r5_at >= lo) & (r5_at < hi)].index
        v = r.reindex(sel).dropna()
        s = summarize(v.values, f"h={h} r5 in [{lo},{hi})"
                                + ("  <-- LIVE 9.1" if lo <= 9.1 < hi else ""))
        if s["n"]:
            s["edge_pp"] = round(100 * (v.mean() - drift), 3)
            s["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
        rows.append(s)
    show(rows, f"MARGINAL r5 buckets at the NFP anchor, GLD h={h} (drift {100*drift:+.3f}%)")
    rows = []
    sel = r5_at[r5_at <= 15].index
    for lbl, m in (("DEEP dd<=-10% (LIVE -18.8%)", dd_at.reindex(sel) <= -0.10),
                   ("shallow dd>-10%", dd_at.reindex(sel) > -0.10)):
        s2 = pd.Index(sel)[m.fillna(False).values]
        v = r.reindex(s2).dropna()
        s = summarize(v.values, f"h={h} {lbl}")
        if s["n"]:
            s["edge_pp"] = round(100 * (v.mean() - drift), 3)
            s["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
        rows.append(s)
    show(rows, f"drawdown split inside the cell, h={h}")

# ==========================================================================
print("\n" + "=" * 78)
print("ATTACK 4 -- era / midterm splits")
print("=" * 78)
for h in (1, 3):
    r = ret_from_entry(gld, h)
    sel = r5_at[r5_at <= 15].index
    v = r.reindex(sel).dropna()
    rows = []
    for lbl, msk in (("pre-2018", [d.year < 2018 for d in v.index]),
                     ("2018+", [d.year >= 2018 for d in v.index]),
                     ("MIDTERM", [d.year % 4 == 2 for d in v.index]),
                     ("non-midterm", [d.year % 4 != 2 for d in v.index])):
        x = v[msk]
        s = summarize(x.values, f"h={h} {lbl}")
        if s["n"]:
            s["sign_p"] = round(sign_test(int((x > 0).sum()), len(x)), 4)
        rows.append(s)
    show(rows, f"splits h={h}")

# ==========================================================================
print("\n" + "=" * 78)
print("ATTACK 5 -- concentration, drop-best, year histogram, bootstrap")
print("=" * 78)
r = ret_from_entry(gld, 1)
sel = r5_at[r5_at <= 15].index
v = r.reindex(sel).dropna()
order = np.argsort(-v.values)
print("  full h=1: %+.3f%% n=%d hit %.1f%% bootstrap P(mean<=0)=%.4f"
      % (100 * v.mean(), len(v), 100 * (v > 0).mean(), bootstrap_p_le0(v.values)))
for k in (1, 2, 3, 5):
    cut = v.drop(index=v.index[order[:k]])
    print("   drop-best-%d: %+.3f%% n=%d hit %.1f%% sign p %.4f  (x cost %.1f, 6bp round trip)"
          % (k, 100 * cut.mean(), len(cut), 100 * (cut > 0).mean(),
             sign_test(int((cut > 0).sum()), len(cut)), 100 * cut.mean() * 100 / 6.0))
print("\n  by year:\n", (100 * v).groupby(v.index.year).agg(["sum", "count", "mean"]).round(2).to_string())
print("  dates and returns:", {str(d.date()): round(100 * x, 2) for d, x in v.items()})

# ==========================================================================
print("\n" + "=" * 78)
print("ATTACK 6 -- multiplicity charge for the buckets the search actually ran")
print("=" * 78)
ts = []
for h in (1, 2, 3):
    for lbl, sel2 in (("all NFP", entry),
                      ("dd<=-10%", dd_at[dd_at <= -0.10].index),
                      ("r5<=15", r5_at[r5_at <= 15].index)):
        vv = ret_from_entry(gld, h).reindex(sel2).dropna()
        if len(vv) > 3:
            ts.append(abs(vv.mean() / (vv.std(ddof=1) / np.sqrt(len(vv)))))
from math import erfc, sqrt
best = max(ts)
p_one = 0.5 * erfc(best / sqrt(2))
print("  b2 attack 4 ran 3 horizons x 3 buckets = %d cells; best |t| = %.3f" % (len(ts), best))
print("  nominal one-sided p = %.4f   Sidak over %d looks = %.4f"
      % (p_one, len(ts), 1 - (1 - p_one) ** len(ts)))
print("  plus the vehicle/leg choices made upstream in b2 (GLD vs GDX vs SLV, miner leg on/off)")
print("\nDONE.")
