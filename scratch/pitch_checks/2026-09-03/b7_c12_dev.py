"""C12 ROUND 3 / final kill attempt -- the two conditioners the registry demands
of exactly this shape of candidate, plus the event reference class.

b5 left C12 alive on definition robustness (50 of 54 neighbour specs positive)
and era stability, with three open objections (top-3 = 61% of total, drop-best-3
= 4.9x cost, live rel bucket [0,5) = +0.090%). What has NOT been run:

  1. The BULL-TAPE SELECTOR check the 2026-09-02 registry entry demands up front
     of any candidate whose state is "calm tape near a high": the above/below
     200-day base-rate split of the gated trigger days. Confirmed three times as
     a kill on other cells; owed here.
  2. The FRAGILITY DIAL condition. Today the dial's 10d-MA of the 63d column is
     87.9 -- among the highest readings in the whole 2016+ series. The 2026-08-21
     credit cell paid +0.408% below dial 50 and -1.511% above it. If C12's
     positive cell is a low-dial cell, today is the wrong day for it.
  3. Event reference class: does the same gate pay into CPI / PPI / FOMC? If it
     pays on all of them the payroll label is decoration (not fatal, but it
     re-prices the novelty); if only NFP, charge the 4-way event search.
  4. Horizon scan 1..10 to choose the horizon from evidence rather than assume it.
  5. episode_paths on the losers.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import (load_prices, load_events, anchor_positions, summarize,
                       show, sign_test, bootstrap_p_le0, horizon_scan,
                       episode_paths, rolling_on_valid)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 230)
ROOT = Path(__file__).resolve().parents[3]

raw = load_prices(["SPY", "^VIX"])
px = pd.DataFrame({t: raw[t]["Close"] for t in ["SPY", "^VIX"]}).dropna(subset=["SPY"])
cal = px.index
spy = px["SPY"]
vix = raw["^VIX"]["Close"].dropna()
rng21 = vix.rolling(21).max() - vix.rolling(21).min()
REL = ((rng21 / vix.rolling(21).mean()).rolling(252).rank(pct=True) * 100).reindex(cal).ffill(limit=3)

sma200 = rolling_on_valid(spy, lambda x: x.rolling(200).mean())
above200 = spy > sma200
hi252 = rolling_on_valid(spy, lambda x: x.rolling(252).max())
dist_hi = spy / hi252 - 1.0

nfp = load_events(["nfp"])["date"]
pos, _ = anchor_positions(cal, nfp, -2)
apos = [i for i in pos if i + 1 < len(cal)]
entry = pd.DatetimeIndex([cal[i + 1] for i in apos])
gate = pd.Series([REL.iloc[i] for i in apos], index=entry)
a200 = pd.Series([bool(above200.iloc[i]) for i in apos], index=entry)
adist = pd.Series([dist_hi.iloc[i] for i in apos], index=entry)
GATED = gate[gate <= 15.0].index


def r(h):
    return spy.shift(-h) / spy - 1.0


print("live 2026-09-02: rel-range pctile %.2f  SPY above 200d %s  dist from 52wh %.2f%%"
      % (REL.iloc[-1], bool(above200.iloc[-1]), 100 * dist_hi.iloc[-1]))

# ==========================================================================
print("\n" + "=" * 78)
print("ATTACK 1 -- BULL-TAPE SELECTOR check (registry 2026-09-02, owed up front)")
print("=" * 78)
base = float(above200.reindex(cal).dropna().mean())
sel_a200 = float(a200.reindex(GATED).mean())
all_anchor_a200 = float(a200.mean())
print("  SPY above its 200d: all sessions %.1f%%  |  all NFP anchors %.1f%%  |  "
      "GATED anchors %.1f%%  (n=%d)"
      % (100 * base, 100 * all_anchor_a200, 100 * sel_a200, len(GATED)))
n_below = int((~a200.reindex(GATED)).sum())
print("  gated anchors BELOW the 200d: %d of %d" % (n_below, len(GATED)))
for h in (1, 3):
    rows = []
    for lbl, m in (("gated & ABOVE 200d", a200.reindex(GATED).fillna(False)),
                   ("gated & BELOW 200d", ~a200.reindex(GATED).fillna(True))):
        s2 = pd.Index(GATED)[m.values]
        v = r(h).reindex(s2).dropna()
        x = summarize(v.values, f"h={h} {lbl}")
        if x["n"]:
            x["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
        rows.append(x)
    # the same split on the UNGATED anchors, so the gate's contribution is visible
    for lbl, m in (("ALL NFP & ABOVE 200d", a200.fillna(False)),
                   ("ALL NFP & BELOW 200d", ~a200.fillna(True))):
        s2 = pd.Index(entry)[m.values]
        v = r(h).reindex(s2).dropna()
        x = summarize(v.values, f"h={h} {lbl}")
        if x["n"]:
            x["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
        rows.append(x)
    show(rows, f"200d base-rate split, h={h}")

print("\n  distance-from-52w-high buckets inside the gated cell (today -1.64%):")
for h in (1,):
    rows = []
    for lo, hi in ((-0.02, 0.001), (-0.05, -0.02), (-0.15, -0.05), (-9, -0.15)):
        s2 = adist.reindex(GATED)
        s2 = pd.Index(GATED)[((s2 > lo) & (s2 <= hi)).fillna(False).values]
        v = r(h).reindex(s2).dropna()
        x = summarize(v.values, f"h={h} dist in ({100*lo:.0f}%,{100*hi:.1f}%]"
                                + ("  <-- LIVE -1.64%" if lo < -0.0164 <= hi else ""))
        if x["n"]:
            x["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
        rows.append(x)
    show(rows, "gated cell by distance from the 52-week high")

# ==========================================================================
print("\n" + "=" * 78)
print("ATTACK 2 -- FRAGILITY DIAL condition (today ma10(63d) = 87.9)")
print("=" * 78)
frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
frag.index = pd.to_datetime(frag.index)
ma10 = frag["63d"].rolling(10).mean()
print("  dial series %s .. %s   today's ma10(63d) = %.1f  (series pctile %.1f)"
      % (frag.index[0].date(), frag.index[-1].date(), ma10.iloc[-1],
         100 * (ma10 <= ma10.iloc[-1]).mean()))
dial_at = pd.Series([ma10.reindex([cal[i]]).iloc[0] for i in apos], index=entry)
have = dial_at.dropna().index
print("  NFP anchors with a dial reading: %d   gated ones: %d"
      % (len(have), len(pd.Index(GATED).intersection(have))))
for h in (1, 3):
    rows = []
    for lo, hi in ((0, 30), (30, 50), (50, 70), (70, 200)):
        s2 = dial_at[(dial_at >= lo) & (dial_at < hi)].index
        for lbl, sset in ((f"GATED dial [{lo},{hi})", pd.Index(GATED).intersection(s2)),
                          (f"all NFP dial [{lo},{hi})", pd.Index(entry).intersection(s2))):
            v = r(h).reindex(sset).dropna()
            x = summarize(v.values, f"h={h} {lbl}"
                          + ("  <-- LIVE 87.9" if lo <= 87.9 < hi else ""))
            if x["n"]:
                x["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
            rows.append(x)
    show(rows, f"fragility-dial buckets, h={h}")
    hi_g = pd.Index(GATED).intersection(dial_at[dial_at >= 70].index)
    if len(hi_g):
        print("   gated anchors at dial >= 70:",
              {str(d.date()): round(100 * r(h).get(d, np.nan), 2) for d in hi_g})
    else:
        print("   gated anchors at dial >= 70: NONE -- today has no precedent in the cell")

# ==========================================================================
print("\n" + "=" * 78)
print("ATTACK 3 -- event reference class: same gate into CPI / PPI / FOMC")
print("=" * 78)
for kind in ("nfp", "cpi", "ppi", "fomc_decision"):
    ev = load_events([kind])["date"]
    p2, _ = anchor_positions(cal, ev, -2)
    p2 = [i for i in p2 if i + 1 < len(cal)]
    ent = pd.DatetimeIndex([cal[i + 1] for i in p2])
    g = pd.Series([REL.iloc[i] for i in p2], index=ent)
    sel = g[g <= 15.0].index
    row = []
    for h in (1, 3):
        v = r(h).reindex(sel).dropna()
        row.append("h=%d %+.3f%% n=%d hit %.1f%% signp %.4f"
                   % (h, 100 * v.mean(), len(v), 100 * (v > 0).mean(),
                      sign_test(int((v > 0).sum()), len(v))))
    print("  %-14s gated pre-print long SPY:  %s" % (kind, "   |   ".join(row)))

# ==========================================================================
print("\n" + "=" * 78)
print("ATTACK 4 -- horizon scan 1..10 (choose the horizon from evidence)")
print("=" * 78)
# horizon_scan/episode_paths take the ANCHOR dates and apply lag themselves;
# GATED holds ENTRY dates, so feed the matching k=-2 anchors instead.
GATED_ANCHOR = pd.DatetimeIndex([cal[i] for i in apos if cal[i + 1] in set(GATED)])
print("  gated anchors (k=-2) fed to the lab: %d" % len(GATED_ANCHOR))
show(horizon_scan(px, GATED_ANCHOR, [("SPY", 1.0)], hs=tuple(range(1, 11)), lag=1,
                  min_gap=5),
     "gated cell, horizon scan (edge_pct = mean minus all-days drift)")

# ==========================================================================
print("\n" + "=" * 78)
print("ATTACK 5 -- episode paths and the losers")
print("=" * 78)
paths = episode_paths(px, GATED_ANCHOR, [("SPY", 1.0)], h=3, lag=1)
paths = (100 * paths).round(2)
worst = paths[1].sort_values().head(8)
print("  worst 8 print-session (day 1) outcomes:")
print(worst.to_string())
print("\n  day-1 distribution: mean %.3f  sd %.3f  p05 %.2f  p95 %.2f  min %.2f"
      % (paths[1].mean(), paths[1].std(), paths[1].quantile(.05),
         paths[1].quantile(.95), paths[1].min()))
print("  fraction of episodes with day-1 <= -1.0%%: %.1f%%   <= -0.5%%: %.1f%%"
      % (100 * (paths[1] <= -1.0).mean(), 100 * (paths[1] <= -0.5).mean()))
print("\n  tomorrow-specific tail: an 08:30 payrolls print is the single largest")
print("  scheduled one-session risk in the month; the cell's worst day-1 was")
print("  2008-06-05 at -3.19%% (the 2008-06-06 payroll shock).")
print("\nDONE.")
