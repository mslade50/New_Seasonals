"""C9 round 1 -- the volatility complex into the payrolls print, out of a
compressed 21-day VIX range.

The plain NFP ladder is CLOSED in this repo on four vehicles, so the ONLY
thing that can survive is the CROSSING with the compressed-range state.
Therefore the load-bearing tests are:
  1. gate attribution: NFP anchors WITH and WITHOUT the compression gate, and
     the compression gate ALONE on non-NFP days.
  2. a placebo anchor ladder, k = -8..+8 sessions from the print.
  3. midterm-year and September splits (both already negative in the repo).
  4. the SVXY leverage break (-1x -> -0.5x on 2018-02-28) as a forced era cut.

Anchor = the session 2 td BEFORE the print (today's analogue, 2026-09-02 with
NFP on 2026-09-04). Entry lag=1 -> MOC on the session 1 td before the print,
so h=1 is the print session's own move.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa: F401,F403,E402
from pitch_lab import (close_panel, fwd_lag, declusters, summarize, sign_test,
                       cluster_note, rolling_on_valid, load_events, show,
                       anchor_positions, bootstrap_p_le0)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)

TK = ["^VIX", "^VIX3M", "SVXY", "UVXY", "SPY"]
px = close_panel(TK)
cal = px["SPY"].dropna().index          # master trading calendar

vix = px["^VIX"]
rng21 = (rolling_on_valid(vix, lambda x: x.rolling(21).max())
         - rolling_on_valid(vix, lambda x: x.rolling(21).min()))
# two normalisations; report both, pick the one that matches the live 8.3
rng_pct_abs = rolling_on_valid(rng21, lambda x: x.rolling(252).rank(pct=True) * 100)
rng_rel = rng21 / rolling_on_valid(vix, lambda x: x.rolling(21).mean())
rng_pct_rel = rolling_on_valid(rng_rel, lambda x: x.rolling(252).rank(pct=True) * 100)

print("=" * 110)
print("LIVE 2026-09-01")
print(f"  VIX {vix.iloc[-1]:.2f}   21d range {rng21.iloc[-1]:.2f}   "
      f"abs-range trailing-252 pctile {rng_pct_abs.iloc[-1]:.1f}   "
      f"rel-range pctile {rng_pct_rel.iloc[-1]:.1f}")
print(f"  VIX3M {px['^VIX3M'].iloc[-1]:.2f}   SVXY {px['SVXY'].iloc[-1]:.2f}   "
      f"UVXY {px['UVXY'].iloc[-1]:.2f}")
print("=" * 110)

RNG = rng_pct_abs if abs(rng_pct_abs.iloc[-1] - 8.3) <= abs(rng_pct_rel.iloc[-1] - 8.3) else rng_pct_rel
print(f"  using the {'ABSOLUTE' if RNG is rng_pct_abs else 'RELATIVE'} range percentile "
      f"(live {RNG.iloc[-1]:.1f})")

nfp = load_events(["nfp"])["date"]
pos, kept = anchor_positions(cal, nfp, -2)
anchors_all = pd.DatetimeIndex([cal[p] for p in pos])
print(f"  NFP prints on the calendar: {len(kept)}  -> anchors at -2 td: {len(anchors_all)}")
print(f"  anchor span {anchors_all[0].date()} .. {anchors_all[-1].date()}")

comp = RNG <= 15.0
comp_anchor = anchors_all[comp.reindex(anchors_all).fillna(False).values]
print(f"  anchors with the 21d VIX range in its bottom 15%: {len(comp_anchor)}")


def cellstat(dates, tkr, h, lag=1, label=""):
    ss = px[tkr].dropna()
    f = fwd_lag(ss, h, lag=lag)
    v = f.reindex(pd.DatetimeIndex(dates)).dropna()
    if len(v) == 0:
        return {"label": label, "n": 0}
    drift = 100 * f.dropna().mean()
    st = summarize(v.values, label)
    st["drift_pct"] = round(drift, 3)
    st["excess_pp"] = round(st["mean_pct"] - drift, 3)
    st["signp"] = round(sign_test(int((v.values > 0).sum()), len(v)), 4)
    return st


# ---------------------------------------------------------------------------
# 1. THE OBJECT: what does the vol complex do into a payrolls print?
# ---------------------------------------------------------------------------
print("\n" + "=" * 110)
print("1. VOL COMPLEX INTO NFP -- anchor = print minus 2 td, entry lag=1")
print("=" * 110)
for tkr in ("^VIX", "^VIX3M", "SVXY", "UVXY", "SPY"):
    rows = []
    for h in (1, 2, 3, 5):
        rows.append(cellstat(anchors_all, tkr, h, label=f"ALL NFP h={h}"))
        rows.append(cellstat(comp_anchor, tkr, h, label=f"COMPRESSED h={h}"))
    show(rows, tkr)

# ---------------------------------------------------------------------------
# 2. GATE ATTRIBUTION -- does the compression gate DO anything?
# ---------------------------------------------------------------------------
print("\n" + "=" * 110)
print("2. GATE ATTRIBUTION  (h chosen per vehicle below; report h=1..3)")
print("=" * 110)
not_comp = anchors_all.difference(comp_anchor)
comp_days_all = cal[comp.reindex(cal).fillna(False).values]
nonnfp_comp = comp_days_all.difference(anchors_all)
for tkr in ("^VIX", "SVXY", "UVXY"):
    for h in (1, 3):
        rows = [cellstat(anchors_all, tkr, h, label="NFP anchors, gate OFF"),
                cellstat(comp_anchor, tkr, h, label="NFP anchors, gate ON"),
                cellstat(not_comp, tkr, h, label="NFP anchors the gate DISCARDS"),
                cellstat(nonnfp_comp, tkr, h, label="compressed range, NO NFP (gate alone)")]
        show(rows, f"{tkr} h={h}")
        on = [r for r in rows if r["label"] == "NFP anchors, gate ON"][0]
        off = [r for r in rows if r["label"] == "NFP anchors, gate OFF"][0]
        disc = [r for r in rows if "DISCARDS" in r["label"]][0]
        if on["n"] and off["n"]:
            print(f"    gate moves {tkr} h={h} from {off['mean_pct']:+.3f}% to "
                  f"{on['mean_pct']:+.3f}% while discarding {off['n']-on['n']} of "
                  f"{off['n']} anchors; the DISCARDED set is {disc['mean_pct']:+.3f}%")

# ---------------------------------------------------------------------------
# 3. PLACEBO ANCHOR LADDER
# ---------------------------------------------------------------------------
print("\n" + "=" * 110)
print("3. PLACEBO ANCHOR LADDER -- slide the anchor k = -8..+8 td from the print,")
print("   compression gate held ON.  A spike at k=-2 that decays either side is")
print("   an event; a plateau or a bigger spike elsewhere is not.")
print("=" * 110)
for tkr in ("^VIX", "SVXY"):
    for h in (1, 3):
        rows = []
        for k in range(-8, 9):
            p2, _ = anchor_positions(cal, nfp, k)
            a2 = pd.DatetimeIndex([cal[p] for p in p2])
            a2 = a2[comp.reindex(a2).fillna(False).values]
            if len(a2) < 5:
                continue
            st = cellstat(a2, tkr, h, label=f"k={k:+d}")
            rows.append(st)
        show(rows, f"{tkr} h={h}, gate ON")
        best = max(rows, key=lambda r: r["excess_pp"])
        true = [r for r in rows if r["label"] == "k=-2"]
        if true:
            rank = sum(1 for r in rows if r["excess_pp"] > true[0]["excess_pp"]) + 1
            print(f"    TRUE anchor k=-2 excess {true[0]['excess_pp']:+.3f}pp ranks "
                  f"{rank} of {len(rows)} rungs; best rung {best['label']} at "
                  f"{best['excess_pp']:+.3f}pp")

# ---------------------------------------------------------------------------
# 4. MIDTERM / SEPTEMBER / ERA
# ---------------------------------------------------------------------------
print("\n" + "=" * 110)
print("4. MIDTERM, SEPTEMBER, AND THE SVXY LEVERAGE BREAK")
print("=" * 110)
for tkr in ("^VIX", "SVXY", "UVXY"):
    for h in (1, 3):
        yr = comp_anchor.year
        mo = comp_anchor.month
        rows = [
            cellstat(comp_anchor, tkr, h, label="gate ON, all"),
            cellstat(comp_anchor[yr % 4 == 2], tkr, h, label="midterm years"),
            cellstat(comp_anchor[yr % 4 != 2], tkr, h, label="non-midterm"),
            cellstat(comp_anchor[mo == 9], tkr, h, label="September prints"),
            cellstat(comp_anchor[mo != 9], tkr, h, label="non-September"),
            cellstat(comp_anchor[(yr % 4 == 2) & (mo == 9)], tkr, h,
                     label="TODAY'S CELL: Sept in a midterm"),
            cellstat(comp_anchor[comp_anchor < pd.Timestamp("2018-02-28")], tkr, h,
                     label="pre 2018-02-28 (SVXY -1x era)"),
            cellstat(comp_anchor[comp_anchor >= pd.Timestamp("2018-02-28")], tkr, h,
                     label="post 2018-02-28 (SVXY -0.5x era)"),
        ]
        show(rows, f"{tkr} h={h}")

# ---------------------------------------------------------------------------
# 5. CONCENTRATION on the best-looking cell + dates
# ---------------------------------------------------------------------------
print("\n" + "=" * 110)
print("5. EPISODE DATES + CONCENTRATION")
print("=" * 110)
print("  compressed-range NFP anchors:",
      ", ".join(str(d.date()) for d in comp_anchor))
for tkr in ("^VIX", "SVXY"):
    for h in (1, 3):
        ss = px[tkr].dropna()
        f = fwd_lag(ss, h, lag=1)
        v = f.reindex(comp_anchor).dropna()
        if len(v) < 3:
            continue
        print(f"\n  {tkr} h={h}: n={len(v)}  " + cluster_note(v.index, v.values, k=2))
        print(f"    bootstrap P(mean<=0) = {bootstrap_p_le0(v.values):.3f}")
