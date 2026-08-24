"""C4 round 1 -- long XLI on an r5-rank washout while a cyclical peer prints a
52-week high.  Adversarial pass: the honest order is

  (0) live state reconstruction, so the numbers in the candidate are mine
  (1) GATE-OFF FIRST (2026-08-19 registry rule): what does the PLAIN state
      "XLI r5 rank <= 5" pay outright, before any peer condition?
  (2) the gate ON: does "+ peer at a 52w high" ADD to that or SUBTRACT from it?
  (3) LEG ATTRIBUTION (2026-08-07 + 2026-08-19 rules): long XLI alone against
      short XLB alone against short XLE alone, before any spread is priced
  (4) the registry-collision measurement the brief demands: is this materially
      the book's own dip-buy family (k=5 washout / h=3, +0.534%)?
  (5) cost at 4 bps/leg, and the book-overlap number (the scanner staged five
      industrial-complex OLV longs this morning)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)

TKS = ["XLI", "XLB", "XLE", "SPY", "XLY", "XLF"]
px = close_panel(TKS)
px = px.dropna(how="any")          # all NYSE sector ETFs, identical calendar
idx = px.index
print("panel %s .. %s  N=%d" % (idx[0].date(), idx[-1].date(), len(idx)))

# ---------------------------------------------------------------- live state
r5 = {t: pct_rank(px[t], 5, 252) for t in TKS}
r21 = {t: pct_rank(px[t], 21, 252) for t in TKS}
hi252 = {t: rolling_on_valid(px[t], lambda x: x.rolling(252).max()) for t in TKS}
offhi = {t: px[t] / hi252[t] - 1.0 for t in TKS}

print("\n--- 0. live state on the last bar (%s) ---" % idx[-1].date())
for t in TKS:
    print("  %-4s r5rank %6.1f  r21rank %6.1f  r5 %+7.2f%%  off-52wh %+7.2f%%"
          % (t, r5[t].iloc[-1], r21[t].iloc[-1],
             100 * (px[t].iloc[-1] / px[t].iloc[-6] - 1.0),
             100 * offhi[t].iloc[-1]))

# ---------------------------------------------------------------- masks
XLI_WASH = r5["XLI"] <= 5.0                       # today 2.4
PEER_HI = {p: offhi[p] >= -0.0025 for p in ("XLB", "XLE")}   # within 0.25%
ANY_PEER_HI = PEER_HI["XLB"] | PEER_HI["XLE"]

print("\n--- mask populations (full history) ---")
print("  XLI r5rank<=5           : %4d days" % int(XLI_WASH.sum()))
print("  XLB at 52wh (<=0.25%%)   : %4d days" % int(PEER_HI["XLB"].sum()))
print("  XLE at 52wh (<=0.25%%)   : %4d days" % int(PEER_HI["XLE"].sum()))
print("  either peer at 52wh     : %4d days" % int(ANY_PEER_HI.sum()))
print("  BOTH conditions (C4)    : %4d days" % int((XLI_WASH & ANY_PEER_HI).sum()))
print("  XLB-only version        : %4d days" % int((XLI_WASH & PEER_HI["XLB"]).sum()))
print("  XLE-only version        : %4d days" % int((XLI_WASH & PEER_HI["XLE"]).sum()))
print("  BOTH peers at 52wh (today's literal state): %4d days"
      % int((XLI_WASH & PEER_HI["XLB"] & PEER_HI["XLE"]).sum()))


def cell(mask, legs, hs=(1, 2, 3, 5, 10), lab="", min_gap=None):
    rows = []
    for h in hs:
        ret = vehicle_ret(px, legs, h, 1)
        valid = ret.notna()
        sig = idx[mask.reindex(idx, fill_value=False).values & valid.values]
        if len(sig) == 0:
            rows.append({"label": f"{lab} h={h}", "n": 0})
            continue
        epi = declusters(sig, min_gap or h, idx)
        r = summarize(ret.loc[epi].values, f"{lab} h={h}")
        in_span = (idx >= sig[0]) & (idx <= sig[-1]) & valid.values
        r["ctrl_a_pct"] = round(100 * ret[in_span].mean(), 3)
        r["ctrl_b_pct"] = round(100 * ret[valid].mean(), 3)
        loc = local_control(idx[valid.values], sig)
        r["ctrl_c_pct"] = round(100 * ret.loc[loc].mean(), 3)
        r["edge_vs_a"] = round(r["mean_pct"] - r["ctrl_a_pct"], 3)
        r["n_days"] = len(sig)
        rows.append(r)
    return rows


# ================================================================= 1. GATE OFF
print("\n" + "=" * 100)
print("1. GATE OFF -- the plain state.  Long XLI on r5 rank <= 5, no peer condition.")
print("=" * 100)
show(cell(XLI_WASH, [("XLI", 1.0)], lab="XLI outright"),
     "1a. long XLI, r5rank<=5, gate OFF")

# ================================================================== 2. GATE ON
print("\n" + "=" * 100)
print("2. GATE ON -- add 'a cyclical peer prints a 52-week high'.")
print("=" * 100)
show(cell(XLI_WASH & ANY_PEER_HI, [("XLI", 1.0)], lab="XLI + peerHi"),
     "2a. long XLI, gate ON (either peer)")
show(cell(XLI_WASH & ~ANY_PEER_HI, [("XLI", 1.0)], lab="XLI + NO peerHi"),
     "2b. COMPLEMENT: long XLI, gate OFF-side (no peer at a high)")
show(cell(XLI_WASH & PEER_HI["XLB"], [("XLI", 1.0)], lab="XLI + XLBhi"),
     "2c. long XLI, XLB-at-high only")
show(cell(XLI_WASH & PEER_HI["XLE"], [("XLI", 1.0)], lab="XLI + XLEhi"),
     "2d. long XLI, XLE-at-high only")

# ============================================================ 3. ATTRIBUTION
print("\n" + "=" * 100)
print("3. LEG ATTRIBUTION on the gated state -- price every leg before any spread.")
print("=" * 100)
M = XLI_WASH & ANY_PEER_HI
for lab, legs in [("LONG XLI alone      ", [("XLI", 1.0)]),
                  ("SHORT XLB alone     ", [("XLB", -1.0)]),
                  ("SHORT XLE alone     ", [("XLE", -1.0)]),
                  ("SHORT SPY alone     ", [("SPY", -1.0)]),
                  ("PAIR XLI - XLB      ", [("XLI", 1.0), ("XLB", -1.0)]),
                  ("PAIR XLI - XLE      ", [("XLI", 1.0), ("XLE", -1.0)]),
                  ("PAIR XLI - SPY      ", [("XLI", 1.0), ("SPY", -1.0)]),
                  ("PAIR XLI - 0.5(B+E) ", [("XLI", 1.0), ("XLB", -0.5), ("XLE", -0.5)])]:
    show(cell(M, legs, hs=(1, 3, 5, 10), lab=lab), None if lab else "")

# ====================================================== 4. REGISTRY COLLISION
print("\n" + "=" * 100)
print("4. REGISTRY COLLISION -- is this the book's own dip-buy family?")
print("   The book's family is 'a generic 5-day washout reversal on liquid names,")
print("   k=5/h=3, +0.534%, t 4.17, 2018+ +0.709%' (registry 2026-08-14).")
print("=" * 100)
rows = []
for t in ["XLI", "XLB", "XLE", "XLY", "XLF", "SPY"]:
    m = pct_rank(px[t], 5, 252) <= 5.0
    ret = fwd_lag(px[t], 3, 1)
    valid = ret.notna()
    sig = idx[m.reindex(idx, fill_value=False).values & valid.values]
    epi = declusters(sig, 3, idx)
    r = summarize(ret.loc[epi].values, f"{t} r5rank<=5 h=3")
    r["ctrl_b_pct"] = round(100 * ret[valid].mean(), 3)
    r["n_days"] = len(sig)
    rows.append(r)
show(rows, "4a. the SAME washout rule across six liquid sector/index vehicles")
print("   -> if XLI is unremarkable inside this row set, C4 is the family, not an idea.")

# 2018+ slice of the plain XLI cell, to line up against the registry's +0.709%
ret3 = fwd_lag(px["XLI"], 3, 1)
sig = idx[XLI_WASH.reindex(idx, fill_value=False).values & ret3.notna().values]
epi = declusters(sig, 3, idx)
show(era_split(epi, ret3.loc[epi].values), "4b. plain XLI washout h=3, era split")

# ======================================================= 5. COST + book overlap
print("\n" + "=" * 100)
print("5. COST and BOOK OVERLAP")
print("=" * 100)
print("  sector ETF ~4 bps round trip; a two-leg pair = 8 bps. Need >=5x -> "
      "outright must clear 20 bps (0.20%), pair must clear 40 bps (0.40%).")
staged_ind = ["LUV", "CHRW", "CMI", "WWD"]      # OLV longs staged this morning
try:
    bpx = close_panel(staged_ind + ["XLI"]).dropna(how="any")
    r = bpx.pct_change().dropna().tail(504)
    bask = r[staged_ind].mean(axis=1)
    print("\n  book overlap: the scanner staged OLV LONGS in %s this morning."
          % ", ".join(staged_ind))
    print("  daily-return corr of that equal-weight basket with XLI (last 504 sessions): "
          "%.3f" % bask.corr(r["XLI"]))
    b = np.polyfit(r["XLI"].values, bask.values, 1)[0]
    print("  basket beta to XLI = %.2f  -> a long-XLI pitch is a %.0f%% duplicate of "
          "exposure the book already staged." % (b, 100 * bask.corr(r["XLI"]) ** 2))
except Exception as e:                                          # pragma: no cover
    print("  book-overlap block failed:", e)

# ==================================================== 6. battery on the pitch
print("\n" + "=" * 100)
variants = {
    "r5rank<=2  + peerHi": (r5["XLI"] <= 2.0) & ANY_PEER_HI,
    "r5rank<=5  + peerHi": XLI_WASH & ANY_PEER_HI,
    "r5rank<=10 + peerHi": (r5["XLI"] <= 10.0) & ANY_PEER_HI,
    "r5rank<=20 + peerHi": (r5["XLI"] <= 20.0) & ANY_PEER_HI,
    "r5rank<=5  no gate ": XLI_WASH,
}
battery(px, M, [("XLI", 1.0), ("XLB", -1.0)], 5,
        "C4 PAIR long XLI / short XLB, gated", 4.0, variants=variants)
battery(px, M, [("XLI", 1.0)], 5,
        "C4 OUTRIGHT long XLI, gated", 4.0, variants=variants)
