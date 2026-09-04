"""C1 DEBT 3 + the three mandatory extras.

DEBT 3 -- the instrument's own record is 13 post-break. SVXY was -1x before
2018-02-28 and -0.5x after, so the 21-anchor NFP record mixes two instruments.
Settle whether the statistical case can rest on ^VIX (no leverage break) with
SVXY only as the vehicle. Post-2018 ^VIX record on the gated NFP set, exact
sign p. If BOTH the post-break vehicle record AND the post-break ^VIX record
are weak, that is a kill.

Because a3 established that the coherent object is the CLEAR-CALENDAR pooled
cell (runway >= 3 sessions to the next scheduled print) and not "payrolls",
every test below is run on BOTH the NFP-only set and the pooled set.

EXTRA 1 -- SVXY closed exactly AT its own-series trailing-252 maximum. Registry
2026-08-11: close_panel unions dates and silently corrupts a rolling 252-day
window, so distance-to-extreme is computed on the SINGLE instrument's own
series here. Then split the gated set on "within 1% of its own 252d high".
Never measured on this cell before.

EXTRA 2 -- registry collision, measured not asserted. Day-level Jaccard between
C1's anchors and the 2026-09-02 kill "'the range has been dead, then it broke'
is monotone in the WRONG direction" (VIX pop >= 8% out of a bottom-15% range,
long SPY at h=10). Precedent: 0.008 between two adjacent vol cells.

EXTRA 3 -- the tail. Gated worst case, and what a 30-60 bps ATR-risk position
loses on a repeat of 2025-07-30 (SVXY -4.13%, ^VIX +21.89%).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa: F401,F403,E402
from pitch_lab import (close_panel, load_prices, fwd_lag, summarize, sign_test,
                       load_events, rolling_on_valid, show, anchor_positions,
                       bootstrap_p_le0, wilder_atr, declusters)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)

RAW = load_prices(["SVXY", "^VIX", "UVXY", "SPY", "^VIX3M"])
px = close_panel(["^VIX", "^VIX3M", "SVXY", "UVXY", "SPY"])
cal = px["SPY"].dropna().index
vix = px["^VIX"]
rng21 = (rolling_on_valid(vix, lambda x: x.rolling(21).max())
         - rolling_on_valid(vix, lambda x: x.rolling(21).min()))
REL = rolling_on_valid(rng21 / rolling_on_valid(vix, lambda x: x.rolling(21).mean()),
                       lambda x: x.rolling(252).rank(pct=True) * 100)
G15 = REL <= 15.0
BREAK = pd.Timestamp("2018-02-28")

KINDS = ("nfp", "cpi", "ppi", "fomc_decision")
EV = {k: load_events([k])["date"] for k in KINDS}
ALL_PRINTS = pd.DatetimeIndex(sorted(pd.concat(list(EV.values())).unique()))
pos = pd.Series(range(len(cal)), index=cal)

svxy_h1 = fwd_lag(px["SVXY"].dropna(), 1, lag=1)
vix_h1 = -fwd_lag(px["^VIX"].dropna(), 1, lag=1)


def anchor_frame(kind, k=-2):
    p, kept = anchor_positions(cal, EV[kind], k)
    rows = []
    for i, ap in enumerate(p):
        pd_ = kept[i]
        nxt = ALL_PRINTS[ALL_PRINTS > pd_]
        if len(nxt) == 0:
            rw = 99
        else:
            pp = pos.get(pd_, int(cal.searchsorted(pd_)))
            pn = pos.get(nxt[0], int(cal.searchsorted(nxt[0])))
            rw = int(pn - pp)
        rows.append({"anchor": cal[ap], "kind": kind, "runway_td": rw})
    df = pd.DataFrame(rows).set_index("anchor")
    df["gate"] = G15.reindex(df.index).fillna(False).values
    df["svxy"] = svxy_h1.reindex(df.index).values
    df["vix"] = vix_h1.reindex(df.index).values
    return df


ALL = pd.concat([anchor_frame(k) for k in KINDS]).sort_index()
G = ALL[ALL["gate"]]
NFP = G[G["kind"] == "nfp"]
POOL = G[G["runway_td"] >= 3]          # the a3 coherent object


def cell(v, label):
    v = pd.Series(v).dropna()
    st = summarize(v.values, label)
    if st["n"]:
        st["signp"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
        st["record"] = f"{int((v>0).sum())}-{int((v<0).sum())}-{int((v==0).sum())}"
    return st


# ===========================================================================
print("=" * 118)
print("DEBT 3 -- ERA SPLIT AT THE SVXY LEVERAGE BREAK (2018-02-28)")
print("=" * 118)
for name, S in (("NFP-only gated", NFP), ("POOLED clear-calendar gated", POOL)):
    rows = []
    for side, col in (("SVXY", "svxy"), ("-^VIX", "vix")):
        s = S[col]
        rows.append(cell(s.values, f"{side} | {name}, full"))
        rows.append(cell(s[S.index < BREAK].values, f"{side} | pre 2018-02-28"))
        rows.append(cell(s[S.index >= BREAK].values, f"{side} | post 2018-02-28"))
    show(rows, name)

print("\nD3b. the decisive number: post-break ^VIX on the gated sets, exact sign test")
for name, S in (("NFP-only", NFP), ("POOLED clear-calendar", POOL),
                ("POOLED all 4 kinds", G)):
    v = S.loc[S.index >= BREAK, "vix"].dropna()
    w = int((v > 0).sum())
    print(f"    {name:26s} n={len(v):3d}  record {w}-{len(v)-w}  "
          f"mean {100*v.mean():+.3f}%  hit {100*w/len(v):.1f}%  "
          f"sign p = {sign_test(w, len(v)):.4f}  "
          f"bootP(mean<=0) = {bootstrap_p_le0(v.values):.4f}")
print("\nD3c. and the post-break SVXY record for contrast")
for name, S in (("NFP-only", NFP), ("POOLED clear-calendar", POOL)):
    v = S.loc[S.index >= BREAK, "svxy"].dropna()
    w = int((v > 0).sum())
    print(f"    {name:26s} n={len(v):3d}  record {w}-{len(v)-w-int((v==0).sum())}"
          f"-{int((v==0).sum())}  mean {100*v.mean():+.3f}%  "
          f"sign p = {sign_test(w, len(v)):.4f}")

print("\nD3d. is the ^VIX leg's post-break edge distinguishable from ALL post-break")
print("     days, and from post-break ungated clear-calendar anchors?")
base = vix_h1[vix_h1.index >= BREAK].dropna()
ung = ALL[(ALL.index >= BREAK) & (ALL["runway_td"] >= 3)]["vix"].dropna()
show([cell(POOL.loc[POOL.index >= BREAK, "vix"].values, "gated clear-calendar, post-break"),
      cell(ung.values, "UNGATED clear-calendar, post-break"),
      cell(base.values, "ALL days, post-break")], "short ^VIX h=1, post 2018-02-28")

# ===========================================================================
print("\n" + "=" * 118)
print("EXTRA 1 -- SVXY AT ITS OWN-SERIES 252-DAY HIGH (never measured on this cell)")
print("=" * 118)
sv = RAW["SVXY"]["Close"].dropna()            # SINGLE-INSTRUMENT series, no panel
sv_hi = sv.rolling(252).max()
sv_dist = sv / sv_hi - 1.0                    # 0.0 = at the high
print(f"   SVXY own-series index length {len(sv)}, {sv.index[0].date()}.."
      f"{sv.index[-1].date()}")
print(f"   LIVE 2026-09-02: SVXY {sv.iloc[-1]:.2f}, own 252d max {sv_hi.iloc[-1]:.2f}, "
      f"distance {100*sv_dist.iloc[-1]:+.3f}%  -> AT the high")
# panel-contaminated version, printed only to show the registry hazard is real
sv_panel = px["SVXY"]
sv_hi_panel = sv_panel.rolling(252).max()
print(f"   (panel-basis contrast, the registry hazard: panel 252d max is "
      f"{'NaN' if pd.isna(sv_hi_panel.iloc[-1]) else f'{sv_hi_panel.iloc[-1]:.2f}'} "
      f"-> the panel basis cannot even STATE the distance today; "
      f"{int(sv_hi_panel.isna().sum() - sv_hi.reindex(sv_panel.index).isna().sum())} "
      f"extra NaN rolling maxima vs the own-series basis)")

for name, S in (("NFP-only gated", NFP), ("POOLED clear-calendar gated", POOL),
                ("POOLED all kinds gated", G)):
    d = sv_dist.reindex(S.index)
    rows = []
    for lbl, m in (("SVXY within 1% of own 252d high", d >= -0.01),
                   ("SVXY within 3%", d >= -0.03),
                   ("SVXY more than 3% below", d < -0.03)):
        v = S.loc[m.fillna(False).values, "svxy"].dropna()
        rows.append(cell(v.values, f"{lbl}"))
    rows.append(cell(S["svxy"].dropna().values, "all (SVXY-covered)"))
    show(rows, f"long SVXY h=1 | {name}, split on distance to its OWN 252d high")

print("\n   dose response over the whole SVXY history (is being at the high good")
print("   or bad for a one-session hold, unconditionally?)")
j = pd.DataFrame({"d": sv_dist, "r": svxy_h1.reindex(sv_dist.index)}).dropna()
rows = []
for lo, hi, lbl in ((-0.01, 1, "within 1% of high"), (-0.03, -0.01, "1-3% below"),
                    (-0.10, -0.03, "3-10% below"), (-1, -0.10, ">10% below")):
    m = (j["d"] > lo) & (j["d"] <= hi)
    rows.append(cell(j.loc[m, "r"].values, lbl))
rows.append(cell(j["r"].values, "all SVXY days"))
show(rows, "long SVXY h=1 by distance to own 252d high (N=%d)" % len(j))
print("   anchors that were within 1% of the high, with their outcomes:")
d_all = sv_dist.reindex(G.index)
sub = G[(d_all >= -0.01).fillna(False).values]
print("   " + ", ".join(f"{i.date()}({r.kind[:3]}) {100*r.svxy:+.2f}%"
                        for i, r in sub.iterrows() if pd.notna(r.svxy)))

# ===========================================================================
print("\n" + "=" * 118)
print("EXTRA 2 -- REGISTRY COLLISION, MEASURED: C1 anchors vs the 2026-09-02")
print("   'dead range then it broke' cell (VIX pop >= 8%, SPY -1.25%..0%, range")
print("   pctile <= 15 on the RATIO definition c7 used). Long SPY h=10 there.")
print("=" * 118)
core = px[["SPY", "^VIX"]].dropna()
r_spy = core["SPY"].pct_change()
r_vix = core["^VIX"].pct_change()
rng_ratio = rolling_on_valid(core["^VIX"],
                             lambda x: x.rolling(21).max() / x.rolling(21).min() - 1)
rng_pct_c7 = rolling_on_valid(rng_ratio.dropna(),
                              lambda x: x.rolling(252).rank(pct=True) * 100).reindex(core.index)
C7 = ((r_vix >= 0.08) & (r_spy > -0.0125) & (r_spy < 0) & (rng_pct_c7 <= 15)).fillna(False)
c7_days = core.index[C7.values]


def jac(a, b):
    a, b = set(pd.DatetimeIndex(a)), set(pd.DatetimeIndex(b))
    u = len(a | b)
    return len(a & b), u, (len(a & b) / u if u else np.nan)


for nm, S in (("C1 NFP-only anchors", NFP.index),
              ("C1 pooled clear-calendar anchors", POOL.index),
              ("all gate-ON days", cal[G15.reindex(cal).fillna(False).values])):
    i, u, jj = jac(S, c7_days)
    print(f"   {nm:34s} n={len(S):5d} vs c7 n={len(c7_days)}: "
          f"intersection {i}, union {u}, Jaccard {jj:.4f}")
print("   c7 trigger days:", ", ".join(str(d.date()) for d in c7_days[-12:]), "...")
print("   NOTE the two cells are near-disjoint BY CONSTRUCTION on the VIX move:")
print(f"     c7 needs VIX +8% or more on the anchor day. On C1's NFP anchors the")
vmv = r_vix.reindex(NFP.index).dropna()
print(f"     same-day VIX move is mean {100*vmv.mean():+.2f}%, max {100*vmv.max():+.2f}%, "
      f"and {int((vmv >= 0.08).sum())} of {len(vmv)} clear +8%.")

# ===========================================================================
print("\n" + "=" * 118)
print("EXTRA 3 -- THE TAIL")
print("=" * 118)
atr = pd.Series(np.asarray(wilder_atr(RAW["SVXY"]["High"], RAW["SVXY"]["Low"],
                                      RAW["SVXY"]["Close"], 14)).ravel(),
                index=RAW["SVXY"].index)
print(f"   SVXY live: close {sv.iloc[-1]:.2f}, Wilder-14 ATR {atr.iloc[-1]:.3f} "
      f"= {100*atr.iloc[-1]/sv.iloc[-1]:.2f}% of price")
for nm, S in (("NFP-only gated", NFP), ("POOLED clear-calendar", POOL),
              ("POOLED all kinds gated", G)):
    v = S["svxy"].dropna()
    vv = S["vix"].dropna()
    print(f"   {nm:26s} SVXY worst {100*v.min():+.2f}% on {v.idxmin().date()}   "
          f"| short-^VIX worst {100*vv.min():+.2f}% on {vv.idxmin().date()}")
ung = ALL[ALL["kind"] == "nfp"]["svxy"].dropna()
print(f"   UNGATED NFP h=1 worst (what the gate is NOT protecting against): "
      f"{100*ung.min():+.2f}% on {ung.idxmin().date()}")

print("\n   what a repeat of 2025-07-30 costs (SVXY -4.125%, ^VIX +21.89%):")
print("   sizing is ATR risk, not notional (pitch convention). stop distance")
print("   is the sizing unit; with no stop the time exit is the only bound, so")
print("   the honest statement is the WORST OBSERVED move against the risk unit.")
atr_pct = float(atr.iloc[-1] / sv.iloc[-1])
for bps in (30, 45, 60):
    risk_dollars = 750_000 * bps / 10_000
    # 1 ATR risk unit -> shares = risk / ATR; a -4.125% day = -x ATR
    move_atr = 0.04125 / atr_pct
    print(f"     {bps} bps risk (= ${risk_dollars:,.0f} per 1 ATR): a -4.125% "
          f"session is -{move_atr:.2f} ATR -> -${risk_dollars*move_atr:,.0f} "
          f"(-{100*risk_dollars*move_atr/750_000:.2f}% NAV)")
worst_pool = POOL["svxy"].dropna().min()
mv = abs(worst_pool) / atr_pct
print(f"   and on the POOLED cell's own worst ({100*worst_pool:.2f}%): "
      f"-{mv:.2f} ATR -> 45 bps risk loses "
      f"-${750_000*0.0045*mv:,.0f} (-{100*0.0045*mv:.2f}% NAV)")
print("\n   ungated tail for scale: SVXY's single worst h=1 print-anchor session")
print(f"     {100*ung.min():.2f}% = -{abs(ung.min())/atr_pct:.2f} ATR at today's ATR%")
print("   NOTE: the 2018-01-31 -13.21% was a -1x SVXY. Today's -0.5x halves it.")

print("\n   left-tail quantiles of the pooled clear-calendar cell (SVXY h=1):")
v = POOL["svxy"].dropna()
for q in (0.01, 0.05, 0.10, 0.25):
    print(f"     q{100*q:>5.1f} = {100*v.quantile(q):+.3f}%")
print(f"     n={len(v)}, mean {100*v.mean():+.3f}%, "
      f"P(loss > 2%) = {100*(v < -0.02).mean():.1f}%, "
      f"P(loss > 3%) = {100*(v < -0.03).mean():.1f}%")
