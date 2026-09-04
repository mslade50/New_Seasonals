"""C5 round 1 -- Long XLI at a triple 5/21/63-day rank floor while SPY is
within 3% of its 52-week high.

Live 2026-09-01: XLI r5 4.0, r21 5.6, r63 6.3, z10 -2.29, -7.39% off its own
52w high, +1.63% above its 200d; SPY -2.07% off its high. 11 declustered
episodes over 7 years.

The reference class is run FIRST and treated as the primary test, because two
directly analogous candidates (watchlist 21, XLI sector washout into a 52-week
high; watchlist 25, SMH at a 63d rank floor) died on exactly this shape inside
the last week.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch_lab import *  # noqa
from _rc import (cochran, dial_series, jaccard, per_name, perm_max_of_n,  # noqa
                 pooled, welch)

pd.set_option("display.width", 250)

SPDRS = ["XLK", "XLV", "XLP", "XLU", "XLI", "XLF", "XLY", "XLE", "XLB"]
WIDE = SPDRS + ["XLRE", "XLC", "SMH", "XBI", "IBB", "KRE", "IHI", "ITB",
                "XME", "XOP", "OIH", "QQQ", "IWM", "DIA", "EFA", "EEM",
                "EWJ", "FXI", "EWZ"]
px = close_panel(sorted(set(WIDE + ["SPY"])))
spy_hi = rolling_on_valid(px["SPY"], lambda x: x.rolling(252).max())
SPY_NEAR = (px["SPY"] / spy_hi - 1.0) >= -0.03

pxd = {t: px[t] for t in px.columns}


def triple(s: pd.Series, k: int = 10) -> pd.Series:
    return ((pct_rank(s, 5) <= k) & (pct_rank(s, 21) <= k)
            & (pct_rank(s, 63) <= k))


MAIN = (triple(px["XLI"]) & SPY_NEAR).fillna(False)
print("LIVE 2026-09-01 XLI: r5 %.1f  r21 %.1f  r63 %.1f | SPY off-high %.2f%%"
      % (pct_rank(px["XLI"], 5).iloc[-1], pct_rank(px["XLI"], 21).iloc[-1],
         pct_rank(px["XLI"], 63).iloc[-1],
         100 * (px["SPY"].iloc[-1] / spy_hi.iloc[-1] - 1)))
print("MAIN fires today:", bool(MAIN.iloc[-1]))

# ---------------------------------------------------------------- 0. horizons
print("\n########## 0. HORIZON SCAN (the pitched horizon comes FROM here) ##########")
ret10 = vehicle_ret(px, [("XLI", 1.0)], 10)
epi_ref = declusters(px.index[MAIN.values].intersection(ret10.dropna().index),
                     10, ret10.dropna().index)
show(horizon_scan(px, px.index[MAIN.values], [("XLI", 1.0)],
                  hs=(1, 2, 3, 5, 7, 10)), "long XLI, episode level, lag=1")
print("  NOTE: I walked a 6-point horizon grid. Any best-h claim owes a x6 "
      "multiplicity correction.")

H, GAP = 10, 10

# ---------------------------------------------------------------- 1. battery
battery(px, MAIN, [("XLI", 1.0)], H, "C5 long XLI triple floor x SPY near high",
        cost_bps=5.0, min_gap=GAP,
        variants={
            "triple<=5 (tighter)": (triple(px["XLI"], 5) & SPY_NEAR).fillna(False),
            "triple<=15 (looser)": (triple(px["XLI"], 15) & SPY_NEAR).fillna(False),
            "triple<=10, SPY within 2%": (triple(px["XLI"]) & ((px["SPY"] / spy_hi - 1) >= -0.02)).fillna(False),
            "triple<=10, SPY within 5%": (triple(px["XLI"]) & ((px["SPY"] / spy_hi - 1) >= -0.05)).fillna(False),
        }, event_kinds=("nfp",))

# --------------------------------------------------- 2. GATE ATTRIBUTION
print("\n########## 2. GATE ATTRIBUTION -- run WITHOUT each leg ##########")
r5, r21, r63 = (pct_rank(px["XLI"], n) for n in (5, 21, 63))
gates = {
    "FULL: r5<=10 & r21<=10 & r63<=10 & SPY<=3% off": MAIN,
    "drop r5   (r21&r63&SPY)": ((r21 <= 10) & (r63 <= 10) & SPY_NEAR),
    "drop r21  (r5&r63&SPY)": ((r5 <= 10) & (r63 <= 10) & SPY_NEAR),
    "drop r63  (r5&r21&SPY)": ((r5 <= 10) & (r21 <= 10) & SPY_NEAR),
    "drop SPY  (bare triple floor)": triple(px["XLI"]),
    "r5 alone <=10": (r5 <= 10),
    "r21 alone <=10": (r21 <= 10),
    "r63 alone <=10": (r63 <= 10),
    "SPY near-high alone": SPY_NEAR,
}
ret = vehicle_ret(px, [("XLI", 1.0)], H)
valid = ret.dropna().index
rows = []
for lbl, m in gates.items():
    t = px.index[m.fillna(False).values].intersection(valid)
    if len(t) == 0:
        rows.append({"label": lbl, "n": 0})
        continue
    e = declusters(t, GAP, valid)
    r = summarize(ret.loc[e].values, lbl)
    r["n_days"] = len(t)
    rows.append(r)
show(rows, f"gate attribution, h={H}, episode level")
full = rows[0]["mean_pct"]
print("  dose of each leg (FULL minus the drop-one form):")
for r in rows[1:5]:
    print(f"    {r['label']:38s} {full - r['mean_pct']:+7.3f} pp  "
          f"(full {full:+.3f} vs {r['mean_pct']:+.3f}, n_epi {r['n']})")

# bull-tape selector check (the watchlist-21 finding)
sma200 = rolling_on_valid(px["SPY"], lambda x: x.rolling(200).mean())
above = (px["SPY"] > sma200).dropna()
tdays = px.index[MAIN.values].intersection(above.index)
print(f"\n  SPY-near-high leg as a bull-tape selector: {100*above.loc[tdays].mean():.1f}% "
      f"of {len(tdays)} trigger days above SPY's 200d, base {100*above.mean():.1f}%. "
      f"N below = {int((~above.loc[tdays]).sum())}")
xli200 = rolling_on_valid(px["XLI"], lambda x: x.rolling(200).mean())
xabove = (px["XLI"] > xli200).dropna()
td2 = px.index[MAIN.values].intersection(xabove.index)
print(f"  XLI itself above its own 200d on trigger days: "
      f"{100*xabove.loc[td2].mean():.1f}% (base {100*xabove.mean():.1f}%). "
      f"LIVE today: XLI +1.63% above its 200d.")

# --------------------------------------- 3. IS THIS WATCHLIST 21 RE-SKINNED?
print("\n########## 3. OVERLAP WITH WATCHLIST 21 (day-level Jaccard) ##########")
xhi = rolling_on_valid(px["XLI"], lambda x: x.rolling(252).max())
xdd = px["XLI"] / xhi - 1.0
W21 = ((pct_rank(px["XLI"], 5) <= 5) & (pct_rank(px["XLI"], 63) >= 30)
       & (pct_rank(px["XLI"], 63) <= 60) & (xdd >= -0.05)).fillna(False)
a = px.index[MAIN.values]
b = px.index[W21.values]
i, u, j = jaccard(a, b)
print(f"  C5 days {len(a)}, watchlist-21 days {len(b)}, intersection {i}, "
      f"union {u}, Jaccard {j:.3f}")
print(f"  -> the two cells are {'THE SAME OBJECT' if j > 0.3 else 'DISJOINT/near-disjoint'}. "
      "W21 requires r63 in [30,60] and XLI within 5% of ITS OWN high; C5 "
      "requires r63<=10 and puts no clause on XLI's own drawdown "
      f"(live -7.39%).")
print(f"  live XLI 52w drawdown -7.39% -> watchlist 21's own 5% leg FAILS today, "
      "which is why C5 exists as a separate object.")

# ------------------------------- 4. WATCHLIST 28's opposite conditioner (r21)
print("\n########## 4. r21 DOSE -- w28 is the r21>=90 side; C5 is the r21<=10 side ##########")
rows = []
for lbl, m in [
    ("r5<=10 & r63<=10 & r21<=10 (C5)", (r5 <= 10) & (r63 <= 10) & (r21 <= 10) & SPY_NEAR),
    ("r5<=10 & r63<=10 & r21 in [10,50)", (r5 <= 10) & (r63 <= 10) & (r21 >= 10) & (r21 < 50) & SPY_NEAR),
    ("r5<=10 & r63<=10 & r21 >= 50", (r5 <= 10) & (r63 <= 10) & (r21 >= 50) & SPY_NEAR),
]:
    t = px.index[m.fillna(False).values].intersection(valid)
    e = declusters(t, GAP, valid) if len(t) else t
    r = summarize(ret.loc[e].values, lbl) if len(e) else {"label": lbl, "n": 0}
    rows.append(r)
show(rows, "r21 ladder inside the r5/r63 floor")

# ---------------------------------------------- 5. REFERENCE CLASS (PRIMARY)
print("\n########## 5. REFERENCE CLASS -- the identical rule on the family ##########")


def mk(k=10):
    def f(_t, s):
        return (triple(s, k) & SPY_NEAR.reindex(s.index, fill_value=False)).fillna(False)
    return f


for fam, name in [(SPDRS, "nine SPDRs"), (WIDE, f"{len(WIDE)}-ETF wide pool")]:
    pn = per_name(pxd, fam, mk(), H, GAP)
    show(pn.sort_values("t_excess", ascending=False), f"per-name, {name}, h={H}")
    co = cochran(pn)
    if co:
        print(f"  Cochran Q = {co['Q']:.2f} on {co['df']} df, p = {co['p']:.4f}, "
              f"I-squared = {co['I2_pct']:.1f}%")
        print(f"  fixed-effect COMMON excess = {co['fe_common_pct']:+.3f} pp "
              f"(se {co['fe_se_pct']:.3f}, t {co['fe_t']:+.2f})")
        print("  -> " + ("HETEROGENEOUS, a member may be special"
                         if co["p"] < 0.10 else
                         "HOMOGENEOUS: nothing distinguishes XLI; honest form is "
                         "POOLED or nothing"))
    ok = pn.dropna(subset=["t_excess"]).sort_values("t_excess", ascending=False)
    ranks = list(ok["tkr"])
    if "XLI" in ranks:
        print(f"  XLI ranks {ranks.index('XLI')+1} of {len(ranks)} by excess-t "
              f"(t_excess {float(ok[ok.tkr=='XLI'].t_excess.iloc[0]):+.2f}); "
              f"leader {ranks[0]}")
    p = pooled(pxd, fam, mk(), H, GAP, f"POOLED {name}")
    print(f"  POOLED: N={p['n']} mean {p['mean_pct']:+.3f}% hit {p['hit']:.1f}% "
          f"t {p['t']:+.2f} worst {p['worst_pct']:+.2f}%")
    pm = perm_max_of_n(pxd, fam, mk(), H, GAP, n_perm=1000)
    xli_exc, xli_t = pm["obs"].get("XLI", (np.nan, np.nan))
    best = max(pm["obs"].items(), key=lambda kv: kv[1][0])
    print(f"  correlation-preserving permutation max-of-{pm['n_names']} "
          f"(common circular offset, {pm['n_perm']} draws):")
    print(f"    observed best excess {100*best[1][0]:+.3f}pp ({best[0]}), "
          f"XLI excess {100*xli_exc:+.3f}pp, XLI |t| {abs(xli_t):.2f}")
    print(f"    P(max excess >= observed best) = "
          f"{(pm['null_exc'] >= best[1][0]).mean():.4f}   "
          f"P(max excess >= XLI's) = {(pm['null_exc'] >= xli_exc).mean():.4f}   "
          f"P(max|t| >= XLI's) = {(pm['null_t'] >= abs(xli_t)).mean():.4f}")

# ---------------------------------------------------------- 6. FRAGILITY DIAL
print("\n########## 6. FRAGILITY DIAL on the trigger episodes ##########")
d = dial_series()
print("  vintage note: rd2_fragility.parquet starts 2016-07-05; rows before "
      "2026-07-02 are the RECOMPUTE vintage, later rows point-in-time appends. "
      "Used as-is.")
epi = declusters(px.index[MAIN.values].intersection(valid), GAP, valid)
dv = d.reindex(epi).dropna()
print(f"  live ma10-63d dial = {d.iloc[-1]:.1f}")
print(f"  episodes with a dial reading: {len(dv)} of {len(epi)} "
      f"(the series only starts 2016)")
if len(dv):
    print("  per-episode dial: " + ", ".join(f"{str(k.date())}={v:.1f}"
                                             for k, v in dv.items()))
    print(f"  MAX historical episode dial = {dv.max():.1f} vs today's {d.iloc[-1]:.1f} "
          f"-> today is {'INSIDE' if d.iloc[-1] <= dv.max() else 'OUTSIDE'} "
          "the historical population")

# ------------------------------------------------------------ 7. BOOK OVERLAP
print("\n########## 7. BOOK OVERLAP ##########")
tr = pd.read_parquet(Path(__file__).resolve().parents[3] / "data"
                     / "backtest_trades_full.parquet")
tr["Signal Date"] = pd.to_datetime(tr["Signal Date"])
eset = set(epi)
win = set()
pos = pd.Series(range(len(px.index)), index=px.index)
for dte in epi:
    p = pos[dte]
    win |= set(px.index[max(0, p - 1):min(len(px.index), p + H + 2)])
sub = tr[tr["Signal Date"].isin(win)]
print(f"  book signals in a [-1,+{H+1}] td window around a C5 episode: {len(sub)}")
if len(sub):
    g = sub.groupby(["Strategy", "Direction"]).agg(
        n=("R_Multiple", "size"), avgR=("R_Multiple", "mean")).round(2)
    print(g.to_string())
xli_tr = tr[tr["Ticker"] == "XLI"]
print(f"  book trades in XLI ever: {len(xli_tr)}")
print(f"  book trades whose SIGNAL DATE is exactly a C5 episode: "
      f"{len(tr[tr['Signal Date'].isin(eset)])}")

print("\n########## 8. COST ##########")
ev = ret.loc[epi].values
print(f"  XLI round trip ~5 bps (top-decile-liquidity SPDR). episode mean "
      f"{100*ev.mean():.3f}% = {10000*ev.mean():.1f} bps -> "
      f"{10000*ev.mean()/5:.1f}x cost")
