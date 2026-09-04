"""C6 round 1 - long the tape's 5-day losers against its winners, cross-sectionally.

This is BOTH a candidate and C1's control.  If the generic cross-sectional
5-day reversal pays as much as C1's XLK-specific rotation cell, C1 has no
sector content.

Primary universe is the 9 sector SPDRs, which is SURVIVORSHIP-FREE (per the
2026-08-24 registry lesson that the 218-name tape and master_prices are
today's survivors).  The liquid single-name universe is run SECOND and
labelled as the biased one.

Registry collision that decides this outright if it holds (2026-08-14):
"a generic 5-day-washout reversal on liquid names (k=5/h=3, +0.534%, t 4.17,
2018+ +0.709%), which is the book's own dip-buy family and must not be
re-dressed as a pitch."  So this check must ALSO establish whether the
dispersion condition adds anything over the unconditional version.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

pd.set_option("display.width", 240)

SECTORS = ["XLK", "XLV", "XLP", "XLU", "XLI", "XLF", "XLY", "XLE", "XLB"]
px = close_panel(SECTORS + ["SPY"])
S = px[SECTORS].dropna(how="any")
print(f"sector panel {S.index[0].date()} -> {S.index[-1].date()}  {S.shape}")

R5 = S.pct_change(5)
K = 2  # long bottom-2, short top-2


def xsec_leg_returns(panel: pd.DataFrame, rank_src: pd.DataFrame, h: int,
                     k: int, lag: int = 1):
    """Equal-weight bottom-k long / top-k short, entered at close D+lag,
    held h sessions.  Returns (long_leg, short_leg, spread) aligned to D."""
    fwd = pd.DataFrame({c: panel[c].shift(-(lag + h)) / panel[c].shift(-lag) - 1.0
                        for c in panel.columns})
    ordr = rank_src.rank(axis=1, method="first")
    n = rank_src.notna().sum(axis=1)
    lo = ordr.le(k, axis=0) & rank_src.notna()
    hi = ordr.gt(n - k, axis=0) & rank_src.notna()
    L = fwd.where(lo).mean(axis=1)
    H = fwd.where(hi).mean(axis=1)
    return L, H, L - H


# cross-sectional dispersion of the 5d return, and its PIT-252 percentile
DISP = (R5.max(axis=1) - R5.min(axis=1)) * 100.0
disp_pit = DISP.dropna().rolling(252).apply(lambda w: (w[:-1] < w[-1]).mean() * 100,
                                            raw=True).reindex(S.index)
print(f"\ntoday: 5d cross-sectional spread (max-min) = {DISP.iloc[-1]:.2f}pp   "
      f"PIT252 pctile = {disp_pit.iloc[-1]:.1f}   "
      f"full-sample pctile = {(DISP.dropna() < DISP.iloc[-1]).mean()*100:.1f}")
print(f"  today's bottom-{K}: {list(R5.iloc[-1].nsmallest(K).index)}   "
      f"top-{K}: {list(R5.iloc[-1].nlargest(K).index)}")

# ------------------------------------------------------------ 1. unconditional
print("\n########## 1. UNCONDITIONAL cross-sectional reversal (all days) ##########")
rows = []
for h in (1, 2, 3, 5, 7, 10):
    L, H, SP = xsec_leg_returns(S, R5, h, K)
    v = SP.dropna()
    e = declusters(v.index, h, v.index)
    rows.append({**summarize(SP.loc[e].values, f"h={h} spread (episodes)"),
                 "long_leg_pct": round(100 * L.loc[e].mean(), 3),
                 "short_leg_pct": round(100 * (-H.loc[e]).mean(), 3),
                 "spy_drift_pct": round(100 * fwd_lag(px["SPY"], h).loc[
                     e.intersection(px.index)].mean(), 3)})
show(rows, f"long bottom-{K} / short top-{K} of 9 SPDRs by 5d return, EVERY DAY")

# ------------------------------------------------------------ 2. conditioned
print("\n########## 2. CONDITIONED ON A DISPERSION EXTREME ##########")
for pct in (80, 90, 95, 99):
    m = (disp_pit >= pct).fillna(False)
    d = S.index[m.values]
    r = []
    for h in (3, 5, 10):
        L, H, SP = xsec_leg_returns(S, R5, h, K)
        v = SP.dropna()
        e = declusters(d.intersection(v.index), h, v.index)
        s = summarize(SP.loc[e].values, f"h={h}")
        s["uncond_pct"] = round(100 * SP.loc[declusters(v.index, h, v.index)].mean(), 3)
        s["edge_pct"] = round(s.get("mean_pct", np.nan) - s["uncond_pct"], 3)
        s["long_leg_pct"] = round(100 * L.loc[e].mean(), 3)
        s["short_leg_pct"] = round(100 * (-H.loc[e]).mean(), 3)
        r.append(s)
    show(r, f"dispersion PIT >= {pct} (N_days={int(m.sum())}, today {disp_pit.iloc[-1]:.1f})")

# ------------------------------------------------------------ 3. vs C1
print("\n########## 3. IS C1 ANYTHING MORE THAN THIS? ##########")
r5x = px.pct_change(5)
SPREAD = (r5x["XLV"] - r5x["XLK"]) * 100.0
C1 = (SPREAD >= 8).fillna(False)
d1 = S.index.intersection(px.index[C1.values])
for h in (3, 5):
    L, H, SP = xsec_leg_returns(S, R5, h, K)
    v = SP.dropna()
    e1 = declusters(d1.intersection(v.index), h, v.index)
    xlk = vehicle_ret(px, [("XLK", 1.0)], h)
    print(f"\n  h={h}, on C1's own trigger days (5d XLV-XLK >= 8pp), N_epi={len(e1)}:")
    print(f"    C6 xsec spread          {100*SP.loc[e1].mean():+.3f}%   "
          f"hit {100*(SP.loc[e1] > 0).mean():.1f}%")
    print(f"    C6 long leg only        {100*L.loc[e1].mean():+.3f}%")
    print(f"    C1 long XLK only        {100*xlk.loc[e1].mean():+.3f}%")
    print(f"    XLK's rank among the 9 by 5d ret on trigger days: "
          f"median {R5.loc[e1].rank(axis=1).loc[:, 'XLK'].median():.1f} of 9  "
          f"(bottom-{K} on {100*(R5.loc[e1].rank(axis=1)['XLK'] <= K).mean():.0f}% of them)")
print(f"\n  TODAY XLK's 5d rank among the 9 = "
      f"{R5.iloc[-1].rank().loc['XLK']:.0f} of 9 (5d {R5.iloc[-1]['XLK']*100:+.2f}%)")

# ------------------------------------------------------------ 4. alphabetical placebo
print("\n########## 4. ALPHABETICAL PLACEBO ##########")
alpha = sorted(SECTORS)
fwd3 = pd.DataFrame({c: S[c].shift(-4) / S[c].shift(-1) - 1.0 for c in SECTORS})
L, H, SP = xsec_leg_returns(S, R5, 3, K)
v = SP.dropna()
e = declusters(v.index, 3, v.index)
placebo = (fwd3[alpha[:K]].mean(axis=1) - fwd3[alpha[-K:]].mean(axis=1))
print(f"  signal-picked bottom{K}-top{K} h=3: {100*SP.loc[e].mean():+.3f}%")
print(f"  alphabetical  {alpha[:K]} - {alpha[-K:]} h=3: {100*placebo.loc[e].mean():+.3f}%")

# ------------------------------------------------------------ 5. liquid universe
print("\n########## 5. LIQUID SINGLE-NAME UNIVERSE (SURVIVORSHIP-BIASED) ##########")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from strategy_config import LIQUID_PLUS_COMMODITIES  # noqa

names = [t for t in LIQUID_PLUS_COMMODITIES if not t.startswith("^")]
LP = close_panel(names)
LP = LP.loc["2005-01-01":]
LP = LP.dropna(axis=1, thresh=int(0.9 * len(LP)))
print(f"  liquid panel {LP.shape[1]} names, {LP.index[0].date()} -> {LP.index[-1].date()}")
LR5 = LP.pct_change(5)
KL = 10
rows = []
for h in (3, 5, 10):
    L, H, SP = xsec_leg_returns(LP, LR5, h, KL)
    v = SP.dropna()
    e = declusters(v.index, h, v.index)
    rows.append({**summarize(SP.loc[e].values, f"h={h} spread"),
                 "long_leg_pct": round(100 * L.loc[e].mean(), 3),
                 "short_leg_pct": round(100 * (-H.loc[e]).mean(), 3)})
show(rows, f"liquid names, long bottom-{KL} / short top-{KL} by 5d return, EVERY DAY")
print("  CAVEAT: master_prices holds today's survivors only (CLAUDE.md ledger")
print("  survivorship note); the long-losers leg is the one this flatters.")
