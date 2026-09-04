"""C1 DEBT 1, follow-up -- the conjunction that a1/B2 turned up.

a1 found the thing that decides C1:

  * the dial is NOT endogenous to this gate. corr(pitch gate, dial) = -0.100,
    mean dial 19.0 with the gate ON against 25.2 OFF. So "the compression
    signal is what is pushing the dial to 88" is FALSE for the pitch's gate
    definition, and today's 87.9 is a genuinely foreign state, not an artifact.
  * at N=2458 day-level the dial DOES carry a one-session short-vol signal and
    it is wrong-signed for C1: corr(dial, next-session long SVXY) = -0.0486
    (t -2.41), short ^VIX -0.0573 (t -2.85).
  * restricted to gate-ON days the cell at dial >= 70 is n=6, long SVXY
    -2.326%, short ^VIX -10.191%.

n=6 decides nothing on its own and the doctrine forbids resting on it. This
script asks whether those six days are SIX draws or ONE episode, whether the
damage is monotone in the dial or an artifact of one window, and where in the
dial range today's 87.9 actually sits relative to the harm.

Everything here is on the RECOMPUTE vintage except the last 44 rows
(2026-07-02+), stated once and not repeated.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa: F401,F403,E402
from pitch_lab import (close_panel, fwd_lag, summarize, sign_test, load_events,
                       rolling_on_valid, show, anchor_positions, declusters,
                       bootstrap_p_le0)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
ROOT = Path(__file__).resolve().parents[3]

px = close_panel(["^VIX", "^VIX3M", "SVXY", "UVXY", "SPY"])
cal = px["SPY"].dropna().index
vix = px["^VIX"]
rng21 = (rolling_on_valid(vix, lambda x: x.rolling(21).max())
         - rolling_on_valid(vix, lambda x: x.rolling(21).min()))
REL = rolling_on_valid(rng21 / rolling_on_valid(vix, lambda x: x.rolling(21).mean()),
                       lambda x: x.rolling(252).rank(pct=True) * 100)
G15 = REL <= 15.0

frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
frag.index = pd.to_datetime(frag.index)
DIAL = frag["63d"].rolling(10).mean()
dv = DIAL.dropna().reindex(DIAL.dropna().index.intersection(cal)).dropna()

svxy_h1 = fwd_lag(px["SVXY"].dropna(), 1, lag=1)
vix_h1 = -fwd_lag(px["^VIX"].dropna(), 1, lag=1)      # short ^VIX

print("=" * 118)
print(f"today dial ma10(63d) = {DIAL.iloc[-1]:.2f}; dial series "
      f"{dv.index[0].date()}..{dv.index[-1].date()} N={len(dv)}")
print("=" * 118)

# ===========================================================================
# 1. EPISODE STRUCTURE OF THE HIGH-DIAL DAYS
# ===========================================================================
print("\n1. HOW MANY INDEPENDENT EPISODES ARE THE HIGH-DIAL DAYS?")
for thr in (60, 70, 80, 85, 87.9):
    d = dv[dv >= thr].index
    if len(d) == 0:
        print(f"   dial >= {thr}: NO DAYS"); continue
    ep = declusters(d, 21, cal)
    yrs = sorted(set(d.year))
    print(f"   dial >= {thr:5.1f}: {len(d):4d} days -> {len(ep):2d} episodes at a "
          f"21td gap; years {yrs}")
    print(f"                   episode starts: "
          f"{', '.join(str(x.date()) for x in ep)}")

print("\n1b. gate ON *and* dial >= 70 -- the exact live conjunction")
gd = pd.DataFrame({"dial": dv,
                   "gate": G15.reindex(dv.index).fillna(False),
                   "svxy": svxy_h1.reindex(dv.index),
                   "vix": vix_h1.reindex(dv.index)})
conj = gd[(gd["gate"]) & (gd["dial"] >= 70)]
print(conj.assign(svxy=(100 * conj["svxy"]).round(2),
                  vix=(100 * conj["vix"]).round(2),
                  dial=conj["dial"].round(1)).to_string())
if len(conj):
    ep = declusters(conj.index, 21, cal)
    print(f"   -> {len(conj)} days but only {len(ep)} independent episodes "
          f"({', '.join(str(x.date()) for x in ep)})")
    print("   n=6 days inside 1-2 windows is ONE observation dressed as six. "
          "It cannot kill and it cannot save.")

# ===========================================================================
# 2. IS THE DIAL DAMAGE MONOTONE, OR ONE WINDOW?
# ===========================================================================
print("\n" + "=" * 118)
print("2. DIAL DAMAGE: monotone, or an artifact of one window?")
print("=" * 118)
j = gd.dropna(subset=["svxy"])
rows = []
for lo, hi in ((0, 20), (20, 40), (40, 55), (55, 70), (70, 80), (80, 999)):
    m = (j["dial"] >= lo) & (j["dial"] < hi)
    st = summarize(j.loc[m, "svxy"].values, f"dial [{lo},{hi})")
    st["signp"] = round(sign_test(int((j.loc[m, "svxy"] > 0).sum()), int(m.sum())), 4)
    st["n_epi"] = len(declusters(j.index[m], 21, cal))
    rows.append(st)
rows.append(summarize(j["svxy"].values, "ALL dial days"))
show(rows, "long SVXY h=1, all dial-covered days, by dial band")

rows = []
for lo, hi in ((0, 20), (20, 40), (40, 55), (55, 70), (70, 80), (80, 999)):
    m = (j["dial"] >= lo) & (j["dial"] < hi)
    st = summarize(j.loc[m, "vix"].values, f"dial [{lo},{hi})")
    st["n_epi"] = len(declusters(j.index[m], 21, cal))
    rows.append(st)
rows.append(summarize(j["vix"].values, "ALL dial days"))
show(rows, "short ^VIX h=1, all dial-covered days, by dial band")

print("\n2b. leave-one-YEAR-out on the corr(dial, next-session SVXY) slope")
for yr in sorted(set(j.index.year)):
    k = j[j.index.year != yr]
    c = np.corrcoef(k["dial"], k["svxy"])[0, 1]
    print(f"    drop {yr}: corr {c:+.4f}  (full {np.corrcoef(j['dial'], j['svxy'])[0,1]:+.4f})")

print("\n2c. the >=80 cell year by year (is it one bad year?)")
hi = j[j["dial"] >= 80]
if len(hi):
    g = hi.groupby(hi.index.year)["svxy"]
    print(pd.DataFrame({"n": g.size(), "mean_pct": (100 * g.mean()).round(3),
                        "hit": (100 * g.apply(lambda x: (x > 0).mean())).round(1)}).to_string())

# ===========================================================================
# 3. THE CONTROL THE DIAL DEBT ACTUALLY NEEDS: is a high dial bad for a
#    short-vol trade held ONE SESSION *around a scheduled print*?
# ===========================================================================
print("\n" + "=" * 118)
print("3. HIGH DIAL x SCHEDULED PRINT -- the anchored version, all four kinds")
print("   (the NFP-only version has n=0 above dial 68, so pool the prints)")
print("=" * 118)
kinds = ("nfp", "cpi", "ppi", "fomc_decision")
allanch = []
for kind in kinds:
    p, _ = anchor_positions(cal, load_events([kind])["date"], -2)
    a = pd.DatetimeIndex([cal[i] for i in p])
    allanch.append(pd.DataFrame({"date": a, "kind": kind}))
AN = pd.concat(allanch).set_index("date").sort_index()
AN["dial"] = DIAL.reindex(AN.index)
AN["gate"] = G15.reindex(AN.index).fillna(False)
AN["svxy"] = svxy_h1.reindex(AN.index)
AN["vix"] = vix_h1.reindex(AN.index)
AN = AN.dropna(subset=["dial", "svxy"])
print(f"   dial-covered print anchors (all 4 kinds, k=-2): N={len(AN)}")
rows = []
for lo, hi in ((0, 30), (30, 50), (50, 70), (70, 999)):
    m = (AN["dial"] >= lo) & (AN["dial"] < hi)
    st = summarize(AN.loc[m, "svxy"].values, f"ALL prints, dial [{lo},{hi})")
    st["signp"] = round(sign_test(int((AN.loc[m, "svxy"] > 0).sum()), int(m.sum())), 4)
    rows.append(st)
show(rows, "long SVXY h=1 at ANY print anchor, by dial (gate irrelevant)")
rows = []
sub = AN[AN["gate"]]
for lo, hi in ((0, 30), (30, 50), (50, 999)):
    m = (sub["dial"] >= lo) & (sub["dial"] < hi)
    st = summarize(sub.loc[m, "svxy"].values, f"gated prints, dial [{lo},{hi})")
    rows.append(st)
show(rows, f"long SVXY h=1 at GATED print anchors (N={len(sub)}), by dial")
print("   gated + dial>=50 dates: " + ", ".join(
    f"{d.date()} {r.kind} {100*r.svxy:+.2f}%"
    for d, r in sub[sub["dial"] >= 50].iterrows()))

# ===========================================================================
# 4. WHAT IS TODAY'S DIAL MADE OF? the proxies say calm, the dial says 88.
# ===========================================================================
print("\n" + "=" * 118)
print("4. FIND THE HISTORICAL DAYS THAT LOOK LIKE TODAY ON THE PROXIES *AND*")
print("   CARRY A HIGH DIAL -- the disagreement is the state, not the dial.")
print("=" * 118)
ts = px["^VIX3M"] / px["^VIX"] - 1.0
hi252 = rolling_on_valid(px["SPY"], lambda x: x.rolling(252).max())
dd_spy = px["SPY"] / hi252 - 1.0
vlp = rolling_on_valid(vix, lambda x: x.rolling(252).rank(pct=True) * 100)
look = pd.DataFrame({"dial": dv, "contango": ts.reindex(dv.index),
                     "spy_dd": dd_spy.reindex(dv.index),
                     "vixpct": vlp.reindex(dv.index),
                     "svxy": svxy_h1.reindex(dv.index),
                     "vix": vix_h1.reindex(dv.index)}).dropna()
m = ((look["contango"] > 0.10) & (look["spy_dd"] >= -0.03)
     & (look["vixpct"] <= 30) & (look["dial"] >= 70))
print(f"   contango>10% AND SPY within 3% of high AND VIX pctile<=30 AND dial>=70:"
      f"  n={int(m.sum())} days, "
      f"{len(declusters(look.index[m], 21, cal))} episodes")
if m.sum():
    print(f"   dates: {', '.join(str(d.date()) for d in look.index[m][:40])}")
    show([summarize(look.loc[m, "svxy"].values, "long SVXY h=1"),
          summarize(look.loc[m, "vix"].values, "short ^VIX h=1"),
          summarize(look["svxy"].values, "all dial days, SVXY")],
         "the calm-tape-high-dial state")
m2 = ((look["contango"] > 0.10) & (look["spy_dd"] >= -0.03)
      & (look["vixpct"] <= 30))
print(f"\n   same benign state WITHOUT the dial leg: n={int(m2.sum())} days")
rows = []
for lo, hi in ((0, 40), (40, 70), (70, 999)):
    mm = m2 & (look["dial"] >= lo) & (look["dial"] < hi)
    st = summarize(look.loc[mm, "svxy"].values, f"benign tape, dial [{lo},{hi})")
    st["n_epi"] = len(declusters(look.index[mm], 21, cal))
    st["signp"] = round(sign_test(int((look.loc[mm, "svxy"] > 0).sum()),
                                  int(mm.sum())), 4)
    rows.append(st)
show(rows, "long SVXY h=1 | benign tape, split by dial  <-- THE DEBT-1 ANSWER")
rows = []
for lo, hi in ((0, 40), (40, 70), (70, 999)):
    mm = m2 & (look["dial"] >= lo) & (look["dial"] < hi)
    rows.append(summarize(look.loc[mm, "vix"].values, f"benign tape, dial [{lo},{hi})"))
show(rows, "short ^VIX h=1 | benign tape, split by dial")
