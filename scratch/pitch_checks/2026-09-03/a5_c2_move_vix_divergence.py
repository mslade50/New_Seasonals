"""C2 -- bond volatility bid while equity volatility is dead, into the labour print.

LIVE: ^MOVE level at the 83.9th trailing-252 percentile with a 5-day return
rank of 93.3 (5d +14.79%), ^TNX AT its 252-day yield high, while the VIX 21-day
range sits at the 4th percentile (rel-range 3.57). Two volatility markets
disagreeing about the same labour number.

STATED UP FRONT, before the data speaks -- which market do I expect to be
wrong, and why:
  I expect the BOND market to be RIGHT and equity vol to be the mispriced leg.
  Payrolls is a front-end rates event first and an equity event second: it
  moves the Fed path directly and equities only through the discount rate. A
  ^MOVE bid ahead of the print is the market with the live catalyst pricing it;
  a 4th-percentile VIX range is the market with no catalyst pricing nothing.
  If that transmission is real, the tradeable expression is SHORT equity vol's
  complacency -- i.e. long ^VIX / short SVXY / short SPY into the print -- and
  it is the exact OPPOSITE side of C1. The two cannot both be right, which is
  the single most useful thing this check can establish.

Registry 2026-08-10 warning honoured: ^MOVE's 5-day RETURN rank and its LEVEL
percentile coincide only 30.7% of the time, so the mechanism must name which
one it needs. Today both are elevated, so both legs are stated and the
sensitivity to each is reported separately.

N is reported honestly at every step. If the joint state is unmeasurable at an
event anchor, that is a KILL (2026-08-07 lesson), not a pass.
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
                       declusters, local_control, bootstrap_p_le0, pct_rank)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)

TK = ["^MOVE", "^VIX", "^VIX3M", "SPY", "TLT", "IEF", "SVXY", "^TNX"]
RAW = load_prices(TK)
px = close_panel(TK)
cal = px["SPY"].dropna().index

# --- states, each on its OWN series (registry 2026-08-11) ------------------
mv = RAW["^MOVE"]["Close"].dropna()
MV_LVL = rolling_on_valid(mv, lambda x: x.rolling(252).rank(pct=True) * 100)
MV_R5 = pct_rank(mv, 5, 252)
vix = RAW["^VIX"]["Close"].dropna()
rng21 = (rolling_on_valid(vix, lambda x: x.rolling(21).max())
         - rolling_on_valid(vix, lambda x: x.rolling(21).min()))
VIX_RNG = rolling_on_valid(rng21 / rolling_on_valid(vix, lambda x: x.rolling(21).mean()),
                           lambda x: x.rolling(252).rank(pct=True) * 100)
tnx = RAW["^TNX"]["Close"].dropna()
TNX_HI = tnx / rolling_on_valid(tnx, lambda x: x.rolling(252).max()) - 1.0

print("=" * 118)
print(f"^MOVE cache: {mv.index[0].date()}..{mv.index[-1].date()}  N={len(mv)}")
print(f"LIVE 2026-09-02: MOVE {mv.iloc[-1]:.2f} lvl-pctile {MV_LVL.iloc[-1]:.1f}  "
      f"r5 rank {MV_R5.iloc[-1]:.1f}  VIX rel-range pctile {VIX_RNG.iloc[-1]:.2f}  "
      f"TNX dist-to-252d-high {100*TNX_HI.iloc[-1]:+.3f}%")
print("=" * 118)

MV_HI = MV_LVL >= 80
MV_POP = MV_R5 >= 90
VIX_DEAD = VIX_RNG <= 15
JOINT = (MV_HI & MV_POP & VIX_DEAD).reindex(cal).fillna(False)
JOINT_LVL = (MV_HI & VIX_DEAD).reindex(cal).fillna(False)
JOINT_POP = (MV_POP & VIX_DEAD).reindex(cal).fillna(False)

print("\n0. HOW OFTEN DOES THE STATE EVEN EXIST? (registry: the two MOVE legs")
print("   coincide only ~31% of the time, so each is priced separately)")
sh = cal[(cal >= mv.index[0]) & (cal >= VIX_RNG.dropna().index[0])]
for lbl, m in (("MOVE lvl>=80", MV_HI), ("MOVE r5>=90", MV_POP),
               ("VIX range<=15", VIX_DEAD),
               ("MOVE lvl>=80 & r5>=90", MV_HI & MV_POP)):
    mm = m.reindex(sh).fillna(False)
    print(f"   {lbl:24s}: {int(mm.sum()):5d} days of {len(sh)} "
          f"({100*mm.mean():.1f}%)")
for lbl, m in (("JOINT (lvl & pop & dead VIX)", JOINT),
               ("lvl & dead VIX", JOINT_LVL), ("pop & dead VIX", JOINT_POP)):
    mm = m.reindex(sh).fillna(False)
    ep = declusters(sh[mm.values], 21, cal)
    print(f"   {lbl:28s}: {int(mm.sum()):5d} days -> {len(ep)} episodes; "
          f"years {sorted(set(sh[mm.values].year))}")
ov = (MV_HI & MV_POP).reindex(sh).fillna(False)
print(f"   MOVE lvl>=80 AND r5>=90 coincidence: {100*ov.sum()/max(1,int(MV_HI.reindex(sh).fillna(False).sum())):.1f}% "
      f"of lvl>=80 days also have r5>=90")

# ---------------------------------------------------------------------------
print("\n" + "=" * 118)
print("1. THE JOINT STATE AT PRINT ANCHORS (k=-2). N reported before any stat.")
print("=" * 118)
KINDS = ("nfp", "cpi", "ppi", "fomc_decision")
EV = {k: load_events([k])["date"] for k in KINDS}
frames = []
for k in KINDS:
    p, kept = anchor_positions(cal, EV[k], -2)
    frames.append(pd.DataFrame({"anchor": [cal[i] for i in p], "kind": k}))
AN = pd.concat(frames).set_index("anchor").sort_index()
for nm, m in (("joint", JOINT), ("lvl_only", JOINT_LVL), ("pop_only", JOINT_POP)):
    AN[nm] = m.reindex(AN.index).fillna(False).values
print("   anchors carrying each state, by kind:")
print(AN.groupby("kind")[["joint", "lvl_only", "pop_only"]].sum().to_string())
print(f"   TOTAL: joint {int(AN['joint'].sum())}, lvl&dead {int(AN['lvl_only'].sum())}, "
      f"pop&dead {int(AN['pop_only'].sum())}")
jn = AN[AN["joint"]]
print(f"   joint-state NFP anchors: {int((jn['kind']=='nfp').sum())}")
if len(jn):
    print("   joint anchor dates: " + ", ".join(
        f"{d.date()}({r.kind[:3]})" for d, r in jn.iterrows()))

# ---------------------------------------------------------------------------
print("\n" + "=" * 118)
print("2. WHAT THE STATE PREDICTS -- h=1..5, day-level (biggest N available)")
print("   sides are stated as the PITCHED direction of the hypothesis:")
print("   long ^VIX, short SVXY, short SPY, long TLT.")
print("=" * 118)
LEGS = (("long ^VIX", "^VIX", 1.0), ("short SVXY", "SVXY", -1.0),
        ("short SPY", "SPY", -1.0), ("long SPY", "SPY", 1.0),
        ("long TLT", "TLT", 1.0), ("long SVXY", "SVXY", 1.0))
for nm, mask in (("JOINT state", JOINT), ("MOVE lvl>=80 & dead VIX", JOINT_LVL),
                 ("MOVE r5>=90 & dead VIX", JOINT_POP)):
    days = cal[mask.values]
    if len(days) < 5:
        print(f"\n   {nm}: n={len(days)} days -- UNMEASURABLE")
        continue
    epi = declusters(days, 21, cal)
    print(f"\n   --- {nm}: {len(days)} days, {len(epi)} episodes ---")
    rows = []
    for lbl, tkr, sgn in LEGS:
        for h in (1, 3, 5):
            f = sgn * fwd_lag(px[tkr].dropna(), h, lag=1)
            v = f.reindex(epi).dropna()
            if len(v) == 0:
                continue
            st = summarize(v.values, f"{lbl} h={h}")
            st["ctrl_all"] = round(100 * f.dropna().mean(), 3)
            st["edge_pp"] = round(st["mean_pct"] - st["ctrl_all"], 3)
            st["signp"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
            rows.append(st)
    show(rows, f"{nm} (episode level)")

# ---------------------------------------------------------------------------
print("\n" + "=" * 118)
print("3. THE ANCHORED VERSION -- the actual C2 trade (state ON at a k=-2 print")
print("   anchor). This is where N decides whether C2 exists at all.")
print("=" * 118)
for nm, col in (("JOINT", "joint"), ("lvl & dead VIX", "lvl_only"),
                ("pop & dead VIX", "pop_only")):
    for scope, sel in (("NFP only", AN["kind"] == "nfp"), ("all 4 kinds", AN["kind"].notna())):
        d = AN.index[(AN[col]) & sel]
        if len(d) < 4:
            print(f"   {nm:16s} {scope:12s}: n={len(d)} -- UNMEASURABLE, no stat run")
            continue
        rows = []
        for lbl, tkr, sgn in (("long ^VIX", "^VIX", 1.0), ("short SVXY", "SVXY", -1.0),
                              ("long SVXY", "SVXY", 1.0), ("short SPY", "SPY", -1.0),
                              ("long TLT", "TLT", 1.0)):
            for h in (1, 3):
                f = sgn * fwd_lag(px[tkr].dropna(), h, lag=1)
                v = f.reindex(pd.DatetimeIndex(d)).dropna()
                if len(v) == 0:
                    continue
                st = summarize(v.values, f"{lbl} h={h}")
                st["signp"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
                rows.append(st)
        show(rows, f"{nm} | {scope} (n_anchors={len(d)}: "
                   f"{', '.join(str(x.date()) for x in d)})")

# ---------------------------------------------------------------------------
print("\n" + "=" * 118)
print("4. DOES BOND VOL LEAD EQUITY VOL AT ALL? -- the mechanism, tested directly")
print("   and independently of any event anchor. If a MOVE pop does not predict")
print("   a VIX move, the whole 'two markets disagree' framing is decoration.")
print("=" * 118)
r_mv5 = (mv / mv.shift(5) - 1.0)
common = px.index.intersection(mv.index)
d = pd.DataFrame({
    "mv5": r_mv5.reindex(common),
    "mv_lvl": MV_LVL.reindex(common),
    "vix_rng": VIX_RNG.reindex(common),
    "vix_f1": fwd_lag(px["^VIX"].dropna(), 1, lag=1).reindex(common),
    "vix_f5": fwd_lag(px["^VIX"].dropna(), 5, lag=1).reindex(common),
    "spy_f5": fwd_lag(px["SPY"].dropna(), 5, lag=1).reindex(common),
    "svxy_f1": fwd_lag(px["SVXY"].dropna(), 1, lag=1).reindex(common),
}).dropna(subset=["mv5", "vix_f5"])
print(f"   N={len(d)}  corr(MOVE 5d return, next-5d ^VIX return) = "
      f"{d['mv5'].corr(d['vix_f5']):+.4f}")
print(f"           corr(MOVE 5d return, next-5d SPY return)  = "
      f"{d['mv5'].corr(d['spy_f5']):+.4f}")
rows = []
q = pd.qcut(d["mv5"], 5, labels=False)
for b in sorted(q.dropna().unique()):
    m = q == b
    st = summarize(d.loc[m, "vix_f5"].values,
                   f"MOVE 5d quintile {int(b)+1} [{100*d.loc[m,'mv5'].min():+.1f},"
                   f"{100*d.loc[m,'mv5'].max():+.1f}]%")
    rows.append(st)
rows.append(summarize(d["vix_f5"].values, "all days"))
show(rows, "forward 5d ^VIX return by MOVE 5-day-move quintile (lag=1)")
print("   the same, restricted to a DEAD VIX range (the live conditioner):")
dd = d[d["vix_rng"] <= 15]
rows = []
if len(dd) > 40:
    q = pd.qcut(dd["mv5"], 3, labels=False)
    for b in sorted(q.dropna().unique()):
        m = q == b
        rows.append(summarize(dd.loc[m, "vix_f5"].values,
                              f"dead range, MOVE 5d tercile {int(b)+1}"))
    rows.append(summarize(dd["vix_f5"].values, "dead range, all"))
    show(rows, f"forward 5d ^VIX | dead range (N={len(dd)})")
    # and the C1-relevant version: does a MOVE pop hurt a short-vol day trade?
    rows = []
    for lbl, m in (("dead range & MOVE r5>=90", dd.index.isin(cal[JOINT_POP.values])),
                   ("dead range & MOVE r5<90", ~dd.index.isin(cal[JOINT_POP.values]))):
        rows.append(summarize(dd.loc[m, "svxy_f1"].values, lbl))
    show(rows, "long SVXY h=1 | dead range, split on the MOVE pop  <-- C1 CROSS-CHECK")

print("\n5. C1 CROSS-CHECK AT THE PRINT ANCHOR: does a bid ^MOVE spoil C1?")
p, _ = anchor_positions(cal, EV["nfp"], -2)
nfp_a = pd.DatetimeIndex([cal[i] for i in p])
nfp_a = nfp_a[VIX_RNG.reindex(nfp_a).fillna(99).values <= 15]
sv1 = fwd_lag(px["SVXY"].dropna(), 1, lag=1)
vx1 = -fwd_lag(px["^VIX"].dropna(), 1, lag=1)
mlv = MV_LVL.reindex(nfp_a)
rows = []
for lbl, m in (("MOVE lvl>=80", mlv >= 80), ("MOVE lvl 50-80", (mlv >= 50) & (mlv < 80)),
               ("MOVE lvl<50", mlv < 50), ("MOVE missing (pre-2003)", mlv.isna())):
    mm = m.fillna(False).values if hasattr(m, "fillna") else m
    for nmm, s in (("SVXY", sv1), ("-^VIX", vx1)):
        v = s.reindex(nfp_a[mm]).dropna()
        st = summarize(v.values, f"{nmm} | gated NFP, {lbl}")
        if st["n"]:
            st["signp"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
        rows.append(st)
show(rows, "C1's gated NFP cell split on the ^MOVE level percentile")
