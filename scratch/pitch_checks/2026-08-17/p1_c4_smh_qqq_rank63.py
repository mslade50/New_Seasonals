"""C4 round 1+2 - SMH at a 63d rank of 1.6 while QQQ's 63d rank is 23.8.

Cell as briefed: a leadership group at the bottom of its own 63d distribution
while the parent index is nowhere near the bottom of its own. Direction not
given, so BOTH sides are measured: long SMH / short QQQ (mean reversion) and
short SMH / long QQQ (stalling leader = early distribution).

Registry debts carried in (all three are prior kills on this exact family):
  * 2026-08-07 "laggard-snapback continuation (SMH/QQQ form)" - pair flat at
    h=5 and the trigger over-selects bear tape by +29pp vs base rate.
  * 2026-08-07 "lag-0 forward returns on a MOC idea" - lag=1 everywhere.
  * 2026-08-13 "the semis laggard OUTRIGHT" - trigger puts SPY below its 200d
    on 59.4% of days vs a 24.2% base rate, and today's state (SMH far ABOVE
    its own 200d) was 4 of 347 trigger days.
  * 2026-08-10 "beta-neutralize a pair before crediting the spread" (GDX/GLD:
    +0.786% equal-dollar, -0.000% at the measured beta of 1.78). MANDATORY
    here: the beta-neutral residual at the measured beta is reported next to
    every equal-dollar number.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

H_LIST = (3, 5, 10)
H_MAIN = 5

pan = close_panel(["SMH", "QQQ", "SPY"]).dropna()
IDX = pan.index
smh, qqq, spy = pan["SMH"], pan["QQQ"], pan["SPY"]
print(f"panel {IDX[0].date()} .. {IDX[-1].date()}  N={len(IDX)}")

r63_smh = pct_rank(smh, 63)
r63_qqq = pct_rank(qqq, 63)
spy200 = spy.rolling(200).mean()
smh200 = smh.rolling(200).mean()
below_spy200 = (spy < spy200)
smh_vs200 = smh / smh200 - 1.0

print(f"\nTODAY (2026-08-14): SMH r63={r63_smh.iloc[-1]:.1f}  QQQ r63={r63_qqq.iloc[-1]:.1f}  "
      f"SMH vs200d={100*smh_vs200.iloc[-1]:+.1f}%  SPY below 200d={bool(below_spy200.iloc[-1])}")

MASK = (r63_smh <= 5) & (r63_qqq >= 20)
MASK = MASK.fillna(False)
sig = IDX[MASK.values]
print(f"trigger days (SMH r63<=5 & QQQ r63>=20): N={len(sig)}  "
      f"span {sig[0].date() if len(sig) else 'n/a'} .. {sig[-1].date() if len(sig) else 'n/a'}")

# --------------------------------------------------------------- 0. IS TODAY IN THE SAMPLE
print("\n" + "=" * 92)
print("0. IS TODAY'S STATE INSIDE THE TRIGGER POPULATION?")
print("=" * 92)
base_bear = float(below_spy200.reindex(IDX).fillna(False).mean())
trig_bear = float(below_spy200.loc[sig].mean())
print(f"SPY below its 200d: trigger days {100*trig_bear:.1f}%  vs base rate {100*base_bear:.1f}%")
v = smh_vs200.loc[sig].dropna()
print(f"SMH vs its own 200d on trigger days: mean {100*v.mean():+.1f}%  median {100*v.median():+.1f}%  "
      f"p90 {100*v.quantile(0.90):+.1f}%  max {100*v.max():+.1f}%")
print(f"  trigger days with SMH >= +15% above its 200d (today = {100*smh_vs200.iloc[-1]:+.1f}%): "
      f"{int((v >= 0.15).sum())} of {len(v)}  ({100*(v>=0.15).mean():.1f}%)")
print(f"  trigger days with SMH >= +25% above its 200d: {int((v >= 0.25).sum())} of {len(v)}")
hi_dates = v.index[v >= 0.15]
if len(hi_dates):
    print("  those dates:", ", ".join(str(d.date()) for d in hi_dates[:20]))
    print("  declustered(5td):", len(declusters(hi_dates, 5, IDX)), "episodes")

# --------------------------------------------------------------- 1. BATTERY, BOTH SIDES
for label, legs in (("LONG SMH / SHORT QQQ (equal $)", [("SMH", 1.0), ("QQQ", -1.0)]),
                    ("SHORT SMH / LONG QQQ (equal $)", [("SMH", -1.0), ("QQQ", 1.0)]),
                    ("LONG SMH outright", [("SMH", 1.0)])):
    variants = {
        "r63smh<=3 & qqq>=20": ((r63_smh <= 3) & (r63_qqq >= 20)).fillna(False),
        "r63smh<=5 & qqq>=20 (BASE)": MASK,
        "r63smh<=10 & qqq>=20": ((r63_smh <= 10) & (r63_qqq >= 20)).fillna(False),
        "r63smh<=20 & qqq>=20": ((r63_smh <= 20) & (r63_qqq >= 20)).fillna(False),
        "r63smh<=5, NO qqq gate": (r63_smh <= 5).fillna(False),
        "r63smh<=5 & qqq>=30": ((r63_smh <= 5) & (r63_qqq >= 30)).fillna(False),
        "r63smh<=5 & qqq>=40": ((r63_smh <= 5) & (r63_qqq >= 40)).fillna(False),
    }
    battery(pan, MASK, legs, H_MAIN, f"C4 {label}", cost_bps=2.0,
            variants=variants, min_gap=10, event_kinds=("opex", "vix_expiry"))

# --------------------------------------------------------------- 2. THE BETA TRAP
print("\n" + "=" * 92)
print("2. BETA-NEUTRAL RESIDUAL (the mandatory GDX/GLD check)")
print("=" * 92)
d_smh = smh.pct_change()
d_qqq = qqq.pct_change()
ok = d_smh.notna() & d_qqq.notna()
beta_full = float(np.polyfit(d_qqq[ok], d_smh[ok], 1)[0])
epi = declusters(sig, 10, IDX)
print(f"full-history daily beta SMH~QQQ = {beta_full:.3f}   (N={int(ok.sum())})")

for h in H_LIST:
    rs = fwd_lag(smh, h, 1)
    rq = fwd_lag(qqq, h, 1)
    okh = rs.notna() & rq.notna()
    beta_h = float(np.polyfit(rq[okh], rs[okh], 1)[0])
    resid = rs - beta_h * rq
    e = epi.intersection(rs.dropna().index)
    rows = [summarize(rs.loc[e].values, f"h={h} SMH leg"),
            summarize(rq.loc[e].values, f"h={h} QQQ leg"),
            summarize((rs - rq).loc[e].values, f"h={h} equal-$ spread"),
            summarize(resid.loc[e].values, f"h={h} BETA-NEUTRAL resid (beta={beta_h:.2f})"),
            summarize(resid[okh].values, f"h={h} resid, all days (control)")]
    show(rows, f"h={h} legs vs spread vs residual, episodes N={len(e)}")

# --------------------------------------------------------------- 3. ROUND 2: DEFINITION NEIGHBOURS
print("\n" + "=" * 92)
print("3. DEFINITION NEIGHBOURHOOD - is 1.6-vs-23.8 a cherry-picked pair of cuts?")
print("=" * 92)
rows = []
for cs in (1, 3, 5, 10, 15, 20):
    for cq in (0, 10, 20, 30):
        m = ((r63_smh <= cs) & (r63_qqq >= cq)).fillna(False)
        s = IDX[m.values]
        if len(s) == 0:
            continue
        e = declusters(s, 10, IDX)
        sp = (fwd_lag(smh, H_MAIN, 1) - fwd_lag(qqq, H_MAIN, 1)).reindex(e).dropna()
        rs = fwd_lag(smh, H_MAIN, 1)
        rq = fwd_lag(qqq, H_MAIN, 1)
        okh = rs.notna() & rq.notna()
        bh = float(np.polyfit(rq[okh], rs[okh], 1)[0])
        rz = (rs - bh * rq).reindex(e).dropna()
        rows.append({"smh_cut": cs, "qqq_cut": cq, "n_days": len(s), "n_epi": len(sp),
                     "spread_pct": round(100 * sp.mean(), 3) if len(sp) else np.nan,
                     "resid_pct": round(100 * rz.mean(), 3) if len(rz) else np.nan,
                     "hit": round(100 * (sp > 0).mean(), 1) if len(sp) else np.nan,
                     "bear_frac": round(100 * float(below_spy200.loc[s].mean()), 1)})
show(rows, f"h={H_MAIN} equal-$ spread and beta-neutral residual across the cut grid")

# --------------------------------------------------------------- 4. ROUND 2: ERA + SEMI CYCLE
print("\n" + "=" * 92)
print("4. ERA + REGIME (semi cycle: SMH above/below its own 200d at trigger)")
print("=" * 92)
sp5 = (fwd_lag(smh, H_MAIN, 1) - fwd_lag(qqq, H_MAIN, 1))
e = epi.intersection(sp5.dropna().index)
show(era_split(e, sp5.loc[e].values), "equal-$ spread, era split (episodes)")
above = smh_vs200.reindex(e) > 0
show([summarize(sp5.loc[e[above.values]].values, "SMH ABOVE its 200d at trigger (today's state)"),
      summarize(sp5.loc[e[~above.values]].values, "SMH BELOW its 200d at trigger")],
     "semi-cycle split")

# --------------------------------------------------------------- 5. BREADTH VERSION
print("\n" + "=" * 92)
print("5. SEMI SINGLE-NAME BREADTH VERSION (does the finding survive the vehicle?)")
print("=" * 92)
names = ["NVDA", "AVGO", "AMAT", "MU", "AMD", "INTC", "TXN", "QCOM", "ADI"]
p2 = close_panel(names + ["QQQ"]).dropna()
rows = []
for t in names:
    r63t = pct_rank(p2[t], 63)
    r63q = pct_rank(p2["QQQ"], 63)
    m = ((r63t <= 5) & (r63q >= 20)).fillna(False)
    s = p2.index[m.values]
    if len(s) < 3:
        rows.append({"ticker": t, "n_epi": 0})
        continue
    e2 = declusters(s, 10, p2.index)
    spx = (fwd_lag(p2[t], H_MAIN, 1) - fwd_lag(p2["QQQ"], H_MAIN, 1)).reindex(e2).dropna()
    r = summarize(spx.values, t)
    rows.append({"ticker": t, "n_epi": r.get("n", 0),
                 "spread_pct": round(r.get("mean_pct", np.nan), 3),
                 "hit": round(r.get("hit", np.nan), 1)})
show(rows, "same cell, each semi single name vs QQQ, equal-$ h=5 (reference class)")
