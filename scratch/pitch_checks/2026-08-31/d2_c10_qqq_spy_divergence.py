"""C10 -- growth-vs-index divergence at a 63-day rank extreme.

Trigger: QQQ 63d return rank (trailing 252d) <= 10 AND SPY within 1.5% of its
52-week high. Live 2026-08-28: QQQ r63 8.7 (63d ret -2.86%), SPY -1.10% off
its high (63d ret +1.96%).

Trades: (i) LONG QQQ / SHORT SPY beta-matched, (ii) LONG QQQ outright.

Standing blockers this must clear:
 - the SPY-ONLY round-trip breakout, killed 2026-08-28: joint -0.048% on 37
   episodes vs own drift +0.457%. OVERLAP with that population is measured
   here explicitly.
 - watchlist 33 pooled finding: the r63-low clause SUBTRACTS -0.106pp over 29
   ETFs and discards 95% of the population.

Kill tests, in order:
 0. population overlap with the 2026-08-28 corpse
 1. LEG ATTRIBUTION (long QQQ alone vs short SPY alone over the same windows)
 2. GATE ATTRIBUTION (bare QQQ r63<=10 / bare SPY-near-high / joint)
 3. full battery on both expressions
 4. threshold ladder on both rungs
 5. era / midterm / concentration / sign / decluster
 6. dial distribution of the trigger population vs today's ma10(63d) = 87.6
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

px = pd.DataFrame({t: s["Close"] for t, s in load_prices(["QQQ", "SPY"]).items()})
px = px.dropna()
IDX = px.index

q, s = px["QQQ"], px["SPY"]
q_r63 = pct_rank(q, 63, 252)
s_r63 = pct_rank(s, 63, 252)
s_hi = rolling_on_valid(s, lambda x: x.rolling(252).max())
s_dist = (s / s_hi - 1.0) * 100.0          # negative = below high
q_hi = rolling_on_valid(q, lambda x: x.rolling(252).max())

M_QLOW = (q_r63 <= 10).fillna(False)
M_SHI = (s_dist >= -1.5).fillna(False)
JOINT = M_QLOW & M_SHI

print("=" * 78)
print("C10  QQQ r63 <= 10  AND  SPY within 1.5% of 52w high")
print("=" * 78)
print(f"span {IDX[0].date()} .. {IDX[-1].date()}  ({len(IDX)} sessions)")
print(f"counts: QQQ r63<=10 {int(M_QLOW.sum())}   SPY within 1.5% {int(M_SHI.sum())}"
      f"   JOINT {int(JOINT.sum())}")
last = IDX[-1]
print(f"live row {last.date()}: QQQ r63 {q_r63.loc[last]:.1f}, SPY dist "
      f"{s_dist.loc[last]:+.2f}%, fires = {bool(JOINT.loc[last])}")

# --------------------------------------------------- 0. overlap with the corpse
print("\n" + "=" * 78)
print("0. POPULATION OVERLAP with the 2026-08-28 SPY-only kill")
print("   (that cell: SPY near its 52w high AND SPY's OWN r63 bottom-quartile)")
print("=" * 78)
for rung, lbl in ((25, "SPY r63<=25 (bottom quartile, the killed form)"),
                  (10, "SPY r63<=10")):
    corpse = ((s_r63 <= rung) & M_SHI).fillna(False)
    inter = int((JOINT & corpse).sum())
    print(f"  {lbl}: N={int(corpse.sum())}   overlap with C10 trigger = {inter} "
          f"days = {100*inter/max(1,int(JOINT.sum())):.1f}% of C10's population")
print(f"  SPY's own r63 on C10 trigger days: median "
      f"{s_r63[JOINT].median():.1f}, "
      f"pct of trigger days with SPY r63<=25 = "
      f"{100*(s_r63[JOINT] <= 25).mean():.1f}%")
print(f"  live 2026-08-28 SPY r63 = {s_r63.loc[last]:.1f} (surface map says 19.0)")

# ------------------------------------------------------------- measured beta
rq, rs = q.pct_change(), s.pct_change()
cov = rq.rolling(252).cov(rs)
var = rs.rolling(252).var()
beta_roll = (cov / var)
beta_full = float(np.polyfit(rs.dropna().values, rq.reindex(rs.dropna().index).values, 1)[0])
print(f"\nMEASURED BETA QQQ vs SPY: full-sample OLS {beta_full:.3f}; "
      f"rolling-252 median {beta_roll.median():.3f}; on trigger days median "
      f"{beta_roll[JOINT].median():.3f}; live {beta_roll.iloc[-1]:.3f}")
BETA = round(float(beta_roll[JOINT].median()), 2)
print(f"pair uses fixed beta {BETA} (trigger-day median); "
      f"a time-varying-beta pair is scored separately below")

# ------------------------------------------------------- 1. LEG ATTRIBUTION
print("\n" + "=" * 78)
print("1. LEG ATTRIBUTION -- what does each leg contribute over the SAME windows?")
print("=" * 78)
for h in (1, 3, 5, 10):
    rQ = fwd_lag(q, h, 1)
    rS = fwd_lag(s, h, 1)
    valid = rQ.notna() & rS.notna()
    trig = IDX[JOINT.values & valid.values]
    epi = declusters(trig, 5, IDX)
    rows = [summarize(rQ.loc[epi].values, "long QQQ leg"),
            summarize((-BETA * rS).loc[epi].values, f"short {BETA}x SPY leg"),
            summarize((rQ - BETA * rS).loc[epi].values, "PAIR (sum)"),
            summarize((rQ - beta_roll * rS).loc[epi].values, "PAIR (rolling beta)")]
    show(rows, f"h={h}  episodes N={len(epi)}")

# ------------------------------------------------------- 2. GATE ATTRIBUTION
print("\n" + "=" * 78)
print("2. GATE ATTRIBUTION (episodes, min_gap 5)")
print("=" * 78)
gates = {"(a) QQQ r63<=10 only": M_QLOW,
         "(b) SPY within 1.5% only": M_SHI,
         "(c) JOINT": JOINT,
         "--- all days": pd.Series(True, index=IDX)}
for h in (3, 5, 10):
    for expr, legs in (("PAIR", [("QQQ", 1.0), ("SPY", -BETA)]),
                       ("QQQ outright", [("QQQ", 1.0)])):
        ret = vehicle_ret(px, legs, h, 1)
        valid = ret.notna()
        rows = []
        for lbl, m in gates.items():
            t = IDX[m.reindex(IDX, fill_value=False).values & valid.values]
            e = declusters(t, 5, IDX)
            r = summarize(ret.loc[e].values, lbl)
            r["n_days"] = len(t)
            rows.append(r)
        show(rows, f"h={h} {expr}")
        a, b, c = (rows[0].get("mean_pct", np.nan), rows[1].get("mean_pct", np.nan),
                   rows[2].get("mean_pct", np.nan))
        print(f"  JOIN VALUE: joint {c:+.3f}% vs better parent {max(a,b):+.3f}% "
              f"-> {c - max(a,b):+.3f}pp")

# ------------------------------------------------------------- 3. batteries
VAR = {}
for qr in (5, 10, 15, 25):
    VAR[f"QQQ r63<={qr} & SPY<=1.5%"] = ((q_r63 <= qr) & M_SHI).fillna(False)
for dd in (0.5, 1.0, 1.5, 3.0):
    VAR[f"QQQ r63<=10 & SPY<={dd}%"] = (M_QLOW & (s_dist >= -dd)).fillna(False)

for expr, legs, cost in (("PAIR long QQQ / short SPY", [("QQQ", 1.0), ("SPY", -BETA)], 3.0),
                         ("LONG QQQ outright", [("QQQ", 1.0)], 3.0)):
    battery(px, JOINT, legs, 5, f"C10 {expr}", cost, variants=VAR, min_gap=5)

# --------------------------------------------------- 5. midterm + year table
print("\n" + "=" * 78)
print("5. MIDTERM / YEAR / DECLUSTER  (h=5)")
print("=" * 78)
for expr, legs in (("PAIR", [("QQQ", 1.0), ("SPY", -BETA)]),
                   ("QQQ outright", [("QQQ", 1.0)])):
    ret = vehicle_ret(px, legs, 5, 1)
    trig = IDX[JOINT.values & ret.notna().values]
    epi = declusters(trig, 5, IDX)
    v = ret.loc[epi].values
    mid = np.array([(d.year % 4) == 2 for d in epi])
    show([summarize(v[mid], f"{expr} midterm (N={int(mid.sum())})"),
          summarize(v[~mid], f"{expr} non-midterm (N={int((~mid).sum())})")], expr)
    print("  years:", dict(pd.Series(v).groupby(epi.year.values).count()))
    for mg in (5, 10, 21):
        e = declusters(trig, mg, IDX)
        print(f"    min_gap={mg}: N={len(e)} mean {100*ret.loc[e].mean():+.3f}% "
              f"hit {100*(ret.loc[e]>0).mean():.1f}%")

# ------------------------------------------------- 6. DIAL DISTRIBUTION
print("\n" + "=" * 78)
print("6. FRAGILITY DIAL DISTRIBUTION OF THE TRIGGER POPULATION")
print("   (data/rd2_fragility.parquet -- SIZING vintage; append-only PIT only")
print("    since 2026-07-02, everything before is the recompute vintage)")
print("=" * 78)
fr = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
fr.index = pd.to_datetime(fr.index)
ma10 = fr["63d"].rolling(10).mean()
today_ma10 = float(ma10.iloc[-1])
trig = IDX[JOINT.values]
have = ma10.reindex(trig).dropna()
print(f"trigger days total {len(trig)}, with a dial reading {len(have)} "
      f"(dial series starts {fr.index.min().date()})")
if len(have):
    print(f"  ma10(63d) on trigger days: min {have.min():.1f}  med {have.median():.1f}"
          f"  p90 {have.quantile(.9):.1f}  MAX {have.max():.1f}")
    print(f"  TODAY = {today_ma10:.1f}   -> today is above the trigger-population "
          f"maximum: {today_ma10 > have.max()}")
    print(f"  trigger days with ma10 >= 85: {int((have >= 85).sum())}")
    print("  dial-bearing trigger dates:",
          ", ".join(f"{d.date()}:{v:.0f}" for d, v in have.items()))
