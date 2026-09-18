"""c3 round 1: watchlist 51 re-run on today's tape + the never-run splits.

Trade: SPY-hedged SHORT SVXY (short SVXY, long b x SPY), signal = one-day
^VIX change <= -10% on close D, entry MOC D+1, exit close D+1+h. 2018-03+
(real -0.5x SVXY). Live: D = 2026-09-17 (^VIX -12.82%), entry = 09-18 close
(September opex / quad witching), D is FOMC k=+1 (decision 09-16).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch_lab import *  # noqa
from kA_common import *  # noqa
import numpy as np
import pandas as pd

px = build_panel()
cal = px.index
vix = px["^VIX"]
vchg = vix / vix.shift(1) - 1
post = pd.Series(cal >= POST, index=cal)
live = pd.Timestamp("2026-09-17")

since_fomc = prev_event_distance(cal, ["fomc_decision"])      # 1 => D is FOMC k=+1
to_opex = next_event_distance(cal, ["opex"])                   # 1 => entry close is opex
to_print = next_event_distance(cal, ["nfp", "cpi", "fomc_decision"])
entry_runway = to_print - 1                                    # sessions from ENTRY close to next print
mid = pd.Series(cal.year % 4 == 2, index=cal)
vts = px["^VIX"] / px["^VIX3M"]

print(f"LIVE {live.date()}: vchg {100*vchg[live]:+.2f}%  since_fomc {since_fomc[live]}  "
      f"to_opex {to_opex[live]}  entry runway to next NFP/CPI/FOMC {entry_runway[live]}  "
      f"VIX/VIX3M {vts[live]:.3f}  VIX {vix[live]:.2f}")

COST = 12.0
res = {}
for h in (1, 2, 3):
    b = hedge_beta(px, "SVXY", h, 1, post)
    res[h] = (b, vehicle_ret(px, [("SVXY", -1.0), ("SPY", b)], h))
    print(f"h={h}: SVXY-on-SPY beta 2018-03+ = {b:.3f}")

# ---------------------------------------------------------------- 1. W51 reproduction + controls
for h in (1, 2, 3):
    b, r = res[h]
    ok = r.notna() & post
    crush = (vchg <= -0.10) & ok
    d = declusters(cal[crush.values], max(h, 3), cal)
    loc = local_control(cal[ok.values], cal[crush.values])
    rows = [rec_row(r.loc[d].values, f"crush<=-10% episodes", COST),
            rec_row(r.loc[cal[crush.values]].values, "crush<=-10% day-level", COST),
            rec_row(r[ok].values, "CTRL all days 2018-03+"),
            rec_row(r.loc[loc].values, "CTRL local +/-126td ex-trigger"),
            rec_row(-vehicle_ret(px, [("SVXY", 1.0)], h).loc[d].values, "raw short SVXY (unhedged), same eps"),
            rec_row(vehicle_ret(px, [("SPY", b)], h).loc[d].values, "the SPY leg alone (b x SPY)")]
    show(rows, f"1. W51 reproduction, hedged short SVXY, h={h}, beta {b:.2f}")

# ---------------------------------------------------------------- 2. dose ladder
for h in (1, 2, 3):
    b, r = res[h]
    ok = r.notna() & post
    rows = []
    for lbl, m in [("[-8,-10)", (vchg <= -0.08) & (vchg > -0.10)),
                   ("[-10,-12)", (vchg <= -0.10) & (vchg > -0.12)),
                   ("[-12,-15)", (vchg <= -0.12) & (vchg > -0.15)),
                   ("<=-15", vchg <= -0.15),
                   (">=10 cell", vchg <= -0.10), (">=12 cell", vchg <= -0.12)]:
        mm = m & ok
        d = declusters(cal[mm.values], max(h, 3), cal)
        rows.append(rec_row(r.loc[d].values, f"dose {lbl}", COST))
    show(rows, f"2. dose ladder h={h}")

# ---------------------------------------------------------------- 3. the new splits
for h in (1, 2, 3):
    b, r = res[h]
    ok = r.notna() & post
    crush = (vchg <= -0.10) & ok
    splits = [
        ("crush & FOMC k=+1", crush & (since_fomc == 1)),
        ("crush NOT FOMC k=+1", crush & (since_fomc != 1)),
        ("crush & opex at D+1 (entry ON opex close)", crush & (to_opex == 1)),
        ("crush & opex in D+1..D+3", crush & to_opex.between(1, 3)),
        ("crush & opex NOT in D+1..D+3", crush & ~to_opex.between(1, 3)),
        ("JOINT FOMC k=+1 & opex D+1..D+3", crush & (since_fomc == 1) & to_opex.between(1, 3)),
        ("crush >=12 & opex D+1..D+3", (vchg <= -0.12) & ok & to_opex.between(1, 3)),
        ("crush >=12 & FOMC k=+1", (vchg <= -0.12) & ok & (since_fomc == 1)),
        ("crush & runway<=1", crush & (entry_runway <= 1)),
        ("crush & runway 2-4", crush & entry_runway.between(2, 4)),
        ("crush & runway 5-9", crush & entry_runway.between(5, 9)),
        ("crush & runway>=10", crush & (entry_runway >= 10)),
        ("crush & runway>=5", crush & (entry_runway >= 5)),
        ("crush & midterm", crush & mid),
        ("crush & non-midterm", crush & ~mid),
        ("crush & VIX/VIX3M<0.85", crush & (vts < 0.85)),
        ("crush & VIX/VIX3M>=0.85", crush & (vts >= 0.85)),
        ("crush & VIX<16 after", crush & (vix < 16)),
        ("crush & VIX>=16 after", crush & (vix >= 16)),
        ("crush & September", crush & pd.Series(cal.month == 9, index=cal)),
    ]
    rows = []
    for lbl, m in splits:
        d = declusters(cal[m.values], max(h, 3), cal)
        x = rec_row(r.loc[d].values, lbl, COST)
        rows.append(x)
    show(rows, f"3. splits, hedged short SVXY h={h}")

# ---------------------------------------------------------------- 4. FOMC offset ladder for crushes
print("\n=== 4. crush cell by FOMC offset (D = decision + j), h=1 and h=2 ===")
for h in (1, 2):
    b, r = res[h]
    ok = r.notna() & post
    rows = []
    for j in range(0, 8):
        m = (vchg <= -0.10) & ok & (since_fomc == j)
        d = cal[m.values]
        rows.append({"j(since FOMC)": j, **{k: v for k, v in rec_row(r.loc[d].values, "").items()
                                          if k in ("n", "mean_pct", "hit", "rec", "sign_p")}})
    print(f"h={h}")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n=== 4b. crush cell by opex distance (to_opex = sessions from D to opex) ===")
for h in (1, 2):
    b, r = res[h]
    ok = r.notna() & post
    rows = []
    for j in range(1, 11):
        m = (vchg <= -0.10) & ok & (to_opex == j)
        d = cal[m.values]
        rows.append({"to_opex": j, **{k: v for k, v in rec_row(r.loc[d].values, "").items()
                                    if k in ("n", "mean_pct", "hit", "rec", "sign_p")}})
    print(f"h={h}")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

# ---------------------------------------------------------------- 5. era split + synthetic pre-break context
print("\n=== 5. era splits, h=1 hedged ===")
b, r = res[1]
ok = r.notna() & post
crush = (vchg <= -0.10) & ok
d = declusters(cal[crush.values], 3, cal)
v = r.loc[d]
for lbl, sel in [("2018-03..2021", v.index < "2022-01-01"), ("2022+", v.index >= "2022-01-01")]:
    show([rec_row(v[sel].values, lbl, COST)])
print("  concentration (h=1, >=10):", signed_concentration(v.index, v.values))
d12 = declusters(cal[((vchg <= -0.12) & ok).values], 3, cal)
print("  concentration (h=1, >=12):", signed_concentration(d12, r.loc[d12].values))

print("\n--- synthetic -0.5x (SVS) pre-break 2011-10..2018-02, hedged at the SVS pre-break beta ---")
pre = pd.Series((cal < BREAK) & (cal >= pd.Timestamp("2011-10-10")), index=cal)
for h in (1, 2, 3):
    bp = hedge_beta(px, "SVS", h, 1, pre)
    rp = vehicle_ret(px, [("SVS", -1.0), ("SPY", bp)], h)
    okp = rp.notna() & pre
    rows = []
    for lbl, m in [(">=10", vchg <= -0.10), (">=12", vchg <= -0.12), ("[10,12)", (vchg <= -0.10) & (vchg > -0.12))]:
        dd = declusters(cal[(m & okp).values], max(h, 3), cal)
        rows.append(rec_row(rp.loc[dd].values, f"PRE-break synth h={h} {lbl} (beta {bp:.2f})", COST))
    rows.append(rec_row(rp[okp].values, f"PRE-break synth all days h={h}"))
    show(rows)

# ---------------------------------------------------------------- 6. live-state neighbours list
print("\n=== 6. all 2018-03+ crush days with opex in D+1..D+3 or FOMC k=+1 (h=1,2,3 hedged) ===")
m = (vchg <= -0.10) & post & (to_opex.between(1, 3) | (since_fomc == 1))
for x in cal[m.values]:
    vals = [res[h][1].get(x, np.nan) for h in (1, 2, 3)]
    print(f"  {x.date()} vchg {100*vchg[x]:+.1f}% VIX {vix[x]:.1f} since_fomc {since_fomc[x]} "
          f"to_opex {to_opex[x]} runway {entry_runway[x]}  h1 {100*vals[0]:+.2f}% h2 {100*vals[1]:+.2f}% h3 {100*vals[2]:+.2f}%")
