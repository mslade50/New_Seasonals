import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

# C3 round 2 on the one split that matches the live tape (midterm 5-0 in the SVXY era):
# does ANY member look like today (dose 7.95%, VIX/VIX3M 0.887, dial ma10 85)? And the
# live neighbourhood in (term structure x dose), plus the parent scored on its own terms.
pd.set_option("future.no_silent_downcasting", True)
TK = ["SPY", "^VIX", "^VIX3M", "SVXY"]
raw = close_panel(TK)
cal = raw["SPY"].dropna().index
px = raw.reindex(cal)
px["^VIX"] = px["^VIX"].ffill(limit=2)
vix = px["^VIX"]
v1 = vix / vix.shift(1) - 1
term = vix / px["^VIX3M"]
frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")["63d"]
dial = frag.rolling(10).mean().reindex(cal).ffill(limit=3)
ERA = pd.Timestamp("2018-03-01")
live = pd.Timestamp("2026-09-14")
print(f"LIVE v1 {100*v1.loc[live]:+.2f}% term {term.loc[live]:.3f} VIX {vix.loc[live]:.2f} dial ma10(63d) {dial.loc[live]:.1f}")

fomc = pd.DatetimeIndex(load_events(["fomc_decision"])["date"])
vxe = set(pd.DatetimeIndex(load_events(["vix_expiry"])["date"]))
fpos, fk = anchor_positions(cal, fomc, 0)
fpos = np.array(fpos)
fk = pd.DatetimeIndex(fk)
sp = fpos - 2
d = cal[sp]
svxy1 = fwd_lag(px["SVXY"], 1, 1)
spy1 = fwd_lag(px["SPY"], 1, 1)
svix1 = -fwd_lag(vix, 1, 1)
svxy2 = fwd_lag(px["SVXY"], 2, 1)
ok = svxy1.notna() & spy1.notna() & (cal >= ERA)
beta = np.polyfit(spy1[ok].values, svxy1[ok].values, 1)[0]
alpha1 = svxy1 - beta * spy1


def rec(v, label):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    r = summarize(v, label)
    if r["n"]:
        w = int((v > 0).sum())
        r["rec"] = f"{w}-{len(v)-w}"
        r["sign_p"] = round(sign_test(w, len(v)), 4)
    return r


V1 = v1.reindex(d).values
TM = term.reindex(d).values
DL = dial.reindex(d).values
mid = np.array([x.year % 4 == 2 for x in fk])
print("\nMIDTERM FOMCs with VIX 1d >= 5% at k=-2 (all eras):")
for x, f, a, b, c, dl in zip(d[(V1 >= .05) & mid], fk[(V1 >= .05) & mid], V1[(V1 >= .05) & mid], TM[(V1 >= .05) & mid],
                            VIXL := vix.reindex(d).values[(V1 >= .05) & mid], DL[(V1 >= .05) & mid]):
    print(f"  {x.date()} FOMC {f.date()} dose {100*a:+.1f}% VIX {c:.1f} term {b:.3f} dial {dl:.1f}  "
          f"SVXY h1 {100*svxy1.loc[x]:+.2f}% alpha {100*alpha1.loc[x]:+.2f}% sVIX {100*svix1.loc[x]:+.2f}% SVXY h2 {100*svxy2.loc[x]:+.2f}% era {'-0.5x' if x >= ERA else 'pre'}")

post = d >= ERA
st = V1 >= .05
rows = []
for lbl, m in [("STATE term<0.9 (LIVE side)", st & (TM < 0.9)), ("STATE term>=0.9", st & (TM >= 0.9)),
               ("STATE term<0.9 & dose [5,10%)", st & (TM < 0.9) & (V1 < .10)),
               ("STATE term<0.95 & dose [6,10%)", (V1 >= .06) & (V1 < .10) & (TM < 0.95)),
               ("STATE dial>=70", st & (DL >= 70)), ("STATE dial<70", st & (DL < 70)),
               ("STATE midterm", st & mid), ("STATE midterm & term<0.9", st & mid & (TM < 0.9)),
               ("PARENT all", np.ones(len(d), bool)), ("PARENT midterm", mid),
               ("PARENT term<0.9", TM < 0.9), ("PARENT dial>=70", DL >= 70)]:
    for nm, s in [("SVXY", svxy1), ("alpha", alpha1)]:
        vals = s.reindex(d).values
        rows.append(rec(vals[m & post], f"{nm}: {lbl}"))
show(rows, "SVXY era (2018-03+), h=1: live-neighbourhood splits and the parent on its own terms")
rows = []
for lbl, m in [("STATE term<0.9", st & (TM < 0.9)), ("STATE term>=0.9", st & (TM >= 0.9)),
               ("STATE midterm & term<0.9", st & mid & (TM < 0.9)), ("STATE midterm & term>=0.9", st & mid & (TM >= 0.9))]:
    vals = svix1.reindex(d).values
    rows.append(rec(vals[m], f"short ^VIX full: {lbl}"))
show(rows, "^VIX index read, full history (VIX3M from 2006-07)")
print(f"\ncost: long SVXY 10 bp RT; hedged (SVXY + {beta:.2f}x SPY short) ~12 bp RT")
