"""A4 -- bond vol popping while equity vol falls, AT the yield level extreme.

Both parents are closed (2026-08-18 twice, 2026-08-26 on the divergence gate),
so the ONLY thing that can live here is the three-way INTERACTION with the
trailing-252 yield maximum.  Count the interaction first; it is expected to be
empty or near-empty, and the budget is deliberately small.

Live state 2026-08-31: ^MOVE +6.13%, ^VIX down, ^TNX at its 252-day max.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
ASOF = pd.Timestamp("2026-08-31")

px = close_panel(["^TNX", "^MOVE", "^VIX", "TLT", "IEF", "SPY"]).dropna(how="any")
px = px[px.index <= ASOF]
idx = px.index
tnx, mv, vx = px["^TNX"], px["^MOVE"], px["^VIX"]

hi252 = rolling_on_valid(tnx, lambda x: x.rolling(252).max())
off_hi = tnx / hi252 - 1.0
LEVEL = off_hi >= -0.0025
dmv = mv.pct_change() * 100
dvx = vx.pct_change() * 100

d = px[["TLT", "IEF"]].pct_change().dropna()
BETA = float(np.polyfit(d["IEF"].values, d["TLT"].values, 1)[0])
FLAT = [("IEF", 1.0), ("TLT", -1.0 / BETA)]
VEH = {"TLT": [("TLT", 1.0)], "IEF": [("IEF", 1.0)], "FLAT": FLAT, "SPY": [("SPY", 1.0)]}

POS = {dd: i for i, dd in enumerate(idx)}


def fast_decluster(sig, gap):
    keep, last = [], -10 ** 9
    for dd in sig:
        p = POS.get(dd)
        if p is None:
            continue
        if p - last >= gap:
            keep.append(dd)
            last = p
    return pd.DatetimeIndex(keep)


print("=" * 110)
print("1. COUNT FIRST on the three-way interaction")
print("=" * 110)
print("  panel %s .. %s  N=%d (^MOVE-limited)" % (idx[0].date(), idx[-1].date(), len(idx)))
print("  TODAY: ^MOVE %+.2f%%  ^VIX %+.2f%%  off-high %+.5f%%"
      % (dmv.iloc[-1], dvx.iloc[-1], 100 * off_hi.iloc[-1]))

DIV = (dmv > 0) & (dvx < 0)
for mvthr in (0.0, 3.0, 5.0, 6.0):
    A = DIV & (dmv >= mvthr)
    B = A & LEVEL
    da, db = idx[A.reindex(idx, fill_value=False).values], idx[B.reindex(idx, fill_value=False).values]
    epa, epb = fast_decluster(da, 10), fast_decluster(db, 10)
    print("  ^MOVE up >= %.0f%% & ^VIX down : %4d days / %3d episodes"
          "   ||  AND ^TNX within 0.25%% of 252-max : %3d days / %2d episodes"
          % (mvthr, len(da), len(epa), len(db), len(epb)))
    if len(epb):
        print("      interaction episode dates:", ", ".join(str(x.date()) for x in epb))

print("\n" + "=" * 110)
print("2. Does the LEVEL gate add anything to the (already dead) divergence?")
print("   gate-attribution: divergence AND level  vs  divergence WITHOUT level")
print("=" * 110)
A = DIV & (dmv >= 3.0)
for h in (5, 10):
    for vk, legs in VEH.items():
        ret = vehicle_ret(px, legs, h, 1)
        valid = ret.notna()
        outs = []
        for lab, M in (("DIV+LEVEL", A & LEVEL), ("DIV only", A & ~LEVEL.fillna(False)),
                       ("LEVEL only", LEVEL.fillna(False) & ~A.fillna(False))):
            sig = idx[M.reindex(idx, fill_value=False).values & valid.values]
            ep = fast_decluster(sig, max(h, 10))
            v = ret.loc[ep].values if len(ep) else np.array([])
            outs.append((lab, len(v), 100 * v.mean() if len(v) else np.nan,
                         100 * (v > 0).mean() if len(v) else np.nan, v))
        base = 100 * float(ret.dropna().mean())
        line = "  h=%2d %-5s | " % (h, vk)
        line += " | ".join("%s N=%2d %+.3f%% hit %.0f%%" % (o[0], o[1], o[2], o[3]) for o in outs)
        line += " | all-days %+.3f%%" % base
        a, b = outs[0][4], outs[1][4]
        if len(a) >= 2 and len(b) >= 2:
            se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
            line += " | gate adds %+.3fpp t %+.2f" % (100 * (a.mean() - b.mean()),
                                                      (a.mean() - b.mean()) / se)
        print(line)
        if len(a) and len(a) < 15:
            w = int((a > 0).sum())
            print("        DIV+LEVEL sign test %d-%d p=%.4f | per-episode %s"
                  % (w, len(a) - w, sign_test(w, len(a)), [round(100 * z, 2) for z in a]))

print("\n" + "=" * 110)
print("3. Today's magnitude inside the interaction's own support")
print("=" * 110)
B = A & LEVEL
sigB = idx[B.reindex(idx, fill_value=False).values]
epB = fast_decluster(sigB, 10)
if len(epB):
    e = dmv.loc[epB].values
    print("  ^MOVE 1d %% at the interaction episodes: min %.2f med %.2f max %.2f | TODAY %.2f = %.0fth pctile"
          % (np.nanmin(e), np.nanmedian(e), np.nanmax(e), dmv.iloc[-1],
             100 * float((e <= dmv.iloc[-1]).mean())))
    print("  episode years:", dict(pd.Series(pd.DatetimeIndex(epB).year).value_counts().sort_index()))
else:
    print("  interaction is EMPTY at this rung.")
