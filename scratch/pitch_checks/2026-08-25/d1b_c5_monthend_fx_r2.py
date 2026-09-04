"""C5 round 2 - price the one thing round 1 left open, and close the cell.

Round 1 found:
  - the pre-specified signed regression is t=-0.75 (DXY) / -0.20 (UUP), R^2 <0.002
  - the terciles are NON-MONOTONE (US-lags -0.027%, middle -0.034%, US-leads
    -0.159% on DXY), so there is no dose response
  - the SESSION the mechanism names, ME-0, pays -0.55bp at a 45.6% hit against
    an all-days base of +0.10bp, and is +3.57bp (WRONG SIGN) from 2020
  - the window's total comes from ME-1/-3/-4, sessions the story does not name

The one open item is the ME-5 session (+5.03bp DXY, 55.9% hit, era-stable).
That came out of a ladder I built and walked, so it owes a multiplicity charge.
Also check the live cells: August, and midterm years.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

px = close_panel(["DX-Y.NYB", "UUP", "SPY", "EFA"])


def me_pos(s):
    ym = pd.Series(list(zip(s.index.year, s.index.month)), index=s.index)
    p = pd.Series(0, index=s.index)
    for _, g in ym.groupby(ym):
        d = g.index
        p.loc[d] = np.arange(len(d) - 1, -1, -1)
    return p


print("=== 1. the ME-5 session, priced as what it is: 1 of 8 walked cells ===")
for t in ["DX-Y.NYB", "UUP"]:
    s = px[t].dropna()
    p = me_pos(s)
    r1 = (s / s.shift(1) - 1.0)
    ts = []
    for k in range(0, 16):
        v = r1[p == k].dropna().values
        if len(v) < 20:
            continue
        ts.append((k, len(v), v.mean(), abs(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))))))
    ts.sort(key=lambda z: -z[3])
    print(f"\n  {t}: 16 session offsets walked, ranked by |t|")
    for k, n, m, tt in ts[:5]:
        print(f"    ME-{k:<2} N={n:>4}  mean={m*1e4:+7.2f}bp  |t|={tt:.2f}")
    best = ts[0]
    # rotation permutation: relocate the month-end anchor to a random offset
    rng = np.random.default_rng(42)
    r = r1.dropna()
    maxt = []
    for _ in range(2000):
        sh = rng.integers(1, len(r) - 1)
        pr = pd.Series(np.roll(p.reindex(r.index).values, sh), index=r.index)
        best_t = 0.0
        for k in range(0, 16):
            v = r[pr == k].values
            if len(v) < 20:
                continue
            tt = abs(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))))
            best_t = max(best_t, tt)
        maxt.append(best_t)
    maxt = np.array(maxt)
    print(f"    rotation null over the SAME 16-cell walk: P(max |t| >= {best[3]:.2f}) = {(maxt >= best[3]).mean():.3f}")

print("\n=== 2. the live cells: August, and midterm years ===")
ANCH = 4
for t in ["DX-Y.NYB", "UUP"]:
    s = px[t].dropna()
    p = me_pos(s)
    rows = []
    for a in s.index[p == ANCH]:
        i = s.index.get_loc(a)
        j = i + ANCH
        if j >= len(s) or p.iloc[j] != 0:
            continue
        rows.append({"d": a, "r": s.iloc[j] / s.iloc[i] - 1.0})
    df = pd.DataFrame(rows)
    df["mon"] = df.d.dt.month
    df["mid"] = (df.d.dt.year % 4 == 2)
    aug = df[df.mon == 8].r.values
    mid = df[df.mid].r.values
    both = df[(df.mon == 8) & df.mid].r.values
    print(f"\n  {t}  all N={len(df)} mean={df.r.mean()*100:+.3f}%")
    for lab, v in [("August", aug), ("midterm", mid), ("August x midterm", both)]:
        if len(v):
            print(f"    {lab:<18} N={len(v):>3} mean={v.mean()*100:+.3f}% med={np.median(v)*100:+.3f}% hit(up)={(v>0).mean()*100:5.1f}%"
                  f"  sign p(down) {sign_test(int((v<0).sum()), len(v)):.4f}")

print("\n=== 3. cost, on the vehicle that actually trades ===")
print("  DX futures round trip ~1.5 bp; UUP round trip ~4 bp (0.20% ER, 2c on ~$27).")
for t, cost in [("DX-Y.NYB", 1.5), ("UUP", 4.0)]:
    s = px[t].dropna()
    p = me_pos(s)
    rows = []
    for a in s.index[p == ANCH]:
        i = s.index.get_loc(a)
        j = i + ANCH
        if j >= len(s) or p.iloc[j] != 0:
            continue
        rows.append(s.iloc[j] / s.iloc[i] - 1.0)
    v = np.array(rows)
    edge = abs(v.mean()) * 1e4
    print(f"  {t:<10} |edge| = {edge:5.2f} bp   cost {cost:4.1f} bp   ratio {edge/cost:4.2f}x   (bar is 5x)")
