"""kB D1 round 1: long the dollar (DX futures, proxy DX-Y.NYB) after a z10 >= 2.0
thrust that closes within 0.5% of its 252 high, h=1..10. Pre-specified LONG
(continuation, breakout-following flow). Job: kill it.

z10 here is the TAPE definition: 10-session return / (21d realized vol * sqrt(10)).
Midterm split pre-committed (the washout pole, W22, is midterm-inverted).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

DX = "DX-Y.NYB"
raw = load_prices([DX, "UUP"])


def tape_z10(c: pd.Series) -> pd.Series:
    c = c.dropna()
    r1 = c.pct_change()
    r10 = c / c.shift(10) - 1.0
    vol = r1.rolling(21).std()
    return r10 / (vol * np.sqrt(10))


for t in (DX, "UUP"):
    c = raw[t]["Close"].dropna()
    z = tape_z10(c)
    hi = c.rolling(252).max()
    print(f"{t}: {c.index[-1].date()} close {c.iloc[-1]:.3f}  tape z10 {z.iloc[-1]:+.2f}  "
          f"lab zscore {zscore(c, 10).iloc[-1]:+.2f}  off 252 high {100*(c.iloc[-1]/hi.iloc[-1]-1):+.2f}%  "
          f"1d {100*(c.iloc[-1]/c.iloc[-2]-1):+.2f}%")

df = raw[DX]
a = wilder_atr(df["High"].to_numpy(), df["Low"].to_numpy(), df["Close"].to_numpy())
print(f"DX-Y.NYB Wilder-14 ATR {a[-1]:.3f} ({100*a[-1]/df['Close'].iloc[-1]:.2f}% of close)")

px = pd.DataFrame({DX: raw[DX]["Close"]}).dropna()
c = px[DX]
z10 = tape_z10(c).reindex(px.index)
hi252 = c.rolling(252).max()
off = c / hi252 - 1.0

cell = (z10 >= 2.0) & (off >= -0.005)
print("\nlast 8 sessions of the rule:")
print(pd.DataFrame({"close": c, "z10": z10.round(2), "off_hi_pct": (100 * off).round(2),
                    "fires": cell}).tail(8).to_string())

variants = {
    "z10>=1.5 & within 0.5%": (z10 >= 1.5) & (off >= -0.005),
    "z10>=2.5 & within 0.5%": (z10 >= 2.5) & (off >= -0.005),
    "z10>=2.0 & AT high (0%)": (z10 >= 2.0) & (off >= -1e-9),
    "z10>=2.0 & within 1.0%": (z10 >= 2.0) & (off >= -0.010),
    "GATE-OFF z10>=2.0 only": (z10 >= 2.0),
    "GATE-OFF within 0.5% only": (off >= -0.005),
    "z10>=2.0 & NOT near high (<-0.5%)": (z10 >= 2.0) & (off < -0.005),
}

for h in (1, 3, 5, 10):
    battery(px, cell, [(DX, 1.0)], h, f"D1 long DX-Y.NYB z10>=2 & within 0.5% of 252 high", 1.5,
            variants=variants if h in (5, 10) else None, event_kinds=("nfp",))

# pre-committed midterm split + rising/falling yield, episodes at h=5 and h=10
tnx = close_panel(["^TNX"])["^TNX"].dropna()
tnx63 = (tnx - tnx.shift(63)).reindex(px.index)
for h in (5, 10):
    ret = vehicle_ret(px, [(DX, 1.0)], h, 1)
    valid = ret.dropna().index
    sig = px.index[cell.fillna(False).values].intersection(valid)
    epi = declusters(sig, h, valid)
    ep = ret.loc[epi]
    mid = np.asarray(epi.year % 4 == 2)
    ris = np.asarray(tnx63.reindex(epi).values > 0)
    base = ret.loc[valid]
    bmid = np.asarray(valid.year % 4 == 2)
    rows = [summarize(ep.values[mid], "midterm episodes"),
            summarize(ep.values[~mid], "non-midterm episodes"),
            summarize(base.values[bmid], "CTRL all days midterm yrs"),
            summarize(base.values[~bmid], "CTRL all days non-midterm"),
            summarize(ep.values[ris], "TNX 63d change > 0"),
            summarize(ep.values[~ris], "TNX 63d change <= 0")]
    show(rows, f"midterm / yield-regime split, h={h} (episodes)")
    for lbl, m in (("midterm", mid), ("non-mid", ~mid)):
        v = ep.values[m]
        w = int((v > 0).sum())
        print(f"  {lbl}: record {w}-{len(v)-w}, sign p {sign_test(w, len(v)):.4f}")
    print("  midterm episode dates:", [str(d.date()) for d in epi[mid]])

# cluster position of today: fresh firing or mid-cluster?
fires = cell.fillna(False)
run = 0
for v in fires.values[::-1]:
    if v:
        run += 1
    else:
        break
print(f"\ntoday's firing is day {run} of its current run")
# fresh-start (first day of run, no firing in prior 10 sessions) vs later-in-run, h=5
ret5 = vehicle_ret(px, [(DX, 1.0)], 5, 1)
prior = fires.shift(1).rolling(10).max().fillna(0).astype(bool)
fresh = fires & ~prior
later = fires & prior
for lbl, m in (("fresh (no firing prior 10td)", fresh), ("later in run", later)):
    s = px.index[m.values].intersection(ret5.dropna().index)
    e = declusters(s, 5, ret5.dropna().index)
    show([summarize(ret5.loc[e].values, f"{lbl} ep"), summarize(ret5.loc[s].values, f"{lbl} days")],
         f"cluster position h=5: {lbl}")
