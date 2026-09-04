"""Sunday idea candidates for Monday 2026-08-31 (month end) / Tuesday 09-01.

Two candidates, both from tonight's tape:

A. IWM into and through month end after a weak week. IWM closed -1.35% on
   Friday, z10 -1.07, 5d rank 20. The pitch-convention shapes available:
     A1  signal Fri (ME-2), entry MOO Mon (ME-1), exit MOC Tue (first Sep
         session): the month-end + turn-of-month bounce from a weak entry.
     A2  signal Fri, entry MOC Mon, exit MOC Tue: the turn only (the brief
         says SPY's September turn is 13-13, -0.26%; is IWM the same?).
   Unconditioned first, then conditioned on the weak entry (5d rank < 25,
   z10 < -1), then the September cut, then era + midterm.

B. GDX second flush inside a parabolic run. Thursday's pitch bought the FIRST
   flush (21d > +25%, day -2.9%, 6-0 cell) and Friday printed -3.9% on top.
   Is a second consecutive flush still a buy, or is it the top?
     B1  21d return >= +20% at the anchor AND anchor day <= -3%, declustered
         at 10 sessions, forward 5 sessions lag-1 (entry MOO Monday).
     B2  same but requiring TWO consecutive down days summing <= -5%.

Conventions: lag-1 forward returns (entry the session after the signal),
close-to-close, declustered where stated, all-days + local controls.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    cluster_note, declusters, era_split, fwd_lag, load_prices, local_control,
    pct_rank, sign_test, summarize, wilder_atr, zscore,
)

px = load_prices(["IWM", "SPY", "GDX", "GLD"])
closes = {k: v["Close"].dropna() for k, v in px.items()}
for k in ("IWM", "GDX"):
    f = px[k]
    atr = pd.Series(wilder_atr(f["High"], f["Low"], f["Close"]), index=f.index)
    print(f"tonight {k}: close {closes[k].iloc[-1]:.2f} | Wilder-14 ATR {atr.iloc[-1]:.4f} "
          f"({atr.iloc[-1] / closes[k].iloc[-1] * 100:.2f}%) | bar {f.index[-1].date()}")


def block(name, r, s, h, lag=1):
    r = r.dropna()
    if len(r) == 0:
        print(f"  {name:<34} n=0")
        return
    st = summarize(r.values)
    nup = int((r > 0).sum())
    allr = fwd_lag(s, h, lag).dropna()
    loc = allr.reindex(local_control(s.index, r.index, 126)).dropna()
    print(f"  {name:<34} n={st['n']:<4} mean={st['mean_pct']:+.3f}%  med={st['median_pct']:+.3f}%  "
          f"{nup}-{len(r)-nup} ({st['hit']:.1f}%)  t={st['t']:+.2f}  sp={sign_test(nup, len(r)):.4f}  "
          f"| all {100*allr.mean():+.3f}%  local {100*loc.mean():+.3f}%  "
          f"| worst {st['worst_pct']:+.2f}% ({r.idxmin().date()})")


def splits(r):
    r = r.dropna()
    v = r.values
    print("    era:", [(e["label"], e["n"], round(e.get("mean_pct", np.nan), 3),
                        round(e.get("hit", np.nan), 1)) for e in era_split(r.index, v)])
    print("    concentration:", cluster_note(r.index, v))
    mid = r[[d.year % 4 == 2 for d in r.index]]
    nu = int((mid > 0).sum())
    if len(mid):
        print(f"    midterm n={len(mid)} mean={100*mid.mean():+.3f}% {nu}-{len(mid)-nu} sp={sign_test(nu, len(mid)):.4f}")


def dist_to_month_end(idx):
    ser = pd.Series(np.arange(len(idx)), index=idx)
    ym = idx.to_period("M")
    d = ser.groupby(ym).transform("max") - ser
    d[ym == ym[-1]] = np.nan
    return d


# ---------------------------------------------------------------- A: IWM
print("\n" + "=" * 100)
print("A. IWM month end after a weak week")
print("=" * 100)
s = closes["IWM"]
f = px["IWM"]
dist = dist_to_month_end(s.index)
r5 = pct_rank(s, 5)
z = zscore(s, 10)
sig = s.index[(dist == 1).values]           # ME-2 sessions (Friday tonight)
print(f"ME-2 signals: {len(sig)}  {sig[0].date()}..{sig[-1].date()}")

# A1: entry MOO Mon, exit MOC Tue  = open(ME-1) -> close(ME+1)
o = f["Open"].reindex(s.index)
pos = pd.Series(np.arange(len(s)), index=s.index)
def moo_to_close(idx, h):
    out = {}
    for d in idx:
        i = pos[d]
        if i + 1 + h - 1 < len(s):
            out[d] = s.iloc[i + h] / o.iloc[i + 1] - 1
    return pd.Series(out)

for h in (1, 2):
    lab = "A1 MOO ME-1 -> MOC ME" if h == 1 else "A1 MOO ME-1 -> MOC ME+1"
    r = moo_to_close(sig, h)
    block(lab, r, s, h)
r = moo_to_close(sig, 2)
weak = r[(r5.reindex(r.index) < 25) & (z.reindex(r.index) < -1)]
block("  A1 h2, weak entry (r5<25 & z<-1)", weak, s, 2)
splits(weak)
weak5 = r[(r5.reindex(r.index) < 25)]
block("  A1 h2, r5<25 only", weak5, s, 2)
sep = r[[d.month == 8 for d in r.index]]
block("  A1 h2, august ME (into sept)", sep, s, 2)

# A2: entry MOC Mon (ME-1), exit MOC Tue (ME+1): lag-1 h1 from ME-1 = signal at ME-1
sig2 = s.index[(dist == 0).values]
r2 = fwd_lag(s, 1, 0).reindex(sig2)   # anchored on ME close, h1 = first session of new month
block("A2 ME close -> first session", r2, s, 1, 0)
splits(r2)
sep2 = r2[[d.month == 8 for d in r2.index]]
block("  A2 into september", sep2, s, 1, 0)
print("    aug->sep by year:", [(d.year, round(100 * x, 2)) for d, x in sep2.dropna().items()])
w2 = r2[(r5.reindex(r2.index) < 25)]
block("  A2 weak (r5<25 at ME close)", w2, s, 1, 0)
# and the ME-1 session itself from Friday's close, conditioned on weak Friday
r_me = fwd_lag(s, 1, 0).reindex(sig)
block("ME session itself (Fri close->Mon close)", r_me, s, 1, 0)
wk = r_me[(r5.reindex(r_me.index) < 25) & (z.reindex(r_me.index) < -1)]
block("  ME session, weak entry", wk, s, 1, 0)
splits(wk)
aug_me = r_me[[d.month == 8 for d in r_me.index]]
block("  ME session, august only", aug_me, s, 1, 0)
print("    aug ME by year:", [(d.year, round(100 * x, 2)) for d, x in aug_me.dropna().items()])

# ---------------------------------------------------------------- B: GDX
print("\n" + "=" * 100)
print("B. GDX second flush inside a parabolic run")
print("=" * 100)
g = closes["GDX"]
ret1 = g.pct_change()
ret21 = g.pct_change(21)
ret2 = g.pct_change(2)
print(f"tonight GDX: 1d {100*ret1.iloc[-1]:+.2f}%  2d {100*ret2.iloc[-1]:+.2f}%  21d {100*ret21.iloc[-1]:+.2f}%")
for thr21, thr1, lab in ((0.20, -0.03, "B1 21d>=20% & day<=-3%"), (0.25, -0.03, "B1 21d>=25% & day<=-3%")):
    m = (ret21 >= thr21) & (ret1 <= thr1)
    idx = declusters(g.index[m.fillna(False).values], 10, g.index)
    for h in (1, 3, 5, 10):
        r = fwd_lag(g, h, 1).reindex(idx)
        block(f"{lab} h{h}", r, g, h)
    r5_ = fwd_lag(g, 5, 1).reindex(idx)
    splits(r5_)
    print("    dates:", [(d.date().isoformat(), round(100 * x, 1)) for d, x in r5_.dropna().items()])
m = (ret21 >= 0.20) & (ret2 <= -0.05) & (ret1 < 0) & (ret1.shift(1) < 0)
idx = declusters(g.index[m.fillna(False).values], 10, g.index)
for h in (1, 3, 5, 10):
    r = fwd_lag(g, h, 1).reindex(idx)
    block(f"B2 21d>=20% & 2 down days <=-5% h{h}", r, g, h)
r5_ = fwd_lag(g, 5, 1).reindex(idx)
print("    dates:", [(d.date().isoformat(), round(100 * x, 1)) for d, x in r5_.dropna().items()])
# gold itself same state
gl = closes["GLD"]
m = (gl.pct_change(21) >= 0.08) & (gl.pct_change() <= -0.03)
idx = declusters(gl.index[m.fillna(False).values], 10, gl.index)
for h in (1, 5):
    block(f"GLD 21d>=8% & day<=-3% h{h}", fwd_lag(gl, h, 1).reindex(idx), gl, h)
