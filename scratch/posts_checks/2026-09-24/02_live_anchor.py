"""Follow-up to 01: where does TONIGHT sit inside each declustered run?

01 showed the current ^TNX run qualified on 2026-09-23 as well, so the
declustered cell-A anchor is 09-23 and tonight is day 2. A Friday MOO entry is
therefore Open[anchor+2]. This prints the recent raw trigger dates for A and B,
and the cell-A stats with the entry pushed one more session (Open[D+2] ->
Close[D+2+h]) on the first-day anchors, which is the exact analogue of buying
Friday's open after a run that started Wednesday.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pitch_lab import (  # noqa: E402
    cluster_note, declusters, load_prices, local_control, sign_test, summarize,
)

ASOF = pd.Timestamp("2026-09-24")
ERA = pd.Timestamp("2018-01-01")
raw = load_prices(["SPY", "IEF", "TLT", "XLU", "^TNX"])
nyse = raw["SPY"].index
nyse = pd.DatetimeIndex(nyse[nyse <= ASOF])
F = {t: raw[t][raw[t].index <= ASOF].astype(float).reindex(nyse) for t in ("IEF", "TLT", "XLU")}

tnx = raw["^TNX"]["Close"].astype(float)
tnx = tnx[tnx.index <= ASOF].dropna()
tidx = tnx.index
bp2 = tnx.diff(2) * 100
tmax = tnx.rolling(252, min_periods=200).max()
hi = tnx >= tmax - 1e-9
trig_mask = (hi & (bp2 >= 15)).fillna(False)
print("=== A: last 8 ^TNX sessions ===")
for d in tidx[-8:]:
    print(f"  {d.date()} TNX {tnx[d]:.3f} bp2 {bp2[d]:+.1f} 52w-high {bool(hi[d])} "
          f"qualifies {bool(trig_mask[d])}")
trig = tidx[trig_mask.values]
epi_all = declusters(trig, 5, tidx)
print(f"  declustered anchors incl. tonight, last 3: {[str(d.date()) for d in epi_all[-3:]]}")

epi = epi_all[(epi_all >= "2003-01-01") & (epi_all < ASOF)]
epi = pd.DatetimeIndex([d for d in epi if d in set(nyse)])


def o2c_lag(t: str, h: int, lag: int) -> pd.Series:
    return F[t]["Close"].shift(-(lag + h)) / F[t]["Open"].shift(-lag) - 1.0


def block(label: str, s: pd.Series, trig_: pd.DatetimeIndex) -> None:
    v = s.reindex(trig_).dropna()
    allv = s[s.index >= "2003-01-01"].dropna()
    loc = s.reindex(local_control(nyse, trig_, 126)).dropna()
    u, dn, n = int((v > 0).sum()), int((v < 0).sum()), len(v)
    sm = summarize(v.values)
    a, b = v[v.index < ERA], v[v.index >= ERA]
    print(f"  {label}: n={n} {u}-{dn} mean {sm['mean_pct']:+.3f}% med "
          f"{sm['median_pct']:+.3f}% t {sm['t']:+.2f} | ctrl all {100*allv.mean():+.3f}% "
          f"local {100*loc.mean():+.3f}% | sign p {sign_test(u, n):.4f} vs base "
          f"{sign_test(u, n, float((allv > 0).mean())):.4f}")
    print(f"      era: pre-2018 n={len(a)} {int((a>0).sum())}-{int((a<0).sum())} "
          f"{100*a.mean():+.2f}% | 2018+ n={len(b)} {int((b>0).sum())}-{int((b<0).sum())} "
          f"{100*b.mean():+.2f}%")
    print(f"      conc: {cluster_note(v.index, v.values)}")
    print("      dates: " + ", ".join(f"{i.date()} {100*x:+.2f}" for i, x in v.items()))


print("\n=== A with entry at Open[anchor+2] (Friday-after-Wednesday-anchor analogue) ===")
for tk in ("IEF", "TLT"):
    for h in (3, 5, 10):
        block(f"{tk} h{h} Open[D+2]->Close[D+2+{h}]", o2c_lag(tk, h, 2), epi)

from pitch_lab import wilder_atr  # noqa: E402

print("\n=== A lag-2 MAE in ATR units (ATR at the Thursday-analogue close D+1; "
      "lows Open[D+2] session .. exit) ===")
for tk in ("IEF", "TLT"):
    f = raw[tk][raw[tk].index <= ASOF].astype(float)
    atr = pd.Series(np.asarray(wilder_atr(f["High"].to_numpy(), f["Low"].to_numpy(),
                                          f["Close"].to_numpy()), float),
                    index=f.index).reindex(nyse)
    for h in (3, 5):
        lows = pd.concat([F[tk]["Low"].shift(-k) for k in range(2, h + 3)], axis=1)
        mae = (lows.min(axis=1, skipna=False) - F[tk]["Open"].shift(-2)) / atr.shift(-1)
        m = mae.reindex(epi).dropna()
        print(f"  {tk} h{h}: median {m.median():+.2f} ATR worst {m.min():+.2f} "
              f"({m.idxmin().date()}) share <= -1 ATR {100*(m <= -1).mean():.0f}% n={len(m)}")

print("\n=== B: last 10 sessions of the XLU state ===")
xc = raw["XLU"]["Close"].astype(float)
xc = xc[xc.index <= ASOF].dropna()
z_ps = xc.pct_change(10) / (xc.pct_change().rolling(21).std() * np.sqrt(10))
z_lit = (xc - xc.rolling(10).mean()) / xc.rolling(10).std()
xlo = xc <= xc.rolling(252).min() + 1e-9
near = (tnx >= 0.995 * tmax).reindex(xc.index)
athi = hi.reindex(xc.index)
for d in xc.index[-10:]:
    print(f"  {d.date()} XLU {xc[d]:.2f} z_ps {z_ps[d]:+.2f} z_lit {z_lit[d]:+.2f} "
          f"52wlow {bool(xlo[d])} TNX near {bool(near[d])} at-high {bool(athi[d])}")
for nm, m in (("z_ps<=-2 & TNX near", (z_ps <= -2) & near.fillna(False)),
              ("z_ps<=-2 uncond", z_ps <= -2),
              ("z_ps<=-2 & XLU 52w low", (z_ps <= -2) & xlo),
              ("z_lit<=-2 & TNX near", (z_lit <= -2) & near.fillna(False))):
    d = xc.index[m.fillna(False).values]
    e = declusters(d, 10, nyse)
    print(f"  {nm}: raw since 2026-08-01 {[str(x.date()) for x in d[d >= '2026-08-01']]} "
          f"| declustered anchors last 2 {[str(x.date()) for x in e[-2:]]}")
