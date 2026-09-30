"""The VIX closed higher on a +1.55% SPY day.

How rare is 'SPY +1.25% or more with the VIX up', and does the weekday explain it
(Sunday's brief: Mondays lift the VIX in every month)? Then what followed, against
ordinary 1.25%+ up days where the VIX fell.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import load_prices, fwd_ret, summarize, era_split, sign_test, cluster_note, show, declusters  # noqa

px = load_prices(["SPY", "^VIX", "QQQ", "IWM", "^VIX3M"])
spy = px["SPY"]["Close"].astype(float)
vix = px["^VIX"]["Close"].astype(float)
df = pd.DataFrame({"spy": spy, "vix": vix}).dropna()
df = df[df.index >= "1999-01-01"]
df["rs"] = df["spy"].pct_change()
df["rv"] = df["vix"].pct_change()
df["dow"] = df.index.dayofweek
today = df.index[-1]
print(f"today {today.date()}: SPY {100*df.rs.iloc[-1]:+.2f}%  VIX {100*df.rv.iloc[-1]:+.2f}%  VIX level {df.vix.iloc[-1]:.2f}")

big = df["rs"] >= 0.0125
both = big & (df["rv"] > 0)
hist = df.index[:-1]
bu = df.index[both & df.index.isin(hist)]
bd = df.index[big & (df["rv"] <= 0) & df.index.isin(hist)]
print(f"\nSPY >= +1.25% days: {int(big[hist].sum())}; with VIX up: {len(bu)} ({100*len(bu)/big[hist].sum():.1f}%)")
tab = pd.crosstab(df.loc[big[big].index.intersection(hist), "dow"],
                  df.loc[big[big].index.intersection(hist), "rv"] > 0)
tab.columns = ["vix_dn", "vix_up"]
tab["share_up"] = (tab["vix_up"] / tab.sum(axis=1) * 100).round(1)
print("by weekday (0=Mon):")
print(tab.to_string())

# with VIX level under 20 (today's regime) and SPY +1.5%
calm = df["vix"].shift(1) < 20
bu_calm = bu[calm.reindex(bu).fillna(False).values]
print(f"\nwith prior VIX < 20: VIX-up big days {len(bu_calm)}, all big days {int((big & calm)[hist].sum())}")
print("list of VIX-up big days with prior VIX<20:")
det = pd.DataFrame({"spy": 100 * df.rs.reindex(bu_calm), "vix": 100 * df.rv.reindex(bu_calm),
                    "vix_lvl": df.vix.reindex(bu_calm), "dow": df.dow.reindex(bu_calm)})
for h in (1, 5, 21):
    det[f"spy_h{h}"] = 100 * fwd_ret(df.spy, h).reindex(bu_calm)
for h in (1, 5):
    det[f"vix_h{h}"] = 100 * fwd_ret(df.vix, h).reindex(bu_calm)
print(det.round(2).to_string())

def block(dates, label):
    rows = []
    for h in (1, 5, 21):
        r = fwd_ret(df.spy, h).reindex(dates).dropna()
        s = summarize(r.values, f"{label} SPY h{h}")
        s["up"] = int((r > 0).sum())
        rows.append(s)
    for h in (1, 5):
        r = fwd_ret(df.vix, h).reindex(dates).dropna()
        s = summarize(r.values, f"{label} VIX h{h}")
        s["up"] = int((r > 0).sum())
        rows.append(s)
    return rows

epi_up = declusters(bu, 5, df.index)
epi_dn = declusters(bd, 5, df.index)
show(block(epi_up, "VIXup") + block(epi_dn, "VIXdn"), "all regimes, declustered 5td")
epi_up_c = declusters(bu_calm, 5, df.index)
epi_dn_c = declusters(bd[calm.reindex(bd).fillna(False).values], 5, df.index)
show(block(epi_up_c, "VIXup calm") + block(epi_dn_c, "VIXdn calm"), "prior VIX<20, declustered 5td")
r5 = fwd_ret(df.spy, 5).reindex(epi_up_c).dropna()
show(era_split(r5.index, r5.values), "era split VIXup calm SPY h5")
print("concentration:", cluster_note(r5.index, r5.values))
a = fwd_ret(df.spy, 5).dropna()
print(f"control SPY all days h5 {100*a.mean():.3f}% hit {100*(a>0).mean():.1f}%")
