"""C5 round 1: long KWEB on a dollar washout (DXY 21d pct_rank <= 2).

Reproduce the recon mask independently, then run the full battery on KWEB at
the two horizons the recon quoted, plus threshold neighbours and a by-year
table. Nothing here is taken from 02_recon2.py on trust.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change

TK = ["DX-Y.NYB", "KWEB", "EEM", "FXI", "SPY", "UUP"]
px = close_panel(TK)
d = px.index
dx = px["DX-Y.NYB"].dropna()

print("panel", d[0].date(), "->", d[-1].date())
for t in TK:
    s = px[t].dropna()
    print(f"  {t:<10} first={s.index[0].date()} last={s.index[-1].date()} n={len(s)}")

r21 = _valid_pct_change(px["DX-Y.NYB"], 21)
rk21 = pct_rank(px["DX-Y.NYB"], 21)
print(f"\nTODAY  DXY 21d ret = {100*r21.iloc[-1]:+.3f}%   rank252 = {rk21.iloc[-1]:.2f}")
print(f"       full-history percentile of that magnitude = "
      f"{100*(r21.dropna() < r21.iloc[-1]).mean():.2f}")

mask = (rk21 <= 2).reindex(d).fillna(False)
sig = d[mask.values]
print(f"\nmask rank<=2: {int(mask.sum())} days, first {sig[0].date()} last {sig[-1].date()}")
epi_all = declusters(sig, 21, d)
print(f"declustered gap21: {len(epi_all)} episodes, years {sorted(set(epi_all.year))}")

variants = {
    "rank<=1": (pct_rank(px["DX-Y.NYB"], 21) <= 1),
    "rank<=2 (pitched)": (pct_rank(px["DX-Y.NYB"], 21) <= 2),
    "rank<=5": (pct_rank(px["DX-Y.NYB"], 21) <= 5),
    "rank<=10": (pct_rank(px["DX-Y.NYB"], 21) <= 10),
    "mag<=-2.32% (today)": (r21 <= -0.02323),
    "mag<=-4%": (r21 <= -0.04),
}
variants = {k: v.reindex(d).fillna(False) for k, v in variants.items()}

for h in (5, 10):
    battery(px, mask, [("KWEB", 1.0)], h,
            f"C5 KWEB long | DXY 21d rank<=2", cost_bps=6.0,
            variants=variants, min_gap=21)

# ---------------------------------------------------------------- by-year
print("\n\n=== C5 by-year, KWEB episodes (gap21) ===")
for h in (5, 10):
    ret = vehicle_ret(px, [("KWEB", 1.0)], h)
    e = pd.DatetimeIndex([x for x in epi_all if not np.isnan(ret.get(x, np.nan))])
    v = ret.loc[e]
    by = v.groupby(v.index.year).agg(["count", "mean", "sum"])
    by[["mean", "sum"]] = (by[["mean", "sum"]] * 100).round(3)
    print(f"\n h={h}  N={len(v)}  total {100*v.sum():+.2f}pp")
    print(by.to_string())
    print(" episodes:", ", ".join(f"{x.date()}:{100*ret[x]:+.1f}" for x in e))

# --------------------------------------------- is the trigger risk-on tape?
print("\n\n=== C5 tape character on trigger days ===")
spy = px["SPY"].dropna()
sma200 = rolling_on_valid(px["SPY"], lambda x: x.rolling(200).mean())
above = (px["SPY"] > sma200)
base = above.dropna()
trig = above.reindex(epi_all).dropna()
print(f" SPY above 200d: base rate {100*base.mean():.1f}% (N={len(base)}) | "
      f"on trigger episodes {100*trig.mean():.1f}% (N={len(trig)})")
sp63 = pct_rank(px["SPY"], 63)
print(f" SPY 63d rank: base median {sp63.dropna().median():.1f} | "
      f"trigger median {sp63.reindex(epi_all).dropna().median():.1f}")
