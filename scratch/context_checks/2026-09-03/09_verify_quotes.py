"""Pin every number that goes in tonight's brief, including the sample starts
the nuggets implicitly claim. Written last so nothing gets quoted from memory.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = close_panel(["^GSPC", "GC=F", "DX-Y.NYB", "^TNX", "BTC-USD", "JPY=X", "IEF"])
for t in ["^GSPC", "GC=F", "DX-Y.NYB", "^TNX", "BTC-USD", "JPY=X", "IEF"]:
    s = px[t].dropna()
    print(f"{t:10s} first {s.index[0].date()}  last {s.index[-1].date()}  n={len(s)}")

raw = load_prices(["JPY=X"])["JPY=X"]
pct = raw["Close"].pct_change().dropna()
rank = int((pct < pct.iloc[-1]).sum()) + 1
print(f"\nUSDJPY today {100*pct.iloc[-1]:+.2f}%; rank {rank} smallest of {len(pct)} "
      f"sessions ({raw.index[0].date()} to {raw.index[-1].date()}); "
      f"percentile {100*rank/len(pct):.2f}")

# 4-leg configuration, per calendar year.
ref = px["^GSPC"].dropna().index
r = {t: px[t].reindex(ref).pct_change(fill_method=None) for t in px.columns}
mask = ((r["^GSPC"] >= 0.01) & (r["GC=F"] >= 0.02) &
        (r["DX-Y.NYB"] < 0) & (r["^TNX"] < 0)).fillna(False)
fires = ref[mask.to_numpy()]
first_valid = max(px[t].dropna().index[0] for t in ("^GSPC", "GC=F", "DX-Y.NYB", "^TNX"))
print(f"\n4-leg cell: sample effectively starts {first_valid.date()}")
print("by year:", pd.Series(fires.year).value_counts().sort_index().to_dict())
print(f"total {len(fires)}; 2026 so far {(fires.year == 2026).sum()}")

# BTC episode sample window.
btc = px["BTC-USD"].dropna()
rank21 = pct_rank(btc, 21, 252)
f = btc.index[(rank21 >= 95).reindex(btc.index).fillna(False).to_numpy()]
dec = declusters(f, 10, btc.index)
print(f"\nBTC episodes: {len(dec)}, from {dec[0].date()} to {dec[-1].date()}")
print(f"BTC 21d return today: {100*(btc.iloc[-1]/btc.iloc[-22]-1):+.2f}%")

# IEF distance-to-low percentile, restated.
ief = px["IEF"].dropna()
dist = 100 * (ief / ief.rolling(252).min() - 1.0)
print(f"\nIEF {dist.iloc[-1]:.2f}% above its 252d low; that is the "
      f"{100*(dist.dropna() <= dist.iloc[-1]).mean():.1f}th percentile of "
      f"{dist.dropna().size} readings since {dist.dropna().index[0].date()}")

# Distinct years behind the 19 pinned-bond payroll eves.
nfp = pd.DatetimeIndex(sorted(set(load_events(["nfp"])["date"]) & set(ref)))
pos = {d: i for i, d in enumerate(ref)}
anc = pd.DatetimeIndex([ref[pos[d] - 1] for d in nfp if pos.get(d, 0) > 0])
tight = pd.DatetimeIndex([d for d in anc if d in dist.index
                          and np.isfinite(dist[d]) and dist[d] <= 1.0])
print(f"pinned-bond payroll eves: {len(tight)} across "
      f"{len(set(tight.year))} distinct years {sorted(set(tight.year))}")
