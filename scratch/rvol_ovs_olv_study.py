"""Does per-ticker realized-vol state predict OVS / OLV trade outcomes?

Two conditioning variables, both point-in-time as of signal date:
  1. rvol EXPANSION: 21d ann. vol / 63d ann. vol (>1 = vol expanding)
  2. rvol LEVEL: 21d vol's percentile within the ticker's own trailing 252d
Buckets per strategy with avgR / N / win%; monthly-clustered top-vs-rest test.
Uses the pre-frag-bands ledger snapshot (full unthrottled sample).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
led = pd.read_parquet(ROOT / "scratch" / "ledger_pre_frag_bands.parquet")
led["Signal Date"] = pd.to_datetime(led["Signal Date"])
sub = led[led.Strategy.isin(["Overbot Vol Spike", "Oversold Low Volume"])].copy()
need = set(sub.Ticker.str.upper().unique())

mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                     columns=["ticker", "date", "Close"])
mps = mp[mp.ticker.str.upper().isin(need)]
px = mps.assign(_d=pd.to_datetime(mps["date"]).values).pivot_table(
    index="_d", columns="ticker", values="Close").sort_index()
px.columns = [str(c).upper() for c in px.columns]
print(f"price coverage: {len([t for t in need if t in px.columns])}/{len(need)} tickers")

ret = px.pct_change()
rv21 = ret.rolling(21).std() * np.sqrt(252)
rv63 = ret.rolling(63).std() * np.sqrt(252)
expansion = rv21 / rv63
# percentile of today's rv21 within the ticker's trailing 252 sessions
lvl_pct = rv21.rolling(252).rank(pct=True)

def lookup(mat, t, d):
    if t not in mat.columns:
        return np.nan
    s = mat[t]
    i = s.index.searchsorted(d, side="right") - 1
    if i < 0 or (d - s.index[i]).days > 7:
        return np.nan
    return s.iloc[i]

sub["exp_ratio"] = [lookup(expansion, t.upper(), d)
                    for t, d in zip(sub.Ticker, sub["Signal Date"])]
sub["lvl_pct"] = [lookup(lvl_pct, t.upper(), d)
                  for t, d in zip(sub.Ticker, sub["Signal Date"])]
sub["ym"] = sub["Signal Date"].dt.to_period("M")

EXP_BANDS = [0, 0.8, 1.0, 1.25, 1.5, 99]
EXP_LABELS = ["<0.8 compressing", "0.8-1.0", "1.0-1.25", "1.25-1.5", ">1.5 exploding"]
LVL_BANDS = [0, 0.25, 0.5, 0.75, 0.9, 1.001]
LVL_LABELS = ["<25 calm", "25-50", "50-75", "75-90", ">90 hot"]

def table(d, col, bands, labels):
    d = d.dropna(subset=[col, "R_Multiple"]).copy()
    d["band"] = pd.cut(d[col], bands, labels=labels, include_lowest=True)
    g = d.groupby("band", observed=False)["R_Multiple"]
    return pd.DataFrame({"N": g.size(), "avgR": g.mean().round(3),
                         "medR": g.median().round(3),
                         "win%": g.apply(lambda s: (s > 0).mean() * 100).round(1)})

def cluster_test(d, mask_hi, label):
    d = d.dropna(subset=["R_Multiple"])
    hi, lo = d[mask_hi], d[~mask_hi]
    if len(hi) < 30:
        print(f"  {label}: N too small ({len(hi)})")
        return
    hm = hi.groupby("ym")["R_Multiple"].mean()
    lm = lo.groupby("ym")["R_Multiple"].mean()
    t, p = stats.ttest_ind(hm, lm, equal_var=False)
    print(f"  {label}: hi {hi.R_Multiple.mean():+.3f} (N={len(hi)}) vs rest "
          f"{lo.R_Multiple.mean():+.3f} (N={len(lo)})  monthly-t={t:+.2f} p={p:.3f}")

for strat in ["Overbot Vol Spike", "Oversold Low Volume"]:
    d = sub[sub.Strategy == strat]
    print("=" * 70)
    print(f"{strat} (N={len(d)}, coverage exp={d.exp_ratio.notna().mean()*100:.0f}%)")
    print("\n-- by rvol EXPANSION (rv21/rv63 at signal) --")
    print(table(d, "exp_ratio", EXP_BANDS, EXP_LABELS).to_string())
    print("\n-- by rvol LEVEL (rv21 pctile in own trailing 252d) --")
    print(table(d, "lvl_pct", LVL_BANDS, LVL_LABELS).to_string())
    print("\nclustered tests:")
    dd = d.dropna(subset=["exp_ratio"])
    cluster_test(dd, dd.exp_ratio > 1.25, "expansion > 1.25")
    cluster_test(dd, dd.exp_ratio < 0.8, "compression < 0.8")
    dl = d.dropna(subset=["lvl_pct"])
    cluster_test(dl, dl.lvl_pct > 0.9, "level > 90th pctile")
    cluster_test(dl, dl.lvl_pct < 0.25, "level < 25th pctile")
    # interaction: both strategies trade vol events; check expansion x year stability
    dd15 = dd[dd.exp_ratio > 1.25]
    if len(dd15) > 40:
        byyr = dd15.groupby(dd15["Signal Date"].dt.year)["R_Multiple"].agg(["size", "mean"])
        print("  expansion>1.25 by year:",
              {int(y): (int(r['size']), round(r['mean'], 2)) for y, r in byyr.iterrows() if r['size'] >= 5})
