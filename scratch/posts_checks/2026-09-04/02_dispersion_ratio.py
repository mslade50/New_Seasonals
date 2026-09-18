"""Single-stock realized vol vs index realized vol. Tonight the risk page says
component RV 30.1% vs SPY RV 7.2% (its own dispersion composite at the 92nd
pctile). Reproduce a simple public version: median 21d realized vol across
the liquid single-stock universe divided by SPY's 21d realized vol, ranked
against its own history since 2001, and SPY forward returns when the ratio
sits in its top decile (declustered 10 td), vs base and vs the bottom decile.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import declusters, fwd_lag, sign_test, summarize  # noqa: E402
from strategy_config import LIQUID_PLUS_COMMODITIES  # noqa: E402

warnings.filterwarnings("ignore")
ASOF = pd.Timestamp("2026-09-04")
ETF_LIKE = {"SPY", "QQQ", "IWM", "DIA", "TLT", "IEF", "LQD", "HYG", "GLD", "SLV", "GDX", "USO",
            "UNG", "DBC", "UUP", "EEM", "EFA", "FXI", "EWZ", "EWJ", "VNQ", "UVXY", "SVXY", "SMH",
            "XBI", "KRE", "XOP", "XME", "XRT", "XHB", "ITB", "IBB", "ARKK", "TQQQ", "SQQQ", "SOXL",
            "SOXS", "TNA", "TZA", "SPXL", "SPXS", "UPRO", "SPXU", "LABU", "LABD", "NUGT", "DUST",
            "JNUG", "JDST", "TMF", "TMV", "UCO", "SCO", "BOIL", "KOLD", "TECL", "TECS", "FAS",
            "FAZ", "ERX", "ERY", "GUSH", "DRIP", "YINN", "YANG", "EDZ", "EDC", "OIH", "KWEB",
            "FXE", "FXY", "USDU", "IAU", "VXX", "VIXY", "BITO", "IBIT", "GBTC", "ETHE", "MSTR"}
stocks = [t for t in LIQUID_PLUS_COMMODITIES
          if t not in ETF_LIKE and "^" not in t and "=" not in t and "-" not in t and "X" != t[-1:] * 0]
stocks = [t for t in stocks if len(t) <= 5 and t.isalpha()]
px = pd.read_parquet(Path("data/master_prices.parquet"), columns=["ticker", "date", "Close"])
px = px[px["ticker"].isin(set(stocks) | {"SPY"})]
close = px.pivot(index="date", columns="ticker", values="Close").sort_index()
close.index = pd.to_datetime(close.index)
have = [t for t in stocks if t in close.columns]
print(f"single-stock universe: {len(have)} names (from {len(stocks)} candidates)")
spy = close["SPY"].dropna()
ret = np.log(close[have]).diff()
rv_stock = ret.rolling(21).std() * np.sqrt(252) * 100
med_rv = rv_stock.median(axis=1)
n_valid = rv_stock.notna().sum(axis=1)
med_rv = med_rv[n_valid >= 60]
spy_rv = (np.log(spy).diff().rolling(21).std() * np.sqrt(252) * 100).reindex(med_rv.index)
ratio = (med_rv / spy_rv).dropna()
ratio = ratio[ratio.index >= "2001-01-01"]
ratio = ratio[ratio.index <= ASOF]
print(f"tonight: median stock 21d RV {med_rv.iloc[-1]:.1f}%  SPY 21d RV {spy_rv.iloc[-1]:.1f}%  "
      f"ratio {ratio.iloc[-1]:.2f}  pctile since 2001 {100*(ratio < ratio.iloc[-1]).mean():.1f}  "
      f"(n days {len(ratio)})")
top = ratio.sort_values(ascending=False).head(15)
print("top 15 ratio days:", [(d.date().isoformat(), round(x, 2)) for d, x in top.items()])
print(f"SPY RV pctile since 2001: {100*(spy_rv.dropna() < spy_rv.iloc[-1]).mean():.1f}   "
      f"median stock RV pctile: {100*(med_rv < med_rv.iloc[-1]).mean():.1f}")
yr = ratio.groupby(ratio.index.year).max()
print("annual max ratio:", [(int(y), round(x, 2)) for y, x in yr.items()])
# days above tonight's ratio by year
above = ratio[ratio >= ratio.iloc[-1]]
print("days at/above tonight by year:", above.groupby(above.index.year).size().to_dict())

q90, q10 = ratio.quantile(0.9), ratio.quantile(0.1)
hi_days = ratio.index[ratio >= q90]
lo_days = ratio.index[ratio <= q10]
hi_dc = declusters(hi_days, 10, spy.index)
lo_dc = declusters(lo_days, 10, spy.index)
print(f"\nratio top decile cut {q90:.2f} ({len(hi_days)} days, {len(hi_dc)} declustered)  "
      f"bottom decile cut {q10:.2f} ({len(lo_days)} days, {len(lo_dc)} dc)")
for h in (5, 10, 21):
    f = fwd_lag(spy, h, 1)
    base = f.reindex(ratio.index).dropna()
    for lab, dd in (("top decile", hi_dc), ("bottom decile", lo_dc)):
        v = f.reindex(dd).dropna()
        st = summarize(v.values)
        nup = int((v > 0).sum())
        print(f"  h={h:<3} {lab:<14} n={st['n']:<4} mean={st['mean_pct']:+.3f}%  med={st['median_pct']:+.3f}%  "
              f"{nup}-{len(v)-nup} ({st['hit']:.1f}%)  t={st['t']:+.2f}  sp={sign_test(nup, len(v)):.4f}  "
              f"| base {100*base.mean():+.3f}% hit {100*(base>0).mean():.1f}%  worst {st['worst_pct']:+.2f}%")
# realized SPY vol after: does index vol expand?
fut_rv = spy_rv.shift(-21)
for lab, dd in (("top decile", hi_dc), ("bottom decile", lo_dc)):
    now = spy_rv.reindex(dd)
    nxt = fut_rv.reindex(dd)
    ok = now.notna() & nxt.notna()
    print(f"  SPY RV 21d later vs now, {lab}: n={int(ok.sum())} higher {int((nxt[ok] > now[ok]).sum())}  "
          f"median change {100*((nxt[ok]/now[ok]).median()-1):+.1f}%")
base_ok = spy_rv.notna() & fut_rv.notna()
print(f"  base: higher {100*(fut_rv[base_ok] > spy_rv[base_ok]).mean():.1f}% of days")
