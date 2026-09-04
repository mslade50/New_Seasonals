"""Compare the saved proxy backtest to the cash-index sleeve. Fresh process so the
original IDEA_UNIVERSE is intact for the index re-sim."""
import sys
import numpy as np
import pandas as pd

ROOT = r"C:\Users\McKinley Slade\dev\New_Seasonals"
sys.path.insert(0, ROOT)
from scripts.resim_seasonal_entry import resim
from scripts.seasonal_sharpe import ratios

PROXY = {"^GSPC": "SPY", "^NDX": "QQQ", "^DJI": "DIA", "^RUT": "IWM",
         "^N225": "EWJ", "^BVSP": "EWZ", "^MXX": "EWW", "^KS11": "EWY",
         "^TWII": "EWT", "^BSESN": "INDA", "^DJT": "IYT",
         "^GDAXI": "EWG", "^FCHI": "EWQ", "^FTSE": "EWU", "^STOXX50E": "FEZ",
         "^SOX": "SOXX", "^MID": "IJH", "^IXIC": "ONEQ", "^VIX": "VXX",
         "^AXJO": "EWA", "^GSPTSE": "EWC", "^HSI": "EWH"}

prx = pd.read_parquet(ROOT + r"\data\seasonal_proxy_backtest.parquet")
prx["exit_date"] = pd.to_datetime(prx["exit_date"])
idx = resim("t1_open", 0, do_dedup=True)          # cash-index sleeve, market-on-open
idx["exit_date"] = pd.to_datetime(idx["exit_date"])


def stats(b):
    if b is None or len(b) == 0:
        return dict(N=0, Win=np.nan, AvgR=np.nan, PF=np.nan, TotR=0, Sharpe=np.nan)
    R = b["R"].astype(float)
    pf = R[R > 0].sum() / abs(R[R < 0].sum()) if (R < 0).any() else np.inf
    full = pd.date_range(b["exit_date"].min().normalize(), b["exit_date"].max().normalize(), freq="B")
    m = b.groupby(b["exit_date"].dt.normalize())["R"].sum().reindex(full, fill_value=0).resample("ME").sum()
    sh, _ = ratios(m, 12)
    return dict(N=len(b), Win=round(100 * (R > 0).mean(), 1), AvgR=round(R.mean(), 3),
                PF=round(pf, 2), TotR=round(R.sum(), 0), Sharpe=round(sh, 2))


print(f"{'pair':16s} | {'------ cash index ------':>30s} | {'---- tradeable proxy ----':>30s}")
print(f"{'':16s} | {'N':>5}{'Win%':>6}{'AvgR':>7}{'PF':>6}{'TotR':>6} | {'N':>5}{'Win%':>6}{'AvgR':>7}{'PF':>6}{'TotR':>6}")
for ix, pr in PROXY.items():
    i = stats(idx[idx.ticker == ix]); p = stats(prx[prx.ticker == pr])
    print(f"{ix+' -> '+pr:16s} | {i['N']:5d}{i['Win']:6.1f}{i['AvgR']:7.3f}{i['PF']:6.2f}{i['TotR']:6.0f} | "
          f"{p['N']:5d}{p['Win']:6.1f}{p['AvgR']:7.3f}{p['PF']:6.2f}{p['TotR']:6.0f}")
ai = stats(idx[idx.ticker.isin(PROXY)]); ap = stats(prx)
print(f"\nAGG cash indices (11): N{ai['N']} AvgR{ai['AvgR']} PF{ai['PF']} TotR{ai['TotR']} Sharpe{ai['Sharpe']}")
print(f"AGG tradeable proxies: N{ap['N']} AvgR{ap['AvgR']} PF{ap['PF']} TotR{ap['TotR']} Sharpe{ap['Sharpe']}")
