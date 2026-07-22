"""Proof-of-concept for the dial-conditioned downside table:
all historical days where the dial-MA (10d MA of 63d) closed within +-3 points
of its CURRENT value -> forward ATR-decline probabilities (low-touch), 1/2/3/5 ATR.
Dial history = data/rd2_fragility.parquet (2016+). SPY OHLC from master_prices.
"""
import os, sys
import numpy as np, pandas as pd

ROOT = r"C:\Users\mckin\UsersmckinNew_Seasonals"
HORIZONS = {"5d":5,"10d":10,"21d":21,"42d":42,"63d":63}; MULTS=[1,2,3,5]; ATR_N=14; BAND=3.0

frag = pd.read_parquet(os.path.join(ROOT,"data","rd2_fragility.parquet"))
frag.index = pd.to_datetime(frag.index)
dial_ma = frag["63d"].rolling(10, min_periods=1).mean()
current = float(dial_ma.iloc[-1])
lo, hi = current-BAND, current+BAND
print(f"current dial-MA (10d of 63d) = {current:.1f}  ->  band [{lo:.1f}, {hi:.1f}]")

mp = pd.read_parquet(os.path.join(ROOT,"data","master_prices.parquet"))
spy = mp[mp["ticker"]=="SPY"].copy(); spy["date"]=pd.to_datetime(spy["date"])
spy = spy.sort_values("date").set_index("date")[["High","Low","Close"]].dropna()
H=spy["High"].to_numpy(float); L=spy["Low"].to_numpy(float); C=spy["Close"].to_numpy(float); n=len(spy)
pc=np.concatenate([[np.nan],C[:-1]]); tr=np.maximum.reduce([H-L,np.abs(H-pc),np.abs(L-pc)])
atr=np.full(n,np.nan); atr[ATR_N]=np.nanmean(tr[1:ATR_N+1])
for i in range(ATR_N+1,n): atr[i]=(atr[i-1]*(ATR_N-1)+tr[i])/ATR_N
atr_valid=np.isfinite(atr)&(atr>0)

def worst_low(N):
    worst=np.zeros(n); valid=np.ones(n,bool)
    for j in range(1,N+1):
        Ls=np.concatenate([L[j:],np.full(j,np.nan)]); worst=np.fmax(worst, C-Ls)
    valid[n-N:]=False; return worst, valid

# dial-band mask on the SPY index
band_dates = dial_ma[(dial_ma>=lo)&(dial_ma<=hi)].index
pos = spy.index.get_indexer(band_dates)
band_mask = np.zeros(n, bool); band_mask[pos[pos>=0]] = True
overlap0 = spy.index.intersection(dial_ma.index)
print(f"dial history overlaps SPY: {overlap0.min().date()} -> {overlap0.max().date()}  "
      f"(SPY ends {spy.index[-1].date()})")
print(f"days in band (with SPY bar): {band_mask.sum()}\n")

rows={}
for lab,N in HORIZONS.items():
    drop,wv=worst_low(N); v=wv&atr_valid&band_mask
    dd=drop[v]/atr[v]
    rows[lab]={k:round(100*np.mean(dd>=k),1) for k in MULTS}
    rows[lab]["n"]=int(v.sum())
df=pd.DataFrame(rows).T[MULTS+["n"]]
print("P(low touches >= k ATR below close) | dial-MA within +-3 of current:")
print(df.to_string())
