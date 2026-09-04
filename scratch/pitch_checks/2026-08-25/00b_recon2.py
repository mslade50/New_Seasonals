"""Recon round 2: the premises the first pass left NaN or unmeasured."""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import pandas as pd, numpy as np

TK = ["SPY","QQQ","IWM","XLK","XLV","XLU","XLE","SMH","OIH","XOP","TLT","IEF","LQD","HYG",
      "^TNX","GLD","GDX","UUP","DX-Y.NYB","EEM","EFA","FXI","^VIX","^VIX3M","SVXY","NVDA","FXE"]
px = close_panel(TK)

def state(t, n=252):
    s = px[t].dropna()
    hi = s.rolling(n, min_periods=n//2).max().iloc[-1]
    lo = s.rolling(n, min_periods=n//2).min().iloc[-1]
    print(f"  {t:<10} last={s.iloc[-1]:>9.2f}  off52wh={(s.iloc[-1]/hi-1)*100:>7.2f}%  above52wl={(s.iloc[-1]/lo-1)*100:>7.2f}%  first={s.index[0].date()}")

print("=== 52w positions (min_periods fixed) ===")
for t in ["XLE","XOP","OIH","HYG","LQD","GLD","GDX","TLT","IEF","SMH","XLK","XLV","XLU","SPY","QQQ","EEM","EFA","DX-Y.NYB","SVXY","NVDA"]:
    state(t)

print("\n=== W5 freshness: TLT/IEF/LQD tight rung, episode-first ===")
def offlow(t, n=252):
    s = px[t]
    lo = s.rolling(n, min_periods=n//2).min()
    return (s/lo - 1) * 100
tl, ie, lq = offlow("TLT"), offlow("IEF"), offlow("LQD")
tight = (tl <= 0.5) & (ie <= 1.0) & (lq <= 1.0)
tight = tight[tl.notna() & ie.notna() & lq.notna()]
print("  today: TLT %.2f%% IEF %.2f%% LQD %.2f%%  -> tight=%s" % (tl.iloc[-1], ie.iloc[-1], lq.iloc[-1], bool(tight.iloc[-1])))
td = tight[tight].index
print("  last 6 tight days:", [str(d.date()) for d in td[-6:]])

print("\n=== EEM / EFA / international relative ===")
r63 = px.pct_change(63, fill_method=None)
r21 = px.pct_change(21, fill_method=None)
for a,b in [("EEM","EFA"),("EEM","SPY"),("FXI","EEM")]:
    sp = (r63[a]-r63[b]).dropna()*100
    win = sp.iloc[-252:]
    print(f"  {a}-{b} 63d = {sp.iloc[-1]:+.2f}pp   PIT252 pctile {(win<sp.iloc[-1]).mean()*100:.1f}  full {(sp<sp.iloc[-1]).mean()*100:.1f}")

print("\n=== VIX term-structure kink (5d change in VIX/VIX3M) ===")
ts = (px["^VIX"]/px["^VIX3M"]).dropna()
d5 = ts.diff(5)
print(f"  ratio {ts.iloc[-1]:.3f}  5d change {d5.iloc[-1]:+.4f}  PIT252 {(d5.iloc[-252:]<d5.iloc[-1]).mean()*100:.1f}")

print("\n=== month-end position (US business calendar) ===")
idx = px.index
me = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).transform("max")
pos = pd.Series([ (idx.get_loc(m) - idx.get_loc(d)) for d,m in zip(idx, me) ], index=idx)
print("  today ME offset (sessions to month-end close):", int(pos.iloc[-1]))

print("\n=== month-end FX: DXY / UUP / FXE across ME-4 -> ME-0 ===")
for t in ["DX-Y.NYB","UUP","FXE","SPY","TLT","GLD"]:
    s = px[t].dropna()
    p = pos.reindex(s.index)
    anch = s.index[(p == 4)]
    rets = []
    for a in anch:
        i = s.index.get_loc(a)
        j = i + 4
        if j < len(s):
            rets.append(s.iloc[j]/s.iloc[i]-1)
    rets = np.array(rets)
    if len(rets):
        print(f"  {t:<10} N={len(rets):>4}  mean={rets.mean()*100:+.3f}%  med={np.median(rets)*100:+.3f}%  hit={(rets>0).mean()*100:.1f}%")

print("\n=== gold: GDX 21d thrust extreme, historical count ===")
g = r21["GDX"].dropna()
print(f"  GDX 21d now {g.iloc[-1]*100:+.2f}%   days >= that, full history: {(g>=g.iloc[-1]).sum()} of {len(g)}")
gg = (r21["GDX"]-r21["GLD"]).dropna()
print(f"  GDX-GLD 21d now {gg.iloc[-1]*100:+.2f}pp  days >=: {(gg>=gg.iloc[-1]).sum()} of {len(gg)}")

print("\n=== NVDA print history in the earnings parquet ===")
ec = pd.read_parquet(ROOT/"data/earnings_calendar.parquet")
print(ec.columns.tolist())
n = ec[ec["symbol"].astype(str).str.upper()=="NVDA"] if "symbol" in ec.columns else None
if n is not None:
    c = [c for c in n.columns if "date" in c.lower()]
    print(n[c].tail(8).to_string())
