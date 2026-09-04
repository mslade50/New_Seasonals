"""Today's value for every active watchlist trigger. Survey input for the
verdict lines in 00_surface_map.md -- one number per parked entry."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import pandas as pd, numpy as np

T = ["TLT","IEF","LQD","HYG","SPY","QQQ","IWM","GLD","SLV","GDX","XLE","XOP","OIH","USO",
     "UUP","DX-Y.NYB","^TNX","^VIX","^SKEW","EEM","EFA","FXI","IHI","XLU","XLK","XLV",
     "XLI","XLF","KRE","XLP","XLY","XLB","XLRE","XLC","SMH","SVXY","^VIX3M","^MOVE","NEM","FCX","XME"]
px = close_panel(T)
last = px.index[-1]
print("panel last date:", last.date())

def r(t, n):
    s = px[t].dropna()
    return s.iloc[-1] / s.iloc[-1 - n] - 1.0

def rank(t, n, lb=252):
    return pct_rank(px[t].dropna(), n, lb).iloc[-1]

def d52h(t):
    s = px[t].dropna()
    return s.iloc[-1] / s.tail(252).max() - 1.0

def d52l(t):
    s = px[t].dropna()
    return s.iloc[-1] / s.tail(252).min() - 1.0

print("\n#0/#5/#12/#18  rates state")
print(f"  TLT ret1 {r('TLT',1)*100:+.2f}%  ret5 {r('TLT',5)*100:+.2f}%  rank5 {rank('TLT',5):.1f} "
      f"rank21 {rank('TLT',21):.1f} | 52wl +{d52l('TLT')*100:.2f}% 52wh {d52h('TLT')*100:+.2f}%")
print(f"  IEF 52wl +{d52l('IEF')*100:.2f}%   LQD 52wl +{d52l('LQD')*100:.2f}%")
print(f"  ^TNX ret21(level) {(px['^TNX'].dropna().iloc[-1]-px['^TNX'].dropna().iloc[-22]):+.3f}pt "
      f"rank21 {rank('^TNX',21):.1f} | pct of trailing-252 high "
      f"{px['^TNX'].dropna().iloc[-1]/px['^TNX'].dropna().tail(252).max()*100:.2f}%")

print("\n#1 credit HYG/LQD 52w extremes")
print(f"  HYG 52wh {d52h('HYG')*100:+.2f}%  LQD 52wl +{d52l('LQD')*100:.2f}%")

print("\n#3 gold/miner thrust")
print(f"  GDX rank5 {rank('GDX',5):.1f} GLD rank5 {rank('GLD',5):.1f} | GLD 52wh {d52h('GLD')*100:+.2f}% "
      f"GLD rank63 {rank('GLD',63):.1f} | GDX 5d {r('GDX',5)*100:+.2f}%")

print("\n#4/#7 energy")
print(f"  USO ret1 {r('USO',1)*100:+.2f}% ret5 {r('USO',5)*100:+.2f}% rank5 {rank('USO',5):.1f} "
      f"rank63 {rank('USO',63):.1f}")
print(f"  XLE 52wh {d52h('XLE')*100:+.2f}% | OIH-XOP 63d spread "
      f"{(r('OIH',63)-r('XOP',63))*100:+.2f}pp")
sp = (px["OIH"].pct_change(63, fill_method=None) - px["XOP"].pct_change(63, fill_method=None)).dropna()
print(f"  OIH-XOP 63d spread PIT pctile(252d) {(sp.tail(252) <= sp.iloc[-1]).mean()*100:.1f}")

print("\n#6/#13 vol")
print(f"  SKEW rank5 {rank('^SKEW',5):.1f} | VIX rank21 {rank('^VIX',21):.1f} "
      f"VIX 1d {r('^VIX',1)*100:+.2f}% | SPY 52wh {d52h('SPY')*100:+.2f}%")
print(f"  VIX3M/VIX {px['^VIX3M'].iloc[-1]/px['^VIX'].iloc[-1]:.4f}  MOVE/VIX {px['^MOVE'].iloc[-1]/px['^VIX'].iloc[-1]:.3f}")

print("\n#8 IHI  |  #9 FXI  |  #11 SPY-TLT joint")
print(f"  IHI rank21 {rank('IHI',21):.1f} | FXI rank5 {rank('FXI',5):.1f} rank21 {rank('FXI',21):.1f}")
print(f"  SPY 52wh {d52h('SPY')*100:+.2f}%  TLT 52wl +{d52l('TLT')*100:.2f}%")

print("\n#14/#16 dollar x rates")
print(f"  DX rank21 {rank('DX-Y.NYB',21):.1f} UUP rank21 {rank('UUP',21):.1f} | "
      f"TNX 21-session level change {(px['^TNX'].dropna().iloc[-1]-px['^TNX'].dropna().iloc[-22]):+.3f}pt | "
      f"TNX rank21 {rank('^TNX',21):.1f}")

print("\n#15 XLV-XLK one-day rotation gap")
gap1 = (px["XLV"].pct_change(fill_method=None) - px["XLK"].pct_change(fill_method=None)).iloc[-1] * 100
gap5 = (px["XLV"].pct_change(5, fill_method=None) - px["XLK"].pct_change(5, fill_method=None)).iloc[-1] * 100
print(f"  1d XLV-XLK {gap1:+.2f}pp | 5d {gap5:+.2f}pp")

print("\n#19 KRE breadth | #22 energy z10 count | #23 new-high breadth")
print(f"  KRE rank5 {rank('KRE',5):.1f} | XLF rank63 {rank('XLF',63):.1f} 52wh {d52h('XLF')*100:+.2f}%")
E = ["XLE","XOP","USO","COP","CVX","VLO","OXY","SLB","EOG","HAL","WMB"]
pe = close_panel(E)
z = {}
for t in E:
    s = pe[t].dropna()
    ret10 = s.pct_change(10, fill_method=None)
    vol21 = s.pct_change(fill_method=None).rolling(21).std() * np.sqrt(10)
    z[t] = (ret10 / vol21).iloc[-1]
print("  energy z10:", {k: round(float(v), 2) for k, v in z.items()})
print(f"  count z10>=2.0: {sum(1 for v in z.values() if v >= 2.0)}")

print("\n#25 sector washout into 52w high (5d rank <= 5 AND within 5% of 52w high)")
SEC = ["XLK","XLV","XLF","XLI","XLY","XLP","XLE","XLU","XLB"]
for t in SEC:
    print(f"   {t}: rank5 {rank(t,5):5.1f}  52wh {d52h(t)*100:+6.2f}%  rank21 {rank(t,21):5.1f}")

print("\n#26 XLU washout x TLT")
print(f"  XLU rank21 {rank('XLU',21):.1f} | TLT rank21 {rank('TLT',21):.1f} rank5 {rank('TLT',5):.1f}")

print("\n#24 OIH outright | metals complex breadth")
M = ["GLD","SLV","GDX","NEM","FCX","XME"]
for t in M:
    print(f"   {t}: rank5 {rank(t,5):5.1f} rank21 {rank(t,21):5.1f} rank63 {rank(t,63):5.1f} 52wh {d52h(t)*100:+7.2f}%")
print(f"  count rank21>=95: {sum(1 for t in M if rank(t,21) >= 95)} of {len(M)}")
