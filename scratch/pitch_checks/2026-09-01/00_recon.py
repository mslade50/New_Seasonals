"""Live readings for every watchlist trigger + structural facts for the surface map."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import pandas as pd, numpy as np

ASOF = pd.Timestamp("2026-08-31")

NAMES = ["SPY","QQQ","IWM","^GSPC","^NDX","TLT","IEF","LQD","HYG","^TNX","^MOVE",
         "GLD","GDX","SLV","NEM","CEF","XME","FCX","USO","UNG","DBC","XLE","XOP","OIH",
         "UUP","DX-Y.NYB","EFA","EEM","FXI","EWZ","EWJ","^VIX","^VIX3M","SVXY","UVXY","^SKEW",
         "XLK","XLF","XLV","XLI","XLY","XLP","XLU","XLB","XLRE","XLC","KRE","ITA","VNQ","SMH","IHI","IBB",
         "COP","CVX","VLO","OXY","SLB","EOG","HAL","WMB","DIA"]
px = load_prices(NAMES)
C = pd.DataFrame({t: px[t]["Close"] for t in px}).dropna(how="all")
C = C[C.index <= ASOF]
print("panel through", C.index[-1].date(), "rows", len(C))

def r(t, n):
    s = C[t].dropna()
    return float(s.iloc[-1]/s.iloc[-1-n] - 1)*100

def rank_n(t, n, lb=252):
    s = C[t].dropna()
    return float(pct_rank(s, n, lb).iloc[-1])

def rank_excl(t, n, lb=252):
    """exclusive-self convention: w[:-1] <= w[-1]."""
    s = C[t].dropna()
    ret = s.pct_change(n)
    w = ret.iloc[-lb:]
    cur = w.iloc[-1]
    prior = w.iloc[:-1].dropna()
    return float((prior <= cur).mean()*100)

def lvl_pct_high(t, lb=252):
    """close as % of trailing-lb max (100 = at the high)."""
    s = C[t].dropna()
    return float(s.iloc[-1] / s.iloc[-lb:].max() * 100)

def lvl_pctile(t, lb=252):
    s = C[t].dropna()
    w = s.iloc[-lb:]
    return float((w <= w.iloc[-1]).mean()*100)

print("\n== W: duration-neutral flattener (trigger: ^TNX within 0.25% of trailing-252 high)")
print("  ^TNX close %.4f  = %.4f%% of trailing-252 max (rung >= 99.75)" % (C['^TNX'].dropna().iloc[-1], lvl_pct_high('^TNX')))
print("  ^TNX level pctile (incl-self) %.2f ; excl-self %.2f" % (lvl_pctile('^TNX'),
      float((C['^TNX'].dropna().iloc[-252:-1] <= C['^TNX'].dropna().iloc[-1]).mean()*100)))
print("  trailing-252 max %.4f on %s" % (C['^TNX'].dropna().iloc[-252:].max(), C['^TNX'].dropna().iloc[-252:].idxmax().date()))
# ^TNX is quoted in PERCENT (4.758 = 4.758%), so an index point change x100 = bp.
# This line multiplied by 10 in the first run and printed every yield change 10x low;
# corrected 2026-09-01 after checker A caught it.
print("  21-session yield change: %+.1f bp" % ((C['^TNX'].dropna().iloc[-1]-C['^TNX'].dropna().iloc[-22])*100))
print("  Jackson Hole 2026-08-28 is BEFORE today -> a hold starting 2026-09-01 does NOT span JH")

print("\n== W: utilities washout w/ long end hit (XLU r21<=5 AND TLT r21<25)")
print("  XLU r21 %.1f   TLT r21 %.1f   TLT r5 %.1f" % (rank_n('XLU',21), rank_n('TLT',21), rank_n('TLT',5)))

print("\n== W: bond vol MID-RANGE band (^MOVE trailing-252 LEVEL pctile in [40,50))")
print("  ^MOVE level %.2f  pctile %.1f" % (C['^MOVE'].dropna().iloc[-1], lvl_pctile('^MOVE')))

print("\n== W: narrow energy thrust cluster (count of 11 at z10>=2.0, arms at 2 or 3)")
eng = ["XLE","XOP","USO","COP","CVX","VLO","OXY","SLB","EOG","HAL","WMB"]
zz = {t: float(zscore(C[t].dropna(),10).iloc[-1]) for t in eng if t in C}
print("  " + "  ".join(f"{t} {v:+.2f}" for t,v in sorted(zz.items(), key=lambda kv:-kv[1])))
print("  COUNT z10>=2.0 :", sum(1 for v in zz.values() if v>=2.0))

print("\n== W: pooled laggard STILL FALLING (r21>=90 AND r63<=10 AND r5<15)")
pool = ["SPY","QQQ","IWM","EFA","EEM","FXI","XLK","XLF","XLV","XLI","XLY","XLP","XLU","XLB","XLRE","XLC",
        "SMH","IBB","IHI","KRE","ITA","VNQ","GDX","XME","XOP","OIH","XLE","DIA","EWJ"]
hits=[]
for t in pool:
    if t not in C: continue
    a,b,c5 = rank_n(t,21), rank_n(t,63), rank_n(t,5)
    if a>=90 and b<=10:
        hits.append((t,a,b,c5))
print("  holders of r21>=90 & r63<=10:", hits if hits else "NONE")
near = sorted([(t, rank_n(t,21), rank_n(t,63), rank_n(t,5)) for t in pool if t in C], key=lambda x:-x[1])[:6]
print("  top r21 in pool:", [(t,round(a,1),round(b,1),round(c,1)) for t,a,b,c in near])

print("\n== W: GLD miner-led thrust (GDX r5>=95 & GLD r5<95)")
print("  GDX r5 %.1f  GLD r5 %.1f" % (rank_n('GDX',5), rank_n('GLD',5)))
print("== W: XLE on crude 1d thrust in 5-6pct band -> USO 1d %+.2f pct" % r("USO",1))
print("== W: SPY on skew spike alone (^SKEW r5>=95) -> %.1f" % rank_n('^SKEW',5))
print("== W: vol pop in calm tape (VIX r21<=25, VIX 1d>=+5%%, SPY 1d>-0.75%%) -> r21 %.1f, VIX 1d %+.2f%%, SPY 1d %+.2f%%"
      % (rank_n('^VIX',21), r('^VIX',1), r('SPY',1)))
print("== W: TLT big up day from low zone (TLT 1d>=+1.5%%, within 4%% of 252-low) -> 1d %+.2f%%, above-low %+.2f%%"
      % (r('TLT',1), float(C['TLT'].dropna().iloc[-1]/C['TLT'].dropna().iloc[-252:].min()-1)*100))
print("== W: IG complex pinned at 252-lows (TLT<=0.5%%, IEF<=1.0%%, LQD<=1.0%% above low)")
for t in ["TLT","IEF","LQD"]:
    s=C[t].dropna(); print("   %s %+.2f%% above trailing-252 low" % (t, float(s.iloc[-1]/s.iloc[-252:].min()-1)*100))
print("== W: HYG at 252-high while SPY off (HYG %+.2f%% off high, SPY %+.2f%% off high, LQD %+.2f%% above low)"
      % (float(C['HYG'].dropna().iloc[-1]/C['HYG'].dropna().iloc[-252:].max()-1)*100,
         float(C['SPY'].dropna().iloc[-1]/C['SPY'].dropna().iloc[-252:].max()-1)*100,
         float(C['LQD'].dropna().iloc[-1]/C['LQD'].dropna().iloc[-252:].min()-1)*100))
print("== W: sector washout into 52w high (r5<=5 AND within 5%% of 52w high)")
for t in ["XLK","XLF","XLV","XLI","XLY","XLP","XLU","XLB","XLRE","XLC","XLE"]:
    s=C[t].dropna(); off=float(s.iloc[-1]/s.iloc[-252:].max()-1)*100
    print("   %-5s r5 %5.1f  off-high %+6.2f%%" % (t, rank_n(t,5), off))
print("== W: SPY at 52w high & TLT at 52w low -> SPY off-high %.2f%%, TLT above-low %.2f%%"
      % (float(C['SPY'].dropna().iloc[-1]/C['SPY'].dropna().iloc[-252:].max()-1)*100,
         float(C['TLT'].dropna().iloc[-1]/C['TLT'].dropna().iloc[-252:].min()-1)*100))
print("== W: rate rise unconfirmed by dollar (TNX r21>=65 & DX r21<=20) -> TNX %.1f  DX %.1f"
      % (rank_n('^TNX',21), rank_n('DX-Y.NYB',21)))
print("== W: XLV-XLK one-day rotation gap >= +3.0pp -> %+.2fpp" % (r('XLV',1)-r('XLK',1)))
print("== W: FXI 5d break inside thrust (FXI r5<=20, r21>=80, EEM 5d>0) -> FXI r5 %.1f r21 %.1f EEM 5d %+.2f%%"
      % (rank_n('FXI',5), rank_n('FXI',21), r('EEM',5)))
print("== W: IHI r21==100 -> %.1f" % rank_n('IHI',21))
print("== W: KRE bank breadth washout -> KRE r5 %.1f" % rank_n('KRE',5))
print("== W: SLV complex break depth bucket -> SLV 1d %+.2f%%, GLD 1d %+.2f%%, GDX 1d %+.2f%%" % (r('SLV',1), r('GLD',1), r('GDX',1)))
