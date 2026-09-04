"""B1 MAP: count-first on the interaction cells today's tape suggests, plus a
data-integrity look at the two tape readings that look impossible."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import pandas as pd, numpy as np
ASOF = pd.Timestamp("2026-08-31")

px = load_prices(["SLV","GLD","SPY","IWM","XLE","XLK","XLV","XLP","XLU","XLI","XLRE","XLY","XLB","XLC","XLF","TLT","IEF","^TNX","^VIX","^MOVE","USO"])
C = pd.DataFrame({t: px[t]["Close"] for t in px}).dropna(how="all"); C = C[C.index<=ASOF]

print("== SLV integrity (tape says -43% off its 52w high with +69% 252d return)")
s = C["SLV"].dropna()
w = s.iloc[-252:]
print("   last %.3f  252d max %.3f on %s  252d min %.3f" % (s.iloc[-1], w.max(), w.idxmax().date(), w.min()))
print("   last 8 bars:"); print(s.tail(8).round(3).to_string())
print("   top 5 bars of the trailing year:"); print(w.nlargest(5).round(3).to_string())

print("\n== defensive/cyclical washout COUNT (sectors with 5d rank <= 15 on the same session)")
SEC = ["XLK","XLF","XLV","XLI","XLY","XLP","XLU","XLB","XLRE","XLC","XLE"]
R5 = pd.DataFrame({t: pct_rank(C[t].dropna(),5) for t in SEC}).dropna()
cnt = (R5 <= 15).sum(axis=1)
print("   today's count:", int(cnt.iloc[-1]), "of", len(SEC), "->", [t for t in SEC if R5[t].iloc[-1]<=15])
print("   historical distribution of the count:"); print(cnt.value_counts().sort_index().to_string())
print("   sessions with count >= today:", int((cnt >= cnt.iloc[-1]).sum()), "of", len(cnt))

print("\n== count-first: FOMC entries (decision-10td) with ^TNX within 0.25pct of its trailing-252 max")
ev = load_events(["fomc_decision"])
fom = pd.DatetimeIndex(sorted(ev["date"].unique()))
tnx = C["^TNX"].dropna()
lvl = tnx / tnx.rolling(252).max() * 100
pos, kept = anchor_positions(tnx.index, fom, offset=-10)
vals = [(kept[i], float(lvl.iloc[p])) for i,p in enumerate(pos) if not np.isnan(lvl.iloc[p])]
hi = [(d,v) for d,v in vals if v >= 99.75]
print("   anchors measured:", len(vals), "| with TNX >= 99.75pct of 252-max at the ENTRY session:", len(hi))
print("   dates:", [str(d.date()) for d,_ in hi])
print("   midterm among them:", [str(d.date()) for d,_ in hi if d.year%4==2])
print("   distribution of entry-session TNX level pct: min %.1f p25 %.1f med %.1f p75 %.1f max %.1f"
      % tuple(np.percentile([v for _,v in vals],[0,25,50,75,100])))

print("\n== count-first: NFP landing on a pre-Labor-Day Friday")
spy = C["SPY"].dropna()
nfp = pd.DatetimeIndex(sorted(load_events(["nfp"])["date"].unique()))
d = pd.Series(spy.index, index=spy.index)
gaps = (d.shift(-1)-d).dt.days
hol_eve = set(spy.index[(gaps==4) & (d.dt.weekday==4)])
sept_nfp_hol = [x for x in nfp if x in hol_eve and x.month==9]
print("   September NFPs that are ALSO a Monday-holiday eve:", len(sept_nfp_hol), [str(x.date()) for x in sept_nfp_hol])
print("   ALL NFPs that are a Monday-holiday eve:", len([x for x in nfp if x in hol_eve]))

print("\n== dispersion state cross-check (component RV vs SPY RV is the risk page's number)")
print("   SPY 21d ann rvol %.1f%%" % (float(C['SPY'].pct_change().iloc[-21:].std()*np.sqrt(252))*100))

print("\n== today's live rates geometry")
print("   ^TNX %.4f = %.4f%% of 252-max | 21d chg %+.1fbp | 63d chg %+.1fbp | 252d chg %+.1fbp"
      % (tnx.iloc[-1], lvl.iloc[-1], (tnx.iloc[-1]-tnx.iloc[-22])*100, (tnx.iloc[-1]-tnx.iloc[-64])*100, (tnx.iloc[-1]-tnx.iloc[-253])*100))
print("   ^MOVE %.2f (1d %+.2f%%) | ^VIX %.2f" % (C['^MOVE'].dropna().iloc[-1], float(C['^MOVE'].dropna().pct_change().iloc[-1])*100, C['^VIX'].dropna().iloc[-1]))
