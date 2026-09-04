"""B1 MAP ONLY: coarse reachability + count-first pass on the two anchors that
are new today (FOMC/VIX-expiry at +10 td) and on the never-swept holiday object.
Numbers here are MAP verdicts, not evidence: anything selected out of this sweep
must be charged for the full grid walked (2026-08-31 registry rule)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import pandas as pd, numpy as np

CLASSES = {"us_large":"SPY","us_small":"IWM","rates":"TLT","rates_belly":"IEF","credit":"HYG",
           "gold":"GLD","miners":"GDX","metals":"SLV","energy":"USO","energy_eq":"XLE",
           "dollar":"UUP","intl_dev":"EFA","intl_em":"EEM","vol_inv":"SVXY","tech":"XLK"}
px = load_prices(sorted(set(CLASSES.values())))
C = pd.DataFrame({t: px[t]["Close"] for t in px}).dropna(how="all")
C = C[C.index <= pd.Timestamp("2026-08-31")]

ev = load_events(["fomc_decision","vix_expiry","nfp","opex"])
print("events cols:", list(ev.columns), "rows", len(ev))
fom = pd.DatetimeIndex(sorted(ev.loc[ev["event"]=="fomc_decision","date"].unique()))
fom = fom[(fom >= C.index[0]) & (fom <= C.index[-1])]
print("FOMC decisions in price window:", len(fom), fom[0].date(), "->", fom[-1].date())

# TODAY'S geometry: entry MOC 2026-09-01 (lag 1 off the 08-31 signal close),
# FOMC 2026-09-16 = 10 sessions after the entry close.
K_ENTRY = -10   # entry session sits 10 td BEFORE the decision
print("\n-- count-first: how many FOMC decisions have a tradeable 10-td-before entry? ", len(fom))
mid = np.array([d.year % 4 == 2 for d in fom])
print("   midterm-year decisions:", int(mid.sum()), "| non-midterm:", int((~mid).sum()))
sep = np.array([d.month == 9 for d in fom])
print("   September decisions:", int(sep.sum()), "| September AND midterm:", int((sep & mid).sum()))
coin = np.array([d in set(pd.DatetimeIndex(ev.loc[ev['event']=='vix_expiry','date'].unique())) for d in fom])
print("   decisions landing ON a vix_expiry date:", int(coin.sum()))

print("\n-- COARSE class sweep, entry MOC at decision-10td, hold to the decision CLOSE (h=10).")
print("   (map only; charged grid = 15 classes x 2 signs x horizons)")
idx = C.index
rows=[]
for name, tic in CLASSES.items():
    s = C[tic].dropna()
    pos, kept = anchor_positions(s.index, fom, offset=K_ENTRY)
    pos = [p for p in pos if p < len(s)-10]
    if len(pos) < 8:
        rows.append({"class":name,"tic":tic,"N":len(pos),"note":"too few"}); continue
    dts = s.index[pos]
    fw = fwd_lag(s, 10, lag=0)          # entry IS the anchored session close
    v = fw.reindex(dts).dropna()
    base = fw.dropna()
    rows.append({"class":name,"tic":tic,"N":len(v),
                 "mean_pct":round(float(v.mean())*100,3),
                 "hit":round(float((v>0).mean())*100,1),
                 "drift_pct":round(float(base.mean())*100,3),
                 "edge_pp":round(float(v.mean()-base.mean())*100,3)})
show(rows, "FOMC-10td -> decision close, by class (MAP)")

print("\n-- midterm-only slice, same geometry")
rows=[]
fm = fom[mid]
for name, tic in CLASSES.items():
    s = C[tic].dropna()
    pos, kept = anchor_positions(s.index, fm, offset=K_ENTRY)
    pos = [p for p in pos if p < len(s)-10]
    if len(pos) < 5: rows.append({"class":name,"N":len(pos),"note":"too few"}); continue
    dts = s.index[pos]; fw = fwd_lag(s,10,lag=0)
    v = fw.reindex(dts).dropna(); base = fw.dropna()
    rows.append({"class":name,"N":len(v),"mean_pct":round(float(v.mean())*100,3),
                 "hit":round(float((v>0).mean())*100,1),
                 "edge_pp":round(float(v.mean()-base.mean())*100,3)})
show(rows, "FOMC-10td -> decision close, MIDTERM ONLY (MAP)")

# ---- the never-swept holiday object: count first
print("\n== HOLIDAY OBJECT count-first (Labor Day = first Monday of September)")
spy = C["SPY"].dropna()
d = pd.Series(spy.index, index=spy.index)
gaps = (d.shift(-1) - d).dt.days
# a Monday holiday shows as a Friday->Tuesday 4-day gap
pre_hol = spy.index[(gaps == 4) & (pd.Series(spy.index, index=spy.index).dt.weekday == 4)]
print("  Friday sessions followed by a 4-day gap (Monday-holiday eves):", len(pre_hol))
labor = [dt for dt in pre_hol if dt.month == 9 and dt.day <= 7]
print("  ... of which Labor Day eves (Sep 1-7 Friday):", len(labor), [str(x.date()) for x in labor])
# the SHORT WEEK object: first session of the post-Labor-Day week
print("  Labor Day weeks available since 1999:", len(labor), "-> count-first says a Labor-Day-specific cell is N<=27")
# today's own position: first session of September
firsts = []
for y in range(1999, 2027):
    m = spy.index[(spy.index.year==y) & (spy.index.month==9)]
    if len(m): firsts.append(m[0])
print("  first September sessions available:", len(firsts))
fw10 = fwd_lag(spy, 10, lag=0)
v = fw10.reindex(pd.DatetimeIndex(firsts)).dropna()
print("  SPY first-Sept-session -> +10td: N=%d mean %+.3f%% hit %.1f%% vs all-days %+.3f%%"
      % (len(v), v.mean()*100, (v>0).mean()*100, fw10.dropna().mean()*100))
