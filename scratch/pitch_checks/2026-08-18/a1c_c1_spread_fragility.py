"""C1 round 2. The spread (long TLT / short SPY into month-end, div-gated)
survived a1b. Fragility battery: threshold ladder (absolute AND rank, because
the pitch quotes an 89.2 percentile), path decomposition (fix the ENTRY at 8
sessions before month-end, vary the exit -> does the mechanism live in the
last few sessions as the flow story claims, or in one print?), and support.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np, pandas as pd

LAG = 1
raw = load_prices(["SPY", "TLT"])
spy, tlt = raw["SPY"]["Close"], raw["TLT"]["Close"]
r21s, r21t = spy.pct_change(21), tlt.pct_change(21)
idx = spy.index.intersection(tlt.index)
px = pd.DataFrame({"SPY": spy.reindex(idx), "TLT": tlt.reindex(idx)}).dropna()
idx = px.index
div = (r21s.reindex(idx) - r21t.reindex(idx))
div_rk = div.rolling(1260).rank(pct=True) * 100.0

ym = pd.Series(idx.year*100 + idx.month, index=idx)
is_last = pd.Series(ym.values, index=idx).ne(pd.Series(ym.values, index=idx).shift(-1))
pos = pd.Series(range(len(idx)), index=idx)

def anchor_exit_me(h):
    t = pos + LAG + h
    m = pd.Series(False, index=idx)
    ok = t < len(idx)
    m.loc[idx[ok.values]] = is_last.values[t[ok.values].values]
    return m

def spread_ret(h):
    return fwd_lag(px["TLT"], h, LAG) - fwd_lag(px["SPY"], h, LAG)

# --- 1. threshold ladder, absolute and rank, h=8 exit==ME
H = 8
sp8 = spread_ret(H); v8 = sp8.notna(); A8 = anchor_exit_me(H)
rows = []
for thr in [-99, 0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.0732, 0.09, 0.11]:
    d = idx[(A8 & (div >= thr).fillna(False)).values & v8.values]
    rows.append(summarize(sp8.loc[d].values, f"div >= {100*thr:+.2f}pp" if thr > -99 else "gate OFF"))
show(rows, "1a. ABSOLUTE threshold ladder, SPREAD, exit==month-end, h=8")
rows = []
for thr in [0, 50, 70, 75, 80, 85, 88, 90, 92, 95]:
    d = idx[(A8 & (div_rk >= thr).fillna(False)).values & v8.values]
    rows.append(summarize(sp8.loc[d].values, f"div rank1260 >= {thr}"))
show(rows, "1b. RANK threshold ladder (live rank = 88.4)")

# --- 2. path decomposition: ENTRY fixed 8 sessions before ME, vary exit h
base = A8  # signal day D; entry D+1 is 8 sessions before the ME close
G = (div >= 0.05).fillna(False)
rows = []
for h in range(1, 11):
    r = fwd_lag(px["TLT"], h, LAG) - fwd_lag(px["SPY"], h, LAG)
    d = idx[(base & G).values & r.notna().values]
    rows.append(summarize(r.loc[d].values, f"hold h={h} from the SAME entry"))
show(rows, "2. PATH: entry fixed 8 sessions before month-end, exit varied (h=8 lands ON the ME close)")

# --- 3. is it the last session alone? entry-fixed marginal day returns
r7 = fwd_lag(px["TLT"], 7, LAG) - fwd_lag(px["SPY"], 7, LAG)
d = idx[(base & G).values & sp8.notna().values & r7.notna().values]
marg = (sp8.loc[d] - r7.loc[d])
print("\n3. marginal contribution of the FINAL session (h=8 minus h=7), same entries:")
show([summarize(marg.values, "final session only"),
      summarize(r7.loc[d].values, "first 7 sessions"),
      summarize(sp8.loc[d].values, "full 8")], "")

# --- 4. entry-offset ladder: exit ALWAYS the ME close, entry k sessions before
rows = []
for h in range(2, 14):
    r = spread_ret(h); A = anchor_exit_me(h)
    d = idx[(A & G).values & r.notna().values]
    rows.append(summarize(r.loc[d].values, f"entry {h} sess before ME, exit ME"))
show(rows, "4. ENTRY-OFFSET ladder (exit always the month-end close). Flow story predicts the LAST few sessions carry it.")

# --- 5. support: where does today sit inside the trigger population?
d = idx[(A8 & G).values & v8.values]
print("\n5. SUPPORT. live: SPY21d +3.95%%, TLT21d -3.36%%, div +7.32pp, rank %.1f" % div_rk.iloc[-1])
print("   trigger-day SPY21d: mean %+.2f%% median %+.2f%% | pctile of live +3.95%%: %.0f"
      % (100*r21s.reindex(d).mean(), 100*r21s.reindex(d).median(),
         100*(r21s.reindex(d) < 0.0395).mean()))
print("   trigger-day TLT21d: mean %+.2f%% median %+.2f%% | pctile of live -3.36%%: %.0f"
      % (100*r21t.reindex(d).mean(), 100*r21t.reindex(d).median(),
         100*(r21t.reindex(d) < -0.0336).mean()))
# how many trigger days had BOTH legs like today (SPY up AND TLT down)?
both = ((r21s.reindex(d) > 0) & (r21t.reindex(d) < 0))
print("   trigger days where SPY21d>0 AND TLT21d<0 (today's shape): %d of %d" % (both.sum(), len(d)))
show([summarize(sp8.loc[d[both.values]].values, "SPY up & TLT down (today's shape)"),
      summarize(sp8.loc[d[~both.values]].values, "other shapes")], "5b. shape split")

# --- 6. AUGUST specifically + midterm
show([summarize(sp8.loc[d[d.month == 8]].values, "August triggers"),
      summarize(sp8.loc[d[d.year % 4 == 2]].values, "midterm years"),
      summarize(sp8.loc[d[(d.year % 4 == 2)]].values if False else sp8.loc[d].values, "all")],
     "6. August / midterm conditioners")
print("   August trigger dates:", ", ".join(str(x.date()) for x in d[d.month == 8]))
