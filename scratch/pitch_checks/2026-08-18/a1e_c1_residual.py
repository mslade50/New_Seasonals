"""C1 round 4. a1d showed the DIVERGENCE is decorative: half of it (TLT 21d
oversold) dominates the whole at matched N. Characterise that residual
honestly so it is recorded as a near-miss and NOT credited to C1, and run
the offset ladder on it (the repo's 8-for-8 killer).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np, pandas as pd

LAG, H = 1, 8
raw = load_prices(["SPY", "TLT"])
spy, tlt = raw["SPY"]["Close"], raw["TLT"]["Close"]
idx = spy.index.intersection(tlt.index)
px = pd.DataFrame({"SPY": spy.reindex(idx), "TLT": tlt.reindex(idx)}).dropna()
idx = px.index
T21 = tlt.pct_change(21).reindex(idx)
ymv = pd.Series(idx.year*100 + idx.month, index=idx)
is_last = ymv.ne(ymv.shift(-1)); is_last.iloc[-1] = False
pos = pd.Series(range(len(idx)), index=idx)

def anchor(h, k=0):
    t = pos + LAG + h + k
    m = pd.Series(False, index=idx); ok = t < len(idx)
    m.loc[idx[ok.values]] = is_last.values[t[ok.values].values]
    return m

def spr(h):
    return fwd_lag(px["TLT"], h, LAG) - fwd_lag(px["SPY"], h, LAG)

G = (T21 <= -0.025).fillna(False)
s8 = spr(H); v = s8.notna(); A = anchor(H)

# threshold ladder on the TLT conditioner
rows = []
for thr in [99, 0.0, -0.01, -0.02, -0.025, -0.0336, -0.045, -0.06, -0.08]:
    g = pd.Series(True, index=idx) if thr == 99 else (T21 <= thr).fillna(False)
    d = idx[(A & g).values & v.values]
    rows.append(summarize(s8.loc[d].values, "gate OFF" if thr == 99 else f"TLT21d <= {100*thr:+.2f}%"))
show(rows, "1. TLT-oversold threshold ladder (spread, month-end anchor, h=8). LIVE = -3.36%")

# EXIT offset ladder (move the exit away from month-end) -- the killer test
rows = []
for k in range(0, 13):
    m = anchor(H, k)
    d = idx[(m & G).values & v.values]
    rows.append(summarize(s8.loc[d].values, f"exit k={k} sessions BEFORE month-end"))
show(rows, "2. EXIT OFFSET LADDER, TLT21d<=-2.5% gate. k=0 is the live cell.")

# ENTRY offset ladder (exit pinned to month-end)
rows = []
for h in range(2, 14):
    r = spr(h); m = anchor(h)
    d = idx[(m & G).values & r.notna().values]
    rows.append(summarize(r.loc[d].values, f"entry {h} sessions before ME"))
show(rows, "3. ENTRY OFFSET LADDER (exit always the month-end close)")

# era / episodes / concentration / cost
d = idx[(A & G).values & v.values]
epi = declusters(d, 21, idx); ve = s8.loc[epi].values; w = int((ve>0).sum())
vv = s8.loc[d]
print("\n4. residual cell: day N=%d mean %+.3f%% t=%.2f | episodes N=%d mean %+.3f%% hit %.1f%% sign p=%.4f boot %.3f"
      % (len(d), 100*vv.mean(), vv.mean()/(vv.std(ddof=1)/np.sqrt(len(d))),
         len(epi), 100*ve.mean(), 100*(ve>0).mean(), sign_test(w,len(epi)), bootstrap_p_le0(ve)))
for cut in ("2018-01-01","2021-01-01"):
    m = d >= pd.Timestamp(cut)
    print("   %s+: N=%d mean %+.3f%% hit %.1f%%" % (cut[:4], m.sum(), 100*vv[m].mean(), 100*(vv[m]>0).mean()))
print("   ", cluster_note(epi, ve))
print("   drop-best-2-years:", end=" ")
byy = pd.Series(ve, index=epi).groupby(epi.year).sum()
bad = byy.nlargest(2).index
keep = ve[~np.isin(epi.year, bad)]
print("N=%d mean %+.3f%% (from %+.3f%%)" % (len(keep), 100*keep.mean(), 100*ve.mean()))
print("   cost: 2 legs x ~2.5 bps = 5 bps vs %.1f bps edge -> %.0fx" % (10000*ve.mean(), 10000*ve.mean()/5))
print("   August triggers:", ", ".join(str(x.date()) for x in d[d.month==8]),
      "| Aug mean %+.3f%%" % (100*vv[d.month==8].mean()))
print("   LIVE TLT 21d = -3.36%%, live div = +7.32pp; today's D is %s (anchor fires? %s)"
      % (idx[-1].date(), bool(A.iloc[-1])))
