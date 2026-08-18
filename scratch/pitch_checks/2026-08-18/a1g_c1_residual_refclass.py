"""C1 round 6. The residual left after killing the divergence construction:
long TLT / short SPY into the month-end close when TLT is 21d OVERSOLD.
It is LIVE today (TLT 21d -3.36%, entry 2026-08-18, exit 2026-08-31, h=9).
Not C1's mechanism. Price it as its own object before calling it a near-miss:
duration reference class (IEF/LQD), the SPY leg's marginal value, and the
local +/-126td control.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np, pandas as pd

LAG, H = 1, 9
raw = load_prices(["SPY", "TLT", "IEF", "LQD"])
idx = raw["SPY"]["Close"].index
for t in ("TLT","IEF","LQD"): idx = idx.intersection(raw[t]["Close"].index)
px = pd.DataFrame({t: raw[t]["Close"].reindex(idx) for t in ("SPY","TLT","IEF","LQD")}).dropna()
idx = px.index
ymv = pd.Series(idx.year*100+idx.month, index=idx)
is_last = ymv.ne(ymv.shift(-1)); is_last.iloc[-1] = False
pos = pd.Series(range(len(idx)), index=idx)
t9 = pos + LAG + H
A = pd.Series(False, index=idx); ok = t9 < len(idx)
A.loc[idx[ok.values]] = is_last.values[t9[ok.values].values]
print("anchor fires on the last row (2026-08-17)?", bool(A.iloc[-1]))

rT = fwd_lag(px["TLT"], H, LAG); rS = fwd_lag(px["SPY"], H, LAG)
rI = fwd_lag(px["IEF"], H, LAG); rL = fwd_lag(px["LQD"], H, LAG)
G = (px["TLT"].pct_change(21) <= -0.025).fillna(False)
v = (rT - rS).notna()
d = idx[(A & G).values & v.values]
loc = local_control(idx[v.values], d, 126)

# 1. reference class down the duration ladder + the local control
rows = []
for lab, r in (("TLT - SPY (the cell)", rT-rS), ("TLT only", rT), ("SHORT SPY only", -rS),
               ("IEF - SPY (lower duration)", rI-rS), ("IEF only", rI),
               ("LQD - SPY (credit+duration)", rL-rS)):
    rows.append(summarize(r.loc[d].values, f"{lab} | TRIGGER (N={len(d)})"))
    rows.append(summarize(r.loc[loc].values, f"   ctrl local +/-126td"))
show(rows, "1. duration reference class + local control, h=9 lag=1")

# 2. does the SHORT SPY leg pay for its risk?
a, b = rT.loc[d].values, rS.loc[d].values
print("2. corr(TLT,SPY) on triggers = %+.3f | beta = %+.3f" % (np.corrcoef(a,b)[0,1], np.polyfit(b,a,1)[0]))
for lab, x in (("TLT only", a), ("spread", a-b)):
    print("   %-10s mean %+.3f%% sd %.3f%% ratio %.3f hit %.1f%% worst %+.2f%%"
          % (lab, 100*x.mean(), 100*x.std(ddof=1), x.mean()/x.std(ddof=1), 100*(x>0).mean(), 100*x.min()))

# 3. threshold plateau at h=9 + era + episodes
T21 = px["TLT"].pct_change(21)
rows = []
for thr in [99, -0.015, -0.02, -0.025, -0.0336, -0.045, -0.06]:
    g = pd.Series(True, index=idx) if thr==99 else (T21 <= thr).fillna(False)
    dd = idx[(A & g).values & v.values]
    rows.append(summarize((rT-rS).loc[dd].values, "gate OFF" if thr==99 else f"TLT21d <= {100*thr:+.2f}%"))
show(rows, "3. threshold plateau at the CORRECT h=9 (live = -3.36%)")
sp = (rT-rS).loc[d]
epi = declusters(d, 21, idx); ve = (rT-rS).loc[epi].values; w = int((ve>0).sum())
print("4. episodes N=%d mean %+.3f%% hit %.1f%% sign p=%.4f boot P(<=0)=%.3f"
      % (len(epi), 100*ve.mean(), 100*(ve>0).mean(), sign_test(w,len(epi)), bootstrap_p_le0(ve)))
for cut in ("2018-01-01","2021-01-01"):
    m = d >= pd.Timestamp(cut); print("   %s+: N=%d mean %+.3f%% hit %.1f%%" % (cut[:4], m.sum(), 100*sp[m].mean(), 100*(sp[m]>0).mean()))
print("  ", cluster_note(epi, ve))
print("   worst episode %+.2f%% on %s" % (100*ve.min(), epi[int(np.argmin(ve))].date()))
