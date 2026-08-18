"""C1 follow-up. The only cell in a1 the gate HELPED was the TLT-SPY spread
(+0.941% gated vs +0.291% gate-off, anchor A). Is that a MONTH-END flow fact
or just "high divergence mean-reverts on any day"? Attribution both ways.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np, pandas as pd

H, LAG = 8, 1
raw = load_prices(["SPY", "TLT"])
spy, tlt = raw["SPY"]["Close"], raw["TLT"]["Close"]
r21s, r21t = spy.pct_change(21), tlt.pct_change(21)
idx = spy.index.intersection(tlt.index)
px = pd.DataFrame({"SPY": spy.reindex(idx), "TLT": tlt.reindex(idx)}).dropna()
idx = px.index
div = (r21s.reindex(idx) - r21t.reindex(idx))

ym = pd.Series(idx.year*100 + idx.month, index=idx)
is_last = pd.Series(ym.values, index=idx).ne(pd.Series(ym.values, index=idx).shift(-1))
pos = pd.Series(range(len(idx)), index=idx)
tgt = pos + LAG + H
exit_is_me = pd.Series(False, index=idx)
ok = tgt < len(idx)
exit_is_me.loc[idx[ok.values]] = is_last.values[tgt[ok.values].values]

rt, rs = fwd_lag(px["TLT"], H, LAG), fwd_lag(px["SPY"], H, LAG)
spread = rt - rs
valid = spread.notna()
G = (div >= 0.05).fillna(False)

def c(mask, lab, r=spread):
    d = idx[mask.values & valid.values]
    return summarize(r.loc[d].values, f"{lab} (N={len(d)})"), d

rows = []
for lab, m in (("ALL days, gate OFF", pd.Series(True, index=idx)),
               ("ALL days, div>=+5pp", G),
               ("month-end anchor, gate OFF", exit_is_me),
               ("month-end anchor, div>=+5pp", exit_is_me & G),
               ("NOT month-end anchor, div>=+5pp", (~exit_is_me) & G)):
    r, _ = c(m, lab)
    rows.append(r)
show(rows, "1. SPREAD long TLT / short SPY, h=8 lag=1: is the gate month-end-specific?")

# declustered episodes for each
for lab, m in (("ALL days div>=+5pp", G), ("month-end x div>=+5pp", exit_is_me & G)):
    d = idx[m.values & valid.values]
    epi = declusters(d, 21, idx)
    v = spread.loc[epi].values
    w = int((v > 0).sum())
    print("\n%s: declustered(21td) N=%d mean %+.3f%% hit %.1f%% sign p=%.4f boot P(<=0)=%.3f"
          % (lab, len(epi), 100*v.mean(), 100*(v>0).mean(), sign_test(w, len(epi)),
             bootstrap_p_le0(v)))
    print("  ", cluster_note(epi, v))

# era split, spread, month-end x gate
d = idx[(exit_is_me & G).values & valid.values]
v = spread.loc[d].values
show([summarize(v[d < pd.Timestamp("2018-01-01")], "pre-2018"),
      summarize(v[d >= pd.Timestamp("2018-01-01")], "2018+"),
      summarize(v[d >= pd.Timestamp("2021-01-01")], "2021+")],
     "2. SPREAD era split, month-end x div>=+5pp")
print("  by year:", pd.Series(v, index=d).groupby(d.year).mean().mul(100).round(2).to_dict())

# offset ladder on the SPREAD (the 8-for-8 killer)
rows = []
for k in range(0, 16):
    m = pd.Series(False, index=idx)
    okk = (tgt + k) < len(idx)
    m.loc[idx[okk.values]] = is_last.values[(tgt + k)[okk.values].values]
    d = idx[(m & G).values & valid.values]
    rows.append(summarize(spread.loc[d].values, f"exit k={k} before ME, div>=+5pp"))
show(rows, "3. SPREAD offset ladder (k=0 = the live cell). Spike or plateau?")

# does the gate itself just proxy 'SPY overbought'? decompose
rows = []
r21s_a = r21s.reindex(idx)
r21t_a = r21t.reindex(idx)
for lab, m in (("SPY 21d >= +3.95% only", (r21s_a >= 0.0395).fillna(False)),
               ("TLT 21d <= -3.36% only", (r21t_a <= -0.0336).fillna(False)),
               ("BOTH (live state)", ((r21s_a >= 0.0395) & (r21t_a <= -0.0336)).fillna(False))):
    d = idx[(exit_is_me & m).values & valid.values]
    rows.append(summarize(spread.loc[d].values, f"ME x {lab}"))
    rows.append(summarize(rt.loc[d].values, f"  -> TLT leg only"))
show(rows, "4. which HALF of the divergence carries it (month-end anchor)?")
