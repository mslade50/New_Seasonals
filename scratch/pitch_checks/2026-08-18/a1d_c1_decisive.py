"""C1 round 3, DECISIVE. Two things a1/a1b/a1c left open.

BUGFIX: is_last_of_month was computed as ym != ym.shift(-1), which flags the
FINAL ROW of the cache (2026-08-17) as a month end. That injected one phantom
trigger (2026-08-04). Fixed here by excluding the last row.

TEST 1 (definition): the mechanism is "rebalance sized by the DIVERGENCE",
which needs BOTH legs. Price each half of the gate separately at matched N.
If one half beats the divergence, the divergence construction is decorative.
TEST 2 (attribution of the final session, which carries half the edge):
is the month's last session an unconditional turn-of-month fact?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np, pandas as pd

LAG, H = 1, 8
raw = load_prices(["SPY", "TLT"])
spy, tlt = raw["SPY"]["Close"], raw["TLT"]["Close"]
r21s, r21t = spy.pct_change(21), tlt.pct_change(21)
idx = spy.index.intersection(tlt.index)
px = pd.DataFrame({"SPY": spy.reindex(idx), "TLT": tlt.reindex(idx)}).dropna()
idx = px.index
div = (r21s.reindex(idx) - r21t.reindex(idx))
S21, T21 = r21s.reindex(idx), r21t.reindex(idx)

ymv = pd.Series(idx.year*100 + idx.month, index=idx)
is_last = ymv.ne(ymv.shift(-1))
is_last.iloc[-1] = False          # BUGFIX: cache end is not a month end
pos = pd.Series(range(len(idx)), index=idx)
t8 = pos + LAG + H
A = pd.Series(False, index=idx)
ok = t8 < len(idx)
A.loc[idx[ok.values]] = is_last.values[t8[ok.values].values]
print("anchor A days after bugfix: %d (was 289)" % A.sum())

rt, rs = fwd_lag(px["TLT"], H, LAG), fwd_lag(px["SPY"], H, LAG)
spread = rt - rs
v = spread.notna()

def cell(m, lab, r=spread):
    d = idx[m.values & v.values]
    out = summarize(r.loc[d].values, f"{lab} (N={len(d)})")
    return out, d

# --- TEST 1: which half of the gate carries it, at MATCHED N ---
rows = []
specs = [("gate OFF", pd.Series(True, index=idx)),
         ("div >= +5.0pp", (div >= 0.050).fillna(False)),
         ("div >= +6.0pp", (div >= 0.060).fillna(False)),
         ("div >= +7.32pp (LIVE)", (div >= 0.0732).fillna(False)),
         ("TLT21d <= -3.36% (LIVE half)", (T21 <= -0.0336).fillna(False)),
         ("TLT21d <= -2.5%", (T21 <= -0.025).fillna(False)),
         ("TLT21d <= -4.5%", (T21 <= -0.045).fillna(False)),
         ("SPY21d >= +3.95% (LIVE half)", (S21 >= 0.0395).fillna(False)),
         ("SPY21d >= +5.5%", (S21 >= 0.055).fillna(False))]
for lab, g in specs:
    r_, d = cell(A & g, lab)
    r_["TLTleg_pct"] = round(100*rt.loc[d].mean(), 3)
    r_["shortSPYleg_pct"] = round(-100*rs.loc[d].mean(), 3)
    rows.append(r_)
show(rows, "1. WHICH HALF? spread at the month-end anchor (bugfixed). matched-N pairs: div>=6pp N~39 vs TLT<=-3.36% N~37")

# --- TEST 2: the final session, conditional vs unconditional ---
r7 = fwd_lag(px["TLT"], 7, LAG) - fwd_lag(px["SPY"], 7, LAG)
G = (div >= 0.05).fillna(False)
d = idx[(A & G).values & v.values & r7.notna().values]
marg_cond = (spread.loc[d] - r7.loc[d])
# unconditional: the month's LAST session spread return, every month
last_days = idx[is_last.values]
lp = pos.reindex(last_days).dropna().astype(int)
one = pd.Series(px["TLT"].pct_change().values, index=idx) - pd.Series(px["SPY"].pct_change().values, index=idx)
uncond_last = one.reindex(last_days).dropna()
show([summarize(marg_cond.values, f"final session, div-gated (N={len(marg_cond)})"),
      summarize(uncond_last.values, f"month's last session, ALL months (N={len(uncond_last)})"),
      summarize(one.dropna().values, "every session, all history")],
     "2. is the final session's +0.45% a GATED fact or the unconditional turn-of-month?")

# --- TEST 3: TLT-oversold gate WITHOUT the month-end anchor (general reversion?) ---
Tg = (T21 <= -0.0336).fillna(False)
rows = []
for lab, m in (("ME anchor x TLT21d<=-3.36%", A & Tg),
               ("NOT ME anchor x TLT21d<=-3.36%", (~A) & Tg),
               ("ME anchor, gate OFF", A),
               ("ALL days", pd.Series(True, index=idx))):
    r_, _ = cell(m, lab)
    rows.append(r_)
show(rows, "3. is the TLT-oversold conditioner month-end-specific?")

# --- TEST 4: bugfixed headline cell, episodes + era + boot ---
d = idx[(A & G).values & v.values]
epi = declusters(d, 21, idx); ve = spread.loc[epi].values
w = int((ve > 0).sum())
print("\n4. BUGFIXED headline (spread, ME anchor, div>=+5pp): day-level N=%d mean %+.3f%% t=%.2f"
      % (len(d), 100*spread.loc[d].mean(),
         spread.loc[d].mean()/(spread.loc[d].std(ddof=1)/np.sqrt(len(d)))))
print("   episodes N=%d mean %+.3f%% hit %.1f%% sign p=%.4f boot P(<=0)=%.3f"
      % (len(epi), 100*ve.mean(), 100*(ve>0).mean(), sign_test(w, len(epi)), bootstrap_p_le0(ve)))
vv = spread.loc[d]
for cut in ("2018-01-01", "2021-01-01"):
    m = d >= pd.Timestamp(cut)
    print("   %s+: N=%d mean %+.3f%% hit %.1f%%" % (cut[:4], m.sum(), 100*vv[m].mean(), 100*(vv[m]>0).mean()))
print("   by year:", pd.Series(vv.values, index=d).groupby(d.year).mean().mul(100).round(2).to_dict())
print("   ", cluster_note(epi, ve))
