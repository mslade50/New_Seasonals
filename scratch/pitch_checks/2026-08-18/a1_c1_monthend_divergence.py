"""C1 - month-end rebalance flow sized by the stock/bond divergence.
Live: SPY 21d +3.95%, TLT 21d -3.36%, div +7.32pp (89.2 pctile since 2000).
Signal close D=2026-08-17, entry MOC D+1=2026-08-18, exit month-end 2026-08-31
= h 8 sessions after the entry close.

Design: h FIXED at 8 with lag=1 so every cell is the SAME trade shape.
Two anchors: (A) exit lands on the month's last session ("8 sessions to
month-end"), (B) tdom(D)==12. Gate = divergence. THE TEST IS ATTRIBUTION.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np, pandas as pd

H, LAG = 8, 1
px_raw = load_prices(["SPY", "TLT"])
spy, tlt = px_raw["SPY"]["Close"], px_raw["TLT"]["Close"]

# 21d returns on each instrument's OWN series, then align
r21_spy = spy.pct_change(21)
r21_tlt = tlt.pct_change(21)
idx = spy.index.intersection(tlt.index)
div = (r21_spy.reindex(idx) - r21_tlt.reindex(idx)).dropna()

px = pd.DataFrame({"SPY": spy.reindex(idx), "TLT": tlt.reindex(idx)}).dropna()
idx = px.index
div = div.reindex(idx)

# trailing-1260d percentile of the divergence (no full-sample lookahead)
div_rk = div.rolling(1260).rank(pct=True) * 100.0
print("live div = %.2fpp | trailing-1260d rank %.1f | full-sample rank %.1f"
      % (100*div.iloc[-1], div_rk.iloc[-1], 100*(div < div.iloc[-1]).mean()))

# trading-day-of-month and sessions-to-month-end
ym = pd.Series(idx.year * 100 + idx.month, index=idx)
tdom = ym.groupby(ym.values).cumcount() + 1
is_last_of_month = pd.Series(ym.values, index=idx).ne(pd.Series(ym.values, index=idx).shift(-1))
pos = pd.Series(range(len(idx)), index=idx)

# anchor A: exit close (pos+LAG+H) is the month's last session
exit_is_me = pd.Series(False, index=idx)
tgt = pos + LAG + H
ok = tgt < len(idx)
exit_is_me.loc[idx[ok.values]] = is_last_of_month.values[tgt[ok.values].values]
# anchor B
anchor_tdom12 = pd.Series(tdom.values == 12, index=idx)

print("anchor A (exit == month-end) days: %d | anchor B (tdom12) days: %d | overlap %d"
      % (exit_is_me.sum(), anchor_tdom12.sum(), (exit_is_me & anchor_tdom12).sum()))

ret_tlt = fwd_lag(px["TLT"], H, LAG)
ret_spy = fwd_lag(px["SPY"], H, LAG)
ret_spread = ret_tlt - ret_spy
valid = ret_tlt.notna()

def cell(mask, label, ret=ret_tlt):
    d = idx[mask.values & valid.values]
    return summarize(ret.loc[d].values, f"{label} (N={len(d)})"), d

GATES = {
    "gate OFF (all anchors)": pd.Series(True, index=idx),
    "div >= +5pp": div >= 0.05,
    "div >= +7.32pp (live)": div >= 0.0732,
    "div rank1260 >= 85": div_rk >= 85,
    "div rank1260 >= 90": div_rk >= 90,
}

for aname, anchor in (("A: exit==month-end", exit_is_me), ("B: tdom==12", anchor_tdom12)):
    rows = []
    for g, gm in GATES.items():
        r, _ = cell(anchor & gm.fillna(False), g)
        rows.append(r)
    show(rows, f"1. LONG TLT h={H} lag=1 | anchor {aname} | GATE ATTRIBUTION")

# --- dose response on anchor A ---
a = exit_is_me & valid
dd = idx[a.values]
dv = div.loc[dd]
rr = ret_tlt.loc[dd]
qs = [-np.inf, -0.05, -0.02, 0.02, 0.05, np.inf]
lbl = ["div < -5pp", "-5..-2", "-2..+2", "+2..+5", "div >= +5pp"]
rows = []
for i in range(len(lbl)):
    m = (dv > qs[i]) & (dv <= qs[i+1])
    rows.append(summarize(rr[m.values].values, f"{lbl[i]} (N={int(m.sum())})"))
show(rows, "2. DOSE RESPONSE in the divergence, anchor A (monotone?)")
# rank correlation
ok2 = dv.notna() & rr.notna()
print("  spearman(div, TLT h8 ret) on anchor A = %.3f (N=%d)"
      % (dv[ok2].rank().corr(rr[ok2].rank()), int(ok2.sum())))

# --- month-of-year control, same trade shape ---
rows = []
for mo in range(1, 13):
    m = (idx.month == mo) & valid.values
    rows.append(summarize(ret_tlt[m].values, f"month {mo:02d}"))
show(rows, "3a. CONTROL: TLT h=8 lag=1 by MONTH-OF-YEAR (all days)")

# --- trading-day-of-month control ---
rows = []
for t in range(1, 22):
    m = (tdom.values == t) & valid.values
    if m.sum() >= 20:
        rows.append(summarize(ret_tlt[m].values, f"tdom {t:02d}"))
show(rows, "3b. CONTROL: TLT h=8 lag=1 by TRADING-DAY-OF-MONTH (all days)")

# --- offset / placebo ladder around month-end ---
rows = []
for k in range(1, 16):
    tgt2 = pos + LAG + H
    # anchor: exit is k sessions BEFORE the month's last session
    m = pd.Series(False, index=idx)
    okk = (tgt2 + k) < len(idx)
    m.loc[idx[okk.values]] = is_last_of_month.values[(tgt2 + k)[okk.values].values]
    for g in ("gate OFF", "div>=+5pp"):
        gm = pd.Series(True, index=idx) if g == "gate OFF" else (div >= 0.05).fillna(False)
        d = idx[(m & gm).values & valid.values]
        r = summarize(ret_tlt.loc[d].values, f"exit k={k} before ME | {g}")
        rows.append(r)
show(rows, "4. PLACEBO LADDER: exit k sessions BEFORE month-end (k=0 is the live cell, see #1)")

# --- era split on the live cell ---
d_live = idx[(exit_is_me & (div >= 0.05).fillna(False)).values & valid.values]
v_live = ret_tlt.loc[d_live].values
show([summarize(v_live[d_live < pd.Timestamp("2018-01-01")], "pre-2018"),
      summarize(v_live[(d_live >= pd.Timestamp("2018-01-01"))], "2018+"),
      summarize(v_live[d_live >= pd.Timestamp("2021-01-01")], "2021+")],
     "5. ERA SPLIT, anchor A x div>=+5pp (the parent dies at 2018+/2021+)")
print("  dates:", ", ".join(str(d.date()) for d in d_live))
print("  by year:", pd.Series(v_live, index=d_live).groupby(d_live.year).mean().mul(100).round(2).to_dict())

# gate-off era split for contrast
d_off = idx[exit_is_me.values & valid.values]
v_off = ret_tlt.loc[d_off].values
show([summarize(v_off[d_off < pd.Timestamp("2018-01-01")], "pre-2018"),
      summarize(v_off[d_off >= pd.Timestamp("2018-01-01")], "2018+"),
      summarize(v_off[d_off >= pd.Timestamp("2021-01-01")], "2021+")],
     "5b. ERA SPLIT, anchor A GATE OFF (contrast)")

# --- the SPY leg and the spread, priced separately ---
rows = []
for g, gm in (("gate OFF", pd.Series(True, index=idx)), ("div>=+5pp", (div >= 0.05).fillna(False))):
    d = idx[(exit_is_me & gm).values & valid.values]
    rows.append(summarize(ret_tlt.loc[d].values, f"LONG TLT | {g}"))
    rows.append(summarize((-ret_spy).loc[d].values, f"SHORT SPY | {g}"))
    rows.append(summarize(ret_spread.loc[d].values, f"SPREAD TLT-SPY | {g}"))
show(rows, "6. LEG PRICING (the 2026-08-12 negative-beta trap)")
d = idx[(exit_is_me & (div >= 0.05).fillna(False)).values & valid.values]
a_, b_ = ret_tlt.loc[d].values, ret_spy.loc[d].values
print("  corr(TLT h8, SPY h8) on the gated cell = %.3f  beta = %.3f (N=%d)"
      % (np.corrcoef(a_, b_)[0,1], np.polyfit(b_, a_, 1)[0], len(d)))
