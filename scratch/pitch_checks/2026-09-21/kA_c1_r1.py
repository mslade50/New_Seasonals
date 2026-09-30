"""kA c1 round 1 - W23 ARMED: long XLU when XLU r21 <= 5 AND TLT r21 < 25.

Definitions re-used verbatim from scratch/pitch_checks/2026-08-25/b1_c3_xlu_washout_tlt_fine.py
(pitch_lab pct_rank on 21d returns, 252 lookback; panel from TLT inception 2002-07-30;
episodes declustered at 21 td). Source re-run on today's data: kA_c1_src_rerun_out.txt.

Round 1: existence vs all-days / own drift / local control; N, worst, era; cost;
tail (NFP/FOMC/CPI in hold); book overlap; live readings; hedged row (XLU - beta*TLT).
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change
import numpy as np
import pandas as pd

TK = ["XLU", "TLT", "IEF", "SPY", "^TNX"]
px = close_panel(TK)
px = px[px.index >= "2002-07-30"]
idx = px.index

rk21 = {t: pct_rank(px[t], 21) for t in TK}
wash = rk21["XLU"] <= 5
tlt_hit = rk21["TLT"] < 25
joint = (wash & tlt_hit).reindex(idx, fill_value=False)

print("=== live readings, last bar", idx[-1].date(), "===")
for t in ["XLU", "TLT", "SPY"]:
    print(f"  {t} r5 {pct_rank(px[t],5).iloc[-1]:.1f}  r21 {rk21[t].iloc[-1]:.2f}  "
          f"r63 {pct_rank(px[t],63).iloc[-1]:.1f}")
sma200 = rolling_on_valid(px["SPY"], lambda x: x.rolling(200).mean())
above200 = px["SPY"] > sma200
print(f"  SPY vs 200d {100*(px['SPY'].iloc[-1]/sma200.iloc[-1]-1):+.2f}%  midterm {idx[-1].year%4==2}")
print(f"  joint mask live: {bool(joint.iloc[-1])}; joint days in last 30 sessions: "
      f"{[str(d.date()) for d in idx[-30:][joint.iloc[-30:].values]]}")
print(f"  wash days in last 30 sessions: {[str(d.date()) for d in idx[-30:][wash.reindex(idx, fill_value=False).iloc[-30:].values]]}")
print(f"  TLT r21 last 10: {[round(v,1) for v in rk21['TLT'].iloc[-10:].values]}")

# ---------------------------------------------------------------- battery
for h in (3, 5):
    battery(px, joint, [("XLU", 1.0)], h, f"c1 long XLU, joint (gap 21)", cost_bps=3.0,
            min_gap=21, event_kinds=("nfp",),
            variants={"XLU r21<=5 alone": wash,
                      "XLU r21<=5 & TLT r21>=25 (complement)": wash & ~tlt_hit,
                      "XLU r21<=3 & TLT<25": (rk21["XLU"] <= 3) & tlt_hit,
                      "XLU r21<=10 & TLT<25": (rk21["XLU"] <= 10) & tlt_hit,
                      "XLU r21<=5 & TLT<15": wash & (rk21["TLT"] < 15),
                      "XLU r21<=5 & TLT<35": wash & (rk21["TLT"] < 35)})

# FOMC / CPI in hold at h=5 (entry 09-21 close -> exit 09-28; neither in the live window)
ret5 = vehicle_ret(px, [("XLU", 1.0)], 5)
d = idx[joint.values].intersection(ret5.dropna().index)
epi = declusters(d, 21, idx)
for k in (("fomc_decision",), ("cpi",)):
    fl = event_in_window(epi, idx, 5, 1, k)
    show([summarize(ret5.loc[epi].values[fl], f"{k[0]} IN (N={fl.sum()})"),
          summarize(ret5.loc[epi].values[~fl], f"{k[0]} OUT (N={(~fl).sum()})")],
         f"h=5 {k[0]}-in-hold split")

# ---------------------------------------------------------------- hedged row
print("\n=== hedged row: XLU - beta_PIT * TLT (trailing 252d daily beta, lag-free at signal) ===")
dr = px[["XLU", "TLT"]].pct_change(fill_method=None)
beta = dr["XLU"].rolling(252).cov(dr["TLT"]) / dr["TLT"].rolling(252).var()
print(f"  beta today {beta.iloc[-1]:+.3f}; full-history median {beta.median():+.3f}")
for h in (3, 5):
    rx = vehicle_ret(px, [("XLU", 1.0)], h)
    rt = vehicle_ret(px, [("TLT", 1.0)], h)
    hed = rx - beta * rt
    d = idx[joint.values].intersection(hed.dropna().index)
    e = declusters(d, 21, idx)
    v = hed.loc[e].values
    base = hed.dropna()
    r = summarize(v, f"h={h} hedged episodes")
    r["all_days_pct"] = 100 * base.mean()
    r["sign_p"] = sign_test(int((v > 0).sum()), len(v))
    r["TLT_leg_pct"] = 100 * rt.loc[e].mean()
    r["XLU_raw_pct"] = 100 * rx.loc[e].mean()
    show([r])

# ---------------------------------------------------------------- book overlap
print("\n=== book overlap ===")
t = pd.read_parquet(Path(__file__).resolve().parents[3] / "data" / "backtest_trades_full.parquet")
xl = t[t["Ticker"] == "XLU"]
print("  ledger XLU trades by strategy:", xl["Strategy"].value_counts().to_dict())
util = ["XLU", "CMS", "PEG", "EXC", "SRE", "PCG", "D", "ETR", "NEE", "SO", "DUK", "AEP", "XEL", "ED", "WEC", "ES"]
rec = t[(t["Ticker"].isin(util)) & (pd.to_datetime(t["Signal Date"]) >= "2026-08-15")]
print("  utility-name ledger rows signalled since 2026-08-15:")
print(rec[["Strategy", "Ticker", "Signal Date", "Entry Date", "Exit Date", "Exit Type", "R_Multiple"]].to_string(index=False)
      if len(rec) else "   (none)")
