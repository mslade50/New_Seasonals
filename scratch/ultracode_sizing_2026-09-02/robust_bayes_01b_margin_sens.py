"""Margin-boundary sensitivity (2026-09-02): the pessimistic IBKR reading.
If IBKR applies the HIGHER of rules-based and portfolio margin to leveraged ETF
legs (long 3x 75%, short 3x 90%) and Reg-T per-share minimums to sub-$17
shorts, the requirement on 3x-fade cluster days changes materially. Recompute
requirement/NAV by day under: TIMS (stock 15 / broad 8 / small 10 / 3x 45),
TIMS+rules-3x (3x long 75, short 90), and TIMS+rules-3x+cheap-short (short
stock < $16.67: max(15%, $5/share); < $5: 100%). Feasibility multiples on max,
p99, p95 days, $750k and $632k NLV. Writes robust_bayes_01b_margin_sens.json.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from strategy_config import LEV3X_ALL  # noqa: E402
NAV, LIVE = 750_000.0, 632_000.0
led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
led = led[led["PnL_flat_750k"].notna()].copy()
BROAD = {"SPY", "QQQ", "DIA", "^GSPC", "^NDX", "VOO", "IVV", "MDY", "IJR", "IJH", "RSP", "EFA", "EEM", "VEA", "VWO"}
SMALL = {"IWM"}; lev = set(LEV3X_ALL)
short = led["Direction"].astype(str).str.lower().str.contains("short|sell")
led["notional"] = (led["Entry Price"] * led["Shares_flat"]).abs()
def r_tims(t, s, p):
    if t in lev: return 0.45
    if t in BROAD: return 0.08
    if t in SMALL: return 0.10
    return 0.15
def r_rules3x(t, s, p):
    if t in lev: return 0.90 if s else 0.75
    return r_tims(t, s, p)
def r_full(t, s, p):
    if t in lev: return 0.90 if s else 0.75
    if s and t not in BROAD and t not in SMALL:
        if p < 5: return 1.0
        if p < 16.67: return max(0.15, 5.0 / p)
    return r_tims(t, s, p)
idx = pd.bdate_range("2003-01-01", "2026-09-01")
reqs = {k: pd.Series(0.0, index=idx) for k in ["tims", "rules3x", "full"]}
fns = dict(tims=r_tims, rules3x=r_rules3x, full=r_full)
for a, b, t, s, p, n in zip(led["Entry Date"], led["Exit Date"], led["Ticker"], short, led["Entry Price"], led["notional"]):
    sl = (idx >= a) & (idx <= b)
    for k, fn in fns.items():
        reqs[k][sl] += n * fn(t, bool(s), float(p))
out = {}
for k, req in reqs.items():
    for wl, sl in [("2003+", np.ones(len(idx), bool)), ("2016+", idx >= "2016-01-01")]:
        q = req[sl] / NAV
        out[f"{k}_{wl}"] = dict(req_max=round(float(q.max()), 3), max_day=str(q.idxmax().date()), req_p99=round(float(q.quantile(.99)), 3),
                                req_p95=round(float(q.quantile(.95)), 3), m_max_750=round(float(1 / q.max()), 2), m_p99_750=round(float(1 / q.quantile(.99)), 2),
                                m_max_live=round(float(LIVE / NAV / q.max()), 2), m_p99_live=round(float(LIVE / NAV / q.quantile(.99)), 2),
                                days_over_50pct=int((q > 0.5).sum()), days_over_70pct=int((q > 0.7).sum()))
df = pd.DataFrame(out).T
pd.set_option("display.width", 250)
print(df.to_string())
# who owns the pessimistic max day
for k in ["rules3x", "full"]:
    d = (reqs[k] / NAV).idxmax()
    o = led[(led["Entry Date"] <= d) & (led["Exit Date"] >= d)]
    print(f"\n{k} max day {d.date()}: gross {o['notional'].sum()/NAV:.2f} NAV; by strategy (NAV share):")
    print((o.groupby("Strategy")["notional"].sum() / NAV).sort_values(ascending=False).round(2).head(5).to_string())
json.dump(out, open(HERE / "robust_bayes_01b_margin_sens.json", "w"), indent=1)
print("\nwrote robust_bayes_01b_margin_sens.json")
