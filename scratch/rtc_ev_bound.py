"""Append a perfect-timing upper bound to rtc_structure_ev.json: value each
episode's structure at INTRINSIC at the worst close within the 63td window
(oracle exit timing, still zero time value — brackets the expiry convention)."""
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EV = os.path.join(_ROOT, "scratch", "rtc_structure_ev.json")

from rtc_structure_ev import (TENOR_TD, HAIRCUT, bs_put, skew_iv,
                              strike_for_delta)  # noqa: E402

mp = pd.read_parquet(os.path.join(_ROOT, "data", "master_prices.parquet"))
mp["date"] = pd.to_datetime(mp["date"])
def series(tkr):
    return mp.loc[mp["ticker"] == tkr, ["date", "Close"]].set_index("date")["Close"].sort_index()
spy = series("SPY")
vix3m = series("^VIX3M").reindex(spy.index).ffill()
irx = (series("^IRX") / 100.0).reindex(spy.index).ffill().fillna(0.04)
spy_np = spy.to_numpy()

d = json.load(open(EV))
T = TENOR_TD / 252.0
for name, cls in d["classes"].items():
    sp_b, tp_b = [], []
    for ds in cls["episode_dates"]:
        t = pd.Timestamp(ds)
        p = spy.index.get_loc(t)
        S = spy_np[p]
        ivb, r = vix3m.iloc[p], irx.iloc[p]
        k30 = strike_for_delta(S, T, r, ivb, -0.30, 0.40)
        k10 = strike_for_delta(S, T, r, ivb, -0.10, 0.40)
        p30 = bs_put(S, k30, T, r, skew_iv(ivb, k30 / S, 0.40))
        p10 = bs_put(S, k10, T, r, skew_iv(ivb, k10 / S, 0.40))
        debit = p30 * (1 + HAIRCUT) - p10 * (1 - HAIRCUT)
        tcost = p10 * (1 + HAIRCUT)
        S_min = spy_np[p + 1: p + TENOR_TD + 1].min()
        sp_pay = max(k30 - S_min, 0.0) - max(k10 - S_min, 0.0)
        tp_pay = max(k10 - S_min, 0.0)
        sp_b.append((sp_pay - debit) / debit)
        tp_b.append((tp_pay - tcost) / tcost)
    cls["oracle_exit_bound"] = {
        "spread_ret_cost_mean": float(np.mean(sp_b)),
        "tail_ret_cost_mean": float(np.mean(tp_b)),
        "note": "intrinsic at worst window close, perfect timing, no time value",
    }
    print(f"{name:16s} oracle spread {np.mean(sp_b):+.3f}  tail {np.mean(tp_b):+.3f}")

json.dump(d, open(EV, "w"), indent=1)
print("updated", EV)
