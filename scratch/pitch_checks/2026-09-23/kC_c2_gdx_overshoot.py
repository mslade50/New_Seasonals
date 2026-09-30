"""C2 round 1: short GDX against beta-GLD after a one-day miner overshoot.
Residual_t = r_GDX_t - beta_{t-1} * r_GLD_t, beta = rolling 63d OLS of GDX on
GLD daily returns (PIT, through t-1). Trigger: residual >= +2.5pp.
Pre-specified sign: REVERSAL (short GDX, long beta GLD), h=1..5, lag=1.
Matched control: GDX up days of similar size (+2.5..+5%) where GLD explains
the move (residual < +1pp)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd


def build():
    px = close_panel(["GDX", "GLD"]).dropna()
    rg, rl = px["GDX"].pct_change(), px["GLD"].pct_change()
    beta = (rg.rolling(63).cov(rl) / rl.rolling(63).var()).shift(1)
    res = rg - beta * rl
    return px, rg, rl, beta, res


def pair_ret(px, beta, h, lag=1):
    return -fwd_lag(px["GDX"], h, lag) + beta * fwd_lag(px["GLD"], h, lag)


def eps(ret, mask, h, idx):
    s = idx[mask.reindex(idx, fill_value=False).values & ret.notna().values]
    e = declusters(s, h, idx)
    return e, ret.loc[e].values


if __name__ == "__main__":
    px, rg, rl, beta, res = build()
    live = pd.DataFrame({"GDX_1d": 100 * rg, "GLD_1d": 100 * rl, "beta63": beta,
                         "resid_pp": 100 * res, "raw_gap_pp": 100 * (rg - rl)}).tail(6)
    print("LIVE STATE CHECK:")
    print(live.round(3).to_string())
    trig = res >= 0.025
    bfix = float(beta[trig].median())
    print(f"\nmedian PIT beta on trigger days = {bfix:.2f}; days={int(trig.sum())}")
    for h in (1, 5):
        battery(px, trig, [("GDX", -1.0), ("GLD", round(bfix, 2))], h,
                f"C2 fixed-beta pair short GDX / long {bfix:.2f} GLD", cost_bps=4.0,
                event_kinds=("nfp", "cpi", "fomc_decision"))
    idx = px.index
    ctrl = (rg >= 0.025) & (rg <= 0.05) & (res < 0.01)
    rows = []
    for h in (1, 2, 3, 5):
        pr = pair_ret(px, beta, h)
        og = -fwd_lag(px["GDX"], h)
        e, v = eps(pr, trig, h, idx)
        em, vm = eps(pr, ctrl, h, idx)
        rows.append(summarize(v, f"h={h} PIT-beta pair CHILD N={len(e)}"))
        rows.append(summarize(vm, f"h={h} PIT-beta pair MATCHED gold-explained N={len(em)}"))
        rows.append(summarize(pr.dropna().values, f"h={h} PIT-beta pair ALL DAYS"))
        rows.append(summarize(og.loc[e].values, f"h={h} outright short GDX CHILD"))
        if h in (1, 5):
            w = int((v > 0).sum())
            print(f"  h={h} child record {w}-{len(v)-w} sign p={sign_test(w, len(v)):.4f}; "
                  f"{cluster_note(e, v)}")
            show(era_split(e, v), f"h={h} PIT pair era split")
    show(rows, "C2 PIT-beta pair vs matched control (episodes)")
