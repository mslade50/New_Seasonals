"""Risk-architect lens, part 2: does the book's own PnL path carry forecast
content? (Kaminski-Lo: a drawdown-triggered cut helps only if PnL is
positively autocorrelated.) Also replays the pod-shop cut template and a
Grossman-Zhou continuous de-lever on the flat daily series, at equal
realized vol, so the cost of each is a number.

Outputs risk_arch_autocorr_dd.json.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from risk_arch_common import NAV, dump, load_spy, load_strategy_daily, sessions


def acf(x: np.ndarray, lags: int) -> np.ndarray:
    x = x - x.mean()
    d = (x ** 2).sum()
    return np.array([(x[:-k] * x[k:]).sum() / d for k in range(1, lags + 1)])


def ljung_box(r: np.ndarray, lags: int) -> tuple[float, float]:
    from math import lgamma
    n = len(r)
    a = acf(r, lags)
    q = n * (n + 2) * sum(a[k - 1] ** 2 / (n - k) for k in range(1, lags + 1))
    # chi2 survival via regularized gamma (no scipy dependency guaranteed)
    try:
        from scipy.stats import chi2
        p = float(chi2.sf(q, lags))
    except Exception:
        p = float("nan")
    return float(q), p


def variance_ratio(r: np.ndarray, q: int) -> float:
    n = len(r)
    mu = r.mean()
    var1 = ((r - mu) ** 2).sum() / (n - 1)
    rq = np.array([r[i:i + q].sum() for i in range(0, n - q + 1)])
    varq = ((rq - q * mu) ** 2).sum() / (n - q)
    return float(varq / (q * var1))


def block_bootstrap_stat(r: np.ndarray, fn, n_boot=1000, block=10, seed=0):
    rng = np.random.default_rng(seed)
    n = len(r)
    vals = []
    for _ in range(n_boot):
        out = np.empty(n)
        i = 0
        while i < n:
            L = rng.geometric(1 / block)
            s = rng.integers(0, n)
            take = min(L, n - i)
            idx = (s + np.arange(take)) % n
            out[i:i + take] = r[idx]
            i += take
        vals.append(fn(out))
    return np.array(vals)


def replay(r: pd.Series, expo: pd.Series) -> dict:
    """Overlay exposure (lag-1 applied) and compare at equal realized vol."""
    e = expo.shift(1).fillna(1.0)
    o = r * e
    scale = r.std() / o.std() if o.std() > 0 else 1.0
    ov = o * scale
    return dict(exposure_mean=float(e.mean()), pnl_raw_pct=float(o.sum() * 100), pnl_base_pct=float(r.sum() * 100),
                sharpe_base=float(r.mean() / r.std() * np.sqrt(252)), sharpe_overlay=float(o.mean() / o.std() * np.sqrt(252)),
                maxdd_base=float(mdd(r) * 100), maxdd_overlay_eqvol=float(mdd(ov) * 100),
                pnl_eqvol_pct=float(ov.sum() * 100), worst_day_base=float(r.min() * 100), worst_day_overlay_eqvol=float(ov.min() * 100))


def mdd(r: pd.Series) -> float:
    eq = r.cumsum()
    return float((eq - eq.cummax()).min())


def main() -> None:
    strat, total = load_strategy_daily()
    spy = load_spy()
    idx = sessions(strat, spy)
    r = (total.reindex(idx) / NAV).astype(float)
    out = {}
    for label, start in (("2003+", "2003-01-01"), ("2010+", "2010-01-01"), ("2016-07+", "2016-07-20")):
        x = r[r.index >= start]
        a = acf(x.values, 20)
        se = 1 / np.sqrt(len(x))
        q5, p5 = ljung_box(x.values, 5)
        q20, p20 = ljung_box(x.values, 20)
        blk = dict(n=int(len(x)), acf_1_5=[float(v) for v in a[:5]], acf_6_20_mean=float(a[5:].mean()),
                   bartlett_se=float(se), ljung_box_q5=q5, p5=p5, ljung_box_q20=q20, p20=p20,
                   vr5=variance_ratio(x.values, 5), vr21=variance_ratio(x.values, 21), vr63=variance_ratio(x.values, 63))
        # squared-return persistence (vol clustering) for contrast
        blk["acf_sq_1_5"] = [float(v) for v in acf((x.values - x.mean()) ** 2, 5)]
        # conditional forward returns
        eq = x.cumsum(); dd = eq - eq.cummax()
        fwd21 = x[::-1].rolling(21).sum()[::-1].shift(-1)
        t21 = x.rolling(21).sum()
        cond = {}
        for name, mask in (("dd_5_10pct", (dd <= -0.05) & (dd > -0.10)), ("dd_gt_10pct", dd <= -0.10),
                           ("dd_gt_5pct", dd <= -0.05), ("t21_bottom_decile", t21 <= t21.quantile(0.10)),
                           ("t21_top_decile", t21 >= t21.quantile(0.90)), ("at_high", dd >= -0.005)):
            f = fwd21[mask].dropna()
            base = fwd21.dropna()
            if len(f) > 10:
                # cluster by non-overlapping 21d blocks for a rough t
                fb = f.iloc[::21]
                cond[name] = dict(days=int(len(f)), fwd21_mean_pct=float(f.mean() * 100), base_fwd21_mean_pct=float(base.mean() * 100),
                                  t_nonoverlap=float(fb.mean() / (fb.std() / np.sqrt(len(fb)))) if len(fb) > 3 else None,
                                  n_nonoverlap=int(len(fb)))
        # losing-streak conditioning
        streak = {}
        neg = (x < 0).astype(int)
        run = neg.groupby((neg != neg.shift()).cumsum()).cumsum()
        for k in (3, 4, 5):
            m = (run >= k)
            nxt = x.shift(-1)[m].dropna()
            nxt5 = x[::-1].rolling(5).sum()[::-1].shift(-1)[m].dropna()
            streak[f"after_{k}_down_days"] = dict(n=int(len(nxt)), next_day_bps=float(nxt.mean() * 1e4), next5_bps=float(nxt5.mean() * 1e4),
                                                  base_day_bps=float(x.mean() * 1e4))
        blk["conditional_fwd"] = cond
        blk["streaks"] = streak
        out[label] = blk

    # pod-shop template and Grossman-Zhou replays, 2010+ and 2016-07+
    reps = {}
    for label, start in (("2010+", "2010-01-01"), ("2016-07+", "2016-07-20")):
        x = r[r.index >= start]
        eq = x.cumsum(); peak = eq.cummax(); dd = eq - peak
        rules = {}
        # pod template: 0.5x while dd from peak <= -5%, 0 while <= -7.5%, restore at new high
        e = pd.Series(1.0, index=x.index)
        state = 1.0
        for i, (d, v) in enumerate(dd.items()):
            if v >= -1e-9:
                state = 1.0
            elif v <= -0.075:
                state = 0.0
            elif v <= -0.05:
                state = min(state, 0.5)
            e.iloc[i] = state
        rules["pod_cut_50_at_5_stop_at_7.5"] = replay(x, e)
        # milder: 0.5x while dd <= -5% until new high
        e2 = pd.Series(1.0, index=x.index); state = 1.0
        for i, v in enumerate(dd.values):
            state = 1.0 if v >= -1e-9 else (0.5 if v <= -0.05 else state)
            e2.iloc[i] = state
        rules["cut_50_at_5_until_new_high"] = replay(x, e2)
        # Grossman-Zhou: exposure = (W - alpha*M)/(W*(1-alpha)) with alpha = 0.8 (floor 20% below peak) and 0.9
        for alpha in (0.8, 0.9):
            W = 1 + eq; M = 1 + peak
            e3 = ((W - alpha * M) / (W * (1 - alpha))).clip(0, 1.5)
            rules[f"grossman_zhou_alpha_{alpha}"] = replay(x, e3)
        # ratchet UP on new highs (compounding proxy): exposure = M/M0 capped 2x
        reps[label] = rules
    out["drawdown_rule_replays"] = reps

    # bootstrap of ACF under the block structure to show what a 10-block bootstrap preserves
    x = r[r.index >= "2010-01-01"].values
    bs = block_bootstrap_stat(x, lambda z: acf(z, 1)[0], n_boot=300, block=10)
    out["bootstrap_acf1_block10_2010+"] = dict(mean=float(bs.mean()), sd=float(bs.std()))
    dump(out, "risk_arch_autocorr_dd.json")

    for k, v in out.items():
        if k in ("drawdown_rule_replays", "bootstrap_acf1_block10_2010+"):
            continue
        print(f"\n== {k}: n={v['n']} acf1-5={[round(a,3) for a in v['acf_1_5']]} (se {v['bartlett_se']:.3f}) acf6-20 mean {v['acf_6_20_mean']:.3f}; LB Q5 {v['ljung_box_q5']:.1f} p={v['p5']:.3f}; Q20 {v['ljung_box_q20']:.1f} p={v['p20']:.3f}; VR5 {v['vr5']:.2f} VR21 {v['vr21']:.2f} VR63 {v['vr63']:.2f}; sq-acf {[round(a,3) for a in v['acf_sq_1_5']]}")
        for c, d in v["conditional_fwd"].items():
            print(f"   {c}: days {d['days']} fwd21 {d['fwd21_mean_pct']:.2f}% vs base {d['base_fwd21_mean_pct']:.2f}% t {d['t_nonoverlap']}")
        for c, d in v["streaks"].items():
            print(f"   {c}: n {d['n']} next day {d['next_day_bps']:.1f} bps (base {d['base_day_bps']:.1f}) next5 {d['next5_bps']:.1f}")
    for label, rules in reps.items():
        print(f"\n== drawdown rules {label}")
        for k, d in rules.items():
            print(f"   {k}: expo {d['exposure_mean']:.2f} Sharpe {d['sharpe_base']:.2f}->{d['sharpe_overlay']:.2f} maxDD {d['maxdd_base']:.1f}->{d['maxdd_overlay_eqvol']:.1f} (eq-vol) PnL {d['pnl_base_pct']:.0f}->{d['pnl_raw_pct']:.0f} raw / {d['pnl_eqvol_pct']:.0f} eq-vol")
    print("\nbootstrap acf1:", out["bootstrap_acf1_block10_2010+"])


if __name__ == "__main__":
    main()
