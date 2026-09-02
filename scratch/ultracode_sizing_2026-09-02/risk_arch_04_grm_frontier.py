"""Risk-architect lens, part 4: the GRM frontier with the measured haircut,
the dial-armed hedge, and the margin wall on one table.

For multiples m of current sizing (GRM = 1.5 m) and haircuts h on the daily
mean (0, 0.40 = estimation_haircut central, 0.50, 0.71 = pessimistic range
edge), reports the log-growth rate and the block-bootstrap 1y/3y drawdown
distribution, unhedged and with the P1 hedge (dial 50/45 hysteresis, 126d
lag-1 beta, 2 bps per arm event) on the live and PIT dial vintages.

Outputs risk_arch_grm_frontier.json.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from risk_arch_common import NAV, dump, load_dial, load_spy, load_strategy_daily, sessions


def hedge_series(r: pd.Series, spy_r: pd.Series, dial: pd.Series, arm=50.0, rel=45.0, win=126, fric=0.0002) -> tuple[pd.Series, pd.Series]:
    d = dial.reindex(r.index).shift(1)
    armed = pd.Series(False, index=r.index)
    st = False
    for i, v in enumerate(d.values):
        if np.isnan(v):
            armed.iloc[i] = st; continue
        if not st and v >= arm:
            st = True
        elif st and v < rel:
            st = False
        armed.iloc[i] = st
    # rolling OLS beta, lag-1
    cov = r.rolling(win).cov(spy_r); var = spy_r.rolling(win).var()
    beta = (cov / var).shift(1).clip(-1, 2).fillna(0.0)
    h = -armed.astype(float) * beta * spy_r
    events = armed.astype(int).diff().abs().fillna(0)
    h = h - events * fric * beta.abs()
    return h, armed


def growth(r: np.ndarray, m: float) -> float:
    x = 1 + m * r
    if (x <= 0).any():
        return float("-inf")
    return float(np.log(x).mean() * 252)


def boot_paths(r: np.ndarray, n_paths: int, horizon: int, block: int = 10, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(r)
    P = np.empty((n_paths, horizon))
    for p in range(n_paths):
        i = 0
        while i < horizon:
            L = rng.geometric(1 / block); s = rng.integers(0, n)
            take = min(L, horizon - i)
            P[p, i:i + take] = r[(s + np.arange(take)) % n]
            i += take
    return P


def dd_stats(P: np.ndarray, m: float) -> dict:
    eq = np.cumprod(1 + m * P, axis=1)
    peak = np.maximum.accumulate(eq, axis=1)
    dd = (eq / peak - 1).min(axis=1)
    term = eq[:, -1]
    return dict(median_dd=float(-np.median(dd) * 100), p95_dd=float(-np.quantile(dd, 0.05) * 100),
                p_dd_gt_10=float((dd < -0.10).mean()), p_dd_gt_15=float((dd < -0.15).mean()), p_dd_gt_20=float((dd < -0.20).mean()),
                p_dd_gt_30=float((dd < -0.30).mean()), median_terminal=float(np.median(term)), p5_terminal=float(np.quantile(term, 0.05)),
                p_loss=float((term < 1).mean()))


def main() -> None:
    strat, total = load_strategy_daily()
    spy = load_spy()
    idx = sessions(strat, spy)
    r = (total.reindex(idx) / NAV).astype(float)
    spy_r = spy.pct_change().reindex(idx).fillna(0.0)
    out: dict = {}
    mults = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    haircuts = [0.0, 0.40, 0.50, 0.71]

    # hedged series on both vintages
    live = load_dial("live"); pit = load_dial("pit")
    windows = {"2016-07+": ("2016-07-20", "2026-08-07", live, "live"),
               "PIT 2018+": ("2018-01-02", "2026-07-02", pit, "pit")}
    for label, (a, b, dial, vint) in windows.items():
        x = r[(r.index >= a) & (r.index <= b)]
        s = spy_r.reindex(x.index)
        h, armed = hedge_series(x, s, dial)
        xh = x + h
        blk = dict(days=int(len(x)), armed_days=int(armed.sum()), hedge_pnl_pct=float(h.sum() * 100),
                   sharpe_unhedged=float(x.mean() / x.std() * np.sqrt(252)), sharpe_hedged=float(xh.mean() / xh.std() * np.sqrt(252)),
                   maxdd_unhedged=float(mdd(x) * 100), maxdd_hedged=float(mdd(xh) * 100),
                   worst_day_unhedged=float(x.min() * 100), worst_day_hedged=float(xh.min() * 100),
                   cvar1_unhedged=float(x[x <= x.quantile(.01)].mean() * 100), cvar1_hedged=float(xh[xh <= xh.quantile(.01)].mean() * 100),
                   armed_sharpe_unhedged=float(x[armed].mean() / x[armed].std() * np.sqrt(252)) if armed.sum() > 20 else None,
                   armed_sharpe_hedged=float(xh[armed].mean() / xh[armed].std() * np.sqrt(252)) if armed.sum() > 20 else None,
                   worst21_unhedged=float(x.rolling(21).sum().min() * 100), worst21_hedged=float(xh.rolling(21).sum().min() * 100))
        table = {}
        for hc in haircuts:
            for hedged, series in (("unhedged", x), ("hedged", xh)):
                z = (series - hc * series.mean()).values
                P1 = boot_paths(z, 2000, 252); P3 = boot_paths(z, 1500, 756, seed=11)
                for m in mults:
                    g = growth(z, m)
                    d1 = dd_stats(P1, m); d3 = dd_stats(P3, m)
                    table[f"h{hc}_{hedged}_m{m}"] = dict(haircut=hc, hedged=hedged, m=m, grm=1.5 * m, growth_ann_pct=g * 100 if np.isfinite(g) else None,
                                                        ann_mean_pct=float(z.mean() * 252 * m * 100), ann_vol_pct=float(z.std() * np.sqrt(252) * m * 100),
                                                        dd1y=d1, dd3y=d3, hist_maxdd_pct=float(mdd(pd.Series(z) * m) * 100), hist_worst_day_pct=float(z.min() * m * 100))
        blk["frontier"] = table
        out[label] = blk

    # long window 2003+ unhedged frontier (no dial before 2016)
    x = r[r.index >= "2003-01-01"]
    table = {}
    for hc in haircuts:
        z = (x - hc * x.mean()).values
        P1 = boot_paths(z, 2000, 252); P3 = boot_paths(z, 1500, 756, seed=11)
        for m in mults:
            g = growth(z, m); d1 = dd_stats(P1, m); d3 = dd_stats(P3, m)
            table[f"h{hc}_m{m}"] = dict(haircut=hc, m=m, grm=1.5 * m, growth_ann_pct=g * 100 if np.isfinite(g) else None,
                                        ann_mean_pct=float(z.mean() * 252 * m * 100), ann_vol_pct=float(z.std() * np.sqrt(252) * m * 100),
                                        dd1y=d1, dd3y=d3, hist_maxdd_pct=float(mdd(pd.Series(z) * m) * 100), hist_worst_day_pct=float(z.min() * m * 100))
    out["2003+_unhedged"] = dict(days=int(len(x)), sharpe=float(x.mean() / x.std() * np.sqrt(252)), frontier=table)

    # theory cross-check: fraction of Kelly and GBM P(ever DD >= x)
    for label in ("2016-07+", "PIT 2018+"):
        blk = out[label]
        a, b = windows[label][0], windows[label][1]
        x = r[(r.index >= a) & (r.index <= b)]
        S = x.mean() / x.std() * np.sqrt(252); vol = x.std() * np.sqrt(252)
        theo = {}
        for hc in haircuts:
            Sh = S * (1 - hc)
            fstar = Sh / vol  # full-Kelly multiple of current
            for m in mults:
                c = m / fstar
                theo[f"h{hc}_m{m}"] = dict(sharpe_h=float(Sh), kelly_fraction=float(c), growth_share=float(c * (2 - c)),
                                          p_ever_dd20=float(0.8 ** (2 / c - 1)) if c > 0 else None, p_ever_dd30=float(0.7 ** (2 / c - 1)) if c > 0 else None,
                                          p_ever_dd50=float(0.5 ** (2 / c - 1)) if c > 0 else None)
        blk["theory"] = theo
    dump(out, "risk_arch_grm_frontier.json")

    for label in ("2016-07+", "PIT 2018+", "2003+_unhedged"):
        blk = out[label]
        print(f"\n== {label}")
        if "hedge_pnl_pct" in blk:
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in blk.items() if k not in ("frontier", "theory")})
        for key, row in blk["frontier"].items():
            if row["m"] in (1.0, 1.5, 2.0, 3.0) and row["haircut"] in (0.0, 0.4, 0.71):
                d1, d3 = row["dd1y"], row["dd3y"]
                print(f"  {key:26s} g {row['growth_ann_pct']:.1f}% vol {row['ann_vol_pct']:.1f}% | 1y medDD {d1['median_dd']:.1f} p95 {d1['p95_dd']:.1f} P>15 {d1['p_dd_gt_15']:.2f} P>20 {d1['p_dd_gt_20']:.2f} | 3y medDD {d3['median_dd']:.1f} P>20 {d3['p_dd_gt_20']:.2f} P>30 {d3['p_dd_gt_30']:.2f} | hist DD {row['hist_maxdd_pct']:.1f} worst day {row['hist_worst_day_pct']:.1f}")
        if "theory" in blk:
            for key, t in blk["theory"].items():
                if key.endswith("m1.5") or key.endswith("m2.0") or key.endswith("m3.0"):
                    print(f"  theory {key}: S_h {t['sharpe_h']:.2f} kelly frac {t['kelly_fraction']:.3f} growth share {t['growth_share']:.2f} P(ever DD20) {t['p_ever_dd20']:.3f} DD30 {t['p_ever_dd30']:.3f}")


def mdd(r: pd.Series) -> float:
    eq = r.cumsum()
    return float((eq - eq.cummax()).min())


if __name__ == "__main__":
    main()
