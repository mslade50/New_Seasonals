"""Risk-architect lens, part 1: size THEMES, not strategies.

Groups the 15 strategies into six themes (risk_arch_common.THEMES), then asks
the only question a theme budget needs answered: what share of book variance
and of tail loss does each theme contribute (Euler decomposition), how does
that shift with the dial, and what would a growth-optimal theme allocation
look like next to the current one.

Outputs risk_arch_theme_budget.json.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from risk_arch_common import (NAV, THEMES, ann_stats, dial_bucket, dump, load_dial, load_spy,
                              load_strategy_daily, sessions)


def euler_shares(X: pd.DataFrame) -> pd.Series:
    """Share of portfolio variance attributable to each column (sum = 1)."""
    book = X.sum(1)
    cov = X.apply(lambda c: np.cov(c, book)[0, 1])
    return cov / book.var()


def cvar_shares(X: pd.DataFrame, q: float = 0.05) -> pd.Series:
    book = X.sum(1)
    tail = book <= book.quantile(q)
    return X[tail].sum() / book[tail].sum()


def shrunk_kelly(X: pd.DataFrame, delta: float = 0.3, mu_shrink: float = 0.5) -> pd.Series:
    mu = X.mean().values
    Sig = np.cov(X.values.T)
    Sig = delta * np.diag(np.diag(Sig)) + (1 - delta) * Sig
    sig = np.sqrt(np.diag(Sig))
    sbar = np.mean(mu / sig)
    mt = mu_shrink * mu + (1 - mu_shrink) * sbar * sig
    w = np.linalg.solve(Sig, mt)
    return pd.Series(w, index=X.columns)


def eff_n(X: pd.DataFrame) -> float:
    v = X.std()
    if (v == 0).any():
        X = X.loc[:, v > 0]
        v = v[v > 0]
    C = X.corr().values
    w = (v / v.sum()).values
    return float(1.0 / (w @ C @ w))


def main() -> None:
    strat, total = load_strategy_daily()
    spy = load_spy()
    idx = sessions(strat, spy)
    strat = strat.loc[idx] / NAV
    spy_r = spy.pct_change().reindex(idx).fillna(0.0)
    theme = pd.DataFrame({t: strat[[s for s in ss if s in strat.columns]].sum(1) for t, ss in THEMES.items()})
    book = theme.sum(1)
    out: dict = {"themes": THEMES, "windows": {}}

    for label, start in (("2010+", "2010-01-01"), ("2016-07+", "2016-07-20")):
        W = theme[theme.index >= start]
        B = W.sum(1)
        s = spy_r.reindex(W.index)
        rows = {}
        for t in W.columns:
            r = W[t]
            active = (r != 0)
            beta = float(np.polyfit(s, r, 1)[0]) if r.std() > 0 else float("nan")
            rows[t] = dict(**ann_stats(r), active_share=float(active.mean()),
                           sharpe_active=float(r[active].mean() / r[active].std() * np.sqrt(252)) if active.sum() > 20 else None,
                           pnl_share=float(r.sum() / B.sum()), var_share=float(euler_shares(W)[t]),
                           cvar5_share=float(cvar_shares(W)[t]), beta_spy=beta,
                           beta_per_active_day=beta / max(active.mean(), 1e-6),
                           maxdd_pct=max_dd_pct(r))
        kelly = shrunk_kelly(W)
        kelly_n = kelly / kelly.abs().mean()
        for t in W.columns:
            rows[t]["kelly_weight_norm"] = float(kelly_n[t])
            rows[t]["kelly_implied_var_share"] = None
        # variance shares if the themes were scaled by the Kelly weights
        Wk = W * kelly_n
        es = euler_shares(Wk)
        for t in W.columns:
            rows[t]["kelly_implied_var_share"] = float(es[t])
        out["windows"][label] = dict(
            book=ann_stats(B), book_maxdd_pct=max_dd_pct(B), eff_n_themes=eff_n(W),
            avg_pair_corr=float(W.corr().values[np.triu_indices(len(W.columns), 1)].mean()),
            corr=W.corr().round(3).to_dict(), themes=rows)

    # by dial bucket (live vintage, lag-1), 2016-07-20+
    dial = load_dial("live").shift(1)
    W = theme[theme.index >= "2016-07-20"]
    d = dial.reindex(W.index)
    bk = dial_bucket(d)
    s = spy_r.reindex(W.index)
    by_bucket = {}
    for b in ["<30", "30-50", "50-65", "65+"]:
        m = (bk == b).values
        if m.sum() < 20:
            continue
        Wb, Bb, sb = W[m], W[m].sum(1), s[m]
        rows = {}
        for t in W.columns:
            r = Wb[t]
            beta = float(np.polyfit(sb, r, 1)[0]) if r.std() > 0 else float("nan")
            rows[t] = dict(mean_bps=float(r.mean() * 1e4), sd_bps=float(r.std() * 1e4),
                           sharpe=float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else None,
                           var_share=float(euler_shares(Wb)[t]), beta_spy=beta,
                           spy_cov_share=float(np.cov(r, sb)[0, 1] / np.cov(Bb, sb)[0, 1]) if np.cov(Bb, sb)[0, 1] != 0 else None)
        bb = np.polyfit(sb, Bb, 1)[0]
        by_bucket[b] = dict(days=int(m.sum()), book_beta=float(bb),
                            book_r2=float(bb ** 2 * sb.var() / Bb.var()), book_sharpe=float(Bb.mean() / Bb.std() * np.sqrt(252)),
                            spy_ann=float(sb.mean() * 252), eff_n_themes=eff_n(Wb),
                            avg_pair_corr=float(Wb.corr().values[np.triu_indices(len(W.columns), 1)].mean()),
                            themes=rows)
    out["by_dial_bucket_live_lag1"] = by_bucket

    # same on PIT dial (2018+)
    pit = load_dial("pit").shift(1)
    Wp = theme[(theme.index >= "2018-01-02") & (theme.index <= "2026-07-02")]
    dp = pit.reindex(Wp.index)
    bkp = dial_bucket(dp)
    sp = spy_r.reindex(Wp.index)
    by_pit = {}
    for b in ["<30", "30-50", "50-65", "65+"]:
        m = (bkp == b).values
        if m.sum() < 20:
            continue
        Wb, Bb, sb = Wp[m], Wp[m].sum(1), sp[m]
        bb = np.polyfit(sb, Bb, 1)[0]
        by_pit[b] = dict(days=int(m.sum()), book_beta=float(bb), book_r2=float(bb ** 2 * sb.var() / Bb.var()),
                         book_sharpe=float(Bb.mean() / Bb.std() * np.sqrt(252)), spy_ann=float(sb.mean() * 252),
                         eff_n_themes=eff_n(Wb),
                         theme_beta={t: float(np.polyfit(sb, Wb[t], 1)[0]) if Wb[t].std() > 0 else None for t in Wb.columns},
                         theme_var_share=euler_shares(Wb).round(4).to_dict())
    out["by_dial_bucket_pit_lag1"] = by_pit

    # theme correlation on the worst 5% book days vs the rest (2010+)
    W = theme[theme.index >= "2010-01-01"]
    B = W.sum(1)
    tail = B <= B.quantile(0.05)
    out["tail_vs_rest_corr_2010+"] = dict(
        tail_days=int(tail.sum()),
        avg_pair_corr_tail=float(W[tail].corr().values[np.triu_indices(len(W.columns), 1)].mean()),
        avg_pair_corr_rest=float(W[~tail].corr().values[np.triu_indices(len(W.columns), 1)].mean()),
        eff_n_tail=eff_n(W[tail]), eff_n_rest=eff_n(W[~tail]),
        theme_mean_bps_tail={t: float(W.loc[tail, t].mean() * 1e4) for t in W.columns},
        theme_share_of_tail_loss={t: float(W.loc[tail, t].sum() / B[tail].sum()) for t in W.columns})

    # marginal contribution to variance of a +10% risk in each theme (Euler, 2010+)
    base_var = B.var()
    mcv = {}
    for t in W.columns:
        W2 = W.copy(); W2[t] = W2[t] * 1.1
        mcv[t] = float((W2.sum(1).var() - base_var) / base_var)
    out["variance_elasticity_2010+"] = mcv
    dump(out, "risk_arch_theme_budget.json")

    # console summary
    for label, blk in out["windows"].items():
        print(f"\n== {label}: book Sharpe {blk['book']['sharpe']:.2f}, effN themes {blk['eff_n_themes']:.2f}, avg pair corr {blk['avg_pair_corr']:.3f}")
        df = pd.DataFrame(blk["themes"]).T[["sharpe", "pnl_share", "var_share", "cvar5_share", "beta_spy", "kelly_weight_norm", "kelly_implied_var_share"]]
        print(df.astype(float).round(3).to_string())
    print("\n== by live dial bucket (lag-1), 2016-07+")
    for b, blk in out["by_dial_bucket_live_lag1"].items():
        print(f"{b}: days {blk['days']}, beta {blk['book_beta']:.2f}, R2 {blk['book_r2']:.2f}, Sharpe {blk['book_sharpe']:.2f}, SPY ann {blk['spy_ann']*100:.1f}%, effN {blk['eff_n_themes']:.2f}")
        print(pd.DataFrame(blk["themes"]).T[["sharpe", "var_share", "beta_spy", "spy_cov_share"]].astype(float).round(3).to_string())
    print("\n== PIT dial buckets")
    for b, blk in out["by_dial_bucket_pit_lag1"].items():
        print(f"{b}: days {blk['days']}, beta {blk['book_beta']:.2f}, R2 {blk['book_r2']:.2f}, Sharpe {blk['book_sharpe']:.2f}, SPY ann {blk['spy_ann']*100:.1f}%, effN {blk['eff_n_themes']:.2f}")
        print("  theme beta:", {k: round(v, 2) for k, v in blk["theme_beta"].items() if v is not None})
        print("  var share:", blk["theme_var_share"])
    print("\n== tail vs rest:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out["tail_vs_rest_corr_2010+"].items() if not isinstance(v, dict)})
    print("  share of tail loss:", {k: round(v, 3) for k, v in out["tail_vs_rest_corr_2010+"]["theme_share_of_tail_loss"].items()})
    print("\n== variance elasticity to +10% theme risk:", {k: round(v, 4) for k, v in mcv.items()})


def max_dd_pct(r: pd.Series) -> float:
    eq = r.cumsum()
    return float((eq - eq.cummax()).min() * 100)


if __name__ == "__main__":
    main()
