"""Follow-ups: (1) sub-period alpha decay, (2) absolute-momentum overlay on the
sector rotation (slot -> cash when blended mom <= 0), (3) sleeve returns in the
book's worst months, (4) drop partial July-2026 bar sensitivity."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mom_rotation import (SECTORS, ETF_COST, load_monthly_closes, stats,  # noqa: E402
                          per_year, backtest)


def beta_alpha(y: pd.Series, x: pd.Series):
    common = y.index.intersection(x.index)
    y, x = y.loc[common], x.loc[common]
    beta = np.cov(y, x)[0, 1] / np.var(x, ddof=1)
    a = y.mean() - beta * x.mean()
    resid = y - beta * x
    t = a / (resid.std() / np.sqrt(len(resid)))
    return beta, a * 12, t, len(common)


def backtest_absfilter(monthly, universe, top_n, cost_side, start='2003-01-31'):
    px = monthly[[c for c in universe if c in monthly.columns]]
    rets = px.pct_change(fill_method=None)
    mom = 0.5 * (px / px.shift(6) - 1) + 0.5 * (px / px.shift(12) - 1)
    hist_ok = px.notna().rolling(13).sum() >= 13
    mom = mom.where(hist_ok)
    dates = px.index
    start_i = max(13, dates.searchsorted(pd.Timestamp(start)))
    net = {}
    prev_w = pd.Series(dtype=float)
    for i in range(start_i, len(dates) - 1):
        t, t1 = dates[i], dates[i + 1]
        m = mom.loc[t].dropna()
        m = m[rets.loc[t1, m.index].notna()]
        if len(m) < top_n:
            continue
        sel = m.nlargest(top_n)
        sel = sel[sel > 0]  # absolute filter: negative-mom slots go to cash (0%)
        w = pd.Series(1.0 / top_n, index=sel.index) if len(sel) else pd.Series(dtype=float)
        g = float((rets.loc[t1, w.index] * w).sum()) if len(w) else 0.0
        if len(prev_w):
            drifted = prev_w * (1 + rets.loc[t, prev_w.index].fillna(0))
            tot = drifted.sum()
            cash_old = 1 - prev_w.sum()
            drifted = drifted / (tot + cash_old) if (tot + cash_old) > 0 else drifted
        else:
            drifted = pd.Series(dtype=float)
        all_idx = w.index.union(drifted.index)
        l1 = float((w.reindex(all_idx, fill_value=0)
                    - drifted.reindex(all_idx, fill_value=0)).abs().sum())
        net[t1] = g - cost_side * l1
        prev_w = w
    return pd.Series(net).sort_index()


def main() -> None:
    ser = pd.read_parquet(ROOT / 'scratch' / 'ultracode_research' / 'mom_monthly_series.parquet')
    # drop the partial July-2026 bar (only 2026-07-01 data behind it)
    ser = ser.loc[:'2026-06-30']
    spy = ser['SPY'].dropna()
    book = ser['BOOK_pct'].dropna()

    print('SUB-PERIOD alpha vs SPY (net, monthly OLS):')
    for name in ['A1 sector top3 blend(6,12)', 'A2 sector top3 12-1', 'C  stock top20 12-1']:
        r = ser[name].dropna()
        for lo, hi in [('2003-01', '2012-12'), ('2013-01', '2026-06'), ('2016-07', '2026-06')]:
            b, a, t, n = beta_alpha(r.loc[lo:hi], spy.loc[lo:hi])
            print(f'  {name:30s} {lo}..{hi}: beta={b:4.2f} alpha={a:+7.2%}/yr t={t:+.2f} N={n}')

    # abs-momentum overlay
    monthly = load_monthly_closes(sorted(set(SECTORS + ['SPY'])))
    absf = backtest_absfilter(monthly, SECTORS, 3, ETF_COST).loc[:'2026-06-30']
    print('\nA1 + ABSOLUTE FILTER (neg-mom slot -> cash):')
    s = stats(absf, 'A1+absfilter')
    print({k: (f'{v:.4f}' if isinstance(v, float) else v) for k, v in s.items()})
    b, a, t, n = beta_alpha(absf, spy)
    print(f'  beta={b:4.2f} alpha={a:+7.2%}/yr t={t:+.2f}')
    py = per_year(absf)
    with pd.option_context('display.float_format', '{:+.1%}'.format):
        print('  per-year:', {y: f'{v:+.1%}' for y, v in py.items()})

    # high-frag months for abs filter
    frag = pd.read_parquet(ROOT / 'data' / 'rd2_fragility.parquet')
    live = frag['63d'].rolling(10, min_periods=1).mean()
    fm = live.groupby(live.index.to_period('M')).mean()
    fm.index = fm.index.to_timestamp('M')
    hi = fm[fm >= 50].index
    lo = fm[fm < 50].index
    h = absf.loc[absf.index.intersection(hi)]
    l = absf.loc[absf.index.intersection(lo)]
    print(f'  high-frag months: avg={h.mean():+.2%} N={len(h)} | rest avg={l.mean():+.2%} N={len(l)}')

    # sleeve in the BOOK's worst months (bottom decile of book monthly pnl, 2016-07+)
    bk = book.loc['2016-07':]
    worst = bk.nsmallest(max(1, len(bk) // 10))
    print(f'\nBOOK worst-decile months (N={len(worst)}, 2016-07+): '
          + ', '.join(f'{d:%Y-%m} {v:+.2%}' for d, v in worst.items()))
    for name in ['A1 sector top3 blend(6,12)', 'B  country top3 blend',
                 'C  stock top20 12-1', 'SPY']:
        r = ser[name].reindex(worst.index)
        print(f'  {name:30s} in book-worst months: avg={r.mean():+.2%} med={r.median():+.2%}')
    r = absf.reindex(worst.index)
    print(f'  {"A1+absfilter":30s} in book-worst months: avg={r.mean():+.2%} med={r.median():+.2%}')

    # summary stats excluding partial bar, sanity re-check
    print('\nRE-CHECK core stats (partial 2026-07 bar dropped):')
    for name in ['A1 sector top3 blend(6,12)', 'A2 sector top3 12-1',
                 'B  country top3 blend', 'C  stock top20 12-1', 'SPY']:
        r = ser[name].dropna()
        s = stats(r, name)
        print(f"  {name:30s} CAGR={s['CAGR']:.2%} Vol={s['Vol']:.2%} "
              f"Sharpe={s['Sharpe']:.2f} MaxDD={s['MaxDD']:.2%}")


if __name__ == '__main__':
    main()
