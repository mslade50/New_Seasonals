"""Cross-sectional momentum rotation prototype — sectors / countries / single stocks.

Rules (stated up front):
- Monthly rebalance at month-end close (last trading day). Signals computed on
  month-end adjusted closes; position held over the following calendar month.
- Variant A1 (sectors): 11 SPDRs, momentum = mean(6m, 12m total return), top 3 EW.
- Variant A2 (sectors): same universe, momentum = 12-1 (t-12 -> t-1), top 3 EW.
- Variant B  (countries): 10 country/regional ETFs in cache, blend mom, top 3 EW.
  RSX hard-dropped after 2022-03-31 (trading halted; flat phantom prices after).
- Variant C  (stocks): 12-1 momentum, top 20 EW, liquid single-stock universe
  (LIQUID_UNIVERSE minus ETFs/indices). SURVIVORSHIP-BIASED — upper bound only.
- Eligibility: >= 13 monthly closes of history at signal time.
- Costs: per-side, charged on L1 weight change vs drifted prior weights.
  ETFs 5 bps/side, stocks 15 bps/side.
- Benchmark: SPY buy-hold (no cost) on the same monthly grid.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from strategy_config import LIQUID_UNIVERSE  # noqa: E402

ETF_OR_INDEX = {
    'DIA', 'IBB', 'IHI', 'ITA', 'ITB', 'IWM', 'IYR', 'KRE', 'QQQ', 'SMH',
    'SPY', 'VNQ', 'XBI', 'XHB', 'XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK',
    'XLP', 'XLU', 'XLV', 'XLY', 'XME', 'XRT', '^GSPC', '^NDX',
}
SECTORS = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']
COUNTRIES = ['EWJ', 'EWW', 'EWZ', 'EWT', 'EWY', 'FXI', 'INDA', 'EEM', 'EFA', 'RSX']
STOCKS = sorted(t for t in LIQUID_UNIVERSE if t not in ETF_OR_INDEX)

ETF_COST = 0.0005   # 5 bps per side
STK_COST = 0.0015   # 15 bps per side


def load_monthly_closes(tickers: list[str]) -> pd.DataFrame:
    mp = pd.read_parquet(ROOT / 'data' / 'master_prices.parquet',
                         columns=['ticker', 'date', 'Close'])
    mp = mp[mp['ticker'].isin(tickers)]
    wide = mp.pivot_table(index='date', columns='ticker', values='Close')
    wide = wide.sort_index().ffill(limit=5)
    # RSX: halted 2022-03; phantom flat prices afterwards -> drop
    if 'RSX' in wide.columns:
        wide.loc[wide.index > '2022-03-31', 'RSX'] = np.nan
    return wide.resample('ME').last()


def backtest(monthly: pd.DataFrame, universe: list[str], top_n: int,
             mom_kind: str, cost_side: float, start: str = '2003-01-31'):
    """Returns (net_returns Series, gross_returns Series, turnover Series, holdings dict)."""
    px = monthly[[c for c in universe if c in monthly.columns]]
    rets = px.pct_change(fill_method=None)
    if mom_kind == 'blend':
        mom = 0.5 * (px / px.shift(6) - 1) + 0.5 * (px / px.shift(12) - 1)
    elif mom_kind == '12-1':
        mom = px.shift(1) / px.shift(12) - 1
    else:
        raise ValueError(mom_kind)
    hist_ok = px.notna().rolling(13).sum() >= 13   # >=13 monthly closes
    mom = mom.where(hist_ok)

    dates = px.index
    start_i = max(13, dates.searchsorted(pd.Timestamp(start)))
    net, gross, tos, hold = {}, {}, {}, {}
    prev_w = pd.Series(dtype=float)
    for i in range(start_i, len(dates) - 1):
        t, t1 = dates[i], dates[i + 1]
        m = mom.loc[t].dropna()
        # need a next-month return to be investable
        m = m[rets.loc[t1, m.index].notna()]
        if len(m) < top_n:
            continue
        sel = m.nlargest(top_n).index
        w = pd.Series(1.0 / top_n, index=sel)
        r_next = rets.loc[t1, sel]
        g = float(r_next.mean())
        # drift prior weights to t close, L1 vs new target
        if len(prev_w):
            drifted = prev_w * (1 + rets.loc[t, prev_w.index].fillna(0))
            drifted = drifted / drifted.sum()
        else:
            drifted = pd.Series(dtype=float)
        all_idx = w.index.union(drifted.index)
        l1 = float((w.reindex(all_idx, fill_value=0)
                    - drifted.reindex(all_idx, fill_value=0)).abs().sum())
        cost = cost_side * l1
        net[t1] = g - cost
        gross[t1] = g
        tos[t1] = l1 / 2.0  # one-way turnover as fraction of book
        hold[t1] = list(sel)
        prev_w = w
    return (pd.Series(net).sort_index(), pd.Series(gross).sort_index(),
            pd.Series(tos).sort_index(), hold)


def stats(r: pd.Series, label: str) -> dict:
    eq = (1 + r).cumprod()
    yrs = len(r) / 12.0
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(12)
    sharpe = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    return {'label': label, 'start': r.index[0].date(), 'end': r.index[-1].date(),
            'n_months': len(r), 'CAGR': cagr, 'Vol': vol, 'Sharpe': sharpe, 'MaxDD': dd}


def per_year(r: pd.Series) -> pd.Series:
    return (1 + r).groupby(r.index.year).prod() - 1


def main() -> None:
    all_tk = sorted(set(SECTORS + COUNTRIES + STOCKS + ['SPY']))
    monthly = load_monthly_closes(all_tk)
    spy = monthly['SPY'].pct_change(fill_method=None).dropna()
    spy = spy[spy.index >= '2003-02-01']

    runs = {
        'A1 sector top3 blend(6,12)': backtest(monthly, SECTORS, 3, 'blend', ETF_COST),
        'A2 sector top3 12-1':        backtest(monthly, SECTORS, 3, '12-1', ETF_COST),
        'B  country top3 blend':      backtest(monthly, COUNTRIES, 3, 'blend', ETF_COST),
        'C  stock top20 12-1':        backtest(monthly, STOCKS, 20, '12-1', STK_COST),
    }

    print('=' * 100)
    print('SUMMARY (net of costs; ETFs 5bps/side, stocks 15bps/side; monthly rebalance)')
    rows = []
    for name, (net, grs, to, _) in runs.items():
        s = stats(net, name)
        s['AvgTO_1way'] = to.mean()
        s['CostDrag_ann'] = (grs - net).mean() * 12
        rows.append(s)
    spy_s = stats(spy, 'SPY buy-hold')
    spy_s['AvgTO_1way'] = 0.0
    spy_s['CostDrag_ann'] = 0.0
    rows.append(spy_s)
    df = pd.DataFrame(rows).set_index('label')
    with pd.option_context('display.float_format', '{:.4f}'.format, 'display.width', 200):
        print(df.to_string())

    # SPY-aligned comparisons + beta/alpha
    print('\nBETA / ALPHA vs SPY (monthly OLS, full common sample):')
    for name, (net, _, _, _) in runs.items():
        common = net.index.intersection(spy.index)
        y, x = net.loc[common], spy.loc[common]
        beta = np.cov(y, x)[0, 1] / np.var(x, ddof=1)
        alpha_m = y.mean() - beta * x.mean()
        resid = y - beta * x
        # HAC-lite: monthly returns, use plain t on alpha with Newey-West lag 3
        t_alpha = alpha_m / (resid.std() / np.sqrt(len(resid)))
        corr = np.corrcoef(y, x)[0, 1]
        print(f'  {name:30s} beta={beta:5.2f} corr={corr:5.2f} '
              f'alpha={alpha_m * 12:+7.2%}/yr (naive t={t_alpha:+.2f}) N={len(common)}')

    # per-year tables
    print('\nPER-YEAR RETURNS (net):')
    py = pd.DataFrame({name: per_year(net) for name, (net, _, _, _) in runs.items()})
    py['SPY'] = per_year(spy)
    with pd.option_context('display.float_format', '{:+.1%}'.format):
        print(py.to_string())

    # momentum crash exhibits
    print('\nMOMENTUM CRASH WINDOWS (monthly, net):')
    for win in [('2009-01', '2009-12'), ('2020-01', '2020-12')]:
        print(f'  window {win[0]}..{win[1]}')
        blk = pd.DataFrame({name: net.loc[win[0]:win[1]]
                            for name, (net, _, _, _) in runs.items()})
        blk['SPY'] = spy.loc[win[0]:win[1]]
        with pd.option_context('display.float_format', '{:+.1%}'.format):
            print(blk.to_string())

    print('\nWORST 8 MONTHS per variant (net):')
    for name, (net, _, _, _) in runs.items():
        w = net.nsmallest(8)
        print(f'  {name}: ' + ', '.join(f'{d:%Y-%m} {v:+.1%}' for d, v in w.items()))

    # ---- book correlation ----
    tr = pd.read_parquet(ROOT / 'data' / 'backtest_trades_full.parquet')
    tr['Exit Date'] = pd.to_datetime(tr['Exit Date'])
    book = tr.groupby(tr['Exit Date'].dt.to_period('M'))['PnL_flat_750k'].sum()
    book.index = book.index.to_timestamp('M')
    book_pct = book / 750_000.0
    print('\nCORRELATION of sleeve monthly returns vs book monthly PnL (flat 750k, exit month):')
    for name, (net, _, _, _) in runs.items():
        common = net.index.intersection(book_pct.index)
        c = np.corrcoef(net.loc[common], book_pct.loc[common])[0, 1]
        print(f'  {name:30s} corr={c:+.3f}  N={len(common)}')
    c_spy = np.corrcoef(spy.loc[spy.index.intersection(book_pct.index)],
                        book_pct.loc[spy.index.intersection(book_pct.index)])[0, 1]
    print(f'  {"SPY buy-hold":30s} corr={c_spy:+.3f}')

    # ---- high-fragility months (2016-07+) ----
    frag = pd.read_parquet(ROOT / 'data' / 'rd2_fragility.parquet')
    live = frag['63d'].rolling(10, min_periods=1).mean()
    fm_mean = live.groupby(live.index.to_period('M')).mean()
    fm_max = live.groupby(live.index.to_period('M')).max()
    fm_mean.index = fm_mean.index.to_timestamp('M')
    fm_max.index = fm_max.index.to_timestamp('M')
    for flag_name, flag in [('month-mean >= 50', fm_mean >= 50),
                            ('month-max >= 50', fm_max >= 50)]:
        hi = flag[flag].index
        lo = flag[~flag].index
        print(f'\nHIGH-FRAGILITY MONTHS ({flag_name}): {len(hi)} high / {len(lo)} rest '
              f'({fm_mean.index[0]:%Y-%m}..{fm_mean.index[-1]:%Y-%m})')
        if len(hi) and flag_name.startswith('month-mean'):
            print('   high months:', ', '.join(f'{d:%Y-%m}' for d in hi))
        for name, (net, _, _, _) in list(runs.items()) + [('SPY buy-hold', (spy, 0, 0, 0)),
                                                          ('BOOK pnl% (flat750k)', (book_pct, 0, 0, 0))]:
            h = net.loc[net.index.intersection(hi)]
            l = net.loc[net.index.intersection(lo)]
            print(f'  {name:30s} high: avg={h.mean():+.2%} med={h.median():+.2%} '
                  f'N={len(h)} | rest: avg={l.mean():+.2%} N={len(l)}')

    # holdings snapshot for sanity
    name = 'A1 sector top3 blend(6,12)'
    hold = runs[name][3]
    hk = sorted(hold)
    print(f'\n{name} — last 6 holdings:')
    for d in hk[-6:]:
        print(f'  {d:%Y-%m}: {hold[d]}')

    # save monthly return series for reuse
    out = pd.DataFrame({name: net for name, (net, _, _, _) in runs.items()})
    out['SPY'] = spy
    out['BOOK_pct'] = book_pct
    out.to_parquet(ROOT / 'scratch' / 'ultracode_research' / 'mom_monthly_series.parquet')
    print('\nsaved mom_monthly_series.parquet')


if __name__ == '__main__':
    main()
