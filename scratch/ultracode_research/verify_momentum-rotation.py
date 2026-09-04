"""Adversarial verification of the momentum-rotation findings.

Independent recompute (fresh code, only the RULE SPEC taken from the report):
- monthly rebalance at month-end close, signal on month-end adjusted closes
- A1: 11 SPDRs, mom = mean(6m, 12m TR), top 3 EW
- A2: same, 12-1 momentum
- B : 10 country ETFs, blend mom, top 3 EW, RSX NaN'd after 2022-03-31
- C : LIQUID_UNIVERSE minus ETFs/indices, 12-1, top 20 EW
- eligibility >= 13 monthly closes at signal time
- costs per side on L1 weight change vs drifted prior weights: 5 bps ETF, 15 bps stock
- sample 2003-02 .. 2026-06 (drop partial 2026-07 bar)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\McKinley Slade\dev\New_Seasonals")
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

START = '2003-02-28'
END = '2026-06-30'


def load_monthly(tickers: list[str]) -> pd.DataFrame:
    mp = pd.read_parquet(ROOT / 'data' / 'master_prices.parquet',
                         columns=['ticker', 'date', 'Close'])
    mp = mp[mp['ticker'].isin(tickers)]
    wide = mp.pivot_table(index='date', columns='ticker', values='Close').sort_index()
    wide.index = pd.to_datetime(wide.index)
    wide = wide.ffill(limit=5)
    if 'RSX' in wide.columns:
        wide.loc[wide.index > '2022-03-31', 'RSX'] = np.nan
    return wide.resample('ME').last()


def rotation(pxm: pd.DataFrame, universe: list[str], top_n: int, kind: str,
             cost_side: float, abs_filter: bool = False):
    px = pxm[[c for c in universe if c in pxm.columns]]
    rets = px.pct_change(fill_method=None)
    if kind == 'blend':
        mom = 0.5 * (px / px.shift(6) - 1.0) + 0.5 * (px / px.shift(12) - 1.0)
    else:  # 12-1
        mom = px.shift(1) / px.shift(12) - 1.0
    eligible = px.notna().cumsum() >= 13

    idx = px.index
    net, gross, oneway = {}, {}, {}
    w_prev = pd.Series(dtype=float)
    for i in range(len(idx) - 1):
        t, t1 = idx[i], idx[i + 1]
        m = mom.loc[t].where(eligible.loc[t])
        m = m.dropna()
        if len(m) < top_n:
            continue
        top = m.nlargest(top_n)
        w_new = pd.Series(1.0 / top_n, index=top.index)
        if abs_filter:
            w_new[top < 0] = 0.0  # neg-mom slot to cash
        # drift prior weights over month t (return from t-1 close to t close)
        if len(w_prev):
            r_t = rets.loc[t].reindex(w_prev.index).fillna(0.0)
            drift = w_prev * (1.0 + r_t)
            # cash remainder stays cash
            tot = drift.sum() + (1.0 - w_prev.sum())
            drift = drift / tot
        else:
            drift = pd.Series(dtype=float)
        all_ix = w_new.index.union(drift.index)
        dw = (w_new.reindex(all_ix, fill_value=0.0)
              - drift.reindex(all_ix, fill_value=0.0))
        l1 = dw.abs().sum()
        cost = cost_side * l1
        r_next = rets.loc[t1].reindex(w_new.index).fillna(0.0)
        g = float((w_new * r_next).sum())
        gross[t1] = g
        net[t1] = g - cost
        oneway[t1] = 0.5 * l1
        w_prev = w_new
    net = pd.Series(net).loc[START:END]
    gross = pd.Series(gross).loc[START:END]
    oneway = pd.Series(oneway).loc[START:END]
    return net, gross, oneway


def stats(r: pd.Series) -> dict:
    eq = (1.0 + r).cumprod()
    yrs = len(r) / 12.0
    cagr = eq.iloc[-1] ** (1.0 / yrs) - 1.0
    vol = r.std() * np.sqrt(12.0)
    sharpe = r.mean() / r.std() * np.sqrt(12.0)
    dd = (eq / eq.cummax() - 1.0).min()
    return dict(cagr=cagr, vol=vol, sharpe=sharpe, maxdd=dd, n=len(r))


def alpha_vs(r: pd.Series, spy: pd.Series) -> tuple[float, float, float, float]:
    df = pd.concat([r, spy], axis=1, join='inner').dropna()
    y, x = df.iloc[:, 0].values, df.iloc[:, 1].values
    X = np.column_stack([np.ones(len(x)), x])
    beta, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - 2
    s2 = (resid @ resid) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    t_alpha = beta[0] / np.sqrt(cov[0, 0])
    corr = np.corrcoef(y, x)[0, 1]
    return beta[0] * 12.0, t_alpha, beta[1], corr


def main() -> None:
    all_tk = sorted(set(SECTORS) | set(COUNTRIES) | set(STOCKS) | {'SPY'})
    pxm = load_monthly(all_tk)
    print(f"stock universe size (LIQUID minus ETF/index): {len(STOCKS)}")
    print(f"monthly px range: {pxm.index[0].date()} .. {pxm.index[-1].date()}")

    spy = pxm['SPY'].pct_change().loc[START:END]

    runs = {}
    runs['A1 sectors blend'], g1, tw1 = rotation(pxm, SECTORS, 3, 'blend', 0.0005)
    runs['A2 sectors 12-1'], _, _ = rotation(pxm, SECTORS, 3, '12-1', 0.0005)
    runs['B countries blend'], _, _ = rotation(pxm, COUNTRIES, 3, 'blend', 0.0005)
    runs['C stocks 12-1'], _, _ = rotation(pxm, STOCKS, 20, '12-1', 0.0015)
    runs['A1+absfilter'], _, _ = rotation(pxm, SECTORS, 3, 'blend', 0.0005,
                                          abs_filter=True)
    runs['SPY buy-hold'] = spy

    print("\n== Performance 2003-02..2026-06 (net) ==")
    for k, r in runs.items():
        s = stats(r.dropna())
        print(f"{k:22s} CAGR {s['cagr']*100:6.2f}%  vol {s['vol']*100:5.1f}%  "
              f"Sharpe {s['sharpe']:5.2f}  maxDD {s['maxdd']*100:6.1f}%  N={s['n']}")
    print(f"A1 avg 1-way turnover/mo: {tw1.mean()*100:.1f}%  "
          f"cost drag/yr: {(g1 - runs['A1 sectors blend']).mean()*12*100:.2f}%")

    print("\n== Alpha vs SPY (monthly OLS, naive t) ==")
    for k in ['A1 sectors blend', 'A2 sectors 12-1', 'B countries blend',
              'C stocks 12-1', 'A1+absfilter']:
        a, t, b, c = alpha_vs(runs[k], spy)
        print(f"{k:22s} full: alpha {a*100:+5.2f}%/yr t={t:+5.2f} beta {b:.2f} corr {c:.2f}")
        a2, t2, *_ = alpha_vs(runs[k].loc['2016-07-01':], spy.loc['2016-07-01':])
        print(f"{'':22s} 2016-07+: alpha {a2*100:+5.2f}%/yr t={t2:+5.2f}")
    a3, t3, *_ = alpha_vs(runs['C stocks 12-1'].loc['2013-01-01':],
                          spy.loc['2013-01-01':])
    print(f"C stocks 2013-2026: alpha {a3*100:+5.2f}%/yr t={t3:+5.2f}")
    a4, t4, *_ = alpha_vs(runs['A1 sectors blend'].loc[:'2012-12-31'],
                          spy.loc[:'2012-12-31'])
    print(f"A1 2003-2012: alpha {a4*100:+5.2f}%/yr t={t4:+5.2f}")

    # ---- fragility months ----
    frag = pd.read_parquet(ROOT / 'data' / 'rd2_fragility.parquet')
    live = frag['63d'].rolling(10, min_periods=1).mean()
    mmean = live.groupby(pd.Grouper(freq='ME')).mean()
    mmean = mmean.loc['2016-07-01':'2026-06-30']
    hi = mmean[mmean >= 50]
    print(f"\n== High-fragility months (63d MA10 month-mean >= 50): N={len(hi)} ==")
    print([d.strftime('%Y-%m') for d in hi.index])

    # book monthly PnL% (flat 750k, by exit month)
    tr = pd.read_parquet(ROOT / 'data' / 'backtest_trades_full.parquet')
    tr['Exit Date'] = pd.to_datetime(tr['Exit Date'])
    book = (tr.groupby(pd.Grouper(key='Exit Date', freq='ME'))['PnL_flat_750k']
              .sum() / 750_000.0)
    book = book.loc[START:END]

    win = mmean.index
    hi_ix, lo_ix = hi.index, mmean.index.difference(hi.index)
    print(f"\n{'series':22s} {'hi-frag avg/mo':>15s} {'rest avg/mo':>12s}")
    for k, r in list(runs.items()) + [('BOOK', book)]:
        rh = r.reindex(hi_ix).dropna()
        rl = r.reindex(lo_ix).dropna()
        print(f"{k:22s} {rh.mean()*100:+14.2f}% {rl.mean()*100:+11.2f}%  "
              f"(N={len(rh)}/{len(rl)})")

    # correlations to book
    print("\n== Corr with book monthly PnL (2003-02..2026-06) ==")
    for k, r in runs.items():
        df = pd.concat([r, book], axis=1, join='inner').dropna()
        print(f"{k:22s} corr {df.corr().iloc[0, 1]:+.3f}  N={len(df)}")

    # book's 12 worst months since 2016-07
    bw = book.loc['2016-07-01':'2026-06-30'].nsmallest(12)
    print(f"\n== Book's 12 worst months since 2016-07 (book avg "
          f"{bw.mean()*100:+.2f}%) ==")
    print([d.strftime('%Y-%m') for d in bw.index])
    for k, r in runs.items():
        rr = r.reindex(bw.index).dropna()
        print(f"{k:22s} avg in those months {rr.mean()*100:+.2f}%  N={len(rr)}")

    # survivorship: delisted counts in master_prices
    mp = pd.read_parquet(ROOT / 'data' / 'master_prices.parquet',
                         columns=['ticker', 'date'])
    last = mp.groupby('ticker')['date'].max()
    n_all = len(last)
    n_dead = int((pd.to_datetime(last) < pd.Timestamp('2026-06-01')).sum())
    print(f"\nmaster_prices tickers: {n_all}, last-date before 2026-06: {n_dead}")

    # looser cut: month-max >= 50
    mmax = live.groupby(pd.Grouper(freq='ME')).max().loc['2016-07-01':'2026-06-30']
    hi2 = mmax[mmax >= 50].index
    print(f"\nlooser cut month-max>=50: N={len(hi2)}")
    for k, r in list(runs.items()) + [('BOOK', book)]:
        rh = r.reindex(hi2).dropna()
        print(f"{k:22s} avg {rh.mean()*100:+.2f}%  N={len(rh)}")


if __name__ == '__main__':
    main()
