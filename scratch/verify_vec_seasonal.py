"""Verify numpy-vectorized seasonal_window_returns / expected_atr_move produce
IDENTICAL output to the originals, and measure the speedup. Only swap into
seasonal_edge.py after this passes."""
import sys, time
import numpy as np
import pandas as pd

ROOT = r"C:\Users\McKinley Slade\dev\New_Seasonals"
sys.path.insert(0, ROOT)
import scripts.seasonal_edge as se


def _pick_positions(doy_arr, years_arr, target_doy, asof_year, cycle_phase, tol, exclude_current):
    m = (doy_arr >= target_doy - tol) & (doy_arr <= target_doy + tol)
    if exclude_current:
        m &= years_arr < asof_year
    if cycle_phase is not None:
        m &= (years_arr % 4) == cycle_phase
    idx = np.flatnonzero(m)
    if idx.size == 0:
        return idx
    # primary year asc, then |doy-target| asc, then position asc (matches idxmin's
    # first-min on chronologically-ordered candidates); one pick per year
    order = np.lexsort((idx, np.abs(doy_arr[idx] - target_doy), years_arr[idx]))
    ys = years_arr[idx][order]
    first = np.ones(ys.size, bool)
    first[1:] = ys[1:] != ys[:-1]
    return idx[order][first]


def _prep(price_df, asof):
    close = price_df["Close"].dropna().sort_index()
    if close.empty:
        return None
    asof = pd.Timestamp(asof).normalize()
    doy = se._trading_doy(close.index).values
    years = close.index.year.values.astype(np.int64)
    le = close.index.values <= np.datetime64(asof)
    if not le.any():
        return None
    target_doy = int(doy[le][-1])
    return close, doy, years, target_doy, asof.year


def swr_vec(price_df, asof, N, cycle_phase_filter=None, doy_tol=2, min_years=3, exclude_current_year=True):
    if price_df is None or price_df.empty:
        return None
    p = _prep(price_df, asof)
    if p is None:
        return None
    close, doy, years, target_doy, asof_year = p
    cv = close.values.astype(float)
    fwd = np.full(cv.shape, np.nan)
    if N < len(cv):
        fwd[:-N] = cv[N:] / cv[:-N] - 1.0
    picks = _pick_positions(doy, years, target_doy, asof_year, cycle_phase_filter, doy_tol, exclude_current_year)
    r = fwd[picks]
    valid = ~np.isnan(r)
    rets = r[valid]
    yrs = years[picks][valid]
    if rets.size < min_years:
        return {"n": int(rets.size), "insufficient": True}
    return {"n": int(rets.size), "mean": float(rets.mean()), "median": float(np.median(rets)),
            "n_down": int((rets < 0).sum()), "n_up": int((rets > 0).sum()),
            "pct_down": float((rets < 0).mean()), "years": [int(y) for y in yrs],
            "rets": [round(float(x), 4) for x in rets]}


def eam_vec(price_df, asof, N, cycle_phase_filter=None, doy_tol=2, min_years=3, exclude_current_year=True):
    if price_df is None or price_df.empty:
        return None
    close = price_df["Close"].dropna().sort_index()
    if close.empty:
        return None
    atr = se.atr_wilder(price_df).reindex(close.index).values.astype(float)
    p = _prep(price_df, asof)
    if p is None:
        return None
    _, doy, years, target_doy, asof_year = p
    cv = close.values.astype(float)
    fwd = np.full(cv.shape, np.nan)
    if N < len(cv):
        fwd[:-N] = (cv[N:] - cv[:-N]) / atr[:-N]
    picks = _pick_positions(doy, years, target_doy, asof_year, cycle_phase_filter, doy_tol, exclude_current_year)
    r = fwd[picks]
    r = r[~np.isnan(r)]
    if r.size < min_years:
        return None
    return float(r.mean())


def eq(a, b, tol=1e-9):
    if a is None and b is None:
        return True
    if (a is None) != (b is None):
        return False
    if isinstance(a, dict):
        if a.get("insufficient") or b.get("insufficient"):
            return bool(a.get("insufficient")) == bool(b.get("insufficient")) and a["n"] == b["n"]
        for k in ("n", "n_down", "n_up"):
            if a[k] != b[k]:
                return False
        for k in ("mean", "median", "pct_down"):
            if abs(a[k] - b[k]) > tol:
                return False
        return a["years"] == b["years"]
    return abs(a - b) < tol


tickers = ["AAPL", "JPM", "GLD", "NG=F", "^FTSE", "TLT", "CL=F", "EURUSD=X"]
prices = se.load_prices(tickers)
asofs = pd.bdate_range("2014-01-01", "2025-12-01", freq="20B")
mism = checks = 0
for t in tickers:
    px = prices.get(se._norm_ticker(t))
    if px is None or len(px) < 400:
        continue
    for a in asofs:
        for N in (5, 10, 21):
            for cyc in (None, a.year % 4):
                checks += 1
                if not eq(se.seasonal_window_returns(px, a, N, cyc), swr_vec(px, a, N, cyc)):
                    mism += 1
                    if mism <= 5:
                        print(f"  SWR mismatch {t} {a.date()} N{N} cyc{cyc}")
                if not eq(se.expected_atr_move(px, a, N, cyc), eam_vec(px, a, N, cyc)):
                    mism += 1
                    if mism <= 5:
                        print(f"  EAM mismatch {t} {a.date()} N{N} cyc{cyc}")
print(f"\nchecks={checks*2}  mismatches={mism}  -> {'IDENTICAL' if mism == 0 else 'DIFFERS'}")

# speed
px = prices[se._norm_ticker("AAPL")]; a = pd.Timestamp("2018-06-15")
def tm(fn, n=300):
    t0 = time.time()
    for _ in range(n):
        fn()
    return (time.time() - t0) / n * 1000
print(f"SWR: orig {tm(lambda: se.seasonal_window_returns(px,a,21)):.2f}ms -> vec {tm(lambda: swr_vec(px,a,21)):.2f}ms")
print(f"EAM: orig {tm(lambda: se.expected_atr_move(px,a,21)):.2f}ms -> vec {tm(lambda: eam_vec(px,a,21)):.2f}ms")
