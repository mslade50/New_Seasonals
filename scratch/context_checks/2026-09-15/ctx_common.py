"""Shared helpers for the 2026-09-15 drills: NYSE calendar from SPY, FOMC k1 anchors, compact reports."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import load_prices, load_events, summarize, era_split, sign_test, cluster_note, pct_rank

TODAY = pd.Timestamp('2026-09-15')


def setup(tickers):
    px = load_prices(sorted(set(tickers) | {'SPY'}))
    nyse = px['SPY']['Close'].dropna().index
    closes = {t: px[t]['Close'].reindex(nyse) for t in tickers if t in px}
    return px, nyse, closes


def fwd(c, h):
    return c.shift(-h) / c - 1


def z10(c):
    r = c.pct_change()
    return c.pct_change(10) / (r.rolling(21).std() * np.sqrt(10))


def fomc_k(nyse, k=1):
    ev = load_events(['fomc_decision'])
    pos = pd.Series(range(len(nyse)), index=nyse)
    out, dec = [], []
    for f in sorted(ev['date'].unique()):
        f = pd.Timestamp(f)
        if f in pos.index and pos[f] >= k and f <= TODAY:
            out.append(nyse[pos[f] - k])
            dec.append(f)
    return pd.DatetimeIndex(out), pd.DatetimeIndex(dec)


def line(label, vals, dates=None, show_era=False):
    v = np.asarray(vals, dtype=float)
    m = ~np.isnan(v)
    s = summarize(v)
    if s['n'] == 0:
        print(f"  {label}: n=0")
        return s
    w = int((v[m] > 0).sum())
    msg = (f"  {label}: n={s['n']} mean {s['mean_pct']:+.3f}% med {s['median_pct']:+.3f}% "
           f"up {w}/{s['n']} t {s['t']:+.2f} sign_up {sign_test(w, s['n']):.4f} sign_dn {sign_test(s['n'] - w, s['n']):.4f}")
    print(msg)
    if show_era and dates is not None:
        d = pd.DatetimeIndex(dates)[m]
        ers = era_split(d, v[m])
        print("     era:", [(e['label'], e['n'], round(e.get('mean_pct', np.nan), 3), round(e.get('hit', np.nan), 1),
                              round(e.get('t', np.nan), 2) if e['n'] > 1 else None) for e in ers])
        print("     conc:", cluster_note(d, v[m]))
    return s
