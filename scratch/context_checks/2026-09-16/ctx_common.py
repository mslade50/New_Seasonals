"""Shared helpers for the 2026-09-16 drills: NYSE calendar from SPY, FOMC anchors, compact reports."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import load_prices, load_events, summarize, era_split, sign_test, cluster_note, pct_rank, declusters

TODAY = pd.Timestamp('2026-09-16')


def setup(tickers):
    px = load_prices(sorted(set(tickers) | {'SPY'}))
    nyse = px['SPY']['Close'].dropna().index
    closes = {t: px[t]['Close'].reindex(nyse) for t in tickers if t in px}
    return px, nyse, closes


def fwd(c, h):
    return c.shift(-h) / c - 1


def back(c, h):
    return c / c.shift(h) - 1


def decisions(nyse, through=TODAY):
    """Scheduled FOMC decision sessions on the NYSE calendar, with the projections (SEP) flag."""
    ev = load_events(['fomc_decision'])
    sep = set(pd.to_datetime(ev.loc[ev['detail'].fillna('').str.contains(r'\*', regex=True), 'date']))
    out = [pd.Timestamp(f) for f in sorted(ev['date'].unique()) if pd.Timestamp(f) in nyse and pd.Timestamp(f) <= through]
    return pd.DatetimeIndex(out), sep


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
