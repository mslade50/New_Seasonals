"""Shared substrate for the 2026-09-07 stage-B1 historical-analogue axis.

Two independent neighbour-set constructions live in 05_analogue_knn.py
(Method A, standardised feature vector + Euclidean k-NN) and
05_analogue_conjunction.py (Method B, a hard conjunction screen with no
distance metric). Everything statistical comes from pitch_lab; this file only
builds the state features both methods share and prints one forward-return
table in the same shape.

Conventions inherited unchanged from pitch_lab:
  - returns are FRACTIONS in, PERCENT out of summarize()
  - entry is lag=1 (state prints on close D, order goes in MOC on close D+1)
  - the small-sample statistic is the exact sign test, not a t-stat

Two things worth stating once, because both methods inherit them:

1. HYG's first bar is 2007-04-11, so any feature reading HYG's 252d high is
   undefined before ~2008-04. The neighbour POOL therefore starts in 2008 and
   the whole exercise is blind to 2000-2007. 05_analogue_knn.py re-runs
   without the credit feature to show what that costs.
2. A day near today is trivially its own nearest neighbour. Candidates inside
   EXCLUDE_TD sessions of the as-of date are dropped so the answer is not
   "last week looked like last week".
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

from pitch_lab import (  # noqa: F401
    load_prices, declusters, summarize, show, sign_test, cluster_note,
    bootstrap_p_le0, pct_rank, fwd_lag, rolling_on_valid,
)

ASOF = pd.Timestamp("2026-09-04")          # freshest bar
EXCLUDE_TD = 21                            # today's own episode cannot vote

INSTRUMENTS = ["SPY", "IWM", "TLT", "HYG", "GLD", "GDX", "USO", "XLE",
               "UUP", "EFA", "SVXY"]
FEATURE_TICKERS = ["SPY", "TLT", "HYG", "^TNX"]
CONTEXT_TICKERS = ["^VIX", "^VIX3M", "^SKEW", "QQQ", "XOP", "DBC", "XLI", "ITA"]

FEATURES = ["spy_d252h", "spy_dsma200", "spy_rank21", "spy_volratio",
            "breadth200", "tnx_d252h", "credit_div"]
FEATURE_LABELS = {
    "spy_d252h":   "SPY dist to 252d high (%)",
    "spy_dsma200": "SPY dist to 200d SMA (%)",
    "spy_rank21":  "SPY 21d return rank (0-100)",
    "spy_volratio": "SPY 21d realised vol / own 63d avg",
    "breadth200":  "breadth: % of fixed universe > 200d SMA",
    "tnx_d252h":   "^TNX dist to 252d high (%)",
    "credit_div":  "TLT dist-to-252d-low minus HYG dist-to-252d-high (pp)",
}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def breadth_universe() -> list[str]:
    """Fixed-membership liquid universe with a bar on/before 2007-01-01.

    build_pitch_state's breadth reads today's 218-name tape; that set cannot
    be walked back through history without membership drift, so the historical
    breadth feature uses the long-lived subset of the SAME universe. It is
    survivorship-biased by construction (only names still in the file today),
    which biases the LEVEL of breadth up in every era equally; the feature is a
    cross-sectional comparison against its own history, so the bias mostly
    cancels. Stated, not fixed.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from strategy_config import LIQUID_PLUS_COMMODITIES
    return sorted(set(LIQUID_PLUS_COMMODITIES))


def load_all() -> tuple[dict[str, pd.DataFrame], list[str]]:
    uni = breadth_universe()
    want = sorted(set(uni) | set(INSTRUMENTS) | set(FEATURE_TICKERS)
                  | set(CONTEXT_TICKERS))
    px = load_prices(want)
    keep = [t for t in uni
            if t in px and px[t].index[0] <= pd.Timestamp("2007-01-01")
            and px[t].index[-1] >= pd.Timestamp("2026-09-01")]
    return px, keep


def align(s: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """Own-calendar series onto the SPY calendar without lookahead."""
    return s.reindex(idx.union(s.index)).ffill().reindex(idx)


def build_features(px: dict[str, pd.DataFrame],
                   breadth_names: list[str]) -> pd.DataFrame:
    spy = px["SPY"]["Close"]
    idx = spy.index[spy.index <= ASOF]

    def dist_hi(s, n=252):
        return (s / s.rolling(n).max() - 1.0) * 100.0

    def dist_lo(s, n=252):
        return (s / s.rolling(n).min() - 1.0) * 100.0

    f = pd.DataFrame(index=idx)
    f["spy_d252h"] = dist_hi(spy).reindex(idx)
    f["spy_dsma200"] = ((spy / spy.rolling(200).mean() - 1.0) * 100.0).reindex(idx)
    f["spy_rank21"] = pct_rank(spy, 21, 252).reindex(idx)

    rv21 = spy.pct_change().rolling(21).std() * np.sqrt(252) * 100.0
    f["spy_rv21_ann"] = rv21.reindex(idx)
    f["spy_volratio"] = (rv21 / rv21.rolling(63).mean()).reindex(idx)

    above = []
    for t in breadth_names:
        c = px[t]["Close"]
        above.append(align((c > c.rolling(200).mean()).astype(float)
                           .where(c.rolling(200).mean().notna()), idx))
    f["breadth200"] = 100.0 * pd.concat(above, axis=1).mean(axis=1)

    tnx = px["^TNX"]["Close"]
    f["tnx_d252h"] = align(dist_hi(tnx), idx)
    f["tnx_dsma200"] = align((tnx / tnx.rolling(200).mean() - 1.0) * 100.0, idx)

    tlt_lo = align(dist_lo(px["TLT"]["Close"]), idx)
    hyg_hi = align(dist_hi(px["HYG"]["Close"]), idx)
    f["tlt_d252l"] = tlt_lo
    f["hyg_d252h"] = hyg_hi
    f["credit_div"] = tlt_lo - hyg_hi
    return f


# ---------------------------------------------------------------------------
# neighbour bookkeeping
# ---------------------------------------------------------------------------
def eligible(f: pd.DataFrame, cols: list[str]) -> pd.DatetimeIndex:
    """Days with every feature defined, outside today's own EXCLUDE_TD window."""
    ok = f[cols].dropna().index
    cutoff = f.index[max(0, len(f.index) - 1 - EXCLUDE_TD)]
    return ok[ok < cutoff]


def year_concentration(dates: pd.DatetimeIndex) -> str:
    yrs = pd.Series(pd.DatetimeIndex(dates).year).value_counts().sort_index()
    tot = int(yrs.sum())
    parts = ", ".join(f"{y}: {n} ({100*n/tot:.0f}%)" for y, n in yrs.items())
    top = yrs.sort_values(ascending=False)
    return (f"{tot} dates over {len(yrs)} calendar years | {parts} | "
            f"top year {top.index[0]} holds {100*top.iloc[0]/tot:.0f}%")


def print_neighbours(dates: pd.DatetimeIndex, dist: pd.Series | None,
                     f: pd.DataFrame, title: str) -> None:
    print(f"\n=== NEIGHBOUR DATES: {title} ===")
    cols = ["spy_d252h", "spy_dsma200", "spy_rank21", "spy_volratio",
            "breadth200", "tnx_d252h", "credit_div"]
    tab = f.loc[dates, cols].copy()
    if dist is not None:
        tab.insert(0, "dist", dist.loc[dates])
    tab.index = [str(d.date()) for d in dates]
    print(tab.round(2).to_string())
    print("  YEAR CONCENTRATION: " + year_concentration(dates))


# ---------------------------------------------------------------------------
# forward returns
# ---------------------------------------------------------------------------
def fwd_table(px: dict[str, pd.DataFrame], dates: pd.DatetimeIndex,
              label: str, hs=(1, 3, 5, 10),
              instruments: list[str] | None = None) -> pd.DataFrame:
    """Per-instrument forward returns at the neighbour dates vs that
    instrument's own all-days drift at the same horizon. lag=1, MOC entry."""
    rows = []
    for tkr in (instruments or INSTRUMENTS):
        s = px[tkr]["Close"]
        for h in hs:
            r = fwd_lag(s, h, lag=1)
            v = r.reindex(pd.DatetimeIndex(dates)).dropna()
            base = r.dropna()
            base = base[base.index <= ASOF]
            if len(v) == 0:
                rows.append({"ticker": tkr, "h": h, "n": 0})
                continue
            d = summarize(v.values, f"{tkr} h={h}")
            w = int((v.values > 0).sum())
            base_hit = float((base.values > 0).mean())
            rows.append({
                "ticker": tkr, "h": h, "n": d["n"],
                "mean_pct": round(d["mean_pct"], 3),
                "med_pct": round(d["median_pct"], 3),
                "hit": round(d["hit"], 1),
                "rec": f"{w}-{d['n'] - w}",
                "sign_p": round(sign_test(w, d["n"], base_hit), 4),
                "allday_pct": round(100 * float(base.mean()), 3),
                "allday_hit": round(100 * base_hit, 1),
                "edge_pct": round(d["mean_pct"] - 100 * float(base.mean()), 3),
                "worst_pct": round(d["worst_pct"], 2),
                "best_pct": round(d["best_pct"], 2),
                "t": round(d["t"], 2) if np.isfinite(d["t"]) else np.nan,
            })
    df = pd.DataFrame(rows)
    print(f"\n=== FORWARD RETURNS: {label} "
          f"(lag=1 MOC entry; sign_p is vs the instrument's OWN all-days "
          f"up-rate, not vs 0.5) ===")
    print(df.to_string(index=False))
    return df


def concentration_note(px: dict[str, pd.DataFrame], dates: pd.DatetimeIndex,
                       tkr: str, h: int) -> str:
    s = px[tkr]["Close"]
    v = fwd_lag(s, h, lag=1).reindex(pd.DatetimeIndex(dates)).dropna()
    if len(v) == 0:
        return "n/a"
    return cluster_note(v.index, v.values, k=2)
