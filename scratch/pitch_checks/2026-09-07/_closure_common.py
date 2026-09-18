"""Shared substrate for the 2026-09-07 post-closure lane survey.

Anchor convention (stated once, used by every 01_closure_lane_*.py script):

- The NYSE session calendar is SPY's own index in master_prices (2000-01-03+).
- A CLOSURE is a gap between consecutive NYSE sessions of >= 3 calendar days.
  gap == 3 is the ordinary Fri->Mon weekend (the CONTROL cell).
  gap >= 4 is the holiday case (the LIVE cell today: Fri 2026-09-04 ->
  Tue 2026-09-08 is 4 days).
- k=0 is the FIRST SESSION AFTER the gap (the anchor). The decision is made
  before that session opens, so both entry forms are genuinely lag=1.
- Two entry forms, each holding exactly h trading sessions:
    MOO(h): Open[k0]  -> Close[k0 + h - 1]
    MOC(h): Close[k0] -> Close[k0 + h]
  They differ by one session of shift; that is the point of reporting both.
- The GAP leg is Open[k0] / Close[k0-1] - 1. NEITHER entry form captures it
  (both are placed after it prints). It is a state variable, not a leg.

Costs are per-proxy round-trip in bps; the slot bar is 3x that.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]   # <repo root>/scratch/pitch_checks/<date> -> repo root
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from pitch_lab import load_prices, local_control, summarize, sign_test, load_events  # noqa: E402

# class -> proxy, exactly the class table the brief specified
PROXIES: list[tuple[str, str]] = [
    ("us_large", "SPY"),
    ("us_small", "IWM"),
    ("rates", "TLT"),
    ("rates", "IEF"),
    ("credit", "HYG"),
    ("credit", "LQD"),
    ("gold", "GLD"),
    ("miners", "GDX"),
    ("metals", "SLV"),
    ("energy", "USO"),
    ("energy", "XLE"),
    ("dollar", "UUP"),
    ("dollar", "DX-Y.NYB"),
    ("international", "EFA"),
    ("international", "EEM"),
    ("volatility", "^VIX"),
    ("volatility", "SVXY"),
]
TICKERS = [t for _, t in PROXIES]

# round-trip cost in bps (two sides). Slot bar = 3x.
COST_RT_BPS = {
    "SPY": 4.0, "IWM": 5.0, "TLT": 5.0, "IEF": 6.0, "HYG": 6.0, "LQD": 6.0,
    "GLD": 5.0, "GDX": 6.0, "SLV": 7.0, "USO": 8.0, "XLE": 5.0, "UUP": 8.0,
    "EFA": 6.0, "EEM": 6.0, "SVXY": 18.0,
    "DX-Y.NYB": np.nan,   # index, not tradeable
    "^VIX": np.nan,       # index, not tradeable
}
HORIZONS = (1, 2, 3, 5, 10)

# unscheduled (non-holiday) closures inside the gap>=4 set -- kept in the
# headline cell because they ARE closures, reported separately as sensitivity
UNSCHEDULED = {pd.Timestamp("2001-09-17"), pd.Timestamp("2012-10-31")}


# ---------------------------------------------------------------------------
def load_panel() -> dict[str, pd.DataFrame]:
    px = load_prices(TICKERS)
    return {t: px[t].sort_index() for t in px}


def nyse_calendar(px: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """SPY's index IS the NYSE session calendar for this survey."""
    return pd.DatetimeIndex(px["SPY"].index)


def closure_table(cal: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per anchor: the first session after a >=3 calendar-day gap."""
    prev = cal[:-1]
    anch = cal[1:]
    gap = (anch - prev).days
    df = pd.DataFrame({"anchor": anch, "prev": prev, "gap_days": gap})
    df = df[df["gap_days"] >= 3].reset_index(drop=True)
    df["kind"] = np.where(df["gap_days"] >= 4, "holiday", "weekend")
    # Labor Day = anchor is the Tuesday after the first Monday of September
    df["labor_day"] = (
        (df["anchor"].dt.month == 9)
        & (df["anchor"].dt.day <= 9)
        & (df["anchor"].dt.weekday == 1)
        & (df["gap_days"] >= 4)
    )
    df["unscheduled"] = df["anchor"].isin(UNSCHEDULED)
    return df


def runway_map(cal: pd.DatetimeIndex,
               kinds=("nfp", "cpi", "ppi", "fomc_decision")) -> pd.Series:
    """Trading sessions from each NYSE session to the NEXT scheduled print
    (strictly after it). NaN when no future print is in the calendar."""
    ev = load_events(list(kinds))["date"]
    pos = pd.Series(range(len(cal)), index=cal)
    # map each print onto the first session on/after it
    ev_pos = np.unique(np.asarray(cal.searchsorted(pd.DatetimeIndex(ev))))
    ev_pos = ev_pos[ev_pos < len(cal)]
    out = np.full(len(cal), np.nan)
    for i in range(len(cal)):
        j = int(np.searchsorted(ev_pos, i, side="right"))
        if j < len(ev_pos):
            out[i] = ev_pos[j] - i
    return pd.Series(out, index=cal)


# ---------------------------------------------------------------------------
def moo_series(df: pd.DataFrame, h: int) -> pd.Series:
    """Open[p] -> Close[p+h-1], aligned to p. Holds h sessions."""
    return df["Close"].shift(-(h - 1)) / df["Open"] - 1.0


def moc_series(df: pd.DataFrame, h: int) -> pd.Series:
    """Close[p] -> Close[p+h], aligned to p. Holds h sessions."""
    return df["Close"].shift(-h) / df["Close"] - 1.0


def gap_series(df: pd.DataFrame) -> pd.Series:
    """Open[p] / Close[p-1] - 1, aligned to p."""
    return df["Open"] / df["Close"].shift(1) - 1.0


def anchors_on(df: pd.DataFrame, dates) -> pd.DatetimeIndex:
    """Anchor dates that exist EXACTLY in this instrument's index.

    Exact membership, not searchsorted: a NYSE closure anchor that the
    instrument did not trade is not that instrument's post-closure session,
    and silently sliding it to the next bar would mint a fake anchor. Callers
    report how many survived.
    """
    idx = pd.DatetimeIndex(df.index)
    d = pd.DatetimeIndex(dates)
    return d.intersection(idx)


def cell(series: pd.Series, dates) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Values of `series` on `dates`, NaNs dropped, with the surviving dates."""
    d = pd.DatetimeIndex(dates).intersection(series.index)
    v = series.loc[d]
    v = v[v.notna()]
    return v.values.astype(float), pd.DatetimeIndex(v.index)


def fmt(r: dict) -> str:
    if not r.get("n"):
        return "n=0"
    return (f"n={r['n']:>4d} mean={r['mean_pct']:+.3f}% med={r['median_pct']:+.3f}% "
            f"hit={r['hit']:.0f}% t={r['t']:+.2f}")
