"""Macro event calendar — load + trading-day join helpers for backtests.

Data lives in data/macro_events.csv (committed), built by
scripts/build_macro_calendar.py from cached Fed/BLS pages. Event types:

    fomc_decision, fomc_intermeeting, fomc_minutes, cpi, nfp, ppi,
    jackson_hole, opex, quad_witching, election

Dates are ACTUAL release/announcement dates (shutdown-delayed BLS releases
carry their real dates; the canceled Oct 2025 CPI/NFP/PPI have no rows).
Rows dated after the build date are scheduled, not actual.

Typical backtest join:

    from macro_calendar import event_window_mask, td_offset
    off = td_offset(px.index, 'fomc_decision')   # signed td distance
    pre_fomc = off.isin([-3, -2, -1])            # 3 sessions before decision
    cpi_day = event_window_mask(px.index, 'cpi', 0, 0)

Offsets are measured on the trading index you pass in (your price data's
own sessions). Events falling on non-trading days (e.g. the Sunday
2020-03-15 emergency cut) map forward to the next session in the index.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

CSV_PATH = Path(__file__).resolve().parent / "data" / "macro_events.csv"

EVENT_TYPES = [
    "fomc_decision", "fomc_intermeeting", "fomc_minutes", "cpi", "nfp",
    "ppi", "jackson_hole", "opex", "quad_witching", "vix_expiry", "election",
]


@lru_cache(maxsize=1)
def _load_raw() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    df["date"] = df["date"].dt.normalize()
    return df


def load_macro_events(events: list[str] | None = None) -> pd.DataFrame:
    """Full calendar (or a subset of event types), sorted by date."""
    df = _load_raw()
    if events is not None:
        unknown = set(events) - set(EVENT_TYPES)
        if unknown:
            raise ValueError(f"unknown event types: {sorted(unknown)}")
        df = df[df["event"].isin(events)]
    return df.reset_index(drop=True)


def event_dates(event: str) -> pd.DatetimeIndex:
    df = load_macro_events([event])
    return pd.DatetimeIndex(df["date"].unique()).sort_values()


def _event_positions(index: pd.DatetimeIndex, event: str) -> np.ndarray:
    """Positions of each event in the trading index, mapped forward to the
    next session for non-trading event dates. Events outside the index
    range are dropped."""
    idx = pd.DatetimeIndex(index).normalize()
    dates = event_dates(event)
    pos = idx.searchsorted(dates, side="left")
    keep = pos < len(idx)
    return np.unique(pos[keep])


def td_offset(index: pd.DatetimeIndex, event: str) -> pd.Series:
    """Signed trading-day distance from each bar to the NEAREST event.

    0 = event day (or the session an off-day event mapped forward to),
    negative = event upcoming (-3 = three sessions before it),
    positive = sessions since the last event. NaN if the index contains
    no occurrence of the event.
    """
    idx = pd.DatetimeIndex(index)
    pos = _event_positions(idx, event)
    if len(pos) == 0:
        return pd.Series(np.nan, index=idx, name=f"td_to_{event}")
    bar = np.arange(len(idx))
    nxt = np.searchsorted(pos, bar, side="left")      # next event >= bar
    prv = nxt - 1                                      # last event <= bar-ish
    d_next = np.where(nxt < len(pos), pos[np.minimum(nxt, len(pos) - 1)] - bar,
                      np.iinfo(np.int32).max)
    d_prev = np.where(prv >= 0, bar - pos[np.maximum(prv, 0)],
                      np.iinfo(np.int32).max)
    off = np.where(d_next <= d_prev, -d_next, d_prev)
    return pd.Series(off, index=idx, name=f"td_to_{event}")


def event_window_mask(index: pd.DatetimeIndex, event: str,
                      before_td: int, after_td: int) -> pd.Series:
    """True for bars from before_td sessions BEFORE each event through
    after_td sessions AFTER it (inclusive; event day = 0 is always in)."""
    idx = pd.DatetimeIndex(index)
    pos = _event_positions(idx, event)
    mask = np.zeros(len(idx), dtype=bool)
    for p in pos:
        mask[max(0, p - before_td): p + after_td + 1] = True
    return pd.Series(mask, index=idx, name=f"{event}_window")


def next_event(asof: pd.Timestamp | str, event: str) -> pd.Timestamp | None:
    """First occurrence strictly after `asof` (calendar basis) — for live
    scans. None if the calendar has nothing scheduled beyond `asof`."""
    dates = event_dates(event)
    ts = pd.Timestamp(asof).normalize()
    after = dates[dates > ts]
    return after[0] if len(after) else None


def add_event_offsets(df: pd.DataFrame,
                      events: list[str] | None = None) -> pd.DataFrame:
    """Return a copy of a date-indexed frame with td_to_{event} columns."""
    events = events or ["fomc_decision", "cpi", "nfp"]
    out = df.copy()
    for ev in events:
        out[f"td_to_{ev}"] = td_offset(df.index, ev)
    return out
