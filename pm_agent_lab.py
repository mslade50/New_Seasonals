"""PM Weekly research helpers: week mechanics, climatology, lab wrappers.

Check scripts (PM_AGENT_HOME/checks/<asof>/*.py, run through
scripts/pm_agent_run_check.py) import this:

    import pm_agent_lab as lab
    px = lab.prices(["SPY", "^VIX"])          # RAW closes from the PM cache
    fwd = lab.fwd_returns(px["SPY"], 5, lag=0) # FRACTIONS
    out = lab.study(mask, fwd, decluster_td=5) # PERCENT summary

Same conventions as risk_agent_lab (whose functions these wrap with the PM
cache directory). Week mechanics and climatology are shared by the state
builder, the publisher and the grader so all three agree on what a "week" is.

Agent-product module: the book and the Risk Agent must not import it.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

import pm_agent_universe as U
import risk_agent_lab as _ral
from trading_calendar import TRADING_DAY

Q10_Z = 1.2815515655446004


# ---------------------------------------------------------------------------
# lab wrappers (PM cache by default)
# ---------------------------------------------------------------------------
def _cd(cache_dir):
    return Path(cache_dir) if cache_dir else U.cache_dir()


def prices(symbols=None, field: str = "Close", start=None, cache_dir=None) -> pd.DataFrame:
    return _ral.prices(symbols, field=field, start=start, cache_dir=_cd(cache_dir))


def ohlc(symbol: str, cache_dir=None) -> pd.DataFrame:
    return _ral.ohlc(symbol, cache_dir=_cd(cache_dir))


def dashboard_history(cache_dir=None) -> pd.DataFrame:
    return _ral.dashboard_history(cache_dir=_cd(cache_dir))


def iv_history(ticker: str, cache_dir=None) -> pd.Series:
    return _ral.iv_history(ticker, cache_dir=_cd(cache_dir))


fwd_returns = _ral.fwd_returns
study = _ral.study


# ---------------------------------------------------------------------------
# week mechanics
# ---------------------------------------------------------------------------
def target_week(asof) -> dict:
    """The ISO week after the one containing `asof` (the anchor session).

    resolves_on is that week's last NYSE session; horizon_td counts sessions in
    (asof, resolves_on]. A holiday-shortened week simply has fewer sessions.
    """
    a = pd.Timestamp(asof).normalize()
    monday = a - pd.Timedelta(days=a.weekday()) + pd.Timedelta(days=7)
    friday = monday + pd.Timedelta(days=4)
    sessions = pd.date_range(monday, friday, freq=TRADING_DAY)
    if len(sessions) == 0:      # a week with no sessions: roll to the next one
        return target_week(friday)
    between = pd.date_range(a + pd.Timedelta(days=1), sessions[-1], freq=TRADING_DAY)
    return {"week_of": str(monday.date()), "first_session": str(sessions[0].date()),
            "resolves_on": str(sessions[-1].date()), "sessions": [str(s.date()) for s in sessions],
            "horizon_td": int(len(between))}


def week_key(asof) -> str:
    """ISO week of the anchor, e.g. 2026-W41. One brief per key."""
    y, w, _ = pd.Timestamp(asof).isocalendar()
    return f"{y}-W{w:02d}"


def prior_week_close(close: pd.Series, asof) -> tuple[pd.Timestamp | None, float | None]:
    """Last bar strictly before the ISO week containing `asof`."""
    a = pd.Timestamp(asof).normalize()
    monday = a - pd.Timedelta(days=a.weekday())
    s = close.dropna()
    s = s[s.index < monday]
    if s.empty:
        return None, None
    return s.index[-1], float(s.iloc[-1])


# ---------------------------------------------------------------------------
# climatology (computed by code; the agent never sets its own baseline)
# ---------------------------------------------------------------------------
def _q(v: np.ndarray, q: float) -> float | None:
    return float(np.quantile(v, q)) if len(v) else None


def climatology(close: pd.Series, asof, horizon_td: int, kind: str,
                years: int = U.CLIMATOLOGY_YEARS) -> dict:
    """Unconditional distribution of h-session moves over the trailing years.

    kind "pct": close-to-close percent return. kind "points": level change.
    Uses every overlapping window for the quantiles; n_independent = n // h is
    the honest sample size.
    """
    s = close.dropna()
    a = pd.Timestamp(asof)
    s = s[(s.index <= a) & (s.index > a - pd.DateOffset(years=years))]
    h = max(int(horizon_td), 1)
    if len(s) <= h + 20:
        return {"n": 0, "n_independent": 0, "p_up": None, "q10": None, "q90": None,
                "median": None, "horizon_td": h, "kind": kind}
    v = s.to_numpy(float)
    moves = (v[h:] / v[:-h] - 1.0) * 100.0 if kind == "pct" else v[h:] - v[:-h]
    moves = moves[np.isfinite(moves)]
    return {"n": int(len(moves)), "n_independent": int(len(moves) // h),
            "p_up": round(float((moves > 0).mean()), 4),
            "q10": round(_q(moves, 0.10), 3), "q90": round(_q(moves, 0.90), 3),
            "median": round(_q(moves, 0.50), 3), "horizon_td": h, "kind": kind,
            "window": [str(s.index[0].date()), str(s.index[-1].date())]}


def implied_band(vix_level: float | None, horizon_td: int) -> dict:
    """Normal-approx q10/q90 of the SPY return implied by VIX (percent).

    A price, not a forecast: VIX embeds a variance risk premium, so this band is
    usually wider than what realises.
    """
    if vix_level is None or not math.isfinite(vix_level) or vix_level <= 0:
        return {}
    sigma = float(vix_level) * math.sqrt(max(int(horizon_td), 1) / 252.0)
    return {"sigma_pct": round(sigma, 3), "q10_pct": round(-Q10_Z * sigma, 3),
            "q90_pct": round(Q10_Z * sigma, 3)}
