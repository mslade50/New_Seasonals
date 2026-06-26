"""Shared trade-ticket simulator for seasonal ideas.

ONE simulator, used by BOTH the forward scorer (score_seasonal_ideas.py) and the
walk-forward backtest (backtest_seasonal_ideas.py), so a live track record and a
historical track record are scored by identical logic (the parity lesson).

A seasonal idea's TICKET is a frozen plan:
    BUY ~53.63 | stop 54.25 (0.6 ATR) | target 52.39 | time-stop 5td | R/R 2.0
parse_ticket() lifts that into structured fields; simulate_ticket() walks the
forward bars and returns the realized outcome.

Conventions (documented because they drive every number downstream):
  - Entry: T+1 open by default (an idea is published after the asof close, so the
    first tradeable price is the next session's open). entry_mode='asof_close'
    enters at the asof close for a faithful-to-the-seasonal-stat sensitivity run.
  - Risk per unit = |ticket_entry - ticket_stop| (the PLANNED risk). Realized R is
    measured against that, so R is comparable to the advertised R/R even when the
    actual fill (open) differs from the ticket's reference entry.
  - Window: the seasonal time-stop = `time_stop_days` trading days after the asof
    (the close[t+N]-close[t] window the edge is measured over). Stops/targets are
    checked intrabar on each bar in the window; if neither triggers, the trade
    time-exits at the last window bar's close.
  - Within-bar tie: stop checked before target (conservative), matching the
    book-wide convention that ambiguous intrabar timing favors the stop.
  - Price basis is the CALLER's choice (raw vs adjusted) — the scorer passes raw
    bars (real-order realism), the backtest passes adjusted bars (scale-invariant
    total-return). The logic here is identical either way.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# BUY ~53.63 | stop 54.25 (0.6 ATR) | target 52.39 | time-stop 5td | R/R 2.0
_TICKET_RE = re.compile(
    r"(BUY|SELL)\s+~?(-?[\d.]+)\s*\|\s*stop\s+(-?[\d.]+).*?"
    r"target\s+(-?[\d.]+).*?time-stop\s+(\d+)\s*td.*?R/R\s+(-?[\d.]+)",
    re.IGNORECASE,
)


def parse_ticket(cand: dict) -> dict | None:
    """Lift a candidate's frozen TICKET into structured fields. Returns None for
    non-tradeable rows (context/regime with no TICKET, or an unparseable string).
    Prefers a structured `ticket` dict if a future emit adds one."""
    if not isinstance(cand, dict):
        return None
    tk = cand.get("ticket")
    if isinstance(tk, dict) and {"entry", "stop", "target"} <= set(tk):
        entry, stop, target = float(tk["entry"]), float(tk["stop"]), float(tk["target"])
        tsd = int(tk.get("time_stop_days") or _horizon_to_days(cand.get("horizon")))
        direction = cand.get("direction", "long")
        rr = float(tk.get("rr") or 0.0)
    else:
        ev = cand.get("evidence") or {}
        s = ev.get("TICKET")
        if not s:
            return None
        m = _TICKET_RE.search(str(s))
        if not m:
            return None
        verb, entry, stop, target, tsd, rr = m.groups()
        direction = "long" if verb.upper() == "BUY" else "short"
        entry, stop, target, rr = float(entry), float(stop), float(target), float(rr)
        tsd = int(tsd)
    if not (np.isfinite(entry) and np.isfinite(stop) and np.isfinite(target)) or tsd <= 0:
        return None
    if abs(entry - stop) <= 0:
        return None
    return {
        "ticker": str(cand.get("ticker", "")).upper(),
        "channel": cand.get("channel"),
        "direction": direction,
        "horizon": cand.get("horizon"),
        "conviction": cand.get("conviction"),
        "p_value": cand.get("p_value"),
        "asof": cand.get("asof"),
        "entry": entry, "stop": stop, "target": target,
        "time_stop_days": tsd, "rr": rr,
        "headline": cand.get("headline"),
    }


def _horizon_to_days(h) -> int:
    if not h:
        return 5
    m = re.search(r"(\d+)", str(h))
    return int(m.group(1)) if m else 5


def _atr_at(df: pd.DataFrame, asof, n: int = 14) -> float:
    """Wilder ATR(n) using bars <= asof (matches seasonal_edge.atr_wilder)."""
    d = df[df.index <= asof]
    if len(d) < n + 1:
        return float("nan")
    h, l, c = d["High"], d["Low"], d["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1.0 / n, adjust=False).mean().iloc[-1])


def simulate_ticket(tk: dict, price_df: pd.DataFrame, asof,
                    entry_mode: str = "t1_open", stop_first: bool = True,
                    entry_atr_mult: float = 0.25) -> dict | None:
    """Walk forward bars and realize the ticket. price_df: OHLC indexed by date
    (raw or adjusted — caller's choice). Returns an outcome dict, or None if the
    idea has not matured yet (no post-asof bars / window not complete)."""
    if price_df is None or price_df.empty:
        return None
    df = price_df.copy()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    asof = pd.Timestamp(asof).normalize()
    fwd = df[df.index > asof]
    if fwd.empty:
        return None  # not matured: no bar after the signal yet

    risk = abs(tk["entry"] - tk["stop"])
    sign = 1.0 if tk["direction"] == "long" else -1.0
    stop, target = tk["stop"], tk["target"]
    n = int(tk["time_stop_days"])

    if entry_mode == "asof_close":
        if asof not in df.index:
            return None
        entry_price = float(df.loc[asof, "Close"])
        entry_date = asof
        window = fwd.iloc[:n]
    elif entry_mode == "limit":
        # Limit on the favorable side of the T+1 open: long buys at
        # open - mult*ATR, short sells at open + mult*ATR. Filled only if the
        # T+1 bar trades through it; otherwise the idea is MISSED (no trade).
        t1 = fwd.iloc[0]
        o = float(t1["Open"])
        atr = _atr_at(df, asof)
        if not np.isfinite(atr) or atr <= 0:
            return None
        if tk["direction"] == "long":
            lim = o - entry_atr_mult * atr
            filled = float(t1["Low"]) <= lim
        else:
            lim = o + entry_atr_mult * atr
            filled = float(t1["High"]) >= lim
        if not filled:
            return {"filled": False, "exit_type": "NoFill", "R": np.nan,
                    "entry_date": pd.Timestamp(fwd.index[0]), "entry_price": round(o, 4),
                    "limit_price": round(float(lim), 4), "exit_date": pd.NaT,
                    "exit_price": np.nan, "mae_R": np.nan, "mfe_R": np.nan,
                    "bars_held": 0, "risk_per_unit": round(float(risk), 4)}
        entry_price = float(lim)
        entry_date = fwd.index[0]
        window = fwd.iloc[:n]
    else:  # t1_open
        entry_price = float(fwd.iloc[0]["Open"])
        entry_date = fwd.index[0]
        window = fwd.iloc[:n]

    if window.empty:
        return None
    # require the window to be complete for a settled outcome (else still open)
    matured = len(fwd) >= n

    exit_price = exit_date = exit_type = None
    mae = mfe = 0.0  # in R, signed against the trade
    for dt, row in window.iterrows():
        hi, lo, close = float(row["High"]), float(row["Low"]), float(row["Close"])
        # running excursion in R (favorable / adverse) using intrabar extremes
        fav = sign * ((hi if sign > 0 else lo) - entry_price) / risk
        adv = sign * ((lo if sign > 0 else hi) - entry_price) / risk
        mfe = max(mfe, fav)
        mae = min(mae, adv)
        if tk["direction"] == "long":
            hit_stop, hit_tgt = lo <= stop, hi >= target
        else:
            hit_stop, hit_tgt = hi >= stop, lo <= target
        order = [("Stop", hit_stop, stop), ("Target", hit_tgt, target)]
        if not stop_first:
            order = order[::-1]
        for etype, hit, lvl in order:
            if hit:
                exit_price, exit_date, exit_type = float(lvl), dt, etype
                break
        if exit_type:
            break

    if exit_type is None:
        if not matured:
            return None  # window not finished — still live
        last = window.iloc[-1]
        exit_price, exit_date, exit_type = float(last["Close"]), window.index[-1], "Time"

    R = sign * (exit_price - entry_price) / risk
    return {
        "filled": True,
        "entry_date": pd.Timestamp(entry_date), "entry_price": round(entry_price, 4),
        "exit_date": pd.Timestamp(exit_date), "exit_price": round(exit_price, 4),
        "exit_type": exit_type, "R": round(float(R), 4),
        "mae_R": round(float(mae), 3), "mfe_R": round(float(mfe), 3),
        "bars_held": int((window.index <= exit_date).sum()),
        "risk_per_unit": round(float(risk), 4),
    }
