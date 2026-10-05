"""Stage A of the Daily Seasonal pipeline: deterministic state assembly.

Design: docs/seasonal_agent_design_2026-09-30.md; live rule:
docs/claude_ref/daily_seasonal.md. Same contract as build_pitch_state.py (and
it reuses that module's blocks rather than copying them): ONE json the
/daily-seasonal skill reads whole, plus a per-ticker tape file read by lookup.
Nothing here decides anything.

    python scripts/build_seasonal_state.py [--asof YYYY-MM-DD] [--out PATH]
                                           [--tape-out PATH] [--no-book]
                                           [--no-board] [--quiet]

Blocks in the payload (on top of the pitch's session/calendar/tape/risk/book/
earnings/pipeline, with earnings widened to 63 sessions):
    ranks          today's rank cross-section outliers (atr_seasonal_ranks
                   .parquet at the REPO ROOT), each with its class, sector,
                   agreeing horizons, expected-path turn, and cycle/all-years
                   counts measured from a T+1 entry (never lag 0)
    calendar_cells turn-of-month, holiday adjacency, weekday-of-month and the
                   macro-event offsets inside the next 10 sessions, with the
                   historical anchor count for each
    board          today's site seasonal-board tickets (evidence.TICKET rows),
                   built in-process with grades "all" and NOTHING written
    history        seasonal-journal fingerprints AND the pitch journal's, so
                   the agent never re-pitches a pitch idea
    watchlist, scoreboard, negative_registry (own file only; the pitch
                   registry is not read)

A missing PRICE cache or RANK file is fatal; everything else degrades to a
warning, as in the pitch builder.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import io
import json
import sys
import warnings as _warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pitch_journal  # noqa: E402
from pitch_grammar import REPEAT_BLOCK_TD  # noqa: E402
from pitch_products import PITCH, SEASONAL  # noqa: E402
from trading_calendar import TRADING_DAY  # noqa: E402
from live_scan_universe import canonical_ticker, exclude_retired_tickers  # noqa: E402

import build_pitch_state as bps  # noqa: E402
from build_context_state import (  # noqa: E402
    event_anchor_dates,
    holiday_adjacent_anchors,
    month_end_position,
    month_window_anchors,
    weekday_month_anchors,
)
from build_pitch_research_index import parse_registry  # noqa: E402
import seasonal_edge as se  # noqa: E402

DEFAULT_OUT = ROOT / "data" / "seasonal_state.json"
DEFAULT_TAPE_OUT = ROOT / "data" / "seasonal_tape.json"
RANKS_PATH = ROOT / "atr_seasonal_ranks.parquet"

EARNINGS_HORIZON_TD = 63
CELL_WINDOW_TD = 10
OUTLIER_HORIZONS = (5, 10, 21, 63)
ALL_HORIZONS = tuple(se.ATR_SZNL_WINDOWS)
HI, LO = 90.0, 10.0              # adjacent-agreement tails
HI_SINGLE, LO_SINGLE = 95.0, 5.0  # a single horizon this extreme qualifies
OUTLIER_CAP = 80
MIN_ADV_USD = 5_000_000.0
ADV_WINDOW = 21
PATH_TURN_TD = 5
CALENDAR_EVENTS = ("opex", "fomc_decision", "cpi", "nfp", "vix_expiry",
                   "quad_witching")

# The pitch skill's stage B1 class table, extended to the other macro/ETF
# names in the rank universe. Anything absent is "equity".
CLASS_MAP: dict[str, str] = {}
for _cls, _names in {
    "us_large": ["SPY", "QQQ", "^GSPC", "^NDX", "DIA", "^DJI", "^IXIC", "^DJT"],
    "us_small": ["IWM", "^RUT", "^MID", "MDY", "IJR"],
    "rates": ["TLT", "IEF", "^TNX", "TIP", "AGG", "SHY", "ZB=F", "ZN=F"],
    "credit": ["HYG", "LQD", "JNK"],
    "gold": ["GLD", "GDX", "GDXJ", "GC=F", "CEF", "IAU"],
    "metals": ["SLV", "SI=F", "HG=F", "PL=F", "PA=F", "CPER"],
    "energy": ["USO", "UNG", "DBC", "CL=F", "NG=F", "BNO"],
    "dollar_fx": ["UUP", "DX-Y.NYB", "EURUSD=X", "JPY=X", "GBPUSD=X",
                  "AUDUSD=X", "NZDUSD=X", "CAD=X", "CHF=X", "USDMXN=X",
                  "USDBRL=X", "USDZAR=X", "USDTRY=X", "FXE", "FXY"],
    "international": ["EFA", "EEM", "FXI", "EWZ", "EWJ", "^FTSE", "^GDAXI",
                      "^FCHI", "^N225", "^HSI", "^STI", "^AXJO", "^KS11",
                      "^TWII", "^BSESN", "^GSPTSE", "^MXX", "^BVSP",
                      "^STOXX50E"],
    "volatility": ["^VIX", "^VIX3M", "^MOVE", "SVXY", "UVXY", "VXX"],
    "agriculture": ["ZC=F", "ZW=F", "ZS=F", "KC=F", "CC=F", "SB=F", "CT=F"],
    "crypto": ["BTC-USD", "ETH-USD"],
}.items():
    for _t in _names:
        CLASS_MAP[_t] = _cls


def asset_class(ticker: str) -> str:
    return CLASS_MAP.get(str(ticker).upper(), "equity")


def has_volume(ticker: str) -> bool:
    """Indices, FX pairs, futures and crypto carry no meaningful share volume,
    so the $ADV floor applies only to listed shares and ETFs."""
    t = str(ticker).upper()
    return not (t.startswith("^") or "=" in t or t.endswith("-USD")
                or t == "DX-Y.NYB")


def _r(value, digits: int = 2):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return round(value, digits) if np.isfinite(value) else None


# ---------------------------------------------------------------------------
# outliers
# ---------------------------------------------------------------------------
def _runs(flags: list[bool]) -> list[list[int]]:
    """Index runs of consecutive True flags."""
    out, cur = [], []
    for i, flag in enumerate(flags):
        if flag:
            cur.append(i)
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def classify_ranks(ranks: dict[int, float]) -> dict | None:
    """Is this rank vector an outlier, on which side, and which horizons agree?

    Outlier: rank <= 10 or >= 90 at any of 5/10/21/63 with at least two
    ADJACENT horizons agreeing in the same tail, OR a single horizon <= 5 /
    >= 95. When both sides qualify, the more extreme side wins and the
    conflict is flagged."""
    candidates = []
    for side, tail, single in (("long", lambda v: v >= HI, lambda v: v >= HI_SINGLE),
                               ("short", lambda v: v <= LO, lambda v: v <= LO_SINGLE)):
        vals = [ranks.get(h) for h in OUTLIER_HORIZONS]
        flags = [v is not None and np.isfinite(v) and tail(v) for v in vals]
        runs = [r for r in _runs(flags) if len(r) >= 2]
        agree = [OUTLIER_HORIZONS[i] for i in max(runs, key=len)] if runs else []
        singles = [OUTLIER_HORIZONS[i] for i, v in enumerate(vals)
                   if v is not None and np.isfinite(v) and single(v)]
        if not agree and not singles:
            continue
        horizons = agree or [max(singles, key=lambda h: abs(ranks[h] - 50))]
        extremity = max(abs(ranks[h] - 50) for h in (agree or singles))
        candidates.append({"side": side, "agree_horizons": horizons,
                           "adjacent": bool(agree), "extremity": extremity})
    if not candidates:
        return None
    candidates.sort(key=lambda c: -c["extremity"])
    best = dict(candidates[0])
    best["conflict"] = len(candidates) > 1
    return best


def price_frame(prices: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    sub = prices[prices["ticker"] == ticker]
    if sub.empty:
        return None
    frame = sub.set_index("date").sort_index()
    frame.index = pd.DatetimeIndex(frame.index).normalize()
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame[["Open", "High", "Low", "Close", "Volume"]].astype(float)


def window_stats(frame: pd.DataFrame, asof: pd.Timestamp, horizon: int,
                 side: str, phase: int | None, entry_lag: int = 1) -> dict:
    """Counts for this calendar window measured from a T+`entry_lag` entry:
    n years, k in the idea's direction, mean % and mean in ATR units (each
    year's move over its own Wilder ATR at the ENTRY bar)."""
    stat = se.seasonal_window_returns(frame, asof, horizon,
                                      cycle_phase_filter=phase,
                                      entry_lag=entry_lag)
    if not stat or stat.get("insufficient"):
        return {"n": (stat or {}).get("n", 0), "k": None, "mean_pct": None,
                "mean_atr": None, "insufficient": True}
    k = stat["n_up"] if side == "long" else stat["n_down"]

    close = frame["Close"].dropna().sort_index()
    atr = se.atr_wilder(frame).reindex(close.index).to_numpy(float)
    cv = close.to_numpy(float)
    doy = se._trading_doy(close.index).to_numpy()
    years = close.index.year.to_numpy().astype(np.int64)
    le = close.index.values <= np.datetime64(pd.Timestamp(asof).normalize())
    picks = se._window_pick_positions(doy, years, int(doy[le][-1]),
                                      pd.Timestamp(asof).year, phase, 2, True)
    moves = []
    for p in picks:
        e, x = p + entry_lag, p + entry_lag + horizon
        if x < cv.size and np.isfinite(atr[e]) and atr[e] > 0:
            moves.append((cv[x] - cv[e]) / atr[e])
    return {"n": stat["n"], "k": int(k),
            "mean_pct": _r(100 * stat["mean"]),
            "mean_atr": _r(np.mean(moves)) if moves else None,
            "years": stat["years"][-10:]}


def path_turn(frame: pd.DataFrame, asof: pd.Timestamp, side: str) -> int | None:
    """Sessions until the expected path's nadir (long) / peak (short), if it
    lands inside the next PATH_TURN_TD; 0 when the path never goes adverse
    (enter now); None when the turn is later or the path is unavailable."""
    path = se.expected_seasonal_path(frame, asof, 2 * PATH_TURN_TD)
    if path is None or not np.isfinite(path).any():
        return None
    signed = path if side == "long" else -path
    idx = int(np.nanargmin(signed))
    if signed[idx] >= 0:
        return 0
    return idx + 1 if idx + 1 <= PATH_TURN_TD else None


def build_ranks(prices: pd.DataFrame, today: pd.Timestamp, warnings: list[str],
                ranks: pd.DataFrame | None = None,
                sectors: dict[str, str] | None = None,
                cap: int = OUTLIER_CAP) -> dict:
    ranks = se.load_seasonal_ranks(str(RANKS_PATH)) if ranks is None else ranks
    cs = se.seasonal_cross_section(today, ranks)
    if cs.empty:
        raise SystemExit(f"FATAL: no seasonal ranks on or before {today.date()}")
    active, retired = exclude_retired_tickers(cs.index, asof=today.date())
    cs = cs[cs.index.map(canonical_ticker).isin(active)]
    if retired:
        warnings.append(f"ranks: confirmed delisted symbols excluded from forward research: {retired}; history retained")
    if cs.empty:
        raise SystemExit(f"FATAL: no active seasonal ranks on or before {today.date()}")
    rank_date = pd.Timestamp(cs["Date"].max()).normalize()
    stale = cs[cs["Date"] < rank_date]
    if len(stale):
        warnings.append(f"ranks: {len(stale)} tickers carry a rank older than "
                        f"{rank_date.date()}")
    if rank_date < today:
        warnings.append(f"ranks: newest rank row {rank_date.date()} is before "
                        f"{today.date()} - the rank file may be stale")

    history = prices[prices["date"] < today]
    flagged = []
    for ticker, row in cs.iterrows():
        vec = {h: float(row[f"atr_sznl_{h}d"]) for h in ALL_HORIZONS
               if pd.notna(row.get(f"atr_sznl_{h}d"))}
        got = classify_ranks(vec)
        if got:
            flagged.append((ticker, vec, got))

    # Liquidity first, on everything flagged, so the cap sees tradeable names.
    tail = (history[history["ticker"].isin([t for t, _, _ in flagged])]
            .sort_values("date").groupby("ticker").tail(ADV_WINDOW))
    adv = (tail["Close"] * tail["Volume"]).groupby(tail["ticker"]).mean()
    kept, dropped_illiquid = [], 0
    for ticker, vec, got in flagged:
        value = adv.get(ticker)
        if has_volume(ticker) and (value is None or not np.isfinite(value)
                                   or value < MIN_ADV_USD):
            dropped_illiquid += 1
            continue
        kept.append((ticker, vec, got, value))
    kept.sort(key=lambda item: -item[2]["extremity"])

    sectors = sectors if sectors is not None else _load_sectors(warnings)
    by_class: dict[str, int] = {}
    by_sector: dict[str, int] = {}
    for ticker, _vec, _got, _adv in kept:
        cls = asset_class(ticker)
        by_class[cls] = by_class.get(cls, 0) + 1
        sec = sectors.get(ticker) or ("n/a" if cls == "equity" else cls)
        by_sector[sec] = by_sector.get(sec, 0) + 1

    phase = se.cycle_phase(today.year)
    outliers = []
    for ticker, vec, got, value in kept[:cap]:
        frame = price_frame(history, ticker)
        entry: dict = {
            "ticker": ticker, "class": asset_class(ticker),
            "sector": sectors.get(ticker),
            "ranks": {str(h): _r(vec.get(h), 1) for h in ALL_HORIZONS},
            "side": got["side"], "agree_horizons": got["agree_horizons"],
            "adjacent_agreement": got["adjacent"],
            "side_conflict": got["conflict"],
            "adv_usd_21d": _r(value, 0),
        }
        if frame is None or len(frame) < 300:
            entry["note"] = "no usable price history"
            outliers.append(entry)
            continue
        h = int(min(got["agree_horizons"]))
        atr = se.atr_wilder(frame).iloc[-1]
        entry.update({
            "stats_horizon_td": h,
            "path_turn_td": path_turn(frame, today, got["side"]),
            "cycle": {"phase": se.cycle_label(today.year),
                      **window_stats(frame, today, h, got["side"], phase)},
            "all_years": window_stats(frame, today, h, got["side"], None),
            "ext": {f"r{w}_pct": _r(se.trailing_return_pctile(frame, w), 1)
                    for w in (5, 10, 21)},
            "atr14": _r(atr, 4),
            "close": _r(frame["Close"].iloc[-1], 4),
            "last_bar": str(frame.index[-1].date()),
        })
        outliers.append(entry)

    return {"rank_date": str(rank_date.date()),
            "rank_file": "atr_seasonal_ranks.parquet",
            "horizons": list(ALL_HORIZONS),
            "n_tickers": int(len(cs)),
            "n_flagged": len(flagged),
            "n_dropped_illiquid": dropped_illiquid,
            "n_outliers": len(kept),
            "cap": cap,
            "rule": ("rank <= 10 or >= 90 at 5/10/21/63 with two adjacent "
                     "horizons agreeing, or one horizon <= 5 / >= 95; $5M "
                     "21d ADV floor on shares/ETFs; stats from a T+1 entry "
                     "over the smallest agreeing horizon"),
            "by_class": dict(sorted(by_class.items())),
            "by_sector": dict(sorted(by_sector.items(), key=lambda kv: -kv[1])),
            "outliers": outliers}


def _load_sectors(warnings: list[str]) -> dict[str, str]:
    try:
        frame = pd.read_parquet(ROOT / "data" / "sector_map.parquet")
        return {str(t).upper(): s for t, s in zip(frame["ticker"], frame["sector"])
                if isinstance(s, str) and s}
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"ranks: sector_map.parquet unavailable ({exc})")
        return {}


# ---------------------------------------------------------------------------
# calendar cells
# ---------------------------------------------------------------------------
def _is_pre_holiday(day: pd.Timestamp) -> bool:
    nxt = (day + TRADING_DAY).normalize()
    return (nxt - day).days > (3 if day.weekday() == 4 else 1)


def _is_post_holiday(day: pd.Timestamp) -> bool:
    prev = (day - TRADING_DAY).normalize()
    return (day - prev).days > (3 if prev.weekday() == 4 else 1)


def build_calendar_cells(today: pd.Timestamp, ref: pd.DatetimeIndex,
                         warnings: list[str]) -> dict:
    """Every calendar cell live inside [today, today + CELL_WINDOW_TD]. Each
    carries the number of historical anchors on the SPY session index (the
    same anchor definitions the Market Context sweep uses) so the agent can
    see how thin a cell is before opening it."""
    window = bps.sessions_from(today, CELL_WINDOW_TD + 1)
    cells: dict[str, dict] = {}

    def add(cell_id: str, kind: str, label: str, day: pd.Timestamp,
            n_hist: int | None, rule: str) -> None:
        cell = cells.setdefault(cell_id, {
            "cell_id": cell_id, "kind": kind, "label": label, "dates": [],
            "td_ahead": [], "n_hist_anchors": n_hist, "anchor_rule": rule})
        cell["dates"].append(str(day.date()))
        cell["td_ahead"].append(bps.td_ahead(today, day))

    n_month_end = len(month_window_anchors(ref, final_n=3, first_n=0))
    n_tom = len(month_window_anchors(ref, final_n=0, first_n=2))
    n_pre = len(holiday_adjacent_anchors(ref, "pre"))
    n_post = len(holiday_adjacent_anchors(ref, "post"))
    names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    for day in window:
        pos = month_end_position(day)
        month = day.strftime("%b %Y")
        if pos["td_from_month_end"] is not None and pos["td_from_month_end"] < 3:
            add(f"month_end|{month}", "month_end",
                f"final 3 sessions of {month}", day, n_month_end,
                "session is one of the month's last 3")
        if pos["td_of_month"] is not None and pos["td_of_month"] <= 2:
            add(f"turn_of_month|{month}", "turn_of_month",
                f"first 2 sessions of {month}", day, n_tom,
                "session is one of the month's first 2")
        if _is_pre_holiday(day):
            add(f"holiday_pre|{day.date()}", "holiday_pre",
                "last session before a market holiday", day, n_pre,
                "next weekday is not a session")
        if _is_post_holiday(day):
            add(f"holiday_post|{day.date()}", "holiday_post",
                "first session after a market holiday", day, n_post,
                "previous weekday was not a session")
        wd = int(day.weekday())
        if wd < 5:
            add(f"weekday_month|{wd}|{day.month}", "weekday_month",
                f"{names[wd]}s in {day.strftime('%B')}", day,
                len(weekday_month_anchors(ref, wd, int(day.month))),
                "bare day-of-week x month cell")

    events = []
    try:
        from macro_calendar import load_macro_events
        frame = load_macro_events(list(CALENDAR_EVENTS))
        end = window[-1]
        live = frame[(frame["date"] >= today) & (frame["date"] <= end)]
        for _, row in live.sort_values("date").iterrows():
            day = pd.Timestamp(row["date"]).normalize()
            k = bps.td_ahead(today, day)
            n_hist = None
            if k >= 1:
                try:
                    n_hist = len(event_anchor_dates(ref, row["event"], k)[0])
                except Exception:  # noqa: BLE001
                    n_hist = None
            events.append({"event": row["event"], "date": str(day.date()),
                           "td_ahead": k, "label": str(row.get("label", "") or ""),
                           "n_hist_anchors_at_offset": n_hist})
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"calendar_cells: macro events unavailable ({exc})")

    months = sorted({pd.Timestamp(d).strftime("%B") for d in window},
                    key=lambda m: pd.Timestamp(f"1 {m} 2000").month)
    return {"window_td": [0, CELL_WINDOW_TD],
            "window": [str(window[0].date()), str(window[-1].date())],
            "cells": sorted(cells.values(), key=lambda c: (c["td_ahead"][0],
                                                           c["cell_id"])),
            "events": events,
            "month_of_year": months,
            "cycle_year": se.cycle_label(today.year),
            "cycle_phase": se.cycle_phase(today.year)}


# ---------------------------------------------------------------------------
# board, history, registries
# ---------------------------------------------------------------------------
BOARD_FIELDS = ("channel", "ticker", "direction", "horizon", "headline",
                "conviction", "p_value", "evidence", "notes")


def build_board(board_asof: pd.Timestamp, warnings: list[str]) -> dict:
    """The site board's tickets for today, in-process with grades 'all'.
    daily_seasonal_ideas.build() is pure (it prints, it writes nothing); the
    file writes and ledger append live in its main(), which is not called."""
    try:
        import daily_seasonal_ideas as dsi
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            _md, payload = dsi.build(board_asof, grades=None)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"board: daily_seasonal_ideas.build failed ({exc})")
        return {"asof": str(board_asof.date()), "rows": [], "error": str(exc)}
    candidates = payload.get("candidates") or []
    active, _ = exclude_retired_tickers(
        [c.get("ticker", "") for c in candidates], asof=board_asof.date())
    candidates = [c for c in candidates if canonical_ticker(c.get("ticker", "")) in active]
    rows = [{k: c.get(k) for k in BOARD_FIELDS}
            for c in candidates
            if (c.get("evidence") or {}).get("TICKET")]
    return {"asof": str(board_asof.date()), "grades": "all",
            "n_candidates": len(candidates),
            "rows": json.loads(json.dumps(rows, default=str))}


def fingerprints_since(path: Path, today: pd.Timestamp,
                       warnings: list[str], label: str) -> dict[str, str]:
    since = str((today - REPEAT_BLOCK_TD * TRADING_DAY).normalize().date())
    try:
        records = pitch_journal.load(path, pull=False) if path.exists() else []
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"history: {label} journal unreadable ({exc})")
        return {}
    return pitch_journal.recent_fingerprints(records, since)


def build_history(today: pd.Timestamp, warnings: list[str],
                  seasonal_path: Path | None = None,
                  pitch_path: Path | None = None) -> dict:
    seasonal_path = seasonal_path or SEASONAL.journal_path
    pitch_path = pitch_path or PITCH.journal_path
    out = bps.build_history(today, warnings, journal_path=seasonal_path)
    out["pitch_recent_fingerprints"] = fingerprints_since(
        pitch_path, today, warnings, "pitch")
    out["pitch_journal"] = "read-only; a pitch fingerprint inside the window "\
        "is blocked here exactly like a seasonal one"
    return out


def registry_block(path: Path, warnings: list[str], label: str,
                   include_text: bool = True) -> dict:
    """The seasonal agent's own negative registry inlined, text and parsed
    entries. The pitch registry is never read (owner decision 2026-09-30)."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        warnings.append(f"{label}: {path.name} unreadable ({exc})")
        return {"path": str(path.name), "entries": []}
    rel = str(path.relative_to(ROOT)).replace("\\", "/") \
        if path.is_relative_to(ROOT) else str(path)
    out = {"path": rel, "entries": parse_registry(path)}
    if include_text:
        out["text"] = text
    return out


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------------------
def build_state(asof: str | None = None, offline: bool = False,
                board: bool = True) -> dict:
    today = (pd.Timestamp(asof) if asof
             else pd.Timestamp.now(tz="America/New_York").tz_localize(None)
             ).normalize()
    warnings: list[str] = []
    if not bps.is_session(today):
        warnings.append(f"session: {today.date()} is not an NYSE session")
    if not bps.PRICES_PATH.exists():
        raise SystemExit(f"FATAL: {bps.PRICES_PATH} missing - nothing to reason about")
    if not RANKS_PATH.exists():
        raise SystemExit(f"FATAL: {RANKS_PATH} missing - the seasonal ranks "
                         f"are the product (pull_scan_caches --set pitch)")
    prices = pd.read_parquet(bps.PRICES_PATH)
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()

    tape = bps.build_tape(prices, today, warnings)
    risk = bps.build_risk(today, warnings)
    expected = str((today - TRADING_DAY).normalize().date())
    if tape["freshest_bar"] < expected:
        warnings.append(f"tape: freshest bar {tape['freshest_bar']} is older "
                        f"than the expected prior session {expected}")

    ranks = build_ranks(prices, today, warnings)
    ref = pd.DatetimeIndex(sorted(prices.loc[(prices["ticker"] == "SPY")
                                             & (prices["date"] < today),
                                             "date"].unique()))
    outlier_tickers = {o["ticker"] for o in ranks["outliers"]}

    return {
        "asof": str(today.date()),
        "product": "seasonal",
        "generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "session": {
            "is_session": bps.is_session(today),
            "prev_session": expected,
            "next_sessions": [str(d.date()) for d in bps.sessions_from(today, 11)],
        },
        "calendar": bps.build_calendar(today),
        "calendar_cells": build_calendar_cells(today, ref, warnings),
        "ranks": ranks,
        "board": (build_board(pd.Timestamp(tape["freshest_bar"]), warnings)
                  if board else {"skipped": True, "rows": []}),
        "tape": tape,
        "risk": risk,
        "book": bps.build_book(today, warnings, offline=offline),
        "earnings": bps.build_earnings(today, warnings,
                                       horizon_td=EARNINGS_HORIZON_TD,
                                       tickers=outlier_tickers),
        "history": build_history(today, warnings),
        "watchlist": bps.build_watchlist(today, warnings,
                                         path=SEASONAL.watchlist_path),
        "scoreboard": _read_json(SEASONAL.scoreboard_path),
        "negative_registry": registry_block(SEASONAL.negative_registry_path,
                                            warnings, "negative_registry"),
        "pipeline": bps.build_pipeline(today, tape, risk, warnings),
        "warnings": warnings,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None, help="override today (YYYY-MM-DD)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--tape-out", default=str(DEFAULT_TAPE_OUT))
    ap.add_argument("--no-book", action="store_true",
                    help="skip Sheets reads (offline dev)")
    ap.add_argument("--no-board", action="store_true",
                    help="skip the in-process seasonal board build")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    state = build_state(args.asof, offline=args.no_book, board=not args.no_board)

    tape_universe = state["tape"].pop("universe")
    tape_path = Path(args.tape_out)
    tape_path.write_text(json.dumps(
        {"asof": state["asof"], "freshest_bar": state["tape"]["freshest_bar"],
         "tickers": tape_universe}, indent=1), encoding="utf-8")
    try:
        state["tape"]["universe_file"] = str(
            tape_path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        state["tape"]["universe_file"] = str(tape_path)
    state["tape"]["universe_count"] = len(tape_universe)

    out = Path(args.out)
    out.write_text(json.dumps(state, indent=1, default=str), encoding="utf-8")

    if not args.quiet:
        ranks = state["ranks"]
        cells = state["calendar_cells"]
        size_kb = out.stat().st_size / 1024
        print(f"Seasonal state {state['asof']} -> {out} ({size_kb:,.0f} KB)")
        print(f"  tape        {len(tape_universe)} tickers -> {tape_path.name}"
              f", freshest {state['tape']['freshest_bar']}")
        frag = state["risk"].get("fragility", {})
        print(f"  risk        dial ma10-63d {frag.get('ma10_63d', 'n/a')}, "
              f"P/C {state['risk'].get('pc_fear', {}).get('state', 'n/a')}")
        sides = {"long": 0, "short": 0}
        for o in ranks["outliers"]:
            sides[o["side"]] += 1
        print(f"  ranks       {ranks['rank_date']}, {ranks['n_tickers']} names, "
              f"{ranks['n_flagged']} flagged, {ranks['n_dropped_illiquid']} "
              f"under $5M ADV, {ranks['n_outliers']} outliers "
              f"({len(ranks['outliers'])} carried: {sides['long']} long / "
              f"{sides['short']} short)")
        print(f"  by class    " + ", ".join(f"{k} {v}" for k, v in
                                           ranks["by_class"].items()))
        top_sec = list(ranks["by_sector"].items())[:6]
        print(f"  by sector   " + ", ".join(f"{k} {v}" for k, v in top_sec))
        kinds: dict[str, int] = {}
        for c in cells["cells"]:
            kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
        print(f"  cells       {len(cells['cells'])} calendar cells in "
              f"{cells['window'][0]}..{cells['window'][1]} "
              f"({', '.join(f'{k} {v}' for k, v in sorted(kinds.items()))}), "
              f"{len(cells['events'])} macro events, {cells['cycle_year']}")
        board = state["board"]
        print(f"  board       {len(board.get('rows', []))} ticket rows "
              f"(asof {board.get('asof', 'n/a')}, "
              f"{board.get('n_candidates', 0)} candidates)")
        print(f"  earnings    {state['earnings'].get('count', 0)} prints "
              f"in {EARNINGS_HORIZON_TD} td")
        hist = state["history"]
        print(f"  history     {len(hist['recent_fingerprints'])} seasonal + "
              f"{len(hist['pitch_recent_fingerprints'])} pitch fingerprints "
              f"inside the repetition window")
        wl = state["watchlist"]
        print(f"  watchlist   {len(wl['entries'])} active, "
              f"{len(wl['expired'])} expired")
        print(f"  registry    own {len(state['negative_registry']['entries'])}"
              f" entries")
        for line in state["warnings"]:
            print(f"  WARNING: {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
