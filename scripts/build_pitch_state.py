"""Stage A of the Daily Pitch pipeline: deterministic state assembly.

Spec: daily_pitch_agent_spec_2026-08-06.html section 3, stage A. Emits ONE
json with everything the ideation stage needs, so the creative work starts
from measured facts instead of from a chat about what the market is doing.
Nothing here decides anything; it only reads what already exists.

    python scripts/build_pitch_state.py [--asof YYYY-MM-DD] [--out PATH]
                                        [--no-book] [--quiet]

Blocks in the payload:
    session       today's and the next sessions on the NYSE calendar
    calendar      macro events inside the horizon with signed td distance
    tape          per-ticker return ranks, z10, ATR, 52w and 200d distances
    risk          fragility dial, P/C fear state, firing signals, exposure leg
    book          staged scanner signals and sleeve state (live broker
                  positions deliberately excluded per McKinley 2026-08-08)
    earnings      liquid names printing inside the horizon
    seasonality   the seasonal board's own view plus month/cycle position
    research      research-doc index and the negative-results registry
    history       what the pitch pipeline itself has pitched recently
    watchlist     parked near-misses from earlier mornings, each with the
                  number that turns it on (data/pitch_watchlist.json)
    pipeline      did the overnight chain actually run, and are the
                  caches current (one line in the email)

Every block except `tape` is best effort: a missing input adds a line to
`warnings` and leaves the block partial, because a morning with no broker
connection should still produce three ideas. A missing PRICE cache is fatal
(there is nothing to reason about).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pitch_journal  # noqa: E402
import cache_io  # noqa: E402
from macro_calendar import EVENT_TYPES, load_macro_events  # noqa: E402
from pitch_lab import load_watchlist  # noqa: E402
from pitch_grammar import REPEAT_BLOCK_TD, wilder_atr  # noqa: E402
from strategy_config import (  # noqa: E402
    ACCOUNT_VALUE,
    LIQUID_PLUS_COMMODITIES,
    STRATEGY_BOOK,
)
from trading_calendar import TRADING_DAY  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from build_pitch_research_index import build as build_research_index  # noqa: E402

DEFAULT_OUT = ROOT / "data" / "pitch_state.json"
DEFAULT_TAPE_OUT = ROOT / "data" / "pitch_tape.json"
PRICES_PATH = ROOT / "data" / "master_prices.parquet"

# Cross-asset instruments the ideation stage reasons about directly. Every one
# is present in master_prices (checked 2026-08-06); anything absent is reported
# in warnings rather than silently dropped.
HEADLINE_TICKERS = [
    "SPY", "QQQ", "IWM", "DIA", "^GSPC", "^NDX",
    "TLT", "IEF", "LQD", "HYG", "^TNX", "^MOVE",
    "GLD", "SLV", "GDX", "USO", "UNG", "DBC",
    "UUP", "DX-Y.NYB", "EEM", "EFA", "FXI", "EWZ", "EWJ", "VNQ",
    "^VIX", "^VIX3M", "^SKEW", "UVXY", "SVXY",
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV",
    "XLY", "SMH", "XBI", "KRE",
]
EARNINGS_HORIZON_TD = 10
CALENDAR_LOOKAHEAD_TD = 15
CALENDAR_LOOKBACK_TD = 5

# Overnight jobs the pitch depends on, checked so the email can say in one
# line whether the chain actually ran. Added after the 2026-08-06 GitHub
# Actions incident, when every PM cron was silently skipped. The local-primary
# design now writes component receipts for local or backup execution, so both
# explicit failures and jobs that never started are visible.
AUTOMATION_RECEIPT_SCHEMA = "automation-receipt.v1"
TRACKED_AUTOMATION_RECEIPTS = [
    ("cboe_am", "put/call", "today"),
    ("master_prices_am", "prices", "today"),
    ("risk_am", "risk dial", "today"),
    ("scan_am", "scan", "today"),
    ("verify_fills", "fills", "previous"),
    ("portfolio_report", "portfolio", "previous"),
    ("earnings_and_grades", "earnings", "previous"),
]


# ---------------------------------------------------------------------------
# calendar helpers
# ---------------------------------------------------------------------------
def sessions_from(day: pd.Timestamp, count: int) -> list[pd.Timestamp]:
    return list(pd.date_range(day, periods=count, freq=TRADING_DAY))


def td_ahead(today: pd.Timestamp, target: pd.Timestamp) -> int:
    """Signed trading-day distance. Positive = target is in the future."""
    today, target = today.normalize(), target.normalize()
    if target >= today:
        return len(pd.date_range(today, target, freq=TRADING_DAY)) - 1
    return -(len(pd.date_range(target, today, freq=TRADING_DAY)) - 1)


def is_session(day: pd.Timestamp) -> bool:
    return bool(len(pd.date_range(day, day, freq=TRADING_DAY)))


def build_calendar(today: pd.Timestamp) -> dict:
    horizon_end = sessions_from(today, CALENDAR_LOOKAHEAD_TD + 1)[-1]
    horizon_start = (today - CALENDAR_LOOKBACK_TD * TRADING_DAY).normalize()
    events = load_macro_events()
    window = events[(events["date"] >= horizon_start)
                    & (events["date"] <= horizon_end)]
    rows = [{"event": r["event"], "date": str(r["date"].date()),
             "td_ahead": td_ahead(today, r["date"]),
             "label": str(r.get("label", "") or "")}
            for _, r in window.sort_values("date").iterrows()]
    next_by_type = {}
    for kind in EVENT_TYPES:
        future = events[(events["event"] == kind) & (events["date"] >= today)]
        if future.empty:
            continue
        nxt = future["date"].iloc[0]
        next_by_type[kind] = {"date": str(nxt.date()),
                              "td_ahead": td_ahead(today, nxt)}
    return {
        "window_td": [-CALENDAR_LOOKBACK_TD, CALENDAR_LOOKAHEAD_TD],
        "events": rows,
        "next_by_type": next_by_type,
        "cycle_year": ("midterm" if today.year % 4 == 2 else
                       "election" if today.year % 4 == 0 else
                       "pre_election" if today.year % 4 == 3 else "post_election"),
        "year_mod_4": today.year % 4,
        "month": today.month,
        "day_of_month": today.day,
        "weekday": today.strftime("%A"),
    }


# ---------------------------------------------------------------------------
# tape
# ---------------------------------------------------------------------------
def _metrics_for(df: pd.DataFrame) -> dict | None:
    """Compact tape metrics from one ticker's OHLCV history (last bar)."""
    if len(df) < 260:
        return None
    close = df["Close"].astype(float)
    ret = {w: close.pct_change(w) for w in (1, 2, 5, 10, 21, 63, 252)}
    vol21 = close.pct_change().rolling(21).std()
    ranks = {w: ret[w].rolling(252).rank(pct=True) * 100.0 for w in (5, 21, 63)}
    atr = wilder_atr(df["High"].to_numpy(), df["Low"].to_numpy(),
                     close.to_numpy())[-1]
    hi52 = close.rolling(252).max().iloc[-1]
    lo52 = close.rolling(252).min().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1]
    z10 = ret[10].iloc[-1] / (vol21.iloc[-1] * np.sqrt(10)) if vol21.iloc[-1] else np.nan
    vol_ratio = (df["Volume"].iloc[-1] / df["Volume"].rolling(63).mean().iloc[-1]
                 if df["Volume"].rolling(63).mean().iloc[-1] else np.nan)

    def _r(value, digits=2):
        value = float(value)
        return None if not np.isfinite(value) else round(value, digits)

    return {
        "date": str(pd.Timestamp(df["date"].iloc[-1]).date()),
        "close": _r(close.iloc[-1], 4),
        "ret_1d": _r(ret[1].iloc[-1] * 100),
        "ret_5d": _r(ret[5].iloc[-1] * 100),
        "ret_21d": _r(ret[21].iloc[-1] * 100),
        "ret_63d": _r(ret[63].iloc[-1] * 100),
        "ret_252d": _r(ret[252].iloc[-1] * 100),
        "rank_5d": _r(ranks[5].iloc[-1], 1),
        "rank_21d": _r(ranks[21].iloc[-1], 1),
        "rank_63d": _r(ranks[63].iloc[-1], 1),
        "z10": _r(z10),
        "atr": _r(atr, 4),
        "atr_pct": _r(atr / close.iloc[-1] * 100),
        "rvol21_ann": _r(vol21.iloc[-1] * np.sqrt(252) * 100, 1),
        "dist_52w_high_pct": _r((close.iloc[-1] / hi52 - 1) * 100),
        "dist_52w_low_pct": _r((close.iloc[-1] / lo52 - 1) * 100),
        "dist_sma200_pct": _r((close.iloc[-1] / sma200 - 1) * 100),
        "vol_vs_63d": _r(vol_ratio),
    }


def build_tape(prices: pd.DataFrame, today: pd.Timestamp,
               warnings: list[str]) -> dict:
    universe = sorted(set(HEADLINE_TICKERS) | set(LIQUID_PLUS_COMMODITIES))
    frame = prices[prices["ticker"].isin(universe)]
    frame = frame[frame["date"] < today].sort_values(["ticker", "date"])
    if frame.empty:
        raise RuntimeError(f"no price rows before {today.date()} in the cache")

    metrics: dict[str, dict] = {}
    for ticker, group in frame.groupby("ticker", sort=False):
        got = _metrics_for(group.tail(400))
        if got:
            metrics[ticker] = got
    missing = [t for t in HEADLINE_TICKERS if t not in metrics]
    if missing:
        warnings.append(f"tape: no metrics for {missing}")

    freshest = max(m["date"] for m in metrics.values())
    stale = sorted(t for t, m in metrics.items() if m["date"] != freshest)
    if stale:
        warnings.append(f"tape: {len(stale)} tickers stale vs {freshest} "
                        f"(first few: {stale[:8]})")

    def _extremes(key: str, count: int = 12) -> dict:
        pool = [(t, m[key]) for t, m in metrics.items()
                if m.get(key) is not None and m["date"] == freshest]
        pool.sort(key=lambda kv: kv[1])
        return {"lowest": [{"ticker": t, key: v} for t, v in pool[:count]],
                "highest": [{"ticker": t, key: v} for t, v in pool[-count:][::-1]]}

    spy = metrics.get("SPY", {})
    return {
        "freshest_bar": freshest,
        "headline": {t: metrics[t] for t in HEADLINE_TICKERS if t in metrics},
        "universe": {t: metrics[t] for t in sorted(metrics)},
        "extremes": {"rank_5d": _extremes("rank_5d"),
                     "rank_21d": _extremes("rank_21d"),
                     "z10": _extremes("z10"),
                     "dist_52w_high_pct": _extremes("dist_52w_high_pct")},
        "breadth": {
            "n_tickers": len(metrics),
            "pct_above_sma200": round(100.0 * np.mean(
                [m["dist_sma200_pct"] > 0 for m in metrics.values()
                 if m.get("dist_sma200_pct") is not None]), 1),
            "pct_rank21_above_50": round(100.0 * np.mean(
                [m["rank_21d"] > 50 for m in metrics.values()
                 if m.get("rank_21d") is not None]), 1),
            "spy_ret_5d": spy.get("ret_5d"),
            "spy_z10": spy.get("z10"),
        },
    }


# ---------------------------------------------------------------------------
# risk state
# ---------------------------------------------------------------------------
def build_risk(today: pd.Timestamp, warnings: list[str]) -> dict:
    out: dict = {}

    frag_path = ROOT / "data" / "rd2_fragility.parquet"
    try:
        frag = pd.read_parquet(frag_path)
        ma10 = frag["63d"].dropna().rolling(10, min_periods=1).mean()
        last_date = pd.Timestamp(frag.index[-1]).normalize()
        out["fragility"] = {
            "as_of": str(last_date.date()),
            "age_td": td_ahead(last_date, today),
            "raw_5d": round(float(frag["5d"].iloc[-1]), 1),
            "raw_21d": round(float(frag["21d"].iloc[-1]), 1),
            "raw_63d": round(float(frag["63d"].iloc[-1]), 1),
            "ma10_63d": round(float(ma10.iloc[-1]), 1),
            "ma10_63d_21d_ago": (round(float(ma10.iloc[-22]), 1)
                                 if len(ma10) > 22 else None),
            "sizing_note": ("10d MA of the 63d column, threshold 50, is the "
                            "ONLY statistic that sizes live orders"),
        }
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"risk: fragility parquet unavailable ({exc})")

    try:
        import pc_fear
        state = pc_fear.fear_state_asof(today)
        out["pc_fear"] = {"state": state["state"], "pctile": state["pct"],
                          "data_date": (str(state["data_date"])
                                        if state["data_date"] else None),
                          "age_bd": state["age_bd"]}
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"risk: P/C fear state unavailable ({exc})")

    env_path = ROOT / "data" / "rd2_environment.json"
    try:
        env = json.loads(env_path.read_text(encoding="utf-8"))
        out["environment_date"] = env.get("date")
        out["price_context"] = env.get("price_ctx")
        out["signals"] = {name: {"on": bool(v.get("on")),
                                 "summary": v.get("summary", "")}
                          for name, v in (env.get("signals") or {}).items()}
        out["signals_on"] = sorted(k for k, v in out["signals"].items() if v["on"])
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"risk: rd2_environment.json unavailable ({exc})")

    try:
        exposure = json.loads((ROOT / "data" / "exposure_state.json")
                              .read_text(encoding="utf-8"))
        out["exposure_leg"] = {"asof": exposure.get("asof"),
                               "mult": exposure.get("mult"),
                               "active_rule": exposure.get("active_rule"),
                               "reason": exposure.get("reason")}
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"risk: exposure_state.json unavailable ({exc})")

    return out


# ---------------------------------------------------------------------------
# book state
# ---------------------------------------------------------------------------
def _read_sheet_tab(tab: str) -> list[dict]:
    from daily_pitch import open_sheet
    return open_sheet().worksheet(tab).get_all_records()


def build_book(today: pd.Timestamp, warnings: list[str],
               offline: bool = False) -> dict:
    out: dict = {"account_value": ACCOUNT_VALUE,
                 "strategy_count": len(STRATEGY_BOOK),
                 "strategy_names": sorted(s["name"] for s in STRATEGY_BOOK)}

    staged: list[dict] = []
    for tab in ([] if offline else ("Order_Staging", "Overflow")):
        try:
            for row in _read_sheet_tab(tab):
                staged.append({
                    "tab": tab,
                    # The staging tabs are written by daily_scan's
                    # save_staging_orders, whose columns are Symbol and
                    # Strategy_Ref. Reading Ticker/Strategy_Name silently
                    # returned None for every row, so the stage-C overlap
                    # check was blind to WHICH names the book staged
                    # (found 2026-08-21, with four gold miners staged short).
                    "strategy": (row.get("Strategy_Ref")
                                 or row.get("Strategy_Name")
                                 or row.get("Strategy")),
                    "ticker": row.get("Symbol") or row.get("Ticker"),
                    "action": row.get("Action"),
                    "quantity": row.get("Quantity"),
                    "entry_type": row.get("Entry_Type") or row.get("Order_Type"),
                    "scan_date": str(row.get("Scan_Date", "")),
                })
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"book: {tab} tab unreadable ({exc})")
    out["staged_signals"] = staged

    for key, path in (("event_sleeve", "event_sleeve_state.json"),
                      ("trend_sleeve", "trend_sleeve_state.json")):
        try:
            out[key] = json.loads((ROOT / "data" / path)
                                  .read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"book: {path} unavailable ({exc})")

    # LIVE BROKER POSITIONS ARE DELIBERATELY EXCLUDED (McKinley, 2026-08-08):
    # the pitch must not know his holdings. Overlap in stage C is judged
    # against the SYSTEMATIC layers only - staged signals, sleeve state and
    # the ledger - all of which are repo-derived. Do not re-add a broker read
    # here.
    return out


# ---------------------------------------------------------------------------
# earnings / seasonality / history
# ---------------------------------------------------------------------------
def build_earnings(today: pd.Timestamp, warnings: list[str]) -> dict:
    try:
        cal = pd.read_parquet(ROOT / "data" / "earnings_calendar.parquet",
                              columns=["ticker", "date"])
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"earnings: calendar unavailable ({exc})")
        return {}
    end = sessions_from(today, EARNINGS_HORIZON_TD + 1)[-1]
    liquid = set(LIQUID_PLUS_COMMODITIES)
    window = cal[(cal["date"] >= today) & (cal["date"] <= end)
                 & (cal["ticker"].isin(liquid))]
    prints = [{"ticker": r["ticker"], "date": str(r["date"].date()),
               "td_ahead": td_ahead(today, r["date"])}
              for _, r in window.sort_values("date").iterrows()]
    return {"horizon_td": EARNINGS_HORIZON_TD, "count": len(prints),
            "prints": prints}


def build_seasonality(today: pd.Timestamp, warnings: list[str]) -> dict:
    out = {"month": today.strftime("%B"), "day_of_month": today.day,
           "cycle_year_mod4": today.year % 4}
    try:
        ideas = json.loads((ROOT / "data" / "daily_seasonal_ideas.json")
                           .read_text(encoding="utf-8"))
        out["board_meta"] = ideas.get("meta", {})
        out["board_candidates"] = [
            {k: c.get(k) for k in ("channel", "ticker", "direction", "horizon",
                                   "headline", "conviction", "p_value",
                                   "evidence", "notes")}
            for c in (ideas.get("candidates") or [])[:25]
        ]
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"seasonality: daily_seasonal_ideas.json unavailable ({exc})")
    return out


def _automation_receipt(job_id: str, run_date: dt.date) -> dict | None:
    key = f"automation/receipts/v1/{run_date.isoformat()}/{job_id}/latest.json"
    local = ROOT / "artifacts" / "daily_pitch" / "automation_receipts" / run_date.isoformat() / f"{job_id}.json"
    if not cache_io.download_to_local(key, str(local)):
        return None
    try:
        receipt = json.loads(local.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (receipt.get("schema_version") != AUTOMATION_RECEIPT_SCHEMA
            or receipt.get("job_id") != job_id
            or receipt.get("run_date_et") != run_date.isoformat()):
        return None
    return receipt


def build_pipeline(today: pd.Timestamp, tape: dict, risk: dict,
                   warnings: list[str], fetch=None) -> dict:
    """Did the overnight chain run, and are the caches current?

    A job counts as green only when the local-primary supervisor (or its
    GitHub fallback) wrote a verified R2 success receipt for the expected ET
    session. This replaces GitHub-run freshness: successful local jobs should
    not need to create a cloud run merely to make the pitch status green.
    """
    prev_session = (today - TRADING_DAY).normalize()
    out: dict = {"since": str(prev_session.date()), "checked": 0, "green": 0,
                 "missing": [], "available": False, "stale": []}

    fetch = fetch or _automation_receipt
    try:
        for job_id, label, date_basis in TRACKED_AUTOMATION_RECEIPTS:
            expected = today.date() if date_basis == "today" else prev_session.date()
            receipt = fetch(job_id, expected)
            out["checked"] += 1
            if receipt and receipt.get("status") == "success":
                out["green"] += 1
            else:
                out["missing"].append({
                    "job": label,
                    "job_id": job_id,
                    "expected_date": expected.isoformat(),
                    "status": (receipt or {}).get("status"),
                })
        out["available"] = True
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"pipeline: automation receipt check unavailable ({exc})")
        out["error"] = str(exc)

    # Cache freshness, from what the rest of the state already measured.
    freshest = tape.get("freshest_bar")
    out["prices_bar"] = freshest
    if freshest and freshest < str(prev_session.date()):
        out["stale"].append(f"prices {freshest}")
    frag = risk.get("fragility") or {}
    out["dial_asof"] = frag.get("as_of")
    if frag.get("as_of") and frag["as_of"] < str(prev_session.date()):
        out["stale"].append(f"dial {frag['as_of']}")
    pc = risk.get("pc_fear") or {}
    out["pc_date"], out["pc_age_bd"] = pc.get("data_date"), pc.get("age_bd")
    if pc.get("state") == "stale":
        out["stale"].append("P/C stale")

    out["ok"] = bool(out["available"] and not out["missing"]
                     and not out["stale"])
    return out


def build_history(today: pd.Timestamp, warnings: list[str]) -> dict:
    try:
        records = pitch_journal.load()
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"history: pitch journal unreadable ({exc})")
        records = []
    cutoff = (today - REPEAT_BLOCK_TD * TRADING_DAY).normalize()
    since = str(cutoff.date())
    ideas = pitch_journal.fold_ideas(records)
    recent = [{"date": r["date"], "title": r.get("title"),
               "grade": r.get("grade"), "fingerprint": r.get("fingerprint"),
               "approve": r.get("approve", ""),
               "outcome_r": (r.get("outcome") or {}).get("r_multiple")}
              for r in ideas if str(r.get("date", "")) >= since]
    killed = [{"date": r.get("date"), "title": r.get("title"),
               "reason": r.get("reason")}
              for r in records
              if r.get("kind") == "killed" and str(r.get("date", "")) >= since]
    return {"repeat_block_td": REPEAT_BLOCK_TD,
            "blocked_since": since,
            "recent_fingerprints": pitch_journal.recent_fingerprints(records,
                                                                    since),
            "recent_ideas": recent,
            "recent_kills": killed,
            "lifetime_pitched": len(ideas)}


def build_watchlist(today: pd.Timestamp, warnings: list[str]) -> dict:
    """Parked near-misses from earlier mornings, each carrying the number
    that turns it on. Stage B1 owes every ACTIVE entry a verdict in the
    surface map (trigger moved -> CHECK, unchanged -> PASS with today's
    value); EXPIRED entries are listed once so the after-publish step prunes
    them from the file."""
    try:
        wl = load_watchlist()
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"watchlist: unreadable ({exc})")
        return {"path": "data/pitch_watchlist.json", "entries": [],
                "expired": []}
    active, expired = [], []
    for e in wl.get("entries", []):
        exp = str(e.get("expires", "") or "")
        (expired if exp and exp < str(today.date()) else active).append(e)
    return {"path": "data/pitch_watchlist.json", "entries": active,
            "expired": expired}


# ---------------------------------------------------------------------------
def build_state(asof: str | None = None, offline: bool = False) -> dict:
    today = (pd.Timestamp(asof) if asof
             else pd.Timestamp.now(tz="America/New_York").tz_localize(None)
             ).normalize()
    warnings: list[str] = []
    if not is_session(today):
        warnings.append(f"session: {today.date()} is not an NYSE session")

    if not PRICES_PATH.exists():
        raise SystemExit(f"FATAL: {PRICES_PATH} missing - nothing to reason about")
    prices = pd.read_parquet(PRICES_PATH)

    tape = build_tape(prices, today, warnings)
    risk = build_risk(today, warnings)
    expected = str((today - TRADING_DAY).normalize().date())
    if tape["freshest_bar"] < expected:
        warnings.append(f"tape: freshest bar {tape['freshest_bar']} is older "
                        f"than the expected prior session {expected} - the "
                        f"price updater may have failed")

    scoreboard = {}
    try:
        scoreboard = json.loads((ROOT / "data" / "pitch_scoreboard.json")
                                .read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        pass

    return {
        "asof": str(today.date()),
        "generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "session": {
            "is_session": is_session(today),
            "prev_session": str((today - TRADING_DAY).normalize().date()),
            "next_sessions": [str(d.date()) for d in sessions_from(today, 11)],
        },
        "calendar": build_calendar(today),
        "tape": tape,
        "risk": risk,
        "book": build_book(today, warnings, offline=offline),
        "earnings": build_earnings(today, warnings),
        "seasonality": build_seasonality(today, warnings),
        "research": build_research_index(),
        "history": build_history(today, warnings),
        "watchlist": build_watchlist(today, warnings),
        "scoreboard": scoreboard,
        "pipeline": build_pipeline(today, tape, risk, warnings),
        "warnings": warnings,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None, help="override today (YYYY-MM-DD)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--tape-out", default=str(DEFAULT_TAPE_OUT))
    ap.add_argument("--no-book", action="store_true",
                    help="skip Sheets and broker reads (offline dev)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    state = build_state(args.asof, offline=args.no_book)

    # The per-ticker table is most of the bytes and is read by lookup, not
    # cover to cover, so it ships as its own file. pitch_state.json stays
    # small enough for the ideation agent to hold whole.
    tape_universe = state["tape"].pop("universe")
    tape_path = Path(args.tape_out)
    tape_path.write_text(json.dumps(
        {"asof": state["asof"], "freshest_bar": state["tape"]["freshest_bar"],
         "tickers": tape_universe}, indent=1), encoding="utf-8")
    state["tape"]["universe_file"] = str(
        tape_path.relative_to(ROOT)).replace("\\", "/")
    state["tape"]["universe_count"] = len(tape_universe)

    out = Path(args.out)
    out.write_text(json.dumps(state, indent=1), encoding="utf-8")

    if not args.quiet:
        size_kb = out.stat().st_size / 1024
        print(f"Pitch state {state['asof']} -> {out} ({size_kb:,.0f} KB)")
        print(f"  tape        {len(tape_universe)} tickers -> {tape_path.name}"
              f", freshest {state['tape']['freshest_bar']}")
        frag = state["risk"].get("fragility", {})
        print(f"  risk        dial ma10-63d {frag.get('ma10_63d', 'n/a')}, "
              f"P/C {state['risk'].get('pc_fear', {}).get('state', 'n/a')}, "
              f"signals on: {state['risk'].get('signals_on', [])}")
        print(f"  calendar    {len(state['calendar']['events'])} events in "
              f"[-{CALENDAR_LOOKBACK_TD}, +{CALENDAR_LOOKAHEAD_TD}] td")
        print(f"  book        {len(state['book'].get('staged_signals', []))} "
              f"staged (live positions excluded by design)")
        print(f"  earnings    {state['earnings'].get('count', 0)} liquid prints "
              f"in {EARNINGS_HORIZON_TD} td")
        print(f"  research    {state['research']['doc_count']} docs, "
              f"{state['research']['negative_registry_count']} dead ends")
        print(f"  history     {len(state['history']['recent_fingerprints'])} "
              f"fingerprints inside the repetition window")
        wl = state["watchlist"]
        print(f"  watchlist   {len(wl['entries'])} active, "
              f"{len(wl['expired'])} expired (prune after publish)")
        for line in state["warnings"]:
            print(f"  WARNING: {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
