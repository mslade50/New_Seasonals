"""Event sleeve — calendar-anchored index trades staged pre-market.

Four trades, frozen in scratch/ultracode_research/event_sleeve_prereg_2026-08-06.md:

  T1 FOMC_DRIFT          long SPY,  MOC td-4 -> MOO decision-day open,
                         non-midterm years, no filter
  T2 FOMC_MIDTERM_SHORT  short SPY, MOC td-4 -> MOO decision-day open,
                         midterm years, SPY rank21 (lag-1) < 50
  T3 SEP_POSTQUAD_SHORT  short IWM, MOC Sep opex -> MOC Sep last session,
                         skip when IWM z10 (lag-1) < -1 (washout bounce)
  T4 DEC_POSTOPEX_LONG   long IWM,  MOC Dec opex -> MOC year last session

Runs pre-market in the daily_screener AM job (after the R2 cache pull,
before order staging); local manual runs work the same way:
    python event_sleeve.py [--dry-run] [--asof YYYY-MM-DD] [--force]

Each run recomputes today's actions from the macro calendar and writes the
`Event` Sheets tab (clear + rewrite; empty when no action today). The tab
is consumed by the pre-market runner event_moo.py (OneDrive trading_ibkr),
which places the auction orders on the primary account.

State (open positions + their scheduled exits) lives in
data/event_sleeve_state.json and round-trips through R2 so GHA runs share
it. EXITS COME FROM STATE, not the calendar: each entry records exit_on +
exit order type, and any run with today >= exit_on stages the exit — a
failed morning run delays an exit by a session instead of dropping it.
All filters use the prior session's close (master_prices pre-market has
yesterday's bar at newest) — lag-1 by construction, matching the prereg.

Known bound (trend-sleeve convention): state marks a position open at
STAGING time. If the staged order was never executed (runner off, order
rejected), clear the position from the state json or the sleeve will
stage a phantom exit later.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from cache_io import download_to_local, head, upload_from_local  # noqa: E402
from macro_calendar import event_dates  # noqa: E402
from strategy_config import ACCOUNT_VALUE  # noqa: E402
from trading_calendar import TRADING_DAY  # noqa: E402

SHEET_NAME = "Trade_Signals_Log"
TAB_NAME = "Event"
STATE_PATH = Path(current_dir) / "data" / "event_sleeve_state.json"
STATE_R2_KEY = "event_sleeve_state.json"
ACTIONS_PATH = Path(current_dir) / "data" / "event_sleeve_last_actions.json"
ACTIONS_R2_KEY = "event_sleeve_last_actions.json"
JOURNAL_PATH = Path(current_dir) / "data" / "event_sleeve_journal.jsonl"
JOURNAL_R2_KEY = "event_sleeve_journal.jsonl"
PRICES_PATH = Path(current_dir) / "data" / "master_prices.parquet"

# Actions that OPEN a position; everything else in the row vocabulary
# (SELL / BUY_TO_COVER) closes one.
ENTRY_ACTIONS = {"BUY", "SELL_SHORT"}

# Staging must finish before the OPG cutoff chain (event_moo submits at
# 9:05, hard OPG cutoff 9:25). Past this time a live run refuses to stage.
STAGING_CUTOFF_ET = datetime.time(9, 0)

# Source of truth for the sleeve (prereg 2026-08-06). %NAV notional, no GRM.
EVENT_SLEEVE = {
    "T1_FOMC_DRIFT": {"ticker": "SPY", "side": "LONG", "nav_frac": 0.25},
    "T2_FOMC_MIDTERM_SHORT": {"ticker": "SPY", "side": "SHORT",
                              "nav_frac": 0.10, "rank21_max": 50},
    "T3_SEP_POSTQUAD_SHORT": {"ticker": "IWM", "side": "SHORT",
                              "nav_frac": 0.15, "z10_skip_below": -1.0},
    "T4_DEC_POSTOPEX_LONG": {"ticker": "IWM", "side": "LONG",
                             "nav_frac": 0.25},
    # V trades (2026-08-06 addendum): defined-risk short vol = LONG SVXY
    # (-0.5x ETP), loss bounded at the position.
    "V2_NOVDEC_VOL": {"ticker": "SVXY", "side": "LONG", "nav_frac": 0.05},
    "V4_POSTOPEX_VOL": {"ticker": "SVXY", "side": "LONG", "nav_frac": 0.10},
}
FOMC_ENTRY_TD_BEFORE = 4   # entry MOC 4 sessions before the decision
V4_EXIT_TD_AFTER = 3       # V4 exits MOC 3 sessions after opex


# Shared NYSE calendar (trading_calendar.py): Columbus/Veterans open,
# Good Friday closed, Saturday New Year's NOT observed on the prior
# Dec 31, Juneteenth from 2022, plus the ad-hoc closure list.
NYSE_BDAY = TRADING_DAY


def is_session(d: pd.Timestamp) -> bool:
    return bool(len(pd.date_range(d, d, freq=NYSE_BDAY)))


def sessions_before(d: pd.Timestamp, n: int) -> pd.Timestamp:
    return (d - n * NYSE_BDAY).normalize()


def last_session_of_month(year: int, month: int) -> pd.Timestamp:
    eom = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
    return eom if is_session(eom) else (eom - NYSE_BDAY).normalize()


def first_session_of_month(year: int, month: int) -> pd.Timestamp:
    som = pd.Timestamp(year, month, 1)
    return som if is_session(som) else (som + NYSE_BDAY).normalize()


def sessions_after(d: pd.Timestamp, n: int) -> pd.Timestamp:
    return (d + n * NYSE_BDAY).normalize()


def load_ticker(tkr: str) -> pd.DataFrame:
    mp = pd.read_parquet(PRICES_PATH, columns=["ticker", "date", "Close"])
    df = mp[mp["ticker"] == tkr].set_index("date").sort_index()[["Close"]]
    df.index = pd.to_datetime(df.index).normalize()
    df = df[~df.index.duplicated(keep="last")]
    r21 = df["Close"].pct_change(21)
    df["rank21"] = r21.rolling(252).rank(pct=True) * 100
    vol21 = df["Close"].pct_change().rolling(21).std()
    df["z10"] = df["Close"].pct_change(10) / (vol21 * np.sqrt(10))
    return df


def journal_prices(tickers) -> dict[str, pd.DataFrame]:
    """Open+Close per ticker for realized_history (MOO legs need the Open)."""
    mp = pd.read_parquet(PRICES_PATH,
                         columns=["ticker", "date", "Open", "Close"])
    out: dict[str, pd.DataFrame] = {}
    for t in sorted(set(tickers)):
        df = mp[mp["ticker"] == t].set_index("date").sort_index()[
            ["Open", "Close"]]
        df.index = pd.to_datetime(df.index).normalize()
        out[t] = df[~df.index.duplicated(keep="last")]
    return out


def load_state() -> dict:
    if not STATE_PATH.exists():
        download_to_local(STATE_R2_KEY, str(STATE_PATH))
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"positions": {}}


def _verified_automation_upload(path: Path, key: str) -> bool:
    ok = bool(upload_from_local(str(path), key))
    if os.environ.get("LOCAL_AUTOMATION_STRICT", "").strip() == "1":
        meta = head(key)
        actual = int((meta or {}).get("ContentLength") or -1)
        expected = path.stat().st_size
        if not ok or actual != expected:
            raise RuntimeError(
                f"Event sleeve R2 verification failed for {key}: "
                f"uploaded={ok}, size={actual}, expected={expected}"
            )
    return ok


def save_state(state: dict, dry_run: bool) -> None:
    if dry_run:
        print("[dry-run] state not saved")
        return
    state["generated"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    _verified_automation_upload(STATE_PATH, STATE_R2_KEY)
    print(f"State saved -> {STATE_PATH} + R2")


def _row(trade: str, cfg: dict, action: str, qty: int, ref_close: float,
         today: pd.Timestamp, order_type: str, note: str) -> dict:
    return {"Trade": trade, "Ticker": cfg["ticker"], "Action": action,
            "Quantity": qty, "Order_Type": order_type,
            "TIF": "OPG" if order_type == "MOO" else "MOC",
            "NAV_Frac": cfg["nav_frac"], "Ref_Close": round(ref_close, 2),
            "Execute_On": str(today.date()), "Note": note,
            "Scan_Source": "Event"}


def compute_actions(today: pd.Timestamp, px: dict[str, pd.DataFrame],
                    state: dict) -> tuple[list[dict], list[str]]:
    """Rows to stage for TODAY (pre-market view: px has data through the
    prior session). Returns (rows, log lines). Mutates state."""
    rows: list[dict] = []
    log: list[str] = []
    today = today.normalize()
    positions = state.setdefault("positions", {})

    def latest(tkr: str) -> pd.Series:
        df = px[tkr]
        prior = df[df.index < today]
        if prior.empty:
            raise RuntimeError(f"no price history before {today.date()} for {tkr}")
        return prior.iloc[-1]

    def stage_entry(trade: str, cfg: dict, note: str,
                    exit_on: pd.Timestamp, exit_order_type: str) -> None:
        if trade in positions:
            log.append(f"{trade}: already open, entry skipped (idempotent)")
            return
        ref = float(latest(cfg["ticker"])["Close"])
        qty = int(cfg["nav_frac"] * ACCOUNT_VALUE / ref)
        action = "BUY" if cfg["side"] == "LONG" else "SELL_SHORT"
        rows.append(_row(trade, cfg, action, qty, ref, today, "MOC", note))
        positions[trade] = {"shares": qty, "entry_date": str(today.date()),
                            "ref_close": ref, "exit_on": str(exit_on.date()),
                            "exit_order_type": exit_order_type}
        log.append(f"{trade}: ENTRY staged {action} {qty} {cfg['ticker']} "
                   f"MOC — {note} (exit {exit_order_type} {exit_on.date()})")

    # ---- exits first, FROM STATE: any open position at/past its exit date.
    # A failed morning run therefore delays an exit by a session instead of
    # dropping it (the calendar day itself is never load-bearing).
    for trade, pos in sorted(positions.items()):
        exit_on = pd.Timestamp(pos["exit_on"])
        if today < exit_on:
            continue
        cfg = EVENT_SLEEVE[trade]
        ot = pos.get("exit_order_type", "MOC")
        late = "" if today == exit_on else f" (LATE — scheduled {exit_on.date()})"
        ref = float(latest(cfg["ticker"])["Close"])
        action = "SELL" if cfg["side"] == "LONG" else "BUY_TO_COVER"
        rows.append(_row(trade, cfg, action, int(pos["shares"]), ref, today,
                         ot, f"scheduled exit{late}"))
        positions.pop(trade)
        log.append(f"{trade}: EXIT staged {action} {pos['shares']} "
                   f"{cfg['ticker']} {ot}{late}")

    # ---- T1 / T2: FOMC entry window ---------------------------------------
    fomc = event_dates("fomc_decision")
    upcoming = fomc[fomc >= today]
    if len(upcoming):
        dec = upcoming[0]
        midterm = dec.year % 4 == 2
        if today == sessions_before(dec, FOMC_ENTRY_TD_BEFORE):
            if not midterm:
                stage_entry("T1_FOMC_DRIFT", EVENT_SLEEVE["T1_FOMC_DRIFT"],
                            f"FOMC {dec.date()} in 4 sessions", dec, "MOO")
            else:
                cfg = EVENT_SLEEVE["T2_FOMC_MIDTERM_SHORT"]
                rank = float(latest(cfg["ticker"])["rank21"])
                if rank < cfg["rank21_max"]:
                    stage_entry("T2_FOMC_MIDTERM_SHORT", cfg,
                                f"midterm FOMC {dec.date()}, rank21 "
                                f"{rank:.0f} < {cfg['rank21_max']}", dec, "MOO")
                else:
                    log.append(f"T2: midterm FOMC {dec.date()} but rank21 "
                               f"{rank:.0f} >= {cfg['rank21_max']} — no trade "
                               f"(overbought tapes excluded)")

    # ---- T3: September post-quad entry ------------------------------------
    opex = event_dates("opex")
    cfg3 = EVENT_SLEEVE["T3_SEP_POSTQUAD_SHORT"]
    sep_opex = [d for d in opex if d.month == 9 and d.year == today.year]
    if sep_opex and today == sep_opex[0]:
        z = float(latest(cfg3["ticker"])["z10"])
        if z < cfg3["z10_skip_below"]:
            log.append(f"T3: Sep opex {today.date()} but z10 {z:+.2f} < "
                       f"{cfg3['z10_skip_below']} — washout, SKIP (bounce "
                       f"regime, see prereg)")
        else:
            stage_entry("T3_SEP_POSTQUAD_SHORT", cfg3,
                        f"Sep opex, z10 {z:+.2f}, short to month-end",
                        last_session_of_month(today.year, 9), "MOC")

    # ---- T4: December post-opex entry -------------------------------------
    cfg4 = EVENT_SLEEVE["T4_DEC_POSTOPEX_LONG"]
    dec_opex = [d for d in opex if d.month == 12 and d.year == today.year]
    if dec_opex and today == dec_opex[0]:
        stage_entry("T4_DEC_POSTOPEX_LONG", cfg4, "Dec opex, long to year-end",
                    last_session_of_month(today.year, 12), "MOC")

    # ---- V2: Nov-Dec short-vol seasonal (ex-midterm) ----------------------
    cfgv2 = EVENT_SLEEVE["V2_NOVDEC_VOL"]
    if today == first_session_of_month(today.year, 11):
        if today.year % 4 == 2:
            log.append("V2: first November session but midterm year — no "
                       "trade (both losing Nov-Dec years were midterms)")
        else:
            stage_entry("V2_NOVDEC_VOL", cfgv2,
                        "Nov-Dec short-vol seasonal (long SVXY)",
                        last_session_of_month(today.year, 12), "MOC")

    # ---- V4: post-opex vol crush (every opex except September; stands
    # down while V2 already holds the short-vol position in Nov/Dec) -------
    cfgv4 = EVENT_SLEEVE["V4_POSTOPEX_VOL"]
    if today in set(opex):
        if today.month == 9:
            log.append("V4: September opex — SKIP by spec (Sep inverts the "
                       "post-quad vol crush; that stress is T3's trade)")
        elif "V2_NOVDEC_VOL" in positions:
            log.append("V4: opex day but V2 already holds the Nov-Dec "
                       "short-vol position — SKIP (no doubling)")
        else:
            stage_entry("V4_POSTOPEX_VOL", cfgv4,
                        f"post-opex vol crush, exit +{V4_EXIT_TD_AFTER} "
                        f"sessions (long SVXY)",
                        sessions_after(today, V4_EXIT_TD_AFTER), "MOC")

    return rows, log


def write_actions_json(rows: list[dict], log: list[str], state: dict,
                       today: pd.Timestamp, dry_run: bool) -> None:
    """Snapshot of what this run staged/skipped, for the scan-email cards."""
    if dry_run:
        print("[dry-run] actions json not written")
        return
    payload = {"asof": str(today.date()), "rows": rows, "log": log,
               "positions": state.get("positions", {}),
               "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    ACTIONS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _verified_automation_upload(ACTIONS_PATH, ACTIONS_R2_KEY)


# ---------------------------------------------------------------------------
# Journal — append-only record of every staged entry/exit, R2-canonical
# (the sleeve runs in GHA where the checkout has no local copy, so every
# append pulls R2 first; pitch_journal convention otherwise). This is what
# lets realized results accrue: the state json pops a position on exit and
# keeps no history.
# ---------------------------------------------------------------------------
def journal_sync_down(path: Path = JOURNAL_PATH) -> None:
    """Pull the journal from R2 when this machine has no local copy. A
    non-default path is a test/dev run and never touches R2."""
    if path != JOURNAL_PATH or path.exists():
        return
    try:
        from cache_io import is_configured
        if is_configured():
            path.parent.mkdir(parents=True, exist_ok=True)
            download_to_local(JOURNAL_R2_KEY, str(path))
    except Exception as exc:  # noqa: BLE001
        print(f"NOTE: event journal R2 pull skipped ({exc})")


def load_journal(path: Path = JOURNAL_PATH, pull: bool = True) -> list[dict]:
    if pull:
        journal_sync_down(path)
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def journal_records_from(rows: list[dict], state: dict) -> list[dict]:
    """One journal record per staged row. Entries carry their scheduled exit
    (from the state position stage_entry just wrote); exits carry lateness."""
    records = []
    for r in rows:
        kind = "entry" if r["Action"] in ENTRY_ACTIONS else "exit"
        rec = {"kind": kind, "trade": r["Trade"], "ticker": r["Ticker"],
               "action": r["Action"], "qty": int(r["Quantity"]),
               "order_type": r["Order_Type"], "date": r["Execute_On"],
               "ref_close": r["Ref_Close"], "note": r["Note"]}
        if kind == "entry":
            pos = state.get("positions", {}).get(r["Trade"], {})
            rec["exit_on"] = pos.get("exit_on")
            rec["exit_order_type"] = pos.get("exit_order_type")
        else:
            rec["late"] = "LATE" in str(r["Note"])
        records.append(rec)
    return records


def backfill_entry_records(state: dict, records: list[dict]) -> list[dict]:
    """Entry records for open state positions the journal is missing. Makes
    the journal self-healing: a position opened before the journal existed
    (V4 SVXY 2026-08-21) or through a missed append gets its entry minted
    from the state on the next run, so the eventual exit can pair. All
    entries are MOC by construction."""
    have = {(r.get("trade"), r.get("date")) for r in records
            if r.get("kind") == "entry"}
    out = []
    for trade, pos in sorted((state.get("positions") or {}).items()):
        if (trade, pos.get("entry_date")) in have:
            continue
        cfg = EVENT_SLEEVE.get(trade)
        if cfg is None:
            continue
        out.append({"kind": "entry", "trade": trade, "ticker": cfg["ticker"],
                    "action": "BUY" if cfg["side"] == "LONG" else "SELL_SHORT",
                    "qty": int(pos.get("shares", 0)), "order_type": "MOC",
                    "date": pos.get("entry_date"),
                    "ref_close": pos.get("ref_close"),
                    "note": "backfilled from state",
                    "exit_on": pos.get("exit_on"),
                    "exit_order_type": pos.get("exit_order_type")})
    return out


def append_journal(records: list[dict], path: Path = JOURNAL_PATH,
                   push: bool = True) -> int:
    if not records:
        return 0
    if path == JOURNAL_PATH:
        journal_sync_down(path)
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps({**record, "written_at": stamp}) + "\n")
    if push and path == JOURNAL_PATH:
        try:
            _verified_automation_upload(path, JOURNAL_R2_KEY)
        except Exception as exc:  # noqa: BLE001
            if os.environ.get("LOCAL_AUTOMATION_STRICT", "").strip() == "1":
                raise
            print(f"NOTE: event journal R2 push skipped ({exc})")
    return len(records)


def realized_history(records: list[dict], px: dict[str, pd.DataFrame]) -> dict:
    """Pair journal entries with their exits and grade each round trip from
    the price cache: MOC legs book at that date's Close, MOO legs at its
    Open. Modeled from bars, not fills — an auction fill differs from the
    consolidated close by noise, and that is the documented basis.

    px: ticker -> DataFrame indexed by normalized date with at least a
    Close column (Open too when any MOO exit exists). Returns
    {"closed": [...], "open": [...]}, both chronological."""
    def leg_px(tkr: str, date: str, order_type: str) -> float | None:
        df = px.get(tkr)
        if df is None:
            return None
        ts = pd.Timestamp(date).normalize()
        if ts not in df.index:
            return None
        col = "Open" if order_type == "MOO" else "Close"
        if col not in df.columns:
            return None
        v = df.loc[ts, col]
        return float(v) if pd.notna(v) else None

    entries: dict[str, dict] = {}
    closed: list[dict] = []
    for rec in records:
        trade = rec.get("trade")
        if rec.get("kind") == "entry":
            entries[trade] = rec
            continue
        if rec.get("kind") != "exit" or trade not in entries:
            continue
        ent = entries.pop(trade)
        side = EVENT_SLEEVE.get(trade, {}).get(
            "side", "LONG" if ent["action"] == "BUY" else "SHORT")
        sign = 1.0 if side == "LONG" else -1.0
        e_px = leg_px(ent["ticker"], ent["date"], ent["order_type"])
        x_px = leg_px(rec["ticker"], rec["date"], rec["order_type"])
        row = {"trade": trade, "ticker": ent["ticker"], "side": side,
               "qty": ent["qty"], "entry_date": ent["date"],
               "exit_date": rec["date"], "entry_px": e_px, "exit_px": x_px,
               "late": bool(rec.get("late"))}
        if e_px and x_px:
            row["ret_pct"] = sign * (x_px / e_px - 1.0) * 100.0
            row["pnl"] = sign * ent["qty"] * (x_px - e_px)
            row["nav_bps"] = row["pnl"] / ACCOUNT_VALUE * 1e4
        closed.append(row)

    open_rows: list[dict] = []
    for trade, ent in entries.items():
        side = EVENT_SLEEVE.get(trade, {}).get(
            "side", "LONG" if ent["action"] == "BUY" else "SHORT")
        sign = 1.0 if side == "LONG" else -1.0
        row = {"trade": trade, "ticker": ent["ticker"], "side": side,
               "qty": ent["qty"], "entry_date": ent["date"],
               "exit_on": ent.get("exit_on"),
               "exit_order_type": ent.get("exit_order_type")}
        e_px = leg_px(ent["ticker"], ent["date"], ent["order_type"])
        df = px.get(ent["ticker"])
        if e_px and df is not None and len(df):
            last = df.index.max()
            m_px = float(df.loc[last, "Close"])
            row.update({"entry_px": e_px, "mark_px": m_px,
                        "mark_date": str(last.date()),
                        "ret_pct": sign * (m_px / e_px - 1.0) * 100.0,
                        "pnl": sign * ent["qty"] * (m_px - e_px)})
        open_rows.append(row)

    closed.sort(key=lambda r: (r["entry_date"], r["trade"]))
    open_rows.sort(key=lambda r: (r["entry_date"], r["trade"]))
    return {"closed": closed, "open": open_rows}


# Static explainers for the scan-email cards (one line of rule, one of
# evidence — prereg event_sleeve_prereg_2026-08-06.md is the source).
CARD_EXPLAINERS = {
    "T1_FOMC_DRIFT": (
        "Long SPY 25% NAV, MOC 4 sessions before a non-midterm FOMC "
        "decision, exit at the decision-day open.",
        "+38 bps/window, t 2.5, 67% hit since 2000 (Lucca-Moench drift; "
        "midterm years invert and are skipped)."),
    "T2_FOMC_MIDTERM_SHORT": (
        "Short SPY 10% NAV into a midterm-year FOMC decision, same window, "
        "ONLY when SPY 21d rank (lag-1) < 50.",
        "+63 bps/window lag-1 basis; overbought tapes flip the edge and "
        "are excluded. Pilot conviction."),
    "T3_SEP_POSTQUAD_SHORT": (
        "Short IWM 15% NAV from Sep opex close to month-end, skipped if "
        "IWM z10 (lag-1) < -1 (washouts bounce).",
        "+185 bps/window, t 2.3, 67% hit since 2000; post-quad drag is a "
        "calm-tape effect."),
    "T4_DEC_POSTOPEX_LONG": (
        "Long IWM 25% NAV from Dec opex close to year-end.",
        "+85 bps/window, t 2.3, 65% hit since 2000; small-cap year-end "
        "rotation (QQQ does not confirm)."),
    "V2_NOVDEC_VOL": (
        "Long SVXY 5% NAV, first November session to year-end, non-midterm "
        "years only.",
        "+11.1% avg, 10 of 11 non-midterm years up; both losing years "
        "(2014, 2018) were midterms."),
    "V4_POSTOPEX_VOL": (
        "Long SVXY 10% NAV, opex close to 3 sessions after, every month "
        "except September; stands down while V2 holds.",
        "+108 bps/window, t 3.6, since 2011 (2021+ t 3.8); September "
        "inverts the crush and is skipped."),
}


# Frozen backtest evidence for the Events tab (prereg 2026-08-06). The page
# renders EXACTLY these numbers — they are transcriptions from
# scratch/ultracode_research/event_sleeve_prereg_2026-08-06.md, never
# recomputed (freeze policy: a replay from today's bars would silently
# diverge from what was actually preregistered, most of all for V2/V4 whose
# backtest used a synthetic -0.5x SVXY series validated 0.9967 vs real).
BACKTEST_EVIDENCE = {
    "T1_FOMC_DRIFT": {
        "n": 150, "avg_bps": 38.2, "t": 2.51, "hit": 0.67,
        "span": "SPY 2000+, lag-1 basis",
        "cross_check": "QQQ +46.5 bps t 2.50. Lucca-Moench (2011) documented "
                       "the drift on 1994-2011 data, so half the sample is "
                       "out-of-sample for the claim.",
        "expected": "~+75 bps NAV/yr gross (8 windows at 25% NAV)",
        "worst": "March 2020 window -959 bps x 25% = ~-2.4% NAV",
        "kill": "pause at cumulative -2.0% NAV; review after 16 windows (~2y)",
        "conviction": "standard",
    },
    "T2_FOMC_MIDTERM_SHORT": {
        "n": 28, "avg_bps": 63.4, "t": 1.53, "hit": 0.54,
        "span": "SPY midterm-year windows 2000+, lag-1",
        "cross_check": "QQQ +97.3 bps t 1.67. rank21>70 tapes FLIP the short "
                       "to a loser and are excluded (threshold 50 = median "
                       "split, not scanned).",
        "expected": "~3-5 windows per midterm year at 10% NAV",
        "worst": "thinnest cell of the six — pilot conviction only",
        "kill": "kill on 4 consecutive losers or cumulative -1.0% NAV; "
                "review after the 2026 cycle",
        "conviction": "pilot",
    },
    "T3_SEP_POSTQUAD_SHORT": {
        "n": 24, "avg_bps": 185.0, "t": 2.27, "hit": 0.67,
        "span": "IWM Sep-opex windows 2000+, lag-1",
        "cross_check": "SPY variant +131.2 bps t 2.30. First-week-of-"
                       "September baseline is flat — this is the expiry-"
                       "anchored back half, not generic September.",
        "expected": "one ~7-session window per year at 15% NAV",
        "worst": "washouts bounce: z10 < -1 skips (would have skipped 2001)",
        "kill": "cumulative -2.0% NAV or 3 losses in any 4 consecutive "
                "years; review at N+5 (2031)",
        "conviction": "standard",
    },
    "T4_DEC_POSTOPEX_LONG": {
        "n": 26, "avg_bps": 85.3, "t": 2.30, "hit": 0.65,
        "span": "IWM Dec-opex windows 2000+, close basis",
        "cross_check": "SPY +65.7 bps t 2.19; QQQ does NOT confirm (-17.7) — "
                       "a small/value year-end rotation, IWM is the carrier.",
        "expected": "one ~7-session window per year at 25% NAV",
        "worst": "no filter by design",
        "kill": "cumulative -2.0% NAV or 3 losses in 4 years; review at N+5",
        "conviction": "standard",
    },
    "V2_NOVDEC_VOL": {
        "n": 11, "avg_bps": 1110.0, "t": None, "hit": 0.91,
        "span": "non-midterm Nov-Dec windows, -0.5x basis",
        "cross_check": "10 of 11 non-midterm years up, +11.1% avg. Midterm "
                       "Novembers went 1-of-3 with both sample losers "
                       "(2014 -5.4%, 2018 -13.4%) and are excluded.",
        "expected": "one 2-month window per non-midterm year at 5% NAV",
        "worst": "a 2018-style repeat ~ -65 bps NAV (the sizing yardstick)",
        "kill": "kill on 2 consecutive losers or cumulative -1.5% NAV; "
                "review at N+5 non-midterm windows (~2032)",
        "conviction": "standard",
    },
    "V4_POSTOPEX_VOL": {
        "n": 164, "avg_bps": 108.0, "t": 3.55, "hit": 0.72,
        "span": "all monthly opex ex-Sep, synthetic -0.5x validated 0.9967 "
                "vs real; hit rate is the 2021-06+ era",
        "cross_check": "+71 bps t 2.2 in the real -0.5x era (2018+); +134 "
                       "t 3.75 since 2021-06. September INVERTS (-65 bps, "
                       "21% hit) and is excluded.",
        "expected": "~11 windows/yr at 10% NAV",
        "worst": "Aug 2015 window -21.5% ~ -2.2% NAV; 2018 year -20.6% "
                 "cumulative ~ -2.1% NAV",
        "kill": "pause at cumulative -2.5% NAV; review after 22 windows (~2y)",
        "conviction": "standard",
    },
}

# Tested and NOT shipped (same studies; do not revive without a fresh
# prereg). Source: the prereg's per-trade notes + the 2026-08-06 vol
# addendum (scratch/svxy_postevent_grid.py, uvxy_event_study.py,
# svxy_defined_risk_study.py, event_sweep_drilldown.py).
REJECTED_STUDIES = [
    {"name": "T1 5d-rank overbought exclusion", "verdict": "dropped",
     "reason": "tested well on SPY but INVERTED on QQQ — not robust, so T1 "
               "ships unfiltered"},
    {"name": "T3 TLT long leg", "verdict": "excluded",
     "reason": "+95 bps t 1.75 standalone, but 2022 showed the duration "
               "regime risk; the equity short carries the edge"},
    {"name": "SVXY leg on T1 (pre-FOMC vol crush)", "verdict": "rejected",
     "reason": "corr 0.78 to the SPY leg and wins in only 23% of SPY-down "
               "windows — duplicates T1 without diversifying it"},
    {"name": "Post-CPI SVXY crush", "verdict": "rejected",
     "reason": "faded after 2018 in the post-event grid"},
    {"name": "Post-FOMC / post-NFP / VIX-expiry SVXY cells",
     "verdict": "rejected",
     "reason": "tested in the same svxy_postevent_grid sweep and failed it"},
    {"name": "Naked UVXY shorts (Nov-Dec)", "verdict": "rejected",
     "reason": "the only strong standalone cell (t 4.5) but 2018 ran +84% "
               "against — unbounded loss at book size; parked pending "
               "options infra"},
    {"name": "Opex washout bounce (long side)", "verdict": "parked",
     "reason": "overlaps the dip-buy family; deferred to a future dip-buy "
               "sizing study, not a sleeve trade"},
]


def sleeve_status_cards(today: pd.Timestamp | None = None) -> list[dict]:
    """Best-effort per-trade status for the scan email. Never raises."""
    today = (today or pd.Timestamp.now(tz="America/New_York")
             .tz_localize(None)).normalize()
    cards: list[dict] = []
    try:
        if not ACTIONS_PATH.exists():
            download_to_local(ACTIONS_R2_KEY, str(ACTIONS_PATH))
        actions = (json.loads(ACTIONS_PATH.read_text(encoding="utf-8"))
                   if ACTIONS_PATH.exists() else {})
        state = load_state()
        positions = state.get("positions", {})
        fresh = actions.get("asof") == str(today.date())
        staged = {r["Trade"]: r for r in actions.get("rows", [])} if fresh else {}
        logs = actions.get("log", []) if fresh else []

        fomc = event_dates("fomc_decision")
        opex = event_dates("opex")

        def next_fomc_entry(midterm: bool):
            for d in fomc[fomc >= today]:
                if (d.year % 4 == 2) == midterm:
                    entry = sessions_before(d, FOMC_ENTRY_TD_BEFORE)
                    if entry >= today:
                        return entry, d
            return None, None

        for trade, cfg in EVENT_SLEEVE.items():
            rule, evidence = CARD_EXPLAINERS[trade]
            card = {"trade": trade, "ticker": cfg["ticker"],
                    "rule": rule, "evidence": evidence}
            skip_line = next((l for l in logs
                              if l.startswith(trade.split("_")[0] + ":")), "")
            if trade in staged:
                r = staged[trade]
                card["status"] = (f"STAGED TODAY — {r['Action']} "
                                  f"{r['Quantity']} {r['Ticker']} "
                                  f"{r['Order_Type']} ({r['Note']})")
                card["kind"] = "staged"
            elif trade in positions:
                pos = positions[trade]
                card["status"] = (f"OPEN — {pos['shares']} {cfg['ticker']} "
                                  f"since {pos['entry_date']}, exit "
                                  f"{pos.get('exit_order_type', 'MOC')} "
                                  f"{pos.get('exit_on', '?')}")
                card["kind"] = "open"
            elif skip_line and ("no trade" in skip_line or "SKIP" in skip_line):
                card["status"] = f"SKIPPED TODAY — {skip_line.split(': ', 1)[-1]}"
                card["kind"] = "skipped"
            else:
                if trade == "T1_FOMC_DRIFT":
                    entry, dec = next_fomc_entry(midterm=False)
                elif trade == "T2_FOMC_MIDTERM_SHORT":
                    entry, dec = next_fomc_entry(midterm=True)
                elif trade == "T3_SEP_POSTQUAD_SHORT":
                    nxt = [d for d in opex if d >= today and d.month == 9]
                    entry, dec = (nxt[0], None) if nxt else (None, None)
                elif trade == "T4_DEC_POSTOPEX_LONG":
                    nxt = [d for d in opex if d >= today and d.month == 12]
                    entry, dec = (nxt[0], None) if nxt else (None, None)
                elif trade == "V2_NOVDEC_VOL":
                    entry, dec = None, None
                    for y in range(today.year, today.year + 5):
                        if y % 4 == 2:
                            continue
                        c = first_session_of_month(y, 11)
                        if c >= today:
                            entry = c
                            break
                else:  # V4: next opex that is not Sep and not V2 territory
                    nxt = [d for d in opex if d >= today and d.month != 9
                           and not (d.month in (11, 12) and d.year % 4 != 2)]
                    entry, dec = (nxt[0], None) if nxt else (None, None)
                if entry is None:
                    card["status"] = "IDLE — no window in calendar range"
                else:
                    extra = f" (FOMC {dec.date()})" if dec is not None else ""
                    card["status"] = f"ARMED — next entry {entry.date()}{extra}"
                card["kind"] = "armed"
            cards.append(card)
    except Exception as e:
        return [{"trade": "event_sleeve", "ticker": "", "rule": "",
                 "evidence": "", "kind": "error",
                 "status": f"status unavailable ({e})"}]
    return cards


def write_sheet(rows: list[dict], dry_run: bool) -> None:
    if dry_run:
        print("[dry-run] Event tab not written")
        return
    try:
        import gspread
        if "GCP_JSON" in os.environ:
            gc = gspread.service_account_from_dict(
                json.loads(os.environ["GCP_JSON"]))
        else:
            gc = gspread.service_account(
                filename=os.path.join(current_dir, "credentials.json"))
        sh = gc.open(SHEET_NAME)
        try:
            ws = sh.worksheet(TAB_NAME)
        except gspread.exceptions.WorksheetNotFound:
            ws = sh.add_worksheet(title=TAB_NAME, rows=20, cols=12)
        ws.clear()
        if not rows:
            expected = [["No event-sleeve action",
                         datetime.datetime.now().strftime("%Y-%m-%d %H:%M")]]
            ws.update(expected)
        else:
            df = pd.DataFrame(rows)
            expected = [df.columns.tolist()] + df.astype(str).values.tolist()
            ws.update(expected)
        if os.environ.get("LOCAL_AUTOMATION_STRICT", "").strip() == "1":
            actual = ws.get_all_values()
            if actual != expected:
                raise RuntimeError(
                    f"Event tab readback mismatch: wrote {len(expected)} rows, "
                    f"read {len(actual)}"
                )
        print(f"Wrote {len(rows)} rows -> '{TAB_NAME}' tab")
    except Exception as e:
        print(f"ERROR: Sheets write failed: {e}")
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print actions, write nothing")
    ap.add_argument("--asof", default=None,
                    help="override 'today' (YYYY-MM-DD) for testing")
    ap.add_argument("--force", action="store_true",
                    help="bypass the pre-market staging cutoff")
    args = ap.parse_args()

    now_et = pd.Timestamp.now(tz="America/New_York").tz_localize(None)
    if args.asof:
        today = pd.Timestamp(args.asof).normalize()
    else:
        today = now_et.normalize()
        if now_et.time() >= STAGING_CUTOFF_ET and not args.force:
            print(f"[CRITICAL] {now_et:%H:%M} ET is past the "
                  f"{STAGING_CUTOFF_ET:%H:%M} staging cutoff — staging "
                  f"NOTHING (--force overrides). The Event tab was not "
                  f"touched; a missed exit self-heals tomorrow.")
            sys.exit(1)
    if not is_session(today):
        print(f"{today.date()} is not a session — nothing to do")
        return

    tickers = sorted({c["ticker"] for c in EVENT_SLEEVE.values()})
    px = {t: load_ticker(t) for t in tickers}
    state = load_state()
    rows, log = compute_actions(today, px, state)

    print(f"Event sleeve {today.date()} — {len(rows)} action(s)")
    for line in log:
        print(f"  {line}")
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))
    write_sheet(rows, args.dry_run)
    save_state(state, args.dry_run)
    write_actions_json(rows, log, state, today, args.dry_run)
    if not args.dry_run:
        recs = journal_records_from(rows, state)
        recs += backfill_entry_records(state, load_journal() + recs)
        if recs:
            n = append_journal(recs)
            print(f"Journal: {n} record(s) appended -> {JOURNAL_PATH.name} + R2")


if __name__ == "__main__":
    main()
