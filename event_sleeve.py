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
from pandas.tseries.holiday import (AbstractHolidayCalendar, GoodFriday,
                                    Holiday, USFederalHolidayCalendar,
                                    sunday_to_monday)
from pandas.tseries.offsets import CustomBusinessDay

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from cache_io import download_to_local, upload_from_local  # noqa: E402
from macro_calendar import event_dates  # noqa: E402
from strategy_config import ACCOUNT_VALUE  # noqa: E402

SHEET_NAME = "Trade_Signals_Log"
TAB_NAME = "Event"
STATE_PATH = Path(current_dir) / "data" / "event_sleeve_state.json"
STATE_R2_KEY = "event_sleeve_state.json"
ACTIONS_PATH = Path(current_dir) / "data" / "event_sleeve_last_actions.json"
ACTIONS_R2_KEY = "event_sleeve_last_actions.json"
PRICES_PATH = Path(current_dir) / "data" / "master_prices.parquet"

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


class NYSEHolidayCalendar(AbstractHolidayCalendar):
    """US federal holidays minus Columbus/Veterans (NYSE trades both),
    plus Good Friday. New Year's uses sunday_to_monday: when Jan 1 falls
    on a Saturday NYSE does NOT close the preceding Dec 31 (the federal
    nearest-weekday rule wrongly kills that session, e.g. 2027-12-31)."""
    rules = ([r for r in USFederalHolidayCalendar.rules
              if r.name not in ("Columbus Day", "Veterans Day",
                                "New Year's Day")]
             + [GoodFriday,
                Holiday("New Year's Day", month=1, day=1,
                        observance=sunday_to_monday)])


NYSE_BDAY = CustomBusinessDay(calendar=NYSEHolidayCalendar())


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


def load_state() -> dict:
    if not STATE_PATH.exists():
        download_to_local(STATE_R2_KEY, str(STATE_PATH))
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"positions": {}}


def save_state(state: dict, dry_run: bool) -> None:
    if dry_run:
        print("[dry-run] state not saved")
        return
    state["generated"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    upload_from_local(str(STATE_PATH), STATE_R2_KEY)
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
    upload_from_local(str(ACTIONS_PATH), ACTIONS_R2_KEY)


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
            ws.update([["No event-sleeve action",
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M")]])
        else:
            df = pd.DataFrame(rows)
            ws.update([df.columns.tolist()] + df.astype(str).values.tolist())
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


if __name__ == "__main__":
    main()
