"""Event sleeve — calendar-anchored index trades staged pre-market.

Four trades, frozen in scratch/ultracode_research/event_sleeve_prereg_2026-08-06.md:

  T1 FOMC_DRIFT          long SPY,  MOC td-4 -> MOO decision-day open,
                         non-midterm years, no filter
  T2 FOMC_MIDTERM_SHORT  short SPY, MOC td-4 -> MOO decision-day open,
                         midterm years, SPY rank21 (lag-1) < 50
  T3 SEP_POSTQUAD_SHORT  short IWM, MOC Sep opex -> MOC Sep last session,
                         skip when IWM z10 (lag-1) < -1 (washout bounce)
  T4 DEC_POSTOPEX_LONG   long IWM,  MOC Dec opex -> MOC year last session

Run pre-market (after the 4:17 AM parquet update, before order_staging):
    python event_sleeve.py [--dry-run] [--asof YYYY-MM-DD]

Each run recomputes today's actions from the macro calendar and writes the
`Event` Sheets tab (clear + rewrite; empty when no action today). Held
positions persist in data/event_sleeve_state.json so exits size correctly
and re-runs are idempotent. All filters use the prior session's close
(master_prices pre-market has yesterday's bar at newest) — lag-1 by
construction, matching the prereg.
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
                                    USFederalHolidayCalendar)
from pandas.tseries.offsets import CustomBusinessDay

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from macro_calendar import event_dates  # noqa: E402
from strategy_config import ACCOUNT_VALUE  # noqa: E402

SHEET_NAME = "Trade_Signals_Log"
TAB_NAME = "Event"
STATE_PATH = Path(current_dir) / "data" / "event_sleeve_state.json"
PRICES_PATH = Path(current_dir) / "data" / "master_prices.parquet"

# Source of truth for the sleeve (prereg 2026-08-06). %NAV notional, no GRM.
EVENT_SLEEVE = {
    "T1_FOMC_DRIFT": {"ticker": "SPY", "side": "LONG", "nav_frac": 0.25},
    "T2_FOMC_MIDTERM_SHORT": {"ticker": "SPY", "side": "SHORT",
                              "nav_frac": 0.10, "rank21_max": 50},
    "T3_SEP_POSTQUAD_SHORT": {"ticker": "IWM", "side": "SHORT",
                              "nav_frac": 0.15, "z10_skip_below": -1.0},
    "T4_DEC_POSTOPEX_LONG": {"ticker": "IWM", "side": "LONG",
                             "nav_frac": 0.25},
}
FOMC_ENTRY_TD_BEFORE = 4   # entry MOC 4 sessions before the decision


class NYSEHolidayCalendar(AbstractHolidayCalendar):
    """US federal holidays minus Columbus/Veterans (NYSE trades both),
    plus Good Friday."""
    rules = [r for r in USFederalHolidayCalendar.rules
             if r.name not in ("Columbus Day", "Veterans Day")] + [GoodFriday]


NYSE_BDAY = CustomBusinessDay(calendar=NYSEHolidayCalendar())


def is_session(d: pd.Timestamp) -> bool:
    return bool(len(pd.date_range(d, d, freq=NYSE_BDAY)))


def sessions_before(d: pd.Timestamp, n: int) -> pd.Timestamp:
    return (d - n * NYSE_BDAY).normalize()


def last_session_of_month(year: int, month: int) -> pd.Timestamp:
    eom = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
    return eom if is_session(eom) else (eom - NYSE_BDAY).normalize()


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
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"positions": {}}


def save_state(state: dict, dry_run: bool) -> None:
    if dry_run:
        print("[dry-run] state not saved")
        return
    state["generated"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"State saved -> {STATE_PATH}")


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
    prior session). Returns (rows, log lines)."""
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

    def shares_for(cfg: dict, ref_close: float) -> int:
        return int(cfg["nav_frac"] * ACCOUNT_VALUE / ref_close)

    def stage_entry(trade: str, cfg: dict, order_type: str, note: str) -> None:
        if trade in positions:
            log.append(f"{trade}: already open, entry skipped (idempotent)")
            return
        ref = float(latest(cfg["ticker"])["Close"])
        qty = shares_for(cfg, ref)
        action = "BUY" if cfg["side"] == "LONG" else "SELL_SHORT"
        rows.append(_row(trade, cfg, action, qty, ref, today, order_type, note))
        positions[trade] = {"shares": qty, "entry_date": str(today.date()),
                            "ref_close": ref}
        log.append(f"{trade}: ENTRY staged {action} {qty} {cfg['ticker']} "
                   f"{order_type} — {note}")

    def stage_exit(trade: str, cfg: dict, order_type: str, note: str) -> None:
        pos = positions.get(trade)
        if not pos:
            log.append(f"{trade}: exit day but no open position — nothing staged")
            return
        ref = float(latest(cfg["ticker"])["Close"])
        action = "SELL" if cfg["side"] == "LONG" else "BUY_TO_COVER"
        rows.append(_row(trade, cfg, action, int(pos["shares"]), ref, today,
                         order_type, note))
        positions.pop(trade)
        log.append(f"{trade}: EXIT staged {action} {pos['shares']} "
                   f"{cfg['ticker']} {order_type} — {note}")

    # ---- T1 / T2: FOMC windows -------------------------------------------
    fomc = event_dates("fomc_decision")
    upcoming = fomc[fomc >= today]
    if len(upcoming):
        dec = upcoming[0]
        midterm = dec.year % 4 == 2
        if today == sessions_before(dec, FOMC_ENTRY_TD_BEFORE):
            if not midterm:
                stage_entry("T1_FOMC_DRIFT", EVENT_SLEEVE["T1_FOMC_DRIFT"],
                            "MOC", f"FOMC {dec.date()} in 4 sessions")
            else:
                cfg = EVENT_SLEEVE["T2_FOMC_MIDTERM_SHORT"]
                rank = float(latest(cfg["ticker"])["rank21"])
                if rank < cfg["rank21_max"]:
                    stage_entry("T2_FOMC_MIDTERM_SHORT", cfg, "MOC",
                                f"midterm FOMC {dec.date()}, rank21 "
                                f"{rank:.0f} < {cfg['rank21_max']}")
                else:
                    log.append(f"T2: midterm FOMC {dec.date()} but rank21 "
                               f"{rank:.0f} >= {cfg['rank21_max']} — no trade "
                               f"(overbought tapes excluded)")
        if today == dec:
            stage_exit("T1_FOMC_DRIFT", EVENT_SLEEVE["T1_FOMC_DRIFT"], "MOO",
                       f"decision day {dec.date()} — exit at the open")
            stage_exit("T2_FOMC_MIDTERM_SHORT",
                       EVENT_SLEEVE["T2_FOMC_MIDTERM_SHORT"], "MOO",
                       f"decision day {dec.date()} — cover at the open")

    # ---- T3: September post-quad short -----------------------------------
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
            stage_entry("T3_SEP_POSTQUAD_SHORT", cfg3, "MOC",
                        f"Sep opex, z10 {z:+.2f}, short to month-end")
    if today == last_session_of_month(today.year, 9):
        stage_exit("T3_SEP_POSTQUAD_SHORT", cfg3, "MOC",
                   "September last session — cover")

    # ---- T4: December post-opex long -------------------------------------
    cfg4 = EVENT_SLEEVE["T4_DEC_POSTOPEX_LONG"]
    dec_opex = [d for d in opex if d.month == 12 and d.year == today.year]
    if dec_opex and today == dec_opex[0]:
        stage_entry("T4_DEC_POSTOPEX_LONG", cfg4, "MOC",
                    "Dec opex, long to year-end")
    if today == last_session_of_month(today.year, 12):
        stage_exit("T4_DEC_POSTOPEX_LONG", cfg4, "MOC",
                   "year last session — exit")

    return rows, log


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
    args = ap.parse_args()

    today = (pd.Timestamp(args.asof) if args.asof
             else pd.Timestamp.now(tz="America/New_York").tz_localize(None))
    today = today.normalize()
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


if __name__ == "__main__":
    main()
