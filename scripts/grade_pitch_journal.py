"""Outcome grader and scoreboard for the Daily Pitch pipeline.

Spec: daily_pitch_agent_spec_2026-08-06.html section 8. Every pitched idea is
replayed against master_prices exactly as it was specced (this is the reason
the grammar is mandatory) and booked at its hypothetical R and dollars.
APPROVED AND DECLINED IDEAS ARE BOTH GRADED, so the scoreboard measures the
filter as well as the pipeline: if declined ideas outperform approved ones,
that is a finding about the reader, not about the ideas.

    python scripts/grade_pitch_journal.py [--regrade] [--dry-run]
                                          [--journal PATH] [--out PATH]

Replay conventions, fixed here so a grade never depends on who ran it:

  entry   MOO fills at the open of Execute_On, MOC at its close. A LIMIT
          fills the first session in its window whose range touches the
          level, at the WORSE of the level and that bar's open (a gap
          through a buy limit fills at the open, which is better, and the
          replay books that honestly). An untouched limit books no_fill.
  exits   stop and target are live from the session AFTER the fill (the
          book-wide day-2 arming convention, and entry-day targets are
          never credited). A bar that touches both books the STOP.
  stop    fills at the worse of the stop and that bar's open, plus 3 bps of
          slippage, plus another 10 bps when the bar gapped through it
          (the book's stop-fill convention). Targets and time exits get no
          slippage.
  time    MOC exits at the close of Time_Exit_Date, MOO at its open (so an
          MOO exit never sees that session's range).
  R       idea dollars divided by the idea's staged ATR risk. A futures leg
          is replayed on its proxy series times the contract multiplier.

Outcomes are APPENDED to the journal as their own records; nothing is ever
edited in place, and a re-grade supersedes an earlier one.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pitch_journal  # noqa: E402
from pitch_grammar import load_prices  # noqa: E402
from trading_calendar import TRADING_DAY  # noqa: E402

DEFAULT_OUT = ROOT / "data" / "pitch_scoreboard.json"
STOP_SLIP_BPS = 3.0
STOP_GAP_SLIP_BPS = 10.0
SCOREBOARD_WINDOW_TD = 60


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------
def _num(value):
    if value in ("", None):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(out) else out


def replay_leg(bars: pd.DataFrame, row: dict) -> dict:
    """Hypothetical fill-to-exit replay of one staged leg."""
    execute_on = pd.Timestamp(row["Execute_On"]).normalize()
    index = bars.index
    if execute_on not in index:
        return {"status": "no_session", "ticker": row["Ticker"]}
    start = index.get_loc(execute_on)
    long = row["Action"] == "BUY"
    direction = 1 if long else -1
    atr = float(row["ATR"])
    qty = int(row["Quantity"])
    mult = float(row.get("Multiplier", 1.0) or 1.0)

    op = bars["Open"].to_numpy(float)
    hi = bars["High"].to_numpy(float)
    lo = bars["Low"].to_numpy(float)
    cl = bars["Close"].to_numpy(float)

    # --- entry ---
    kind = str(row["Entry_Type"]).upper()
    if kind == "MOO":
        fill_i, fill = start, op[start]
    elif kind == "MOC":
        fill_i, fill = start, cl[start]
    else:
        limit = _num(row["Limit_Price"])
        if limit is None:  # OPEN-anchored: the level is set by today's open
            limit = op[start] + float(row["Entry_Offset_ATR"]) * atr
        expire = pd.Timestamp(row["Entry_Expire_Date"]).normalize()
        fill_i = fill = None
        for i in range(start, len(index)):
            if index[i] > expire:
                break
            touched = lo[i] <= limit if long else hi[i] >= limit
            if touched:
                fill_i = i
                fill = min(limit, op[i]) if long else max(limit, op[i])
                break
        if fill_i is None:
            # Terminal no_fill only once the whole window has been SEEN.
            # Grading runs with prices through yesterday, so a multi-session
            # window (fill_window_td > 1) exhausts the available bars long
            # before it expires; booking terminal 0R then would permanently
            # hide a day-2/3 fill from the scoreboard (no_fill is never
            # re-graded). Still-open windows stay "open" and re-grade daily.
            if index[-1] < expire:
                return {"status": "open", "ticker": row["Ticker"],
                        "limit": round(limit, 4), "qty": qty}
            return {"status": "no_fill", "ticker": row["Ticker"],
                    "limit": round(limit, 4), "qty": qty, "pnl": 0.0}

    # --- exit ---
    exit_on = pd.Timestamp(row["Time_Exit_Date"]).normalize()
    later = index[index >= exit_on]
    if len(later) == 0:
        return {"status": "open", "ticker": row["Ticker"],
                "entry_date": str(index[fill_i].date()),
                "entry_price": round(fill, 4), "qty": qty}
    exit_i = index.get_loc(later[0])
    time_order = str(row.get("Time_Exit_Order", "MOC")).upper()
    stop = _num(row["Stop_Price"])
    target = _num(row["Target_Price"])
    # Stop/target only exist as prices when the entry level was known up
    # front; an MOO/MOC idea with ATR distances is graded off its actual fill.
    if stop is None and _num(row["Stop_ATR"]) is not None:
        stop = fill - direction * float(row["Stop_ATR"]) * atr
    if target is None and _num(row["Target_ATR"]) is not None:
        target = fill + direction * float(row["Target_ATR"]) * atr

    last_scan = exit_i - 1 if time_order == "MOO" else exit_i
    exit_price, exit_kind, exit_i_used = None, None, exit_i
    for j in range(fill_i + 1, last_scan + 1):
        stop_hit = stop is not None and (lo[j] <= stop if long else hi[j] >= stop)
        tgt_hit = target is not None and (hi[j] >= target if long else lo[j] <= target)
        if stop_hit:  # a bar that touches both books the stop
            raw = min(stop, op[j]) if long else max(stop, op[j])
            gapped = (op[j] < stop) if long else (op[j] > stop)
            slip = STOP_SLIP_BPS + (STOP_GAP_SLIP_BPS if gapped else 0.0)
            exit_price = raw * (1 - direction * slip / 1e4)
            exit_kind = "stop_gap" if gapped else "stop"
            exit_i_used = j
            break
        if tgt_hit:
            exit_price, exit_kind, exit_i_used = target, "target", j
            break
    if exit_price is None:
        exit_price = op[exit_i] if time_order == "MOO" else cl[exit_i]
        exit_kind = f"time_{time_order.lower()}"
        exit_i_used = exit_i

    span = slice(fill_i, exit_i_used + 1)
    if long:
        mae = (fill - np.nanmin(lo[span])) / atr
        mfe = (np.nanmax(hi[span]) - fill) / atr
    else:
        mae = (np.nanmax(hi[span]) - fill) / atr
        mfe = (fill - np.nanmin(lo[span])) / atr

    pnl = direction * (exit_price - fill) * qty * mult
    return {
        "status": "closed", "ticker": row["Ticker"],
        "entry_date": str(index[fill_i].date()), "entry_price": round(fill, 4),
        "exit_date": str(index[exit_i_used].date()),
        "exit_price": round(float(exit_price), 4), "exit_kind": exit_kind,
        "qty": qty, "pnl": round(float(pnl), 2),
        "mae_atr": round(float(mae), 2), "mfe_atr": round(float(mfe), 2),
        "held_td": int(exit_i_used - fill_i),
    }


def replay_idea(idea: dict, bars_by_ticker: dict[str, pd.DataFrame]) -> dict:
    legs = []
    for row in idea.get("orders", []):
        key = str(row.get("Proxy_Ticker") or row["Ticker"]).upper()
        bars = bars_by_ticker.get(key)
        if bars is None or bars.empty:
            legs.append({"status": "no_prices", "ticker": row["Ticker"]})
            continue
        legs.append(replay_leg(bars, row))

    statuses = {leg["status"] for leg in legs}
    if "open" in statuses or "no_prices" in statuses or "no_session" in statuses:
        status = "open" if "open" in statuses else "ungradeable"
        return {"status": status, "legs": legs}

    closed = [leg for leg in legs if leg["status"] == "closed"]
    if not closed:
        status = "no_fill"
    elif len(closed) < len(legs):
        status = "partial_fill"
    else:
        status = "closed"

    pnl = sum(leg.get("pnl", 0.0) for leg in legs)
    risk = sum(float(r["Risk_Amt"]) for r in idea.get("orders", []))
    return {
        "status": status,
        "pnl": round(pnl, 2),
        "risk": round(risk, 2),
        "r_multiple": round(pnl / risk, 3) if risk else None,
        "mae_atr": (round(max(leg["mae_atr"] for leg in closed), 2)
                    if closed else None),
        "mfe_atr": (round(max(leg["mfe_atr"] for leg in closed), 2)
                    if closed else None),
        "exit_kinds": sorted({leg["exit_kind"] for leg in closed}),
        "legs": legs,
    }


# ---------------------------------------------------------------------------
# scoreboard
# ---------------------------------------------------------------------------
def _bucket(ideas: list[dict]) -> dict:
    graded = [i for i in ideas
              if (i.get("outcome") or {}).get("r_multiple") is not None]
    rs = [i["outcome"]["r_multiple"] for i in graded]
    filled = [i for i in ideas
              if (i.get("outcome") or {}).get("status") in ("closed",
                                                            "partial_fill")]
    return {
        "n": len(ideas),
        "graded": len(graded),
        "fill_rate": round(100.0 * len(filled) / len(ideas), 1) if ideas else None,
        "avg_r": round(float(np.mean(rs)), 3) if rs else None,
        "median_r": round(float(np.median(rs)), 3) if rs else None,
        "hit_rate": round(100.0 * float(np.mean([r > 0 for r in rs])), 1) if rs else None,
        "total_pnl": round(sum(i["outcome"]["pnl"] for i in graded), 2),
    }


def build_scoreboard(ideas: list[dict], today: pd.Timestamp) -> dict:
    cutoff = str((today - SCOREBOARD_WINDOW_TD * TRADING_DAY).normalize().date())
    window = [i for i in ideas if str(i.get("date", "")) >= cutoff]

    def _split(pool: list[dict], key) -> dict:
        groups: dict[str, list[dict]] = {}
        for idea in pool:
            groups.setdefault(str(key(idea)), []).append(idea)
        return {k: _bucket(v) for k, v in sorted(groups.items())}

    approved = [i for i in window if pitch_journal.approved(i)]
    declined = [i for i in window if not pitch_journal.approved(i)]
    gap = None
    if approved and declined:
        yes, no = _bucket(approved), _bucket(declined)
        if yes["avg_r"] is not None and no["avg_r"] is not None:
            gap = {"approved_n": yes["graded"], "approved_avg_r": yes["avg_r"],
                   "declined_n": no["graded"], "declined_avg_r": no["avg_r"],
                   "edge_of_the_filter": round(yes["avg_r"] - no["avg_r"], 3)}

    rolling = _bucket(window)
    rolling["by_grade"] = _split(window, lambda i: i.get("grade", "?"))
    rolling["by_axis"] = _split(window, lambda i: i.get("novelty_axis", "?"))
    # Which model wrote the idea. The bake-off could not rank two models
    # off one adjudicated disagreement; this can, off realized outcomes.
    rolling["by_model"] = _split(window, lambda i: i.get("model") or "unknown")
    rolling["approved_vs_declined"] = gap
    rolling["window_td"] = SCOREBOARD_WINDOW_TD
    rolling["since"] = cutoff

    lifetime = _bucket(ideas)
    lifetime["by_grade"] = _split(ideas, lambda i: i.get("grade", "?"))
    lifetime["by_model"] = _split(ideas, lambda i: i.get("model") or "unknown")
    return {"generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "asof": str(today.date()),
            "rolling_60d": rolling, "lifetime": lifetime}


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", default=str(pitch_journal.JOURNAL_PATH))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--regrade", action="store_true",
                    help="re-book ideas that already carry an outcome")
    ap.add_argument("--asof", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    journal_path = Path(args.journal)
    records = pitch_journal.load(journal_path)
    ideas = pitch_journal.fold_ideas(records)
    if not ideas:
        print("No pitched ideas in the journal yet - nothing to grade.")
        return 0

    today = pd.Timestamp(args.asof or dt.date.today()).normalize()
    prices = load_prices()
    wanted = {str(r.get("Proxy_Ticker") or r["Ticker"]).upper()
              for idea in ideas for r in idea.get("orders", [])}
    bars_by_ticker = {}
    for ticker, group in prices[prices["ticker"].isin(wanted)].groupby("ticker"):
        frame = group.sort_values("date").set_index("date")
        frame.index = pd.DatetimeIndex(frame.index).normalize()
        bars_by_ticker[ticker] = frame[~frame.index.duplicated(keep="last")]

    todo = [i for i in ideas
            if args.regrade or (i.get("outcome") or {}).get("status")
            in (None, "open", "ungradeable")]
    outcomes, still_open = [], 0
    for idea in todo:
        result = replay_idea(idea, bars_by_ticker)
        if result["status"] in ("open", "ungradeable"):
            still_open += 1
            continue
        outcomes.append({"kind": "outcome", "idea_id": idea["idea_id"],
                         "date": idea["date"], "outcome": result,
                         "graded_at": dt.datetime.now().isoformat(
                             timespec="seconds")})
        r = result.get("r_multiple")
        print(f"  {idea['idea_id']} [{idea.get('grade')}] "
              f"{str(idea.get('title'))[:52]:<52} {result['status']:<13} "
              f"{'' if r is None else f'{r:+.2f}R'} "
              f"{'' if not result.get('exit_kinds') else result['exit_kinds']}")

    print(f"Graded {len(outcomes)} idea(s); {still_open} still open; "
          f"{len(ideas) - len(todo)} already booked")

    if args.dry_run:
        print("[dry-run] no journal append, no scoreboard write")
        return 0

    if outcomes:
        pitch_journal.append(outcomes, journal_path)

    refreshed = pitch_journal.fold_ideas(pitch_journal.load(journal_path,
                                                            pull=False))
    scoreboard = build_scoreboard(refreshed, today)
    Path(args.out).write_text(json.dumps(scoreboard, indent=1), encoding="utf-8")
    roll = scoreboard["rolling_60d"]
    print(f"Scoreboard -> {args.out}: {roll['n']} pitched in the window, "
          f"{roll['graded']} graded, avg "
          f"{roll['avg_r'] if roll['avg_r'] is not None else 'n/a'}R, "
          f"hit {roll['hit_rate'] if roll['hit_rate'] is not None else 'n/a'}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
