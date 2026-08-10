"""Outcome grader and scoreboard for the Daily Posts pipeline.

Every IDEA draft is replayed against master_prices with the pitch grader's
pessimistic conventions - the replay itself is IMPORTED from
grade_pitch_journal so the two products can never drift apart on fill
mechanics (day-2 stop arming, stop-and-target bar books the stop, gapped
stops fill at the open + 13 bps, entry-day targets never credited).

POSTED AND UNPOSTED DRAFTS ARE BOTH GRADED. The unposted bucket is the
public account's version of the pitch's declined bucket: it measures the
human filter, and it is also the honesty reserve - the scoreboard the
account posts can say "including the ideas we drafted and sat on".

    python scripts/posts_scoreboard.py [--journal PATH] [--out PATH]
                                       [--regrade] [--dry-run] [--asof D]
                                       [--format-post]

--format-post prints a ready-to-paste accountability post (text only,
neutral numbers; the drafting skill wraps voice around it).
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
sys.path.insert(0, str(ROOT / "scripts"))

import posts_journal  # noqa: E402
from posts_grammar import derive_order_row  # noqa: E402
from pitch_grammar import load_prices  # noqa: E402
from trading_calendar import TRADING_DAY  # noqa: E402
from grade_pitch_journal import replay_leg  # noqa: E402

DEFAULT_OUT = ROOT / "data" / "posts_scoreboard.json"
WINDOW_TD = 60


def _bucket(drafts: list[dict]) -> dict:
    graded = [d for d in drafts
              if (d.get("outcome") or {}).get("r_multiple") is not None]
    rs = [d["outcome"]["r_multiple"] for d in graded]
    return {
        "n": len(drafts),
        "graded": len(graded),
        "avg_r": round(float(np.mean(rs)), 3) if rs else None,
        "median_r": round(float(np.median(rs)), 3) if rs else None,
        "hit_rate": round(100.0 * float(np.mean([r > 0 for r in rs])), 1) if rs else None,
        "total_r": round(float(np.sum(rs)), 2) if rs else None,
        "no_fill": sum(1 for d in drafts
                       if (d.get("outcome") or {}).get("status") == "no_fill"),
    }


def build_scoreboard(drafts: list[dict], today: pd.Timestamp) -> dict:
    ideas = [d for d in drafts if d.get("type") == "idea"]
    cutoff = str((today - WINDOW_TD * TRADING_DAY).normalize().date())
    window = [d for d in ideas if str(d.get("date", "")) >= cutoff]
    posted = [d for d in window if d.get("posted")]
    unposted = [d for d in window if not d.get("posted")]

    rolling = _bucket(window)
    rolling["posted"] = _bucket(posted)
    rolling["unposted"] = _bucket(unposted)
    rolling["window_td"] = WINDOW_TD
    rolling["since"] = cutoff

    lifetime = _bucket(ideas)
    lifetime["posted"] = _bucket([d for d in ideas if d.get("posted")])
    lifetime["unposted"] = _bucket([d for d in ideas if not d.get("posted")])
    return {"generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "asof": str(today.date()),
            "rolling_60d": rolling, "lifetime": lifetime}


def format_post(board: dict) -> str:
    """Neutral accountability numbers for a scoreboard post. Voice is the
    skill's job; the numbers are this one's."""
    roll = board["rolling_60d"]
    posted = roll["posted"]
    lines = [f"scoreboard, trailing 60 trading days (since {roll['since']}):"]
    if posted["graded"]:
        lines.append(
            f"posted ideas: {posted['graded']} graded, "
            f"avg {posted['avg_r']:+.2f}R, hit {posted['hit_rate']:.0f}%, "
            f"total {posted['total_r']:+.1f}R")
    else:
        lines.append("posted ideas: none graded yet")
    unposted = roll["unposted"]
    if unposted["graded"]:
        lines.append(
            f"ideas drafted and NOT posted: {unposted['graded']} graded, "
            f"avg {unposted['avg_r']:+.2f}R (the filter is on the record too)")
    lines.append("every idea carried its entry, time stop and evidence when "
                 "posted. grading is pessimistic: both-touch bars book the "
                 "stop, gapped stops fill at the open plus slippage.")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", default=str(posts_journal.JOURNAL_PATH))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--regrade", action="store_true")
    ap.add_argument("--asof", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--format-post", action="store_true")
    args = ap.parse_args()

    journal_path = Path(args.journal)
    drafts = posts_journal.fold_drafts(posts_journal.load(journal_path))
    ideas = [d for d in drafts if d.get("type") == "idea" and d.get("idea")]
    today = pd.Timestamp(args.asof or dt.date.today()).normalize()

    todo = [d for d in ideas
            if args.regrade or (d.get("outcome") or {}).get("status")
            in (None, "open", "ungradeable")]
    outcomes = []
    if todo:
        prices = load_prices()
        wanted = {str((d["idea"].get("proxy_ticker") or d["idea"]["ticker"])).upper()
                  for d in todo}
        bars_by_ticker = {}
        for ticker, group in prices[prices["ticker"].isin(wanted)].groupby("ticker"):
            frame = group.sort_values("date").set_index("date")
            frame.index = pd.DatetimeIndex(frame.index).normalize()
            bars_by_ticker[ticker] = frame[~frame.index.duplicated(keep="last")]

        still_open = 0
        for draft in todo:
            row = derive_order_row(draft)
            key = str(row.get("Proxy_Ticker") or row["Ticker"]).upper()
            bars = bars_by_ticker.get(key)
            if bars is None or bars.empty:
                still_open += 1
                continue
            leg = replay_leg(bars, row)
            if leg["status"] in ("open", "no_session"):
                still_open += 1
                continue
            risk = float(row["Risk_Amt"])
            pnl = leg.get("pnl", 0.0)
            outcome = {**leg,
                       "r_multiple": (round(pnl / risk, 3)
                                      if risk and leg["status"] == "closed"
                                      else (0.0 if leg["status"] == "no_fill"
                                            else None))}
            # A no-fill idea books 0R deliberately: the audience saw an entry
            # that never triggered, which is a null result, not a loss.
            outcomes.append({"kind": "outcome", "draft_id": draft["draft_id"],
                             "date": draft["date"], "outcome": outcome,
                             "graded_at": dt.datetime.now().isoformat(
                                 timespec="seconds")})
            r = outcome.get("r_multiple")
            print(f"  {draft['draft_id']} {row['Ticker']:<8} "
                  f"{leg['status']:<10} "
                  f"{'' if r is None else f'{r:+.2f}R'} "
                  f"{leg.get('exit_kind', '')}")
        print(f"graded {len(outcomes)}; {still_open} still open/unpriced; "
              f"{len(ideas) - len(todo)} already booked")

    if args.dry_run:
        print("[dry-run] no journal append, no scoreboard write")
        return 0
    if outcomes:
        posts_journal.append(outcomes, journal_path)

    refreshed = posts_journal.fold_drafts(posts_journal.load(journal_path,
                                                             pull=False))
    board = build_scoreboard(refreshed, today)
    Path(args.out).write_text(json.dumps(board, indent=1), encoding="utf-8")
    roll = board["rolling_60d"]
    print(f"scoreboard -> {args.out}: {roll['n']} idea(s) in window, "
          f"{roll['graded']} graded, posted avg "
          f"{roll['posted']['avg_r'] if roll['posted']['avg_r'] is not None else 'n/a'}R")
    if args.format_post:
        print()
        print(format_post(board))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
