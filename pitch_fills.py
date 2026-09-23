"""Daily Pitch approvals derived from live FILLS.

The private site's Pitch tab lets McKinley stage a pitch idea into IBKR by
hand, after the 9:05/9:32 runner windows. Those orders carry the same
orderRef strategy `pitch_moo.py` stamps, `Pitch-{idea_id}`, so a fill under
that tag is proof the idea was traded. This module turns such fills into ONE
`approval` record per idea (`approve: "Y"`, `source: "fills"`), so the
scoreboard's approved bucket counts site-staged ideas too. There is no write
path from the site; a staged limit that never filled does not count, by
decision.

Source: `data/live_fills.parquet` (R2-canonical, harvested postclose by
`scripts/harvest_fills.py`). Children of a bracket carry the parent's ref, so
only ENTRY fills (fill side agrees with the ref's action) are counted.

Ordering with the tab capture: `daily_pitch.capture_approvals` reads the
Pitch tab the morning after an idea, which runs AFTER the grader. An idea is
only eligible here once its tab answer is on the journal (any non-fills
approval record, blank included) or it is older than
`TAB_WINDOW_TD` sessions. Otherwise a pitch_moo-placed idea would get a
fills record first and the tab's own `Y` would be swallowed by the
(kind, date, idea_id, approve) dedupe, losing where the approval came from.
"""
from __future__ import annotations

import datetime as dt
import os
import re
from pathlib import Path

import pandas as pd

import pitch_journal

ROOT = Path(__file__).resolve().parent
FILLS_PATH = ROOT / "data" / "live_fills.parquet"
FILLS_R2_KEY = "live_fills.parquet"
STRATEGY_RE = re.compile(r"^Pitch-(\d{4}-\d{2}-\d{2}-\d+)$")
# pitch_moo stamps a short entry's orderRef action as SELL_SHORT; site-staged
# shorts carry SELL. Both are entries when the fill is a sale.
ENTRY_SIDES = {("BUY", "BOT"), ("SELL", "SLD"), ("SELL_SHORT", "SLD")}
TAB_WINDOW_TD = 2
STALE_DAYS = 5


def idea_id_from_strategy(strategy) -> str | None:
    match = STRATEGY_RE.match(str(strategy or "").strip())
    return match.group(1) if match else None


def refresh_fills(path: Path = FILLS_PATH) -> None:
    """Pull the canonical store from R2 over the local copy. Only the default
    path syncs, and a failed pull leaves any local copy untouched."""
    if path != FILLS_PATH:
        return
    from cache_io import download_to_local, is_configured
    if not is_configured():
        print("[pitch-fills] R2 not configured; using the local fills copy if any")
        return
    tmp = path.with_suffix(".parquet.pitchdl")
    try:
        if download_to_local(FILLS_R2_KEY, str(tmp)):
            pd.read_parquet(tmp)  # refuse to install an unreadable download
            os.replace(tmp, path)
        else:
            print("[pitch-fills] WARNING: R2 fills pull failed; using the local copy if any")
    except Exception as exc:  # noqa: BLE001
        print(f"[pitch-fills] WARNING: R2 fills pull unusable ({exc}); using the local copy if any")
    finally:
        if tmp.exists():
            tmp.unlink()


def load_fills(path: Path = FILLS_PATH, today: pd.Timestamp | None = None) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[pitch-fills] WARNING: no fills store at {path}; no fills approvals this run")
        return None
    fills = pd.read_parquet(path)
    newest = fills["session_date"].dropna().astype(str).max() if len(fills) else None
    today = pd.Timestamp(today or dt.date.today()).normalize()
    if newest is None or (today - pd.Timestamp(newest)).days > STALE_DAYS:
        print(f"[pitch-fills] WARNING: fills store newest session {newest} is stale; "
              f"approvals may be missing")
    return fills


def summarize_pitch_fills(fills: pd.DataFrame) -> dict[str, dict]:
    """idea_id -> entry-fill summary (first session, accounts, qty, VWAPs)."""
    if fills is None or fills.empty:
        return {}
    frame = fills.copy()
    frame["idea_id"] = frame["strategy"].map(idea_id_from_strategy)
    frame = frame[frame["idea_id"].notna()]
    pairs = zip(frame["ref_action"].astype(str).str.upper(),
                frame["side"].astype(str).str.upper())
    entry = pd.Series([pair in ENTRY_SIDES for pair in pairs], index=frame.index, dtype=bool)
    frame = frame[entry]
    frame = frame[pd.to_numeric(frame["qty"], errors="coerce") > 0]
    out: dict[str, dict] = {}
    for idea_id, group in frame.groupby("idea_id"):
        qty = group["qty"].astype(float)
        notional = (qty * group["price"].astype(float)).sum()
        symbols = {}
        for symbol, leg in group.groupby("symbol"):
            leg_qty = leg["qty"].astype(float)
            symbols[str(symbol)] = {
                "side": str(leg["side"].iloc[0]),
                "qty": float(leg_qty.sum()),
                "vwap": round(float((leg_qty * leg["price"].astype(float)).sum()
                                    / leg_qty.sum()), 4),
            }
        out[str(idea_id)] = {
            "first_fill_session": str(group["session_date"].astype(str).min()),
            "accounts": sorted({str(a) for a in group["account"].dropna()}),
            "total_qty": float(qty.sum()),
            "vwap": round(float(notional / qty.sum()), 4),
            "symbols": symbols,
            "n_fills": int(len(group)),
        }
    return out


def approval_records(records: list[dict], summary: dict[str, dict],
                     today: pd.Timestamp) -> list[dict]:
    from trading_calendar import TRADING_DAY

    ideas = {i["idea_id"]: i for i in pitch_journal.fold_ideas(records)}
    tab_seen = {r.get("idea_id") for r in records
                if r.get("kind") == "approval" and r.get("source") != "fills"}
    existing = {(r.get("kind"), r.get("date"), r.get("idea_id"), r.get("approve"))
                for r in records}
    window_cutoff = str((today - TAB_WINDOW_TD * TRADING_DAY).normalize().date())
    out = []
    for idea_id, info in sorted(summary.items()):
        idea = ideas.get(idea_id)
        if idea is None:
            print(f"[pitch-fills] WARNING: fills tagged Pitch-{idea_id} match no journaled "
                  f"idea; skipped")
            continue
        if pitch_journal.approved(idea):
            continue
        date = str(idea.get("date", ""))
        if idea_id not in tab_seen and date > window_cutoff:
            print(f"[pitch-fills] {idea_id} filled; waiting for the tab capture "
                  f"before journaling it")
            continue
        if ("approval", date, idea_id, "Y") in existing:
            continue
        out.append({"kind": "approval", "idea_id": idea_id, "date": date,
                    "approve": "Y", "source": "fills",
                    "captured_at": dt.datetime.now().isoformat(timespec="seconds"),
                    "fills": info})
    return out


def append_fills_approvals(journal_path: Path, today: pd.Timestamp,
                           fills_path: Path | None = None,
                           dry_run: bool = False) -> int:
    """Morning hook. A non-default journal never touches R2 for either file."""
    default = journal_path == pitch_journal.JOURNAL_PATH
    if fills_path is None:
        fills_path = FILLS_PATH if default else journal_path.parent / "live_fills.parquet"
    if default:
        refresh_fills(fills_path)
    fills = load_fills(fills_path, today)
    summary = summarize_pitch_fills(fills)
    records = pitch_journal.load(journal_path, pull=default)
    new = approval_records(records, summary, today)
    for record in new:
        info = record["fills"]
        print(f"[pitch-fills] {record['idea_id']} approved from fills: "
              f"{info['total_qty']:g} sh {','.join(info['symbols'])} "
              f"first {info['first_fill_session']} {'/'.join(info['accounts'])}")
    if dry_run or not new:
        return len(new)
    return pitch_journal.append(new, journal_path)
