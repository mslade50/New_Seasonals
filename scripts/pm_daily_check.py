"""PM daily check-in: code-only exceptions on the systematic book.

    python scripts/pm_daily_check.py [--today YYYY-MM-DD] [--no-send] [--no-r2] [--no-sync]

Weekdays 08:30 ET on the trading desktop. Quiet by default: every run journals
a `check_in` record (so silence proves it ran), and an email goes out only when
an exception fires. No LLM; the catalogue is fixed:

  job_failed        an automation receipt with status failure (prior session or today)
  sleeve_task       an ENABLED sleeve task Missing or with a non-zero last result
  exit_missed       expected-exit obligations with missed > 0
  fills_store       live_fills gap, Primary incomplete, or last session behind
  snapshot_missing  no Primary broker snapshot for the prior session
  delivery_missing  Pitch / Seasonal (today) or Risk Agent (prior-session asof) not sent
  position_flip     a Primary stock position on the wrong side of its only
                    strategy tag (e.g. short after selling out a BUY-tagged entry)
  nlv_move          Primary NLV moved more than NLV_MOVE in one session (or a flow)

Deliberately NOT here: stopless/unprotected position alerts (owner, 2026-10-09:
those are intentional), staleness harmonisation (each consumer keeps its own
rule), sizing or dial proposals. Doc: docs/claude_ref/pm_agent.md
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pm_agent_book as B  # noqa: E402
import pm_agent_data as pad  # noqa: E402
import pm_agent_journal as J  # noqa: E402
import pm_agent_universe as U  # noqa: E402
from trading_calendar import TRADING_DAY, NYSE_HOLIDAYS  # noqa: E402

NLV_MOVE = 0.03
SCHEMA = "pm_agent_checkin.v1"


def is_session(d: dt.date) -> bool:
    return d.weekday() < 5 and pd.Timestamp(d) not in set(NYSE_HOLIDAYS)


def prev_session(d: dt.date) -> dt.date:
    return (pd.Timestamp(d) - TRADING_DAY).date()


def _j(key: str, cache_dir=None):
    try:
        return json.loads(pad.local_path(key, cache_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def position_flips(snapshot: dict | None, fills: pd.DataFrame | None, lookback_days: int = 30) -> list[dict]:
    a = B.primary_account(snapshot or {})
    if not a or fills is None or fills.empty:
        return []
    f = fills[fills["account_key"] == "primary"].copy()
    f["session_date"] = f["session_date"].astype(str)
    lo = str((pd.Timestamp(snapshot.get("session") or dt.date.today()) - pd.Timedelta(days=lookback_days)).date())
    f = f[f["session_date"] >= lo]
    out = []
    for p in a.get("positions") or []:
        if p.get("sec_type") != "STK" or not p.get("position"):
            continue
        g = f[f["symbol"] == p["symbol"]]
        # strategy entry tags are SYMBOL|BUY|Strategy|date; EXEC|<uuid>|... close
        # tags parse a uuid into ref_action and are not entries
        acts = set(g["ref_action"].dropna().astype(str).str.upper()) & {"BUY", "SELL"}
        strat = sorted(set(g.loc[g["ref_action"].astype(str).str.upper().isin(["BUY", "SELL"]),
                                 "strategy"].dropna().astype(str)) - {""})
        if p["position"] < 0 and acts == {"BUY"}:
            side, tag = "short", "BUY"
        elif p["position"] > 0 and acts == {"SELL"}:
            side, tag = "long", "SELL"
        else:
            continue
        out.append({"symbol": p["symbol"], "position": p["position"],
                    "market_value": p.get("market_value"), "side": side, "only_tag": tag,
                    "strategies": strat})
    return out


def exceptions(today: dt.date, cache_dir: Path | None = None, keys: dict | None = None) -> tuple[list[dict], dict]:
    ps = prev_session(today)
    keys = keys or B.load_index(cache_dir)
    exc: list[dict] = []
    facts: dict = {"today": str(today), "prev_session": str(ps)}

    def add(kind, key, msg, **extra):
        exc.append({"kind": kind, "key": key, "message": msg, **extra})

    # job receipts
    rec = B.load_snapshots(keys.get("receipts") or [], cache_dir)
    fails: dict[str, list] = {}
    for j in rec:
        if j.get("status") == "failure":
            fails.setdefault(j.get("job_id"), []).append(j.get("run_date_et"))
    for job, days in sorted(fails.items()):
        recent = [d for d in days if d in (str(ps), str(today))]
        if recent:
            add("job_failed", f"job:{job}", f"{job} failed on {max(recent)} "
                f"({len(days)} failure day(s) in the last {B.RECEIPT_DAYS} days)", days=sorted(days))
    # sleeve runtime
    for t in B.runtime_issues(_j("ops/sleeve_runtime_status.json", cache_dir)):
        add("sleeve_task", f"task:{t['task']}", f"sleeve task {t['task']} is {t['state']} "
            f"(last result {t['last_result']}, last run {t['last_run_at']})")
    # expected exits
    ee = _j("ops/expected_exit_status.json", cache_dir) or {}
    missed = ((ee.get("counts") or {}).get("missed") or 0)
    if missed:
        add("exit_missed", "exits", f"{missed} expected exit(s) missed (status {ee.get('status')})")
    # fills store
    fh = B.fills_health(_j("live_fills_status.json", cache_dir))
    facts["fills"] = fh
    if fh["gap"]:
        add("fills_store", "fills:gap", f"live_fills reports a GAP: {fh['gap_reason']}")
    if fh["primary_complete"] is False:
        add("fills_store", "fills:primary", f"live_fills Primary coverage incomplete: {fh['primary_error']}")
    if fh["last_session"] and fh["last_session"] < str(ps):
        add("fills_store", "fills:behind", f"live_fills last session {fh['last_session']} is behind {ps}")
    # broker snapshot
    snaps = keys.get("snapshots") or []
    if not any(k.endswith(f"/{ps}.json") for k in snaps):
        add("snapshot_missing", "snapshot", f"no Primary broker snapshot for {ps} (ops/olv_capacity)")
    # deliveries
    for label, key in (("Daily Pitch", f"pitch_delivery_receipts/{today}.json"),
                       ("Daily Seasonal", f"seasonal_agent_delivery_receipts/{today}.json"),
                       ("Risk Agent", f"risk_agent/delivery_receipts/{ps}.json")):
        r = _j(key, cache_dir) if key in (keys.get("deliveries") or []) else None
        if not r or r.get("status") != "sent":
            add("delivery_missing", f"delivery:{label}", f"{label} has no sent delivery receipt "
                f"({key.rsplit('/', 1)[-1][:-5]}; status {None if not r else r.get('status')})")
    # positions and NLV
    latest = _j("ops/olv_capacity/latest.json", cache_dir)
    fp = pad.local_path("live_fills.parquet", cache_dir)
    fills = pd.read_parquet(fp) if fp.exists() else None
    for f in position_flips(latest, fills):
        add("position_flip", f"flip:{f['symbol']}",
            f"{f['symbol']} is {f['side']} {abs(f['position']):g} sh (${abs(f['market_value'] or 0):,.0f}) "
            f"but its only strategy tag in 30 days is a {f['only_tag']} entry ({', '.join(f['strategies'])}): "
            "possible over-filled exit", **f)
    nav = B.nav_series(B.load_snapshots(snaps, cache_dir))
    if len(nav) >= 2:
        chg = nav.iloc[-1] / nav.iloc[-2] - 1
        facts["nlv"] = {"last": B._r(nav.iloc[-1], 0), "date": str(nav.index[-1].date()), "chg_pct": B._r(100 * chg)}
        if abs(chg) > NLV_MOVE:
            add("nlv_move", "nlv", f"Primary NLV moved {100 * chg:+.1f}% on {nav.index[-1].date()} "
                f"(to {nav.iloc[-1]:,.0f}); a flow or a large P&L day")
    if latest:
        facts["exposure"] = {k: v for k, v in B.live_exposure(latest).items() if k in
                             ("nlv", "gross_pct", "net_pct", "n_positions", "working_orders")}
    return exc, facts


def streaks(records: list[dict], exc: list[dict]) -> None:
    """Annotate each exception with how many consecutive prior check-ins carried it."""
    prior = [r for r in records if r.get("kind") == "check_in"]
    for e in exc:
        n = 0
        for r in reversed(prior):
            if any(x.get("key") == e["key"] for x in r.get("exceptions") or []):
                n += 1
            else:
                break
        e["days_running"] = n + 1


def render(today: dt.date, exc: list[dict], facts: dict) -> tuple[str, str]:
    import html as h
    subject = f"PM check-in {today}: {len(exc)} exception(s)"
    rows = "".join(f"<tr><td style='padding:4px 8px;'><b>[{h.escape(e['kind'])}]</b></td>"
                   f"<td style='padding:4px 8px;'>{h.escape(e['message'])}"
                   f"{' <i>(day ' + str(e['days_running']) + ')</i>' if e.get('days_running', 1) > 1 else ''}"
                   f"</td></tr>" for e in exc)
    ex = facts.get("exposure") or {}
    nl = facts.get("nlv") or {}
    foot = (f"Primary NLV {nl.get('last')} ({nl.get('chg_pct')}% on {nl.get('date')}) | gross {ex.get('gross_pct')}% "
            f"net {ex.get('net_pct')}% | {ex.get('n_positions')} stock/futures positions | "
            f"{ex.get('working_orders')} working orders")
    body = ("<div style='font-family:Arial,Helvetica,sans-serif;max-width:760px;'>"
            f"<h3 style='margin:0 0 6px;'>PM check-in {today}</h3>"
            f"<table style='border-collapse:collapse;font-size:13px;'>{rows}</table>"
            f"<p style='font-size:12px;color:#6b7280;margin-top:12px;'>{h.escape(foot)}</p>"
            "<p style='font-size:12px;color:#6b7280;'>Read-only check. Nothing here changes a rule or places "
            "an order. Quiet days send no email.</p></div>")
    return subject, body


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--today", default=None)
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--no-send", action="store_true")
    ap.add_argument("--no-r2", action="store_true")
    ap.add_argument("--journal", default=None)
    a = ap.parse_args(argv)
    today = dt.date.fromisoformat(a.today) if a.today else dt.datetime.now(
        __import__("zoneinfo").ZoneInfo("America/New_York")).date()
    if not is_session(today):
        print(f"[pm_check] {today} is not a session; nothing to check")
        return 0
    use_r2 = not a.no_r2
    keys = None
    if not a.no_sync:
        keys = B.sync_book(today, with_ledger=False)["keys"]
    journal = Path(a.journal or U.journal_path())
    records = J.load(journal, pull=use_r2)
    if any(r.get("kind") == "check_in" and r.get("date") == str(today) for r in records):
        print(f"[pm_check] already checked in for {today}")
        return 0
    exc, facts = exceptions(today, keys=keys)
    streaks(records, exc)
    rec = {"kind": "check_in", "schema": SCHEMA, "date": str(today), "exceptions": exc, "facts": facts,
           "n_exceptions": len(exc)}
    sent = None
    if exc and not a.no_send:
        import weekly_pm_agent as W
        subject, html = render(today, exc, facts)
        path = U.home() / "checkin_receipts" / f"{today}.json"
        try:
            receipt, should = W.reserve_receipt(str(today), f"PMC-{today}", subject, html,
                                                W.email_recipients(), path, False)
            if should:
                ok = W.send_email(subject, html, W.email_recipients())
                W.complete_receipt(receipt, path, False, ok)
                sent = ok
        except W.ReceiptError as exc_:
            print(f"[pm_check] DELIVERY BLOCKED: {exc_}")
            sent = False
    rec["emailed"] = sent
    J.append([rec], journal, push=use_r2)
    out = U.home() / "checkin_latest.json"
    out.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    if use_r2:
        import cache_io
        if cache_io.is_configured():
            cache_io.upload_from_local(str(out), U.R2_PREFIX + "checkin_latest.json")
    print(f"[pm_check] {today}: {len(exc)} exception(s); emailed={sent}")
    for e in exc:
        print(f"  [{e['kind']}] {e['message']} (day {e.get('days_running', 1)})")
    return 1 if (exc and sent is False) else 0


if __name__ == "__main__":
    raise SystemExit(main())
