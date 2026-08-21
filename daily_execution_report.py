"""Nightly 5 PM ET execution report — live PRIMARY-account positions by email.

Mirrors the site Execution tab's positions table for the primary IBKR account.
Data source: the execution-broker Durable Object's /book snapshot (pushed by
the local exec_agent from a read-only IBKR connection), so the numbers are
actual broker fills, not the modeled portfolio report.

Options positions (sec_type OPT) are excluded (listed in a footer line). Each
remaining position is enriched from its own working exit legs in the same
snapshot:
  - target      = closing LMT leg(s) lmtPrice
  - stop        = closing STP leg auxPrice (NA when no stop is working)
  - time stop   = closing MKT leg's goodAfterTime date
  - strategy    = 3rd pipe field of any leg's orderRef
                  (eq_order_entry stamps SYMBOL|ACTION|Strategy_Ref|Staged_Date
                  on every leg, so a filled entry's working exits keep the tag)
Positions with no strategy-tagged legs fall back to "Event Sleeve (T1..V4)"
when the symbol is held in event_sleeve_state.json (R2 — event orders are
parent-only MKT MOC/MOO with no exit legs, so orderRef attribution can never
see them), then "Trend Sleeve" when the symbol is in trend_sleeve_state.json
(R2), else "Discretionary". Both sleeves' states are also reconciled against
the live book (mismatch = staged order never executed -> phantom future
exits); event positions entered TODAY are skipped (the MOC fill and the book
push race the 4:30 report).

Scheduling: .github/workflows/execution_report.yml runs at 20:30 AND 21:30 UTC
on weekdays; exactly one of the two sends at ~4:30 PM ET year-round. The gate
decides by WHICH cron fired (GHA_SCHEDULE env = github.event.schedule) and
whether the date is in daylight time — not by the actual clock, because GHA's
shared cron queue can start runs 60-90 min late (both 2026-07-08 and -09 runs
arrived past 5 PM ET and the old hour==16 gate dropped the email). Without
GHA_SCHEDULE (local runs) the legacy hour==16 gate applies. --force bypasses
the gate for manual runs.

Env:
  EXEC_BROKER_URL  broker base URL (default the deployed workers.dev URL)
  STATUS_TOKEN     broker read bearer token (required)
  EMAIL_USER / EMAIL_PASS  Gmail SMTP creds (same as the other reports)
  RECIPIENTS       comma-separated recipient list
                   (default mckinleyslade@gmail.com)
"""
import argparse
import json
import os
import smtplib
import sys
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

ET = ZoneInfo("America/New_York")
DEFAULT_BROKER_URL = "https://execution-broker.mckinleyslade.workers.dev"
DEFAULT_RECIPIENTS = "mckinleyslade@gmail.com"
TREND_STATE_KEY = "trend_sleeve_state.json"
TREND_STATE_LOCAL = Path(__file__).resolve().parent / "data" / TREND_STATE_KEY
EVENT_STATE_KEY = "event_sleeve_state.json"
EVENT_STATE_LOCAL = Path(__file__).resolve().parent / "data" / EVENT_STATE_KEY

# Order statuses that mean the leg is no longer working.
DEAD_ORDER_STATUSES = {"Cancelled", "ApiCancelled", "Filled", "Inactive"}


def et_now() -> datetime:
    return datetime.now(tz=ET)


# Must match the cron strings in .github/workflows/execution_report.yml.
EDT_CRON = "30 20 * * 1-5"  # 4:30 PM ET while daylight time
EST_CRON = "30 21 * * 1-5"  # 4:30 PM ET while standard time


def cron_should_send(cron: str, now: datetime) -> bool:
    in_dst = bool(now.dst())
    if cron == EDT_CRON:
        return in_dst
    if cron == EST_CRON:
        return not in_dst
    # Unrecognized cron (workflow edited without updating the constants):
    # send rather than silently drop — a duplicate email is visible, a
    # missing one is not.
    print(f"WARNING: unrecognized cron {cron!r} — sending anyway.")
    return True


def fetch_book(base_url: str, token: str) -> dict | None:
    r = requests.get(
        f"{base_url.rstrip('/')}/book",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json().get("book")


def load_trend_state() -> dict:
    """Full trend-sleeve state dict from R2 (best effort, {} when unavailable)."""
    try:
        from cache_io import download_to_local, is_configured
        if is_configured():
            TREND_STATE_LOCAL.parent.mkdir(parents=True, exist_ok=True)
            download_to_local(TREND_STATE_KEY, str(TREND_STATE_LOCAL))
    except Exception as e:  # noqa: BLE001
        print(f"NOTE: trend state R2 pull skipped ({e})")
    try:
        return json.loads(TREND_STATE_LOCAL.read_text())
    except Exception:  # noqa: BLE001
        return {}


def load_trend_symbols(state: dict | None = None) -> set[str]:
    """Symbols the trend sleeve currently holds (R2 state, best effort)."""
    if state is None:
        state = load_trend_state()
    return {
        str(t).upper()
        for t, v in (state.get("positions") or {}).items()
        if int(v.get("shares", 0)) > 0
    }


def reconcile_trend(state: dict, positions: list[dict]) -> list[str]:
    """Trend state target shares vs the live book (2026-07-16).

    The month-end run computes DELTAS against this state — if staged orders
    were never executed (or doubled), every future rebalance is wrong and
    nothing surfaced it. Compares each state symbol's shares to the live STK
    position. Skipped until the state's Execute_On (the first session after
    'generated') has passed, so a freshly staged rebalance never false-alarms.
    Returns human-readable mismatch strings; empty = reconciled or skipped.
    """
    targets = {
        str(t).upper(): int(v.get("shares", 0))
        for t, v in (state.get("positions") or {}).items()
        if int(v.get("shares", 0)) > 0
    }
    if not targets:
        return []
    gen = str(state.get("generated", "")).split(" ")[0]
    if gen:
        try:
            from trading_calendar import TRADING_DAY
            import pandas as pd
            execute_on = (pd.Timestamp(gen) + TRADING_DAY).normalize()
            if pd.Timestamp(et_now().date()) < execute_on:
                print(f"NOTE: trend reconciliation skipped — state generated {gen}, "
                      f"Execute_On {execute_on.date()} not reached")
                return []
        except Exception as e:  # noqa: BLE001
            print(f"NOTE: trend reconciliation date gate failed ({e}) — checking anyway")
    live = {}
    for p in positions:
        if p.get("sec_type") == "STK":
            live[str(p.get("symbol", "")).upper()] = int(p.get("position") or 0)
    issues = []
    for sym, want in sorted(targets.items()):
        got = live.get(sym, 0)
        if got != want:
            issues.append(f"{sym}: state says {want:,} sh, live book has {got:,}")
    return issues


def load_event_state() -> dict:
    """Event-sleeve state dict from R2 (best effort, {} when unavailable)."""
    try:
        from cache_io import download_to_local, is_configured
        if is_configured():
            EVENT_STATE_LOCAL.parent.mkdir(parents=True, exist_ok=True)
            download_to_local(EVENT_STATE_KEY, str(EVENT_STATE_LOCAL))
    except Exception as e:  # noqa: BLE001
        print(f"NOTE: event state R2 pull skipped ({e})")
    try:
        return json.loads(EVENT_STATE_LOCAL.read_text())
    except Exception:  # noqa: BLE001
        return {}


def event_symbol_map(state: dict | None = None) -> dict[str, str]:
    """SYMBOL -> 'Event Sleeve (V4)' label for open event positions. The
    sleeve keys positions by trade id (V4_POSTOPEX_VOL); the short prefix is
    unique per trade and keeps the strategy column narrow. Two trades can
    never hold the same symbol at once (V2/V4 mutual stand-down), but join
    defensively if they ever do."""
    if state is None:
        state = load_event_state()
    by_sym: dict[str, list[str]] = {}
    for trade, pos in (state.get("positions") or {}).items():
        try:
            from event_sleeve import EVENT_SLEEVE
            sym = EVENT_SLEEVE[trade]["ticker"].upper()
        except Exception:  # noqa: BLE001
            continue
        if int(pos.get("shares", 0)) > 0:
            by_sym.setdefault(sym, []).append(str(trade).split("_")[0])
    return {sym: f"Event Sleeve ({'+'.join(sorted(set(t)))})"
            for sym, t in by_sym.items()}


def reconcile_event(state: dict, positions: list[dict]) -> list[str]:
    """Event state vs the live book. State marks positions open at STAGING
    time, so an order the runner never placed leaves a phantom that later
    stages a phantom exit — this is what makes that loud. Positions entered
    TODAY are skipped: the entry MOC fills at 4:00 and the agent's book push
    can race the 4:30 report."""
    issues = []
    today = et_now().strftime("%Y-%m-%d")
    live = {}
    for p in positions:
        if p.get("sec_type") == "STK":
            live[str(p.get("symbol", "")).upper()] = int(p.get("position") or 0)
    for trade, pos in sorted((state.get("positions") or {}).items()):
        if str(pos.get("entry_date", "")) >= today:
            continue
        try:
            from event_sleeve import EVENT_SLEEVE
            cfg = EVENT_SLEEVE[trade]
        except Exception:  # noqa: BLE001
            continue
        sign = 1 if cfg["side"] == "LONG" else -1
        want = sign * int(pos.get("shares", 0))
        got = live.get(cfg["ticker"].upper(), 0)
        # Only a SHORTFALL in the event direction alarms: the book can hold
        # the same symbol on top of the sleeve (SPY/IWM overlap), so an
        # excess is indistinguishable from systematic/discretionary shares.
        if sign * got < sign * want:
            issues.append(f"{trade}: state says {want:,} sh "
                          f"{cfg['ticker']}, live book has {got:,}")
    return issues


def parse_ref(ref: str | None) -> tuple[str | None, str | None]:
    """orderRef 'SYMBOL|ACTION|Strategy_Ref|Staged_Date[|tranche]' ->
    (strategy, staged_date)."""
    if not ref:
        return None, None
    parts = str(ref).split("|")
    if len(parts) < 4:
        return None, None
    strat = parts[2].strip() or None
    staged = parts[3].strip() or None
    return strat, staged


def parse_ib_date(s: str | None) -> str | None:
    """Leading YYYYMMDD of an IB time string -> ISO date."""
    if not s:
        return None
    head = str(s).strip()[:8]
    if len(head) == 8 and head.isdigit():
        return f"{head[:4]}-{head[4:6]}-{head[6:8]}"
    return None


def exit_legs_for(pos: dict, orders: list[dict]) -> list[dict]:
    """Working orders that close this position: same symbol + sec_type,
    opposite side, still-alive status."""
    closing = "SELL" if (pos.get("position") or 0) > 0 else "BUY"
    return [
        o for o in orders
        if o.get("symbol") == pos.get("symbol")
        and o.get("sec_type") == pos.get("sec_type")
        and str(o.get("action", "")).upper() == closing
        and str(o.get("status", "")) not in DEAD_ORDER_STATUSES
    ]


def enrich_position(pos: dict, orders: list[dict], trend_syms: set[str],
                    event_map: dict[str, str] | None = None) -> dict:
    legs = exit_legs_for(pos, orders)
    event_map = event_map or {}

    targets = sorted({o["lmt"] for o in legs
                      if o.get("order_type") == "LMT" and o.get("lmt")})
    # STP legs: prefer the GTC protective stop; the OVS EOD-DD leg is GTD and
    # expires intraday so it should never be working at 5 PM anyway.
    stops = [o for o in legs if str(o.get("order_type", "")).startswith("STP")]
    stops = [o for o in stops if o.get("tif") == "GTC"] or stops
    stop = stops[0].get("aux") if stops else None

    time_stop = None
    for o in legs:
        if o.get("order_type") == "MKT" and o.get("good_after"):
            d = parse_ib_date(o.get("good_after"))
            if d and (time_stop is None or d < time_stop):
                time_stop = d

    strategies, staged_dates = [], []
    for o in legs:
        strat, staged = parse_ref(o.get("order_ref"))
        if strat and strat not in strategies:
            strategies.append(strat)
        if staged:
            staged_dates.append(staged)

    sym_u = str(pos.get("symbol", "")).upper()
    if strategies:
        strategy = " + ".join(strategies)
    elif sym_u in event_map:
        strategy = event_map[sym_u]
    elif sym_u in trend_syms:
        strategy = "Trend Sleeve"
    else:
        strategy = "Discretionary"

    qty = pos.get("position") or 0
    avg = pos.get("avg_cost")
    upnl = pos.get("unrealized_pnl")
    basis = abs((avg or 0) * qty)
    pnl_pct = (upnl / basis * 100.0) if (upnl is not None and basis) else None

    return {
        "symbol": pos.get("symbol"),
        "sec_type": pos.get("sec_type"),
        "qty": qty,
        "avg_cost": avg,
        "last": pos.get("market_price"),
        "mkt_value": pos.get("market_value"),
        "upnl": upnl,
        "pnl_pct": pnl_pct,
        "targets": targets,
        "stop": stop,
        "time_stop": time_stop,
        "strategy": strategy,
        "staged": min(staged_dates) if staged_dates else None,
    }


def build_rows(account: dict, trend_syms: set[str],
               event_map: dict[str, str] | None = None) -> tuple[list[dict], list[str]]:
    positions = account.get("positions") or []
    orders = account.get("orders") or []
    opt_excluded = [
        p.get("symbol") for p in positions if p.get("sec_type") == "OPT"
    ]
    rows = [
        enrich_position(p, orders, trend_syms, event_map)
        for p in positions if p.get("sec_type") != "OPT"
    ]

    # Strategy positions first (alpha), then Event Sleeve, Trend Sleeve,
    # Discretionary.
    def rank(strategy: str) -> int:
        if strategy.startswith("Event Sleeve"):
            return 1
        return {"Trend Sleeve": 2, "Discretionary": 3}.get(strategy, 0)

    rows.sort(key=lambda r: (rank(r["strategy"]),
                             r["strategy"], r["symbol"] or ""))
    return rows, opt_excluded


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _fmt_num(v, dec=2):
    return f"{v:,.{dec}f}" if isinstance(v, (int, float)) else "&mdash;"


def _fmt_money(v):
    if not isinstance(v, (int, float)):
        return "&mdash;"
    sign = "-" if v < 0 else ""
    return f"{sign}${abs(v):,.0f}"


def _pnl_color(v):
    if not isinstance(v, (int, float)):
        return "#555"
    return "#0a7a2f" if v >= 0 else "#b02a1e"


def build_html(rows: list[dict], opt_excluded: list[str], nlv,
               now: datetime, trend_warnings: list[str] | None = None,
               event_warnings: list[str] | None = None) -> str:
    td = "padding:6px 10px;border-bottom:1px solid #e3e3e3;font-size:13px;"
    tdr = td + "text-align:right;white-space:nowrap;"
    th = ("padding:6px 10px;border-bottom:2px solid #444;font-size:12px;"
          "text-align:left;color:#333;")
    thr = th + "text-align:right;"

    body_rows = []
    for r in rows:
        tgt = " / ".join(_fmt_num(t) for t in r["targets"]) if r["targets"] else "&mdash;"
        qty = r["qty"]
        qty_s = f"{qty:,.0f}" if isinstance(qty, (int, float)) else "&mdash;"
        fut_badge = (" <span style='color:#888;font-size:11px'>FUT</span>"
                     if r["sec_type"] == "FUT" else "")
        body_rows.append(
            "<tr>"
            f"<td style='{td}font-weight:600'>{r['symbol']}{fut_badge}</td>"
            f"<td style='{td}'>{r['strategy']}</td>"
            f"<td style='{tdr}'>{qty_s}</td>"
            f"<td style='{tdr}'>{_fmt_num(r['avg_cost'])}</td>"
            f"<td style='{tdr}'>{_fmt_num(r['last'])}</td>"
            f"<td style='{tdr}'>{_fmt_money(r['mkt_value'])}</td>"
            f"<td style='{tdr}color:{_pnl_color(r['upnl'])}'>{_fmt_money(r['upnl'])}</td>"
            f"<td style='{tdr}color:{_pnl_color(r['pnl_pct'])}'>"
            f"{_fmt_num(r['pnl_pct']) + '%' if r['pnl_pct'] is not None else '&mdash;'}</td>"
            f"<td style='{tdr}'>{tgt}</td>"
            f"<td style='{tdr}'>{_fmt_num(r['stop'])}</td>"
            f"<td style='{tdr}'>{r['time_stop'] or '&mdash;'}</td>"
            f"<td style='{tdr}'>{r['staged'] or '&mdash;'}</td>"
            "</tr>"
        )

    tot_mv = sum(r["mkt_value"] for r in rows if isinstance(r["mkt_value"], (int, float)))
    tot_pnl = sum(r["upnl"] for r in rows if isinstance(r["upnl"], (int, float)))
    totals = (
        "<tr>"
        f"<td style='{td}font-weight:700'>TOTAL</td>"
        f"<td style='{td}'></td><td style='{tdr}'></td><td style='{tdr}'></td>"
        f"<td style='{tdr}'></td>"
        f"<td style='{tdr}font-weight:700'>{_fmt_money(tot_mv)}</td>"
        f"<td style='{tdr}font-weight:700;color:{_pnl_color(tot_pnl)}'>{_fmt_money(tot_pnl)}</td>"
        f"<td colspan='5' style='{td}'></td>"
        "</tr>"
    )

    opt_note = ""
    if opt_excluded:
        opt_note = (
            f"<p style='color:#888;font-size:12px'>{len(opt_excluded)} options "
            f"position(s) excluded: {', '.join(sorted(set(opt_excluded)))}.</p>"
        )

    return f"""
<div style="font-family:Segoe UI,Arial,sans-serif;max-width:1000px">
  <h2 style="margin-bottom:2px">Position Sheet &mdash; Primary Account</h2>
  <p style="color:#555;margin-top:2px;font-size:13px">
    {now.strftime('%A %Y-%m-%d')} &nbsp;|&nbsp;
    NLV: <b>{_fmt_money(nlv)}</b> &nbsp;|&nbsp; {len(rows)} position(s)
  </p>
  {('<div style="background:#fdecea;border:1px solid #b02a1e;border-radius:4px;'
    'padding:8px 12px;margin:8px 0"><b style="color:#b02a1e">TREND SLEEVE '
    'MISMATCH</b> — state targets vs live book diverge; next month-end '
    'rebalance deltas will be WRONG until reconciled (fix positions or clear '
    'the state):<br>'
    + '<br>'.join(trend_warnings) + '</div>') if trend_warnings else ''}
  {('<div style="background:#fdecea;border:1px solid #b02a1e;border-radius:4px;'
    'padding:8px 12px;margin:8px 0"><b style="color:#b02a1e">EVENT SLEEVE '
    'MISMATCH</b> — state holds shares the live book is missing; the staged '
    'order likely never executed and the sleeve will stage a phantom exit '
    '(fill the position or clear it from event_sleeve_state.json):<br>'
    + '<br>'.join(event_warnings) + '</div>') if event_warnings else ''}
  <table style="border-collapse:collapse;width:100%">
    <tr>
      <th style="{th}">Symbol</th><th style="{th}">Strategy</th>
      <th style="{thr}">Qty</th><th style="{thr}">Avg</th>
      <th style="{thr}">Last</th><th style="{thr}">Mkt Val</th>
      <th style="{thr}">PnL $</th><th style="{thr}">PnL %</th>
      <th style="{thr}">Target</th><th style="{thr}">Stop</th>
      <th style="{thr}">Time Stop</th><th style="{thr}">Staged</th>
    </tr>
    {''.join(body_rows) if body_rows else
     f"<tr><td colspan='12' style='{td}color:#888'>No open non-option positions.</td></tr>"}
    {totals if body_rows else ''}
  </table>
  {opt_note}
  <p style="color:#aaa;font-size:11px;margin-top:14px">
    Source: execution-broker /book (read-only IBKR snapshot). Stops shown are
    working STP legs; targets are working LMT exit legs; time stop is the MKT
    exit leg's activation date. "Event Sleeve" / "Trend Sleeve" = untagged
    position whose symbol the sleeve state holds (their orders are parent-only,
    exits sleeve-scheduled); "Discretionary" = no strategy-tagged exit legs and
    no sleeve claim.
  </p>
</div>"""


def send_email(subject: str, html: str) -> bool:
    sender = os.environ.get("EMAIL_USER")
    password = os.environ.get("EMAIL_PASS")
    recipients = [
        a.strip() for a in
        os.environ.get("RECIPIENTS", DEFAULT_RECIPIENTS).split(",")
        if a.strip()
    ]
    if not sender or not password:
        print("EMAIL_USER/EMAIL_PASS not set — skipping email send.")
        return False
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(html, "html"))
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, recipients, msg.as_string())
    print(f"Email sent to {', '.join(recipients)}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="skip the 4 PM ET hour gate")
    ap.add_argument("--html-out", default=None,
                    help="also write the rendered HTML to this path")
    ap.add_argument("--no-send", action="store_true",
                    help="render only, never email (implies --force useful for dev)")
    args = ap.parse_args()

    now = et_now()
    if not args.force and not args.no_send:
        cron = os.environ.get("GHA_SCHEDULE", "")
        if cron:
            if not cron_should_send(cron, now):
                print(f"Cron {cron!r} is the DST twin for this date "
                      f"(dst={bool(now.dst())}) — exiting.")
                return 0
        elif now.hour != 16:
            print(f"ET hour is {now.hour}, not 16 — DST twin cron, exiting.")
            return 0

    base_url = os.environ.get("EXEC_BROKER_URL", DEFAULT_BROKER_URL)
    token = os.environ.get("STATUS_TOKEN", "")
    subject_date = now.strftime("%Y-%m-%d")

    try:
        if not token:
            raise RuntimeError("STATUS_TOKEN not set")
        book = fetch_book(base_url, token)
        if not book:
            raise RuntimeError("broker returned no book (agent never pushed one?)")
        primary = next((a for a in book.get("accounts", [])
                        if a.get("key") == "primary"), None)
        if primary is None:
            raise RuntimeError("no 'primary' account in book")
        if primary.get("error"):
            raise RuntimeError(f"primary account error: {primary['error']}")
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {e}")
        send_email(f"Execution Report FAILED — {subject_date}",
                   f"<p>Nightly execution report could not fetch the live book:"
                   f"</p><pre>{e}</pre>")
        return 1

    trend_state = load_trend_state()
    trend_syms = load_trend_symbols(trend_state)
    trend_warnings = reconcile_trend(trend_state, primary.get("positions") or [])
    if trend_warnings:
        print("TREND RECONCILIATION MISMATCH:")
        for w in trend_warnings:
            print(f"  {w}")
    event_state = load_event_state()
    event_map = event_symbol_map(event_state)
    event_warnings = reconcile_event(event_state, primary.get("positions") or [])
    if event_warnings:
        print("EVENT RECONCILIATION MISMATCH:")
        for w in event_warnings:
            print(f"  {w}")
    rows, opt_excluded = build_rows(primary, trend_syms, event_map)
    html = build_html(rows, opt_excluded, primary.get("nlv"), now,
                      trend_warnings=trend_warnings,
                      event_warnings=event_warnings)

    if args.html_out:
        Path(args.html_out).write_text(html, encoding="utf-8")
        print(f"HTML written to {args.html_out}")

    if not args.no_send:
        send_email("Position Sheet", html)
    return 0


if __name__ == "__main__":
    sys.exit(main())
