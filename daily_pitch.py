"""Daily Pitch publisher — validate, deliver, stage, journal.

Spec: daily_pitch_agent_spec_2026-08-06.html sections 4-8. The creative work
(invent, falsify, polish) happens in the `/daily-pitch` skill; this module is
the deterministic tail that everything depends on. It refuses to publish
anything the grammar rejects, so a malformed or half-finished pipeline run
produces a loud failure instead of three plausible-looking cards.

    python daily_pitch.py --ideas data/pitch_ideas.json [--dry-run]
                          [--no-send] [--html-out out.html] [--asof DATE]
                          [--validate-only]

What one run does, in order:
  1. read the agent's ideas json and the price cache;
  2. validate the whole payload against pitch_grammar (exactly three ideas,
     one grade-C cap, a time stop on every idea, fresh evidence, the
     repetition rule against the journal);
  3. derive per-leg order specs (sizes, limits, exit dates);
  4. capture yesterday's Approve cells off the Pitch tab BEFORE it is
     overwritten, and journal them;
  5. render and send the Daily Pitch email;
  6. clear and rewrite the Pitch tab with today's rows;
  7. append the idea and kill records to data/pitch_journal.jsonl.

Order matters: approvals are captured before the tab is rewritten, and the
journal is written last so a failed send never records a delivery that did
not happen.

Env: EMAIL_USER / EMAIL_PASS (same Gmail creds as the other reports),
PITCH_RECIPIENTS (defaults to mckinleyslade@gmail.com), GCP_JSON or
credentials.json for Sheets.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import smtplib
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd

import pitch_journal
from pitch_grammar import (
    IDEA_COUNT,
    REPEAT_BLOCK_TD,
    STAND_DOWN_MIN_AXES,
    STAND_DOWN_MIN_CANDIDATES,
    STAND_DOWN_MIN_KILLED,
    build_orders,
    check_risk_budget,
    entry_label,
    exit_label,
    fingerprint,
    is_stand_down,
    load_prices,
    price_context,
    validate_payload,
)
from trading_calendar import TRADING_DAY

ROOT = Path(__file__).resolve().parent
SHEET_NAME = "Trade_Signals_Log"
TAB_NAME = "Pitch"
DEFAULT_IDEAS = ROOT / "data" / "pitch_ideas.json"
SCOREBOARD_PATH = ROOT / "data" / "pitch_scoreboard.json"
STATE_PATH = ROOT / "data" / "pitch_state.json"
DEFAULT_RECIPIENTS = "mckinleyslade@gmail.com"

TAB_COLUMNS = [
    "Idea_Id", "Approve", "Leg", "Ticker", "Sec_Type", "Contract", "Action",
    "Entry_Type", "Entry_Anchor", "Entry_Offset_ATR", "Order_Type", "TIF",
    "Limit_Price", "Quantity", "Stop_ATR", "Target_ATR", "Stop_Price",
    "Target_Price", "Time_Exit_Date", "Time_Exit_Order", "Entry_Expire_Date",
    "Horizon_TD", "Grade", "Ref_Close", "ATR", "Ref_Date", "Risk_Amt",
    "Notional", "Sizing_Note", "Place_Pass", "Manual_Only", "Place_Note",
    "Proxy_Ticker", "Execute_On", "Scan_Source",
]

GRADE_MEANING = {
    "A": "N >= 50, |t| >= 2.5, holds across eras, verified fresh this morning",
    "B": "real pattern, N 15-50 or single era, verified fresh this morning",
    "C": "context, not statistics: N < 15 fingerprint or analogue reasoning",
}
GRADE_COLOR = {"A": "#0a7a2f", "B": "#8a6d00", "C": "#7a4a8a"}


# ---------------------------------------------------------------------------
# load + validate
# ---------------------------------------------------------------------------
def load_ideas(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def leg_tickers(payload: dict) -> list[str]:
    """Every price series the payload needs: traded tickers plus the proxies
    a futures leg is levelled and graded on."""
    wanted: set[str] = set()
    for idea in payload.get("ideas", []):
        for leg in idea.get("legs", []) or []:
            proxy = str(leg.get("proxy_ticker", "")).strip().upper()
            wanted.add(proxy or str(leg.get("ticker", "")).strip().upper())
    return sorted(t for t in wanted if t)


def prepare(payload: dict, asof: pd.Timestamp, prices: pd.DataFrame,
            journal_records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Validate the payload and derive its order rows. Raises SystemExit with
    every error at once, because a half-valid pitch is not publishable and
    fixing one error at a time wastes a morning."""
    since = str((asof - REPEAT_BLOCK_TD * TRADING_DAY).normalize().date())
    recent = pitch_journal.recent_fingerprints(journal_records, since)
    errors = validate_payload(payload, recent)

    contexts: dict[str, dict] = {}
    if not errors:
        for ticker in leg_tickers(payload):
            try:
                contexts[ticker] = price_context(prices, ticker, asof)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"price context: {exc}")

    rows: list[dict] = []
    enriched: list[dict] = []
    if not errors:
        for rank, idea in enumerate(payload["ideas"], 1):
            idea_id = f"{asof.date()}-{rank}"
            try:
                idea_rows = build_orders(idea, contexts, asof, idea_id)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"idea {rank}: sizing failed ({exc})")
                continue
            rows.extend(idea_rows)
            enriched.append({**idea, "rank": rank, "idea_id": idea_id,
                             "fingerprint": fingerprint(idea),
                             "orders": idea_rows})
        errors.extend(check_risk_budget(rows))

    if errors:
        print(f"PITCH VALIDATION FAILED ({len(errors)} problem(s)) - "
              f"nothing published:")
        for line in errors:
            print(f"  - {line}")
        raise SystemExit(2)
    return enriched, rows


# ---------------------------------------------------------------------------
# Sheets
# ---------------------------------------------------------------------------
def credentials_path() -> Path | None:
    """First service-account file that exists, in preference order.

    The pitch runs LOCALLY at 7 AM, unlike the GHA jobs that read GCP_JSON, and
    this repo has no committed credentials.json. The working copy lives with
    the IBKR tooling in OneDrive, so that is the practical fallback; without it
    the Pitch tab never gets written and the approval loop is dead.
    """
    candidates = [
        os.environ.get("GCP_CREDENTIALS_FILE"),
        ROOT / "credentials.json",
        Path(os.environ.get("USERPROFILE", Path.home()))
        / "OneDrive" / "trading_ibkr" / "credentials.json",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    return None


def open_sheet(sheet_name: str = SHEET_NAME):
    import gspread
    if "GCP_JSON" in os.environ:
        client = gspread.service_account_from_dict(
            json.loads(os.environ["GCP_JSON"]))
    else:
        path = credentials_path()
        if path is None:
            raise RuntimeError(
                "no Sheets credentials: set GCP_JSON, or place credentials.json "
                "in the repo root, or set GCP_CREDENTIALS_FILE")
        client = gspread.service_account(filename=str(path))
    return client.open(sheet_name)


def capture_approvals(sheet, today: pd.Timestamp) -> list[dict]:
    """Approve cells from the tab as it stands, i.e. the PREVIOUS run's rows.

    The tab is cleared and rewritten every morning, so this is the only
    window in which McKinley's answer is readable. One approval record per
    idea (legs of one idea share a verdict; the first non-blank leg wins,
    and a disagreement is reported rather than guessed at)."""
    import gspread
    try:
        rows = sheet.worksheet(TAB_NAME).get_all_records()
    except gspread.WorksheetNotFound:
        return []
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: could not read prior Pitch tab ({exc}) - approvals "
              f"for that day are lost")
        return []

    by_idea: dict[str, dict] = {}
    for row in rows:
        idea_id = str(row.get("Idea_Id", "")).strip()
        if not idea_id:
            continue
        answer = str(row.get("Approve", "")).strip().upper()
        seen = by_idea.setdefault(idea_id, {"answers": set(),
                                            "date": str(row.get("Execute_On", ""))})
        if answer:
            seen["answers"].add(answer)

    records = []
    for idea_id, seen in sorted(by_idea.items()):
        answers = seen["answers"]
        if not answers:
            answer = ""
        elif len(answers) == 1:
            answer = next(iter(answers))
        else:
            answer = "CONFLICT:" + "/".join(sorted(answers))
            print(f"WARNING: {idea_id} legs disagree on Approve ({answers}) - "
                  f"journaled as a conflict")
        if str(seen["date"]) == str(today.date()):
            continue  # a same-day re-run; today's own rows, not an answer
        records.append({"kind": "approval", "idea_id": idea_id,
                        "date": seen["date"], "approve": answer,
                        "captured_at": dt.datetime.now().isoformat(
                            timespec="seconds")})
    return records


def write_tab(sheet, rows: list[dict]) -> None:
    import gspread
    try:
        worksheet = sheet.worksheet(TAB_NAME)
    except gspread.WorksheetNotFound:
        worksheet = sheet.add_worksheet(title=TAB_NAME, rows=60,
                                        cols=len(TAB_COLUMNS))
    worksheet.clear()
    # rows == [] is the stand-down case: header only, nothing approvable.
    # Building the frame from an empty list would KeyError on the column
    # selection, and leaving the tab as-is would leave yesterday's orders
    # live on a morning that pitched nothing.
    body = (pd.DataFrame(rows)[TAB_COLUMNS].astype(str).values.tolist()
            if rows else [])
    worksheet.update([TAB_COLUMNS] + body)
    print(f"Wrote {len(rows)} row(s) -> '{TAB_NAME}' tab")


# ---------------------------------------------------------------------------
# email
# ---------------------------------------------------------------------------
def _esc(text) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def _order_line(row: dict) -> str:
    side = "BUY" if row["Action"] == "BUY" else "SHORT"
    # A futures row is hand-entered, so the card has to carry the exact
    # contract (spec section 7). Naming the ticker alone is not an order.
    instrument = row["Ticker"]
    if str(row.get("Sec_Type", "")).upper() == "FUT" and row.get("Contract"):
        instrument = f"{row['Contract']}"
    bits = [f"{side} {row['Quantity']:,} {instrument}"]
    if row["Order_Type"] == "LMT":
        if row["Limit_Price"] != "":
            bits.append(f"LMT {row['Limit_Price']:.2f}")
        else:
            sign = "+" if float(row["Entry_Offset_ATR"]) >= 0 else "-"
            bits.append(f"LMT at open {sign}"
                        f"{abs(float(row['Entry_Offset_ATR'])):g} ATR "
                        f"(1 ATR = {row['ATR']:.2f})")
        bits.append(f"good through {row['Entry_Expire_Date']}")
    else:
        bits.append(f"MKT {row['TIF']}")
    if row["Stop_Price"] != "":
        bits.append(f"stop {row['Stop_Price']:.2f}")
    if row["Target_Price"] != "":
        bits.append(f"target {row['Target_Price']:.2f}")
    bits.append(f"time exit {row['Time_Exit_Date']} {row['Time_Exit_Order']}")
    return ", ".join(bits)


def _evidence_table(table) -> str:
    if not isinstance(table, list) or not table:
        return ""
    cell = ("padding:3px 8px;border-bottom:1px solid #e6e6e6;font-size:12px;"
            "font-family:Consolas,monospace")
    head = ("padding:3px 8px;border-bottom:2px solid #999;font-size:11px;"
            "text-align:left;color:#444")
    body = []
    for i, line in enumerate(table):
        style = head if i == 0 else cell
        cells = "".join(f"<td style='{style}'>{_esc(v)}</td>" for v in line)
        body.append(f"<tr>{cells}</tr>")
    return ("<table style='border-collapse:collapse;margin:8px 0'>"
            + "".join(body) + "</table>")


def render_card(idea: dict) -> str:
    grade = str(idea["grade"]).upper()
    color = GRADE_COLOR.get(grade, "#555")
    ev = idea.get("evidence", {}) or {}
    rows = idea["orders"]
    manual = [r for r in rows if r["Manual_Only"]]
    pass_names = sorted({r["Place_Pass"] for r in rows})

    place_note = ""
    if manual:
        place_note = (
            "<div style='background:#fdf6e3;border-left:4px solid #b58900;"
            "padding:6px 10px;margin:8px 0;font-size:12px'><b>Manual only.</b> "
            f"{_esc(manual[0]['Place_Note'])}. The Pitch tab carries the row "
            "for the record; the approval runner will not place it.</div>")
    elif "open" in pass_names:
        place_note = (
            "<div style='background:#eef4fb;border-left:4px solid #3b6ea5;"
            "padding:6px 10px;margin:8px 0;font-size:12px'>Priced at the live "
            "open, so the runner places this in its post-open pass, not the "
            "9:05 auction pass.</div>")

    label = " / ".join(
        f"{'long' if r['Action'] == 'BUY' else 'short'} {r['Ticker']}"
        for r in rows)
    risk = sum(r["Risk_Amt"] for r in rows)
    notional = sum(r["Notional"] for r in rows)

    field = ("margin:6px 0;font-size:13px;line-height:1.5")
    tag = "color:#666;font-weight:600;font-size:11px;letter-spacing:.06em"

    return f"""
<div style="border:1px solid #d8d8d8;border-radius:6px;padding:14px 16px;
            margin:0 0 18px 0;background:#fff">
  <div style="display:block;margin-bottom:6px">
    <span style="background:{color};color:#fff;border-radius:3px;padding:2px 7px;
                 font-size:11px;font-weight:700">GRADE {grade}</span>
    <span style="font-size:16px;font-weight:700;margin-left:8px">
      #{idea['rank']} {_esc(idea['title'])}</span>
    <span style="color:#777;font-size:12px;margin-left:8px">
      horizon {idea['horizon_td']} td &nbsp;|&nbsp; {_esc(label)}</span>
  </div>
  <div style="{field}"><span style="{tag}">ENTRY</span>
    &nbsp;{_esc(entry_label(idea))}</div>
  <div style="{field}"><span style="{tag}">EXIT</span>
    &nbsp;{_esc(exit_label(idea))}</div>
  <div style="{field}"><span style="{tag}">SIZE</span>
    &nbsp;{_esc(rows[0]['Sizing_Note'])} &nbsp;=&nbsp;
    ${risk:,.0f} at risk, ${notional:,.0f} gross</div>
  <div style="{field}"><span style="{tag}">ORDER</span><br>
    {'<br>'.join(f"&nbsp;&nbsp;{_esc(_order_line(r))}" for r in rows)}</div>
  {place_note}
  <div style="{field}"><span style="{tag}">THESIS</span>
    &nbsp;{_esc(idea['thesis'])}</div>
  <div style="{field}"><span style="{tag}">EVIDENCE</span>
    &nbsp;{_esc(ev.get('summary', ''))}
    <div style="color:#666;font-size:12px;margin-top:3px">
      N = {_esc(ev.get('n'))}{', t = ' + _esc(ev.get('t_stat')) if ev.get('t_stat') is not None else ''}
      &nbsp;|&nbsp; window {_esc(ev.get('window', 'n/a'))}
      &nbsp;|&nbsp; control: {_esc(ev.get('control', ''))}
      &nbsp;|&nbsp; eras: {_esc(ev.get('era_note', ''))}
    </div>
    {_evidence_table(ev.get('table'))}
    <div style="color:#999;font-size:11px">checked this morning:
      {_esc(ev.get('script', ''))}</div>
  </div>
  <div style="{field}"><span style="{tag}">SURVIVED</span>
    &nbsp;{_esc(idea.get('survived', ''))}</div>
  <div style="{field}"><span style="{tag}">WHAT KILLS IT</span>
    &nbsp;{_esc(idea['what_kills_it'])}</div>
  <div style="{field}"><span style="{tag}">OVERLAP</span>
    &nbsp;{_esc(idea['overlap'])}</div>
</div>"""


def render_scoreboard(scoreboard: dict) -> str:
    if not scoreboard or not scoreboard.get("rolling_60d"):
        return ("<p style='color:#888;font-size:12px'>Scoreboard: no graded "
                "history yet. Outcomes start accruing once the first pitched "
                "ideas reach their time exits.</p>")
    roll = scoreboard["rolling_60d"]
    by_grade = " &nbsp;|&nbsp; ".join(
        f"{g}: {v['n']} ideas, {v['hit_rate']:.0f}% hit, avg {v['avg_r']:+.2f}R"
        for g, v in sorted((roll.get("by_grade") or {}).items()))
    gap = roll.get("approved_vs_declined")
    gap_line = ""
    if gap:
        gap_line = (f" &nbsp;|&nbsp; approved {gap['approved_avg_r']:+.2f}R "
                    f"(n {gap['approved_n']}) vs declined "
                    f"{gap['declined_avg_r']:+.2f}R (n {gap['declined_n']})")
    by_model = ""
    models = {k: v for k, v in (roll.get("by_model") or {}).items()
              if v.get("avg_r") is not None}
    if len(models) > 1:
        by_model = ("<br><b>By model:</b> " + " &nbsp;|&nbsp; ".join(
            f"{m}: {v['graded']} graded, avg {v['avg_r']:+.2f}R, "
            f"{v['hit_rate']:.0f}% hit"
            for m, v in sorted(models.items())))
    return (f"<p style='color:#555;font-size:12px'><b>Scoreboard, rolling 60 "
            f"days:</b> {roll['n']} pitched, {roll['graded']} graded, avg "
            f"{roll['avg_r']:+.2f}R. {by_grade}{gap_line}{by_model}</p>")


def render_pipeline_line(pipeline: dict | None) -> str:
    """One line: did the overnight chain run, and are the caches current?

    Added after the 2026-08-06 Actions incident. A job that runs and fails is
    already loud; a job that never STARTS leaves no trace at all, which is how
    an entire evening of crons went missing unnoticed. Green days are printed
    too, because silence would otherwise be ambiguous between "all good" and
    "the check itself is broken".
    """
    if not pipeline:
        return ""
    bits = []
    if pipeline.get("available"):
        bits.append(f"{pipeline['green']}/{pipeline['checked']} overnight jobs ran")
        if pipeline.get("missing"):
            names = ", ".join(m["job"] for m in pipeline["missing"])
            bits.append(f"<b>MISSING: {_esc(names)}</b>")
    else:
        bits.append("overnight run check unavailable "
                    f"({_esc(pipeline.get('error', 'unknown'))})")
    bits.append(f"prices {_esc(pipeline.get('prices_bar') or 'n/a')}")
    bits.append(f"dial {_esc(pipeline.get('dial_asof') or 'n/a')}")
    age = pipeline.get("pc_age_bd")
    bits.append(f"P/C {_esc(pipeline.get('pc_date') or 'n/a')}"
                + (f" ({age} bd)" if age is not None else ""))
    stale = pipeline.get("stale") or []
    if pipeline.get("ok"):
        verdict = "all current"
    elif stale:
        verdict = "<b>STALE: " + _esc(", ".join(stale)) + "</b>"
    else:
        verdict = "check the warnings above"
    color = "#0a7a2f" if pipeline.get("ok") else "#b02a1e"
    return (f"<p style='color:{color};font-size:12px;margin:4px 0'>"
            f"Pipeline: {' &nbsp;|&nbsp; '.join(bits)} &mdash; {verdict}</p>")


def render_email(payload: dict, ideas: list[dict], asof: pd.Timestamp,
                 scoreboard: dict, state: dict | None) -> str:
    state = state or {}
    cards = "".join(render_card(idea) for idea in ideas)

    killed = payload.get("killed") or []
    killed_html = ""
    if killed:
        lines = "<br>".join(
            f"killed: {_esc(k.get('title', ''))} &mdash; {_esc(k.get('reason', ''))}"
            for k in killed)
        killed_html = (f"<p style='color:#777;font-size:12px'>"
                       f"<b>Finalists killed in falsification:</b><br>{lines}</p>")

    context = ""
    if state:
        risk = state.get("risk", {})
        frag = risk.get("fragility", {})
        pc = risk.get("pc_fear", {})
        signals = ", ".join(risk.get("signals_on") or []) or "none"
        pct = pc.get("pctile")
        pc_pct = f" ({pct:.0f}%ile)" if isinstance(pct, (int, float)) else ""
        context = (
            f"<p style='color:#555;font-size:12px'>"
            f"Dial (10d MA 63d): <b>{frag.get('ma10_63d', 'n/a')}</b> as of "
            f"{frag.get('as_of', 'n/a')} &nbsp;|&nbsp; P/C fear: "
            f"<b>{pc.get('state', 'n/a')}</b>{pc_pct}"
            f" &nbsp;|&nbsp; fragility signals on: {signals}"
            f" &nbsp;|&nbsp; freshest bar "
            f"{state.get('tape', {}).get('freshest_bar', 'n/a')}</p>")

    warnings = (state or {}).get("warnings") or []
    warn_html = ""
    if warnings:
        warn_html = (
            "<div style='background:#fdecea;border:1px solid #b02a1e;"
            "border-radius:4px;padding:8px 12px;margin:10px 0;font-size:12px'>"
            "<b style='color:#b02a1e'>STATE WARNINGS</b><br>"
            + "<br>".join(_esc(w) for w in warnings) + "</div>")

    grades = "".join(
        f"<li><b>{g}</b> &mdash; {GRADE_MEANING[g]}</li>" for g in "ABC")

    return f"""
<div style="font-family:Segoe UI,Arial,sans-serif;max-width:920px;color:#1a1a1a">
  <h2 style="margin-bottom:2px">Daily Pitch &mdash; {asof.strftime('%A %Y-%m-%d')}</h2>
  <p style="color:#555;margin-top:2px;font-size:13px">
    Three ideas, invented from repo context and interrogated against the data
    this morning. Nothing here is a book signal; the scanner trades those
    separately. Approve by typing Y in the Pitch tab.</p>
  {context}
  {render_pipeline_line(state.get('pipeline'))}
  {warn_html}
  {cards}
  {killed_html}
  {render_scoreboard(scoreboard)}
  <p style="color:#888;font-size:11px;margin-top:14px">
    Grades: <ul style="color:#888;font-size:11px;margin:4px 0">{grades}</ul>
    Sizes assume ${state.get('book', {}).get('account_value', 750000):,.0f} NAV.
    ATR is Wilder-14 on the traded instrument (the systematic book uses a
    simple 14d ATR; a pitch level is never a scanner level). Ideas are graded
    against their stated entry and exit whether or not they were approved, and
    the scoreboard above is the result.
  </p>
</div>"""


def render_stand_down(payload: dict, asof: pd.Timestamp, scoreboard: dict,
                      state: dict | None) -> str:
    """The email for a morning that shipped nothing.

    It leads with the near-misses and the number each one turned on, because
    that is the part McKinley can argue with. A bare 'no trades today' is
    indistinguishable from a broken task, which is exactly the hole this
    closes.
    """
    state = state or {}
    block = payload.get("stand_down") or {}

    closest = "".join(
        f"<div style='border-left:3px solid #8a6d00;padding:6px 12px;"
        f"margin:10px 0;background:#fffdf5'>"
        f"<div style='font-weight:600'>{_esc(near.get('title', ''))}</div>"
        f"<div style='font-size:13px;color:#444;margin-top:3px'>"
        f"<b>Decisive number:</b> {_esc(near.get('decisive', ''))}</div>"
        f"<div style='font-size:13px;color:#444;margin-top:3px'>"
        f"<b>Why it died:</b> {_esc(near.get('why_died', ''))}</div>"
        + (f"<div style='font-size:12px;color:#777;margin-top:3px'>"
           f"{_esc(near.get('script', ''))}</div>"
           if near.get("script") else "")
        + "</div>"
        for near in (block.get("closest") or []))

    killed = payload.get("killed") or []
    killed_html = ""
    if killed:
        lines = "<br>".join(
            f"<b>{_esc(k.get('title', ''))}</b> &mdash; {_esc(k.get('reason', ''))}"
            for k in killed)
        killed_html = (
            f"<p style='color:#555;font-size:12px;margin-top:16px'>"
            f"<b>Everything killed today ({len(killed)}):</b><br>{lines}</p>")

    axes = ", ".join(_esc(str(a)) for a in (block.get("axes") or []))
    context = ""
    if state:
        risk = state.get("risk", {})
        frag = risk.get("fragility", {})
        pc = risk.get("pc_fear", {})
        signals = ", ".join(risk.get("signals_on") or []) or "none"
        pct = pc.get("pctile")
        pc_pct = f" ({pct:.0f}%ile)" if isinstance(pct, (int, float)) else ""
        context = (
            f"<p style='color:#555;font-size:12px'>"
            f"Dial (10d MA 63d): <b>{frag.get('ma10_63d', 'n/a')}</b> as of "
            f"{frag.get('as_of', 'n/a')} &nbsp;|&nbsp; P/C fear: "
            f"<b>{pc.get('state', 'n/a')}</b>{pc_pct}"
            f" &nbsp;|&nbsp; fragility signals on: {signals}"
            f" &nbsp;|&nbsp; freshest bar "
            f"{state.get('tape', {}).get('freshest_bar', 'n/a')}</p>")

    warnings = (state or {}).get("warnings") or []
    warn_html = ""
    if warnings:
        warn_html = (
            "<div style='background:#fdecea;border:1px solid #b02a1e;"
            "border-radius:4px;padding:8px 12px;margin:10px 0;font-size:12px'>"
            "<b style='color:#b02a1e'>STATE WARNINGS</b><br>"
            + "<br>".join(_esc(w) for w in warnings) + "</div>")

    return f"""
<div style="font-family:Segoe UI,Arial,sans-serif;max-width:920px;color:#1a1a1a">
  <h2 style="margin-bottom:2px">Daily Pitch &mdash; {asof.strftime('%A %Y-%m-%d')}
    &mdash; <span style="color:#b02a1e">NO TRADES</span></h2>
  <p style="color:#555;margin-top:2px;font-size:13px">
    {block.get('candidates_considered', 0)} candidates across
    {len(block.get('axes') or [])} novelty axes, all killed in falsification.
    The Pitch tab is empty and there is nothing to approve. This is a verdict,
    not a failed run.</p>
  {context}
  {render_pipeline_line(state.get('pipeline'))}
  {warn_html}
  <div style="background:#f6f6f4;border-radius:4px;padding:10px 14px;
              margin:12px 0;font-size:13px;line-height:1.5">
    {_esc(block.get('reason', ''))}
  </div>
  <h3 style="margin-bottom:2px;font-size:15px">Closest to shipping</h3>
  {closest}
  <p style="color:#777;font-size:12px">Axes swept: {axes}<br>
     Checks written this morning: {_esc(block.get('checks_dir', ''))}</p>
  {killed_html}
  {render_scoreboard(scoreboard)}
  <p style="color:#888;font-size:11px;margin-top:14px">
    A stand-down has to clear a higher bar than shipping does: at least
    {STAND_DOWN_MIN_CANDIDATES} candidates over {STAND_DOWN_MIN_AXES} axes,
    {STAND_DOWN_MIN_KILLED} named kills, and every near-miss written down with
    the number it turned on. If that ever looks like an easy out, the floors
    are in pitch_grammar.py.
  </p>
</div>"""


def dotenv_values(path: Path | None = None) -> dict[str, str]:
    """KEY=value pairs from the repo .env, or {} when it is absent.

    The GHA reports get EMAIL_USER / EMAIL_PASS from repo secrets, but the
    pitch runs locally off Task Scheduler, whose environment carries neither.
    Without this the 7 AM run would print "skipping email send" and deliver
    nothing while every other step reported success. Environment wins; the
    file is only a fallback.
    """
    path = path or (ROOT / ".env")
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def smtp_credentials() -> tuple[str | None, str | None]:
    env = os.environ
    sender, password = env.get("EMAIL_USER"), env.get("EMAIL_PASS")
    if sender and password:
        return sender, password
    fallback = dotenv_values()
    return (sender or fallback.get("EMAIL_USER"),
            password or fallback.get("EMAIL_PASS"))


def send_email(subject: str, html: str) -> bool:
    sender, password = smtp_credentials()
    recipients = [a.strip() for a in os.environ.get(
        "PITCH_RECIPIENTS", DEFAULT_RECIPIENTS).split(",") if a.strip()]
    if not sender or not password:
        print("EMAIL_USER/EMAIL_PASS not set in the environment or .env - "
              "skipping email send. THE PITCH WAS NOT DELIVERED.")
        return False
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(html, "html"))
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender, password)
            server.sendmail(sender, recipients, msg.as_string())
    except smtplib.SMTPException as exc:
        # A rejected app password looks identical to a healthy run from the
        # outside: the tab still gets written and the journal still records a
        # delivery. Say it loudly instead.
        print(f"EMAIL SEND FAILED ({exc}) - THE PITCH WAS NOT DELIVERED. "
              f"The Pitch tab and journal below still reflect the run.")
        return False
    print(f"Email sent to {', '.join(recipients)}")
    return True


# ---------------------------------------------------------------------------
def run_identity(model: str | None = None,
                 effort: str | None = None) -> tuple[str, str]:
    """Which model and effort produced this pitch.

    run_daily_pitch.bat exports PITCH_MODEL / PITCH_EFFORT before invoking the
    agent, and this process is a descendant of it, so the stamp is the tier
    that was actually requested rather than something the agent self-reports.
    A manual run with neither set records 'unknown' rather than guessing a
    default, because a wrong stamp is worse than an absent one: the scoreboard
    splits on this field.
    """
    return (model or os.environ.get("PITCH_MODEL") or "unknown",
            effort or os.environ.get("PITCH_EFFORT") or "unknown")


def journal_records(payload: dict, ideas: list[dict], asof: pd.Timestamp,
                    model: str | None = None,
                    effort: str | None = None) -> list[dict]:
    date = str(asof.date())
    model, effort = run_identity(model, effort)
    records = [{
        "model": model, "effort": effort,
        "kind": "idea", "date": date, "idea_id": idea["idea_id"],
        "rank": idea["rank"], "fingerprint": idea["fingerprint"],
        "title": idea["title"], "grade": str(idea["grade"]).upper(),
        "horizon_td": idea["horizon_td"],
        "novelty_axis": idea.get("novelty_axis"),
        "spec": {k: idea.get(k) for k in ("legs", "entry", "exit", "sizing")},
        "orders": idea["orders"],
        "thesis": idea["thesis"], "evidence": idea.get("evidence"),
        "survived": idea.get("survived"),
        "what_kills_it": idea["what_kills_it"], "overlap": idea["overlap"],
        "changed_since": idea.get("changed_since", ""),
        "place_pass": idea["orders"][0]["Place_Pass"],
    } for idea in ideas]
    records += killed_records(payload, asof, model, effort)
    return records


def killed_records(payload: dict, asof: pd.Timestamp, model: str,
                   effort: str) -> list[dict]:
    return [{"kind": "killed", "date": str(asof.date()), "model": model,
             "effort": effort,
             "title": k.get("title", ""), "reason": k.get("reason", ""),
             "novelty_axis": k.get("novelty_axis")}
            for k in (payload.get("killed") or [])]


def stand_down_records(payload: dict, asof: pd.Timestamp,
                       model: str | None = None,
                       effort: str | None = None) -> list[dict]:
    """One stand_down record plus a killed record per named kill.

    The kills are journaled here for the same reason they are on a shipping
    day: the negative registry holds the reusable lesson, but the journal is
    what the scoreboard and the model split read. An all-kill morning used to
    write neither.
    """
    model, effort = run_identity(model, effort)
    block = payload.get("stand_down") or {}
    return [{
        "kind": "stand_down", "date": str(asof.date()),
        "model": model, "effort": effort,
        "reason": block.get("reason", ""),
        "candidates_considered": block.get("candidates_considered"),
        "axes": block.get("axes") or [],
        "checks_dir": block.get("checks_dir", ""),
        "closest": block.get("closest") or [],
        "killed_n": len(payload.get("killed") or []),
    }] + killed_records(payload, asof, model, effort)


def load_context(asof: pd.Timestamp) -> tuple[dict, dict | None]:
    """The scoreboard and this morning's state, shared by both publish paths."""
    scoreboard = {}
    if SCOREBOARD_PATH.exists():
        scoreboard = json.loads(SCOREBOARD_PATH.read_text(encoding="utf-8"))
    state = None
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if state.get("asof") != str(asof.date()):
            print(f"WARNING: pitch_state.json is dated {state.get('asof')}, "
                  f"not {asof.date()} - the context header may be stale")
    return scoreboard, state


def publish_stand_down(payload: dict, asof: pd.Timestamp, journal_path: Path,
                       args) -> int:
    """Ship the verdict that nothing shipped.

    Same shape as the normal path with one difference that matters: the Pitch
    tab is cleared to a header row. Leaving yesterday's rows up on a
    no-trade morning would leave live, approvable orders on the sheet for
    ideas nobody pitched today.
    """
    errors = validate_payload(payload)
    if errors:
        print(f"STAND-DOWN VALIDATION FAILED ({len(errors)} problem(s)) - "
              f"nothing published:")
        for line in errors:
            print(f"  - {line}")
        return 2

    block = payload["stand_down"]
    model, effort = run_identity(args.model, args.effort)
    print(f"Daily Pitch {asof.date()} - STAND-DOWN, nothing shipped "
          f"[model {model}, effort {effort}]")
    print(f"  {block['candidates_considered']} candidates over "
          f"{len(block['axes'])} axes, {len(payload.get('killed') or [])} "
          f"named kills")
    for near in block["closest"]:
        print(f"  closest: {near['title']} - {near['decisive']}")
    if args.validate_only:
        print("Validation only - nothing published.")
        return 0

    scoreboard, state = load_context(asof)
    html = render_stand_down(payload, asof, scoreboard, state)
    if args.html_out:
        Path(args.html_out).write_text(html, encoding="utf-8")
        print(f"HTML written to {args.html_out}")
    if args.dry_run:
        print("[dry-run] no email, no Sheets write, no journal append")
        return 0

    approvals: list[dict] = []
    sheet = None
    try:
        sheet = open_sheet()
        approvals = capture_approvals(sheet, asof)
        if approvals:
            answered = sum(1 for a in approvals if a["approve"])
            print(f"Captured {len(approvals)} prior approval row(s), "
                  f"{answered} answered")
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: Sheets unavailable ({exc}) - approvals not captured "
              f"and the Pitch tab will not be cleared")

    if not args.no_send:
        send_email(f"Daily Pitch - {asof.date()} - NO TRADES "
                   f"({len(payload.get('killed') or [])} killed)", html)

    if sheet is not None:
        try:
            write_tab(sheet, [])
            print("Pitch tab cleared - nothing to approve today")
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: Pitch tab clear failed ({exc}) - yesterday's rows "
                  f"may still be approvable")
            return 1

    written = pitch_journal.append(
        approvals + stand_down_records(payload, asof, args.model, args.effort),
        journal_path)
    print(f"Journaled {written} record(s) -> {journal_path.name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ideas", default=str(DEFAULT_IDEAS))
    ap.add_argument("--asof", default=None, help="override today (YYYY-MM-DD)")
    ap.add_argument("--validate-only", action="store_true",
                    help="validate and size, then stop")
    ap.add_argument("--dry-run", action="store_true",
                    help="render but never email, write the tab, or journal")
    ap.add_argument("--no-send", action="store_true", help="skip the email only")
    ap.add_argument("--html-out", default=None)
    ap.add_argument("--model", default=None,
                    help="model stamp; defaults to $PITCH_MODEL")
    ap.add_argument("--effort", default=None,
                    help="effort stamp; defaults to $PITCH_EFFORT")
    ap.add_argument("--journal", default=str(pitch_journal.JOURNAL_PATH),
                    help="journal path; a non-default path never touches R2")
    args = ap.parse_args()

    payload = load_ideas(Path(args.ideas))
    asof = pd.Timestamp(args.asof or payload.get("asof")
                        or dt.date.today()).normalize()
    if payload.get("asof") and str(asof.date()) != str(payload["asof"]):
        print(f"NOTE: publishing as {asof.date()} while the ideas file says "
              f"{payload['asof']}")

    journal_path = Path(args.journal)
    records = pitch_journal.load(journal_path)

    if is_stand_down(payload):
        return publish_stand_down(payload, asof, journal_path, args)

    ideas, rows = prepare(payload, asof, load_prices(), records)

    _model, _effort = run_identity(args.model, args.effort)
    print(f"Daily Pitch {asof.date()} - {len(ideas)} ideas, "
          f"{len(rows)} order rows [model {_model}, effort {_effort}]")
    for idea in ideas:
        placement = idea["orders"][0]["Place_Pass"]
        print(f"  #{idea['rank']} [{idea['grade']}] {idea['title']} "
              f"({idea['horizon_td']} td, {placement})")
    if args.validate_only:
        print("Validation only - nothing published.")
        return 0

    scoreboard, state = load_context(asof)
    html = render_email(payload, ideas, asof, scoreboard, state)
    if args.html_out:
        Path(args.html_out).write_text(html, encoding="utf-8")
        print(f"HTML written to {args.html_out}")

    if args.dry_run:
        print("[dry-run] no email, no Sheets write, no journal append")
        return 0

    approvals: list[dict] = []
    sheet = None
    try:
        sheet = open_sheet()
        approvals = capture_approvals(sheet, asof)
        if approvals:
            answered = sum(1 for a in approvals if a["approve"])
            print(f"Captured {len(approvals)} prior approval row(s), "
                  f"{answered} answered")
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: Sheets unavailable ({exc}) - approvals not captured "
              f"and the Pitch tab will not be rewritten")

    if not args.no_send:
        send_email(f"Daily Pitch - {asof.date()} - {IDEA_COUNT} ideas", html)

    if sheet is not None:
        try:
            write_tab(sheet, rows)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: Pitch tab write failed ({exc}) - the email went out "
                  f"but there is nothing to approve against")
            return 1

    written = pitch_journal.append(
        approvals + journal_records(payload, ideas, asof, args.model,
                                    args.effort), journal_path)
    print(f"Journaled {written} record(s) -> {journal_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
