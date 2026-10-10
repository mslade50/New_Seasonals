"""PM Weekly publisher: validate, lock forecasts, attach the readout, deliver.

The judgement (survey, recap, forecasts) happens in the `/pm-agent` skill,
which writes PM_AGENT_HOME/brief.json. This module is the deterministic tail
and refuses anything pm_agent_grammar rejects.

    python weekly_pm_agent.py [--brief PATH] [--state PATH] [--validate-only]
                              [--no-send] [--no-r2] [--html-out PATH]

Flow: load state + brief + journal; refuse a second brief for the same ISO
week; validate; journal the brief and one `forecast` record per claim (anchor
close, resolves_on, climatology and a sha256, all from the state, none from the
agent); push the journal to R2. Only THEN read the Risk Agent's published
today.json, so the PM's forecasts are locked before it sees the Risk Agent's.
Render, email once behind a receipt, write today.json, upload pm_agent/today.json.

A brief published after the target week's first session opens (09:30 ET) is
delivered, but its forecasts are journaled `scored: false`.

Env: EMAIL_USER / EMAIL_PASS, PM_AGENT_RECIPIENTS (defaults to the pitch
recipient), PM_AGENT_MODEL / PM_AGENT_EFFORT (stamped on every record),
PM_AGENT_HOME. Agent-product module: the book and the Risk Agent must not
import it. Doc: docs/claude_ref/pm_agent.md
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html as _html
import json
import os
import smtplib
import sys
import uuid
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pm_agent_data as pad  # noqa: E402
import pm_agent_grammar as grammar  # noqa: E402
import pm_agent_journal as J  # noqa: E402
import pm_agent_universe as U  # noqa: E402

ET = ZoneInfo("America/New_York")
RECEIPT_SCHEMA = "pm-agent-delivery.v1"
RA_TODAY_KEY = "risk_agent/today.json"
DEFAULT_RECIPIENTS = "mckinleyslade@gmail.com"


# ===========================================================================
# helpers
# ===========================================================================
def _read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _esc(text) -> str:
    return _html.escape(str(text if text is not None else ""), quote=True)


def _num(v, default=None):
    try:
        f = float(v)
        return f if f == f and abs(f) != float("inf") else default
    except (TypeError, ValueError):
        return default


def _fmt(v, digits=2, signed=True, suffix="") -> str:
    f = _num(v)
    if f is None:
        return "-"
    return (f"{f:+.{digits}f}" if signed else f"{f:.{digits}f}") + suffix


def deadline(target_week: dict) -> dt.datetime:
    """09:30 ET on the target week's first session."""
    d = dt.date.fromisoformat(target_week["first_session"])
    return dt.datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET)


def forecast_id(week: str, claim: str) -> str:
    return f"PMF-{week}-{claim}"


def forecast_records(forecasts: list[dict], state: dict, *, week: str, scored: bool,
                     model: str, effort: str, now: str) -> list[dict]:
    out = []
    for f in forecasts:
        anchor = (state.get("anchors") or {}).get(f["symbol"]) or {}
        core = {"claim_type": f["claim_type"], "symbol": f["symbol"], "unit": f["unit"],
                "anchor_date": anchor.get("date"), "anchor_value": anchor.get("close"),
                "resolves_on": f["resolves_on"], "horizon_td": f["horizon_td"],
                "p_up": f["p_up"], "q10": f["q10"], "q90": f["q90"]}
        sha = hashlib.sha256(json.dumps(core, sort_keys=True).encode("utf-8")).hexdigest()
        out.append({"kind": "forecast", "forecast_id": forecast_id(week, f["claim_type"]),
                    "asof": state["asof"], "week": week, **core,
                    "climatology": f["climatology"], "evidence_n": f["evidence_n"],
                    "evidence_script": f["evidence_script"], "scored": scored,
                    "model": model, "effort": effort, "sha256": sha, "locked_at": now})
    return out


# ===========================================================================
# Risk Agent readout (read AFTER our forecasts are journaled)
# ===========================================================================
def fetch_ra_today(use_r2: bool) -> dict | None:
    """Download, read and delete: no copy is left where a later PM session
    could read the Risk Agent's forecasts before writing its own."""
    if not use_r2:
        return None
    pad._check(RA_TODAY_KEY)
    import tempfile
    fd, tmp = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        import cache_io
        if not cache_io.is_configured() or not cache_io.download_to_local(RA_TODAY_KEY, tmp):
            return None
        return _read_json(Path(tmp))
    except Exception as exc:  # noqa: BLE001 - readout is optional
        print(f"Risk Agent readout unavailable: {exc}")
        return None
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def ra_readout(ra: dict | None, pm_forecasts: list[dict]) -> dict:
    if not isinstance(ra, dict) or not ra.get("asof"):
        return {"available": False}
    fcs = {int(f.get("horizon_td")): f for f in ra.get("forecasts") or []
           if isinstance(f, dict) and isinstance(f.get("horizon_td"), (int, float))}
    pm_spy = next((f for f in pm_forecasts if f["claim_type"] == "spy_week_return"), None)
    hl = (ra.get("scoreboard") or {}).get("headline") or {}
    posture = ra.get("posture") or {}
    ra5 = fcs.get(5) or {}
    return {"available": True, "asof": ra.get("asof"), "mode": ra.get("mode"),
            "posture": posture.get("summary"), "net_beta": posture.get("net_beta"),
            "cash_pct": posture.get("cash_pct"),
            "forecast_5": {k: ra5.get(k) for k in ("p_up", "q10_pct", "q90_pct")} if ra5 else None,
            "pm_spy": {k: pm_spy[k] for k in ("p_up", "q10", "q90")} if pm_spy else None,
            "positions": len(((ra.get("book") or {}).get("positions")) or []),
            "scoreboard": {k: hl.get(k) for k in ("nav", "total_return_pct", "vs_spy_pct",
                                                   "brier_5", "brier_21", "n_marks")}}


# ===========================================================================
# receipts (own namespace)
# ===========================================================================
class ReceiptError(RuntimeError):
    pass


def r2_upload(local: Path, key: str) -> bool:
    if not U.r2_key_allowed(key) or not key.startswith(U.R2_PREFIX):
        raise ReceiptError(f"refusing to write outside {U.R2_PREFIX}: {key}")
    try:
        import cache_io
    except Exception as exc:  # noqa: BLE001
        print(f"R2 unavailable: {exc}")
        return False
    if not cache_io.is_configured():
        print("R2 is not configured")
        return False
    return bool(cache_io.upload_from_local(str(local), key))


def read_receipt(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReceiptError(f"receipt unreadable at {path}: {exc}") from exc
    if rec.get("schema") != RECEIPT_SCHEMA:
        raise ReceiptError(f"receipt at {path} has schema {rec.get('schema')!r}")
    return rec


def _write_receipt(rec: dict, path: Path, use_r2: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    if use_r2 and not r2_upload(path, f"{U.R2_PREFIX}delivery_receipts/{rec['week']}.json"):
        raise ReceiptError("could not mirror the delivery receipt to R2")


def reserve_receipt(week: str, decision_id: str, subject: str, html: str,
                    recipients: list[str], path: Path, use_r2: bool) -> tuple[dict, bool]:
    existing = read_receipt(path)
    if existing is not None:
        if existing.get("status") == "sent":
            return existing, False
        raise ReceiptError(f"delivery receipt for {week} is {existing.get('status')}; "
                           "resolve the SMTP outcome before any rerun")
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    digest = hashlib.sha256(json.dumps({"s": subject, "h": html, "r": sorted(recipients)},
                                       sort_keys=True).encode("utf-8")).hexdigest()
    rec = {"schema": RECEIPT_SCHEMA, "status": "sending", "week": week,
           "decision_id": decision_id, "delivery_id": str(uuid.uuid4()),
           "message_digest": digest, "subject": subject, "recipients": sorted(recipients),
           "created_at": now, "updated_at": now}
    _write_receipt(rec, path, use_r2)
    return rec, True


def complete_receipt(rec: dict, path: Path, use_r2: bool, sent: bool) -> dict:
    out = dict(rec)
    out["status"] = "sent" if sent else "ambiguous"
    out["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    if sent:
        out["sent_at"] = out["updated_at"]
    else:
        out["ambiguity_reason"] = "SMTP did not confirm delivery"
    try:
        _write_receipt(out, path, use_r2)
    except ReceiptError:
        out["status"] = "ambiguous"
        _write_receipt(out, path, False)
        raise
    return out


def email_recipients() -> list[str]:
    raw = os.environ.get("PM_AGENT_RECIPIENTS") or os.environ.get("PITCH_RECIPIENTS") \
        or DEFAULT_RECIPIENTS
    return [a.strip() for a in raw.split(",") if a.strip()]


def send_email(subject: str, html: str, recipients: list[str]) -> bool:
    from daily_pitch import smtp_credentials
    sender, password = smtp_credentials()
    if not sender or not password:
        print("EMAIL_USER/EMAIL_PASS not set - skipping send. THE PM WEEKLY WAS NOT DELIVERED.")
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
            refused = server.sendmail(sender, recipients, msg.as_string())
            if refused:
                print(f"EMAIL SEND PARTIAL FAILURE ({sorted(refused)}) - delivery is ambiguous.")
                return False
    except (smtplib.SMTPException, OSError) as exc:
        print(f"EMAIL SEND FAILED ({exc}) - THE PM WEEKLY WAS NOT DELIVERED.")
        return False
    print(f"Email sent to {', '.join(recipients)}")
    return True


# ===========================================================================
# email
# ===========================================================================
_TD = "padding:4px 8px;border-bottom:1px solid #e5e7eb;font-size:13px;"
_TH = "padding:4px 8px;text-align:left;font-size:11px;color:#6b7280;text-transform:uppercase;"


def _table(headers, rows) -> str:
    if not rows:
        return '<p style="color:#6b7280;font-size:13px;margin:4px 0;">None.</p>'
    head = "".join(f'<th style="{_TH}">{_esc(h)}</th>' for h in headers)
    body = "".join("<tr>" + "".join(f'<td style="{_TD}">{c}</td>' for c in r) + "</tr>" for r in rows)
    return f'<table style="border-collapse:collapse;width:100%;"><tr>{head}</tr>{body}</table>'


def _h(title: str) -> str:
    return f'<h3 style="margin:22px 0 6px;font-size:15px;color:#111827;">{_esc(title)}</h3>'


def _p(text: str, style: str = "font-size:14px;margin:4px 0;") -> str:
    return f'<p style="{style}">{_esc(text)}</p>'


TAPE_ROWS = ("SPY", "QQQ", "IWM", "TLT", "HYG", "GLD", "USO", "UUP", "EEM", "BTC-USD")


def render_tape(state: dict) -> str:
    sy = (state.get("recap") or {}).get("symbols") or {}
    rows = []
    for s in TAPE_ROWS:
        r = sy.get(s)
        if r:
            rows.append([_esc(s), _esc(_fmt(r.get("ret_week_pct"), suffix="%")),
                         _esc(_fmt(r.get("ret_4w_pct"), suffix="%")),
                         _esc(_fmt(r.get("dist_200d_pct"), suffix="%"))])
    for s in ("^VIX", "^TNX"):
        r = sy.get(s)
        if r:
            u = " bp" if r.get("unit") == "bps" else " pt"
            rows.append([_esc(s), _esc(_fmt(r.get("chg_week"), suffix=u)),
                         _esc(_fmt(r.get("chg_4w"), suffix=u)), _esc(f"level {r.get('last')}")])
    lead = (state.get("recap") or {}).get("leaders_week") or []
    lag = (state.get("recap") or {}).get("laggards_week") or []
    tail = (f'<div style="font-size:12px;color:#6b7280;margin-top:4px;">Leaders: {_esc(", ".join(lead))} | '
            f'Laggards: {_esc(", ".join(lag))}</div>') if lead else ""
    return _table(["Symbol", "Week", "4 weeks", "vs 200d"], rows) + tail


def render_scoreboard(sb: dict | None) -> str:
    claims = (sb or {}).get("claims") or {}
    rows = []
    for c, st in claims.items():
        if not st.get("n"):
            continue
        rows.append([_esc(c), _esc(st["n"]), _esc(_fmt(st.get("brier"), 3, False)),
                     _esc(_fmt(st.get("brier_clim"), 3, False)), _esc(_fmt(st.get("brier_skill"), 2)),
                     _esc(_fmt(100 * st["inside_q10_q90"], 0, False, "%"))])
    recent = [r for r in (sb or {}).get("recent") or [] if r.get("status") != "open"][-4:]
    rec_rows = [[_esc(r.get("asof")), _esc(r.get("claim_type")), _esc(r.get("p_up")),
                 _esc(f"[{r.get('q10')}, {r.get('q90')}]"), _esc(r.get("status")),
                 _esc(_fmt(r.get("value")))] for r in recent]
    if not rows and not rec_rows:
        return _p("No resolved forecasts yet. Skill is reported once outcomes are in.",
                  "font-size:13px;color:#6b7280;")
    out = _table(["Claim", "N", "Brier", "Brier clim", "Skill", "Inside q10-q90 (80% target)"], rows)
    if rec_rows:
        out += _table(["Asof", "Claim", "P(up)", "q10-q90", "Status", "Outcome"], rec_rows)
    return out


def render_forecasts(validated: list[dict], payload: dict, state: dict, scored: bool) -> str:
    by = {f.get("claim_type"): f for f in payload.get("forecasts") or [] if isinstance(f, dict)}
    implied = ((state.get("climatology") or {}).get("spy_week_return") or {}).get("vix_implied") or {}
    rows, notes = [], []
    for f in validated:
        c = f["climatology"]
        u = "%" if f["unit"] == "pct" else " pt"
        rows.append([_esc(f["claim_type"]), _esc(f"{f['p_up']:.2f}"),
                     _esc(f"{f['q10']:+.2f}{u} / {f['q90']:+.2f}{u}"),
                     _esc(f"{_fmt(c.get('p_up'), 2, False)} | {_fmt(c.get('q10'))} / {_fmt(c.get('q90'))}"),
                     _esc(f["resolves_on"])])
        src = by.get(f["claim_type"]) or {}
        notes.append(f'<div style="font-size:13px;margin:6px 0;"><b>{_esc(f["claim_type"])}.</b> '
                     f'{_esc(src.get("why"))} <i>Change my mind:</i> {_esc(src.get("change_my_mind"))} '
                     f'<span style="color:#6b7280;">Evidence (n={_esc(f["evidence_n"])}): '
                     f'{_esc((src.get("evidence") or {}).get("summary"))}</span></div>')
    out = _table(["Claim", "P(up)", "q10 / q90", "Climatology p | q10 / q90", "Resolves"], rows)
    if implied:
        out += _p(f"VIX-implied SPY band over the horizon: {implied.get('q10_pct')}% / "
                  f"{implied.get('q90_pct')}% (a price, not a forecast).",
                  "font-size:12px;color:#6b7280;margin:4px 0;")
    if not scored:
        out += _p("Published after the target week opened: these forecasts are journaled but not scored.",
                  "font-size:12px;color:#92400e;")
    return out + "".join(notes)


def render_ra(ro: dict) -> str:
    if not ro.get("available"):
        return _p("No Risk Agent output available.", "font-size:13px;color:#6b7280;")
    f5, pm = ro.get("forecast_5") or {}, ro.get("pm_spy") or {}
    sb = ro.get("scoreboard") or {}
    lines = [f"As of {ro.get('asof')} ({ro.get('mode')}): {ro.get('posture') or '-'}",
             f"Net beta {ro.get('net_beta')} | cash {ro.get('cash_pct')}% | {ro.get('positions')} open position(s)",
             f"SPY 5 td: Risk Agent p(up) {f5.get('p_up')} [{f5.get('q10_pct')}, {f5.get('q90_pct')}]% "
             f"vs PM week p(up) {pm.get('p_up')} [{pm.get('q10')}, {pm.get('q90')}]%",
             f"Sleeve NAV {sb.get('nav')} | return {_fmt(sb.get('total_return_pct'), suffix='%')} | "
             f"vs SPY {_fmt(sb.get('vs_spy_pct'), suffix='%')} | Brier 5d {sb.get('brier_5')} | marks {sb.get('n_marks')}"]
    return ("".join(_p(t, "font-size:13px;margin:3px 0;") for t in lines)
            + _p("Read after the PM forecasts above were locked. The Risk Agent never sees this brief.",
                 "font-size:12px;color:#6b7280;"))


def render_email(payload: dict, state: dict, validated: list[dict], ro: dict, *,
                 week: str, scored: bool, warnings: list[str], model: str, effort: str) -> tuple[str, str]:
    tw = state.get("target_week") or {}
    if payload.get("mode") == "stand_down":
        subject = f"PM Weekly {week}: DATA HOLD"
    else:
        head = str(payload.get("headline") or "").strip()
        if len(head) > 90:
            head = head[:90].rsplit(" ", 1)[0].rstrip(",;:") + " ..."
        subject = f"PM Weekly {week}: {head}"
    out = ['<div style="font-family:Arial,Helvetica,sans-serif;max-width:760px;color:#111827;">',
           f'<h2 style="margin:0 0 4px;">PM Weekly, week of {_esc(tw.get("week_of"))}</h2>',
           f'<div style="font-size:12px;color:#6b7280;">Data through {_esc(state.get("asof"))}. '
           f'Market-only and read-only: nothing here changes a rule or places an order. '
           f'Model {_esc(model)} / {_esc(effort)}.</div>']
    if payload.get("mode") == "stand_down":
        out += [_h("Data hold"), _p(payload.get("reason"))]
    else:
        out.append(_p(payload.get("headline"), "font-size:16px;font-weight:700;margin:14px 0 4px;"))
    if warnings:
        out.append(_h("Data warnings"))
        out.append("<ul style='font-size:13px;color:#92400e;'>"
                   + "".join(f"<li>{_esc(w)}</li>" for w in warnings[:12]) + "</ul>")
    if payload.get("mode") != "stand_down":
        out.append(_h("What happened"))
        out.append(render_tape(state))
        for r in payload.get("recap") or []:
            out.append(f'<div style="font-size:14px;margin:6px 0;"><b>{_esc(r.get("topic"))}:</b> '
                       f'{_esc(r.get("text"))}</div>')
    out.append(_h("Scoreboard (past PM forecasts)"))
    out.append(render_scoreboard(state.get("scoreboard")))
    if payload.get("mode") != "stand_down":
        nw = payload.get("next_week") or {}
        out.append(_h(f"Next week ({tw.get('first_session')} to {tw.get('resolves_on')}, {tw.get('horizon_td')} sessions)"))
        cal = nw.get("calendar") or []
        if cal:
            out.append("<ul style='font-size:13px;'>" + "".join(f"<li>{_esc(c)}</li>" for c in cal) + "</ul>")
        out.append(f'<div style="font-size:14px;margin:6px 0;"><b>Base case:</b> {_esc(nw.get("base_case"))}</div>')
        out.append(f'<div style="font-size:14px;margin:6px 0;"><b>Alternative:</b> {_esc(nw.get("alt_case"))}</div>')
        out.append(_h("Forecasts (locked, graded against climatology)"))
        out.append(render_forecasts(validated, payload, state, scored))
    out.append(_h("Risk Agent readout"))
    out.append(render_ra(ro))
    if payload.get("mode") != "stand_down":
        watch = payload.get("watch") or []
        if watch:
            out.append(_h("Watch"))
            out.append("<ul style='font-size:13px;'>" + "".join(
                f"<li><b>{_esc(w.get('item'))}</b>: {_esc(w.get('trigger'))}</li>" for w in watch) + "</ul>")
        qs = payload.get("questions") or []
        if qs:
            out.append(_h("Food for thought"))
            out.append("<ul style='font-size:13px;'>" + "".join(
                f"<li>{_esc(q.get('question'))} <span style='color:#6b7280;'>{_esc(q.get('why_it_matters'))}</span></li>"
                for q in qs) + "</ul>")
        gaps = payload.get("data_gaps") or []
        if gaps:
            out.append(_p("Data gaps: " + "; ".join(str(g) for g in gaps), "font-size:12px;color:#6b7280;"))
    out.append("</div>")
    return subject, "".join(out)


# ===========================================================================
# main
# ===========================================================================
def build_ctx(state: dict, checks_dir: Path) -> dict:
    vix = ((state.get("anchors") or {}).get("^VIX") or {}).get("close")
    return {"asof": state.get("asof"), "week": state.get("week"),
            "target_week": state.get("target_week") or {},
            "climatology": state.get("climatology") or {}, "checks_dir": checks_dir,
            "vix_last": vix}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--brief", default=None)
    ap.add_argument("--state", default=None)
    ap.add_argument("--journal", default=None)
    ap.add_argument("--checks-root", default=None)
    ap.add_argument("--today-out", default=None)
    ap.add_argument("--receipt-dir", default=None)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--no-send", action="store_true")
    ap.add_argument("--no-r2", action="store_true")
    ap.add_argument("--html-out", default=None)
    ap.add_argument("--now", default=None, help="ISO timestamp (tests)")
    args = ap.parse_args(argv)

    state_path = Path(args.state or U.state_path())
    state = _read_json(state_path)
    if not isinstance(state, dict) or not state.get("asof") or not state.get("week"):
        print(f"FAILED: no usable state at {state_path}")
        return 2
    payload = _read_json(Path(args.brief or U.brief_path()))
    if not isinstance(payload, dict):
        print(f"FAILED: no usable brief at {args.brief or U.brief_path()}")
        return 2
    asof, week = str(state["asof"]), str(state["week"])
    journal = Path(args.journal or U.journal_path())
    use_r2 = not args.no_r2
    records = J.load(journal, pull=use_r2 and not args.validate_only)
    if not args.validate_only and J.verdicts_for(records, week):
        print(f"REFUSED: the journal already has a brief or stand-down for {week}.")
        return 2
    checks_dir = Path(args.checks_root or U.checks_root()) / asof
    result = grammar.validate_brief(payload, build_ctx(state, checks_dir))
    warnings = list(result["warnings"])
    if result["errors"]:
        print(f"REJECTED: {len(result['errors'])} validation error(s)")
        for e in result["errors"]:
            print(f"  ERROR {e}")
        return 2
    now = dt.datetime.fromisoformat(args.now) if args.now else dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=dt.timezone.utc)
    scored = now < deadline(state["target_week"])
    if not scored:
        warnings.append("published after the target week's first open: forecasts are not scored")
    if args.validate_only:
        print(f"OK: brief for {week} (asof {asof}) validates ({payload.get('mode')}); scored={scored}")
        for f in result["forecasts"]:
            print("  forecast " + json.dumps(f, sort_keys=True))
        for w in warnings:
            print(f"  warning {w}")
        return 0
    return publish(payload, result, state, state_path, records, journal, week, scored, warnings, args)


def publish(payload, result, state, state_path, records, journal, week, scored, warnings, args) -> int:
    use_r2 = not args.no_r2
    model = os.environ.get("PM_AGENT_MODEL") or "unknown"
    effort = os.environ.get("PM_AGENT_EFFORT") or "unknown"
    decision_id = f"PMW-{week}"
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    warnings = warnings + [str(w) for w in state.get("warnings") or []]
    stand = payload.get("mode") == "stand_down"
    head = {"kind": "stand_down" if stand else "brief", "asof": state["asof"], "week": week,
            "decision_id": decision_id, "model": model, "effort": effort,
            "state_sha256": hashlib.sha256(Path(state_path).read_bytes()).hexdigest(),
            "state_built_at": state.get("built_at"), "warnings": warnings,
            "written_at": stamp, "payload": payload}
    recs = [head] + forecast_records(result["forecasts"], state, week=week, scored=scored,
                                     model=model, effort=effort, now=stamp)
    failed = False
    try:
        J.append(recs, journal, push=use_r2)
    except Exception as exc:  # noqa: BLE001
        print(f"JOURNAL WRITE/PUSH FAILED: {exc}")
        if not J.verdicts_for(J.load(journal), week):
            return 1
        failed = True
    print(f"Journaled {len(recs)} record(s) for {week}")

    ro = ra_readout(fetch_ra_today(use_r2), result["forecasts"])
    subject, html = render_email(payload, state, result["forecasts"], ro, week=week, scored=scored,
                                 warnings=warnings, model=model, effort=effort)
    if args.html_out:
        Path(args.html_out).write_text(html, encoding="utf-8")
    if args.no_send:
        print(f"--no-send: skipping email ({subject})")
    else:
        path = Path(args.receipt_dir or U.receipt_dir()) / f"{week}.json"
        recipients = email_recipients()
        try:
            receipt, should_send = reserve_receipt(week, decision_id, subject, html, recipients, path, use_r2)
        except ReceiptError as exc:
            print(f"DELIVERY BLOCKED: {exc}")
            failed = True
        else:
            if should_send:
                ok = send_email(subject, html, recipients)
                try:
                    complete_receipt(receipt, path, use_r2, ok)
                except ReceiptError as exc:
                    print(f"DELIVERY RECEIPT FAILED: {exc}")
                    failed = True
                failed = failed or not ok
            else:
                print(f"Email already confirmed sent for {week}; skipping.")

    today = {"schema_version": "pm_agent.today.v1", "week": week, "asof": state["asof"],
             "decision_id": decision_id, "mode": payload.get("mode"), "model": model, "effort": effort,
             "target_week": state.get("target_week"), "payload": payload,
             "forecasts": result["forecasts"], "scored": scored, "risk_agent_readout": ro,
             "warnings": warnings, "published_at": stamp}
    tpath = Path(args.today_out or U.today_path())
    tpath.parent.mkdir(parents=True, exist_ok=True)
    tpath.write_text(json.dumps(today, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Wrote {tpath}")
    if use_r2:
        if r2_upload(tpath, U.R2_PREFIX + "today.json"):
            print(f"Uploaded R2 {U.R2_PREFIX}today.json")
        else:
            print(f"R2 UPLOAD FAILED for {U.R2_PREFIX}today.json")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
