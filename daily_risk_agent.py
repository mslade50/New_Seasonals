"""Risk Agent publisher: validate, journal, deliver, publish to the site.

The creative work (survey, forecast, select, falsify) happens in the
`/risk-agent` skill, which writes data/risk_agent_decision.json. This module is
the deterministic tail. It refuses to publish anything risk_agent_grammar
rejects, so a malformed run produces a loud failure instead of a plausible
looking email.

    python daily_risk_agent.py [--decision PATH] [--state PATH] [--chains PATH]
                               [--validate-only] [--no-send] [--no-r2]

Flow: load state + chains + journal, replay the journal into the paper book,
build the validator ctx, validate. Errors exit 2. A clean decision is then
journaled (decision + sized orders, or a stand_down), emailed once behind a
local receipt, written to data/risk_agent_today.json and uploaded to R2
(risk_agent/today.json, risk_agent/journal.jsonl). A second publish for the
same asof is refused.

Env: EMAIL_USER / EMAIL_PASS (or repo .env), RISK_AGENT_RECIPIENTS (defaults to
the pitch recipient), RISK_AGENT_MODEL / RISK_AGENT_EFFORT (stamped on the
decision record; the bat exports them).

Agent-product module: the book must not import it. Doc: docs/claude_ref/risk_agent.md
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

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import risk_agent_grammar as grammar  # noqa: E402

DATA = ROOT / "data"
DEFAULT_DECISION = DATA / "risk_agent_decision.json"
DEFAULT_STATE = DATA / "risk_agent_state.json"
DEFAULT_CHAINS = DATA / "risk_agent_chains.json"
DEFAULT_SCOREBOARD = DATA / "risk_agent_scoreboard.json"
TODAY_PATH = DATA / "risk_agent_today.json"
TODAY_R2_KEY = "risk_agent/today.json"
RECEIPT_DIR = DATA / "risk_agent_delivery_receipts"
RECEIPT_R2_PREFIX = "risk_agent/delivery_receipts"
CHECKS_ROOT = ROOT / "scratch" / "risk_agent_checks"
DEFAULT_RECIPIENTS = "mckinleyslade@gmail.com"
RECEIPT_SCHEMA = "risk-agent-delivery.v1"
VERDICT_KINDS = ("decision", "stand_down")

ledger = None  # risk_agent_ledger, imported lazily so tests can inject a fake


def get_ledger():
    global ledger
    if ledger is None:
        import risk_agent_ledger
        ledger = risk_agent_ledger
    return ledger


# ===========================================================================
# helpers
# ===========================================================================

def _read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _clean(text) -> str:
    """No em dashes in anything we render or ship."""
    return str(text if text is not None else "").replace("—", " - ")


def _esc(text) -> str:
    return _html.escape(_clean(text), quote=True)


def _num(value, default=None):
    try:
        out = float(value)
        return out if out == out and abs(out) != float("inf") else default
    except (TypeError, ValueError):
        return default


def _money(value, signed=False) -> str:
    v = _num(value)
    if v is None:
        return "-"
    return f"{v:+,.0f}" if signed else f"{v:,.0f}"


def _pct(value, digits=1, signed=True) -> str:
    v = _num(value)
    if v is None:
        return "-"
    return f"{v:+.{digits}f}%" if signed else f"{v:.{digits}f}%"


def _px(value):
    """Prices arrive as float32-widened floats (61.86000061035156); show cents."""
    v = _num(value)
    return v if v is None else round(v, 2)


def _frac_pct(value, digits=1) -> str:
    """A fraction (0.034) shown as a percent."""
    v = _num(value)
    return "-" if v is None else f"{v * 100:+.{digits}f}%"


def record_asof(record: dict) -> str:
    return str(record.get("asof") or record.get("date") or "")


def verdict_records(records, asof: str) -> list[dict]:
    return [r for r in records if r.get("kind") in VERDICT_KINDS and record_asof(r) == str(asof)]


def instrument_label(inst: dict) -> str:
    kind = (inst or {}).get("type")
    if kind == "etf":
        return str(inst.get("symbol"))
    if kind == "future":
        return f"{inst.get('root')} {inst.get('contract_month')}"
    if kind == "option_structure":
        legs = " / ".join(f"{l.get('qty'):+d} {l.get('right')} {l.get('strike'):g} {l.get('expiry')}"
                          if isinstance(l.get("strike"), (int, float))
                          else f"{l.get('qty')} {l.get('right')} {l.get('strike')} {l.get('expiry')}"
                          for l in inst.get("legs") or [])
        return f"{inst.get('underlying')} x{inst.get('structure_qty')}: {legs}"
    return "?"


# ===========================================================================
# receipts (own minimal protocol; pitch_delivery has no namespace for us)
# ===========================================================================

class ReceiptError(RuntimeError):
    pass


def receipt_path(asof: str) -> Path:
    return RECEIPT_DIR / f"{asof}.json"


def read_receipt(asof: str, path: Path | None = None) -> dict | None:
    p = Path(path) if path else receipt_path(asof)
    if not p.exists():
        return None
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReceiptError(f"receipt unreadable at {p}: {exc}") from exc
    if rec.get("schema") != RECEIPT_SCHEMA:
        raise ReceiptError(f"receipt at {p} has unsupported schema {rec.get('schema')!r}")
    return rec


def _write_receipt(rec: dict, path: Path, use_r2: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    if use_r2:
        if not r2_upload(path, f"{RECEIPT_R2_PREFIX}/{rec['date']}.json"):
            raise ReceiptError("could not mirror the delivery receipt to R2")


def reserve_receipt(asof: str, decision_id: str, subject: str, html: str,
                    recipients: list[str], path: Path, use_r2: bool) -> tuple[dict, bool]:
    """(receipt, should_send). A sent receipt skips; any other state blocks."""
    existing = read_receipt(asof, path)
    if existing is not None:
        if existing.get("status") == "sent":
            return existing, False
        raise ReceiptError(f"delivery receipt for {asof} is {existing.get('status')}; "
                           "resolve the SMTP outcome before any rerun")
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    digest = hashlib.sha256(json.dumps(
        {"s": subject, "h": html, "r": sorted(recipients)}, sort_keys=True).encode("utf-8")).hexdigest()
    rec = {"schema": RECEIPT_SCHEMA, "status": "sending", "date": asof,
           "decision_id": decision_id, "delivery_id": str(uuid.uuid4()),
           "message_digest": digest, "subject": subject,
           "recipients": sorted(recipients), "created_at": now, "updated_at": now}
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
    except ReceiptError as exc:
        out["status"] = "ambiguous"
        out["ambiguity_reason"] = f"receipt persistence failed after SMTP outcome: {exc}"
        _write_receipt(out, path, False)
        raise
    return out


# ===========================================================================
# R2 / SMTP (thin; tests monkeypatch these)
# ===========================================================================

def r2_upload(local: Path, key: str) -> bool:
    try:
        import cache_io
    except Exception as exc:  # noqa: BLE001
        print(f"R2 unavailable: {exc}")
        return False
    if not cache_io.is_configured():
        print("R2 is not configured")
        return False
    return bool(cache_io.upload_from_local(str(local), key))


def email_recipients() -> list[str]:
    raw = os.environ.get("RISK_AGENT_RECIPIENTS") or os.environ.get("PITCH_RECIPIENTS") \
        or DEFAULT_RECIPIENTS
    return [a.strip() for a in raw.split(",") if a.strip()]


def send_email(subject: str, html: str, recipients: list[str]) -> bool:
    # Lazy: daily_pitch pulls pandas and the pitch grammar; only credentials
    # lookup is reused (no Sheets side effects at import time).
    from daily_pitch import smtp_credentials
    sender, password = smtp_credentials()
    if not sender or not password:
        print("EMAIL_USER/EMAIL_PASS not set - skipping send. THE RISK AGENT EMAIL WAS NOT DELIVERED.")
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
        print(f"EMAIL SEND FAILED ({exc}) - THE RISK AGENT EMAIL WAS NOT DELIVERED.")
        return False
    print(f"Email sent to {', '.join(recipients)}")
    return True


# ===========================================================================
# views over the replayed book
# ===========================================================================

def _first(d: dict, *keys):
    for k in keys:
        if d.get(k) is not None:
            return d[k]
    return None


def position_view(pid: str, p: dict) -> dict:
    return {
        "id": pid,
        "kind": p.get("kind"),
        "symbol": _first(p, "symbol", "underlying", "root", "series"),
        "side": p.get("side"),
        "qty": _first(p, "qty", "quantity", "structure_qty"),
        "entry": _px(_first(p, "entry_price", "avg_price", "fill_price", "entry")),
        "mark": _px(_first(p, "mark", "mark_price", "last_mark")),
        "pnl": _first(p, "unrealized_pnl", "pnl", "unrealized"),
        "risk_bps": p.get("risk_bps"),
        "notional": p.get("notional"),
        "stop": _first(p, "stop", "stop_price"),
        "target": _first(p, "target", "target_price"),
        "time_exit": _first(p, "time_exit_date", "exit_date", "time_exit", "time_td"),
        "stale_mark": bool(p.get("stale_mark")),
    }


def book_snapshot(book: dict) -> dict:
    positions = [position_view(pid, p) for pid, p in (book.get("positions") or {}).items()]
    return {"nav": book.get("nav"), "cash": book.get("cash"),
            "realized_pnl": book.get("realized_pnl"),
            "last_mark_date": book.get("last_mark_date"),
            "pending": len(book.get("pending") or []),
            "positions": positions}


def order_view(order: dict, pos: dict | None) -> dict:
    pos = pos or {}
    inst = pos.get("instrument") or {}
    out = {
        "id": order.get("position_id"),
        "kind": order.get("kind"),
        "instrument": instrument_label(inst) if inst else (order.get("symbol") or order.get("underlying")),
        "side": order.get("side") or ("long" if order.get("kind") == "option" else None),
        "qty": order.get("qty") if order.get("qty") is not None else order.get("structure_qty"),
        "ref_price": order.get("ref_price"),
        "risk_bps": order.get("risk_bps"),
        "notional": order.get("notional"),
        "entry": order.get("entry"),
        "exit": order.get("exit"),
        "thesis": pos.get("thesis"),
        "evidence": (pos.get("evidence") or {}).get("summary"),
        "survived": pos.get("survived"),
        "what_kills_it": pos.get("what_kills_it"),
        "forecast": pos.get("forecast"),
    }
    gate = order.get("gate")
    if gate:
        out["option"] = {"status": gate.get("status"), "max_loss": gate.get("max_loss"),
                         "max_loss_fraction": gate.get("max_loss_fraction"),
                         "loss_bound": gate.get("loss_bound"),
                         "stress_move": gate.get("stress_move"),
                         "legs": [{k: l.get(k) for k in ("right", "strike", "expiry", "qty", "bid", "ask")}
                                  for l in order.get("legs") or []]}
    return out


def verdict_views(payload: dict, held: dict) -> list[dict]:
    out = []
    for pos in payload.get("positions") or []:
        if pos.get("action") in ("hold", "adjust", "close"):
            h = held.get(pos.get("id")) or {}
            out.append({"id": pos.get("id"), "action": pos.get("action"),
                        "symbol": _first(h, "symbol", "underlying", "root"),
                        "reason": pos.get("reason"), "exit": pos.get("exit")})
    return out


# ===========================================================================
# email
# ===========================================================================

_TD = "padding:4px 8px;border-bottom:1px solid #e5e7eb;font-size:13px;"
_TH = "padding:4px 8px;text-align:left;font-size:11px;color:#6b7280;text-transform:uppercase;"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return '<p style="color:#6b7280;font-size:13px;margin:4px 0;">None.</p>'
    head = "".join(f'<th style="{_TH}">{_esc(h)}</th>' for h in headers)
    body = "".join("<tr>" + "".join(f'<td style="{_TD}">{c}</td>' for c in r) + "</tr>" for r in rows)
    return (f'<table style="border-collapse:collapse;width:100%;"><tr>{head}</tr>{body}</table>')


def _h(title: str) -> str:
    return f'<h3 style="margin:22px 0 6px;font-size:15px;color:#111827;">{_esc(title)}</h3>'


def _exit_label(ex: dict | None) -> str:
    ex = ex or {}
    parts = []
    if ex.get("stop") is not None:
        parts.append(f"stop {ex['stop']}")
    if ex.get("target") is not None:
        parts.append(f"target {ex['target']}")
    if ex.get("time_td") is not None:
        parts.append(f"time {ex['time_td']} td")
    return ", ".join(parts) or "-"


def _entry_label(e: dict | None) -> str:
    e = e or {}
    if e.get("type") == "LIMIT":
        return f"LIMIT {e.get('limit')} ({e.get('fill_window_td', 1)} td)"
    return str(e.get("type") or "-")


def render_card(view: dict) -> str:
    opt = view.get("option")
    lines = [
        f'<div style="font-size:15px;font-weight:700;">{_esc(view.get("instrument"))} '
        f'<span style="color:#6b7280;font-weight:400;">{_esc(view.get("side") or "")} '
        f'{_esc(view.get("qty"))} @ {_esc(view.get("ref_price") if view.get("ref_price") is not None else "chain")}</span></div>',
        f'<div style="font-size:12px;color:#374151;margin:2px 0 6px;">'
        f'Entry {_esc(_entry_label(view.get("entry")))} | Exit {_esc(_exit_label(view.get("exit")))} | '
        f'Risk {_esc(view.get("risk_bps"))} bps | Notional {_money(view.get("notional"))}</div>',
    ]
    if opt:
        lines.append(
            f'<div style="font-size:12px;margin-bottom:6px;"><b>{_esc(opt.get("status"))}</b> '
            f'certified max loss ${_money(opt.get("max_loss"))} ({100 * (_num(opt.get("max_loss_fraction")) or 0):.2f}% of capital; cap 5%'
            f'{", stress +%.0f%%" % (100 * opt["stress_move"]) if opt.get("stress_move") else ""})</div>')
    for label, key in (("Thesis", "thesis"), ("Evidence", "evidence"),
                       ("Survived", "survived"), ("What kills it", "what_kills_it")):
        if view.get(key):
            lines.append(f'<div style="font-size:13px;margin:3px 0;"><b>{label}:</b> {_esc(view[key])}</div>')
    fc = view.get("forecast") or {}
    if fc:
        lines.append(f'<div style="font-size:12px;color:#6b7280;">Forecast {_esc(fc.get("horizon_td"))} td: '
                     f'expected {_esc(fc.get("expected_return_pct"))}%, p(win) {_esc(fc.get("p_win"))}</div>')
    return ('<div style="border:1px solid #d1d5db;border-radius:6px;padding:10px 12px;margin:8px 0;">'
            + "".join(lines) + "</div>")


def render_scoreboard(sb: dict | None) -> str:
    # grade_risk_agent.scoreboard()["headline"]: percent units, fixed names.
    sb = (sb or {}).get("headline") or {}
    if not sb:
        return ""
    items = []
    for label, keys, fmt in (
            ("NAV", ("nav",), _money),
            ("Return", ("total_return_pct",), _pct),
            ("Max drawdown", ("max_drawdown_pct",), _pct),
            ("vs SPY", ("vs_spy_pct",), _pct),
            ("Sharpe", ("sharpe",), lambda v: f"{_num(v, 0):.2f}"),
            ("Brier 5d", ("brier_5",), lambda v: f"{_num(v, 0):.3f}"),
            ("Brier 21d", ("brier_21",), lambda v: f"{_num(v, 0):.3f}")):
        v = _first(sb, *keys)
        if v is not None and _num(v) is not None:
            items.append(f"{label} {fmt(v)}")
    if not items:
        return ""
    return (f'<p style="margin-top:22px;padding-top:8px;border-top:1px solid #e5e7eb;font-size:12px;'
            f'color:#6b7280;">Scoreboard: {_esc(" | ".join(items))}</p>')


def render_email(payload: dict, *, orders_views: list[dict], verdicts: list[dict],
                 book_after: dict, snapshot: dict, scoreboard: dict | None,
                 warnings: list[str], asof: str, model: str, effort: str) -> tuple[str, str]:
    """(subject, html)."""
    stand = payload.get("mode") == "stand_down"
    if stand:
        subject = f"Risk Agent {asof}: DATA HOLD"
    else:
        subject = f"Risk Agent {asof}: {_clean((payload.get('posture') or {}).get('summary', ''))[:60].strip()}"
    posture = payload.get("posture") or {}
    out = ['<div style="font-family:Arial,Helvetica,sans-serif;max-width:760px;color:#111827;">',
           f'<h2 style="margin:0 0 4px;">Risk Agent {_esc(asof)}</h2>',
           f'<div style="font-size:12px;color:#6b7280;">Independent $200k paper sleeve. Model {_esc(model)} / {_esc(effort)}. '
           'Nothing here places live orders.</div>']
    if stand:
        out.append(_h("Data hold"))
        out.append(f'<p style="font-size:14px;">{_esc(payload.get("reason"))}</p>')
    if posture:
        out.append(_h("Posture"))
        out.append(f'<p style="font-size:14px;margin:4px 0;">{_esc(posture.get("summary"))}</p>')
        out.append(f'<div style="font-size:13px;color:#374151;">Net beta {_esc(posture.get("net_beta"))} | '
                   f'cash {_esc(posture.get("cash_pct"))}%</div>')
    if warnings:
        out.append(_h("Data warnings"))
        out.append("<ul style='font-size:13px;color:#92400e;'>" + "".join(f"<li>{_esc(w)}</li>" for w in warnings) + "</ul>")
    fcs = payload.get("forecasts") or []
    if fcs:
        out.append(_h("SPY forecasts"))
        out.append(_table(["Horizon", "P(up)", "q10", "q90", "Basis"],
                          [[_esc(f"{f.get('horizon_td')} td"), _esc(f.get("p_up")),
                            _esc(_pct(f.get("q10_pct"))), _esc(_pct(f.get("q90_pct"))),
                            _esc(f.get("basis") or "")] for f in sorted(fcs, key=lambda x: x.get("horizon_td") or 0)]))
    out.append(_h("New positions"))
    out.append("".join(render_card(v) for v in orders_views)
               or '<p style="color:#6b7280;font-size:13px;">None.</p>')
    out.append(_h("Verdicts on held positions"))
    out.append(_table(["Position", "Symbol", "Verdict", "Reason"],
                      [[_esc(v["id"]), _esc(v.get("symbol") or ""), f"<b>{_esc(v['action'])}</b>",
                        _esc(v.get("reason"))] for v in verdicts]))
    out.append(_h("Book after today's orders"))
    out.append(f'<div style="font-size:13px;">Positions {_esc(book_after.get("positions_after"))} | '
               f'risk {_esc(book_after.get("risk_bps_after"))} bps | '
               f'gross {_money(book_after.get("gross_notional_after"))} = {_esc(book_after.get("gross_x_nav"))}x NAV</div>')
    out.append(_h("Current paper book (marks and P&L)"))
    rows = []
    for p in snapshot.get("positions") or []:
        rows.append([_esc(p["id"]), _esc(p.get("symbol") or ""), _esc(p.get("side") or ""),
                     _esc(p.get("qty")), _esc(p.get("entry")), _esc(p.get("mark")) + (" (stale)" if p.get("stale_mark") else ""),
                     _money(p.get("pnl"), signed=True)])
    out.append(_table(["ID", "Symbol", "Side", "Qty", "Entry", "Mark", "P&L $"], rows))
    out.append(f'<div style="font-size:12px;color:#6b7280;margin-top:4px;">NAV {_money(snapshot.get("nav"))} | '
               f'cash {_money(snapshot.get("cash"))} | realised {_money(snapshot.get("realized_pnl"), signed=True)} | '
               f'{_esc(snapshot.get("pending"))} pending order(s)</div>')
    rej = payload.get("considered_and_rejected") or []
    if rej:
        out.append(_h("Considered and rejected"))
        out.append("<ul style='font-size:13px;'>" + "".join(
            f"<li><b>{_esc(r.get('idea'))}</b>: {_esc(r.get('reason'))}</li>" for r in rej) + "</ul>")
    wl = payload.get("watchlist") or []
    if wl:
        out.append(_h("Watchlist"))
        out.append("<ul style='font-size:13px;'>" + "".join(
            f"<li><b>{_esc(w.get('idea'))}</b>: {_esc(w.get('trigger'))} (expires {_esc(w.get('expires'))})</li>"
            for w in wl) + "</ul>")
    out.append(render_scoreboard(scoreboard))
    out.append("</div>")
    return subject, "".join(out)


# ===========================================================================
# main
# ===========================================================================

def build_ctx(state: dict, chains: dict, book: dict, checks_dir: Path | None) -> dict:
    lg = get_ledger()
    return {"asof": state.get("asof"), "nav": book.get("nav"),
            "positions": lg.validator_positions(book),
            "quotes": state.get("quotes") or {}, "chains": chains,
            "stress": state.get("stress") or {}, "checks_dir": checks_dir}


def load_chains(path: Path) -> dict:
    raw = _read_json(path, {}) or {}
    if isinstance(raw.get("chains"), dict):
        raw = raw["chains"]
    return raw


def site_payload(payload: dict, *, decision_id: str, model: str, effort: str, asof: str,
                 orders_views, verdicts, snapshot, scoreboard, warnings, book_after) -> dict:
    return {
        "schema_version": "risk_agent.today.v1",
        "asof": asof, "decision_id": decision_id, "mode": payload.get("mode"),
        "reason": payload.get("reason"), "model": model, "effort": effort,
        "posture": payload.get("posture") or {},
        "forecasts": payload.get("forecasts") or [],
        "new_orders": orders_views, "verdicts": verdicts,
        "book": snapshot, "book_after": book_after,
        "considered_and_rejected": payload.get("considered_and_rejected") or [],
        "watchlist": payload.get("watchlist") or [],
        "scoreboard": scoreboard or {}, "warnings": warnings,
        "published_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--decision", default=str(DEFAULT_DECISION))
    ap.add_argument("--state", default=str(DEFAULT_STATE))
    ap.add_argument("--chains", default=str(DEFAULT_CHAINS))
    ap.add_argument("--journal", default=None, help="journal path (tests/dev)")
    ap.add_argument("--checks-root", default=str(CHECKS_ROOT))
    ap.add_argument("--today-out", default=str(TODAY_PATH))
    ap.add_argument("--receipt-dir", default=None)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--no-send", action="store_true")
    ap.add_argument("--no-r2", action="store_true")
    ap.add_argument("--html-out", help="also write the rendered email HTML here (preview)")
    args = ap.parse_args(argv)

    lg = get_ledger()
    state_path = Path(args.state)
    state = _read_json(state_path)
    if not isinstance(state, dict) or not state.get("asof"):
        print(f"FAILED: no usable state at {state_path}")
        return 2
    payload = _read_json(Path(args.decision))
    if not isinstance(payload, dict):
        print(f"FAILED: no usable decision at {args.decision}")
        return 2
    asof = str(state["asof"])
    chains = load_chains(Path(args.chains))
    journal_path = Path(args.journal) if args.journal else Path(lg.JOURNAL_PATH)
    records = lg.load(journal_path, pull=False)
    if not args.validate_only and verdict_records(records, asof):
        print(f"REFUSED: the journal already has a decision or stand-down for {asof}. "
              "Publishing twice is not allowed.")
        return 2
    book = lg.replay(records)
    checks_dir = Path(args.checks_root) / asof
    result = validate_and_size(payload, state, chains, book, checks_dir)

    if result["errors"]:
        print(f"REJECTED: {len(result['errors'])} validation error(s)")
        for e in result["errors"]:
            print(f"  ERROR {e}")
        for w in result["warnings"]:
            print(f"  warning {w}")
        return 2

    if args.validate_only:
        print(f"OK: decision for {asof} validates ({payload.get('mode')})")
        for o in result["orders"]:
            print("  order " + json.dumps(o, sort_keys=True, default=str))
        print("  book " + json.dumps(result["book"], sort_keys=True))
        for w in result["warnings"]:
            print(f"  warning {w}")
        return 0

    return publish(payload, result, state, state_path, book, records, journal_path, asof, args)


def validate_and_size(payload, state, chains, book, checks_dir) -> dict:
    ctx = build_ctx(state, chains, book, checks_dir)
    return grammar.validate_decision(payload, ctx)


def publish(payload, result, state, state_path, book, records, journal_path, asof, args) -> int:
    lg = get_ledger()
    use_r2 = not args.no_r2
    model = os.environ.get("RISK_AGENT_MODEL") or "unknown"
    effort = os.environ.get("RISK_AGENT_EFFORT") or "unknown"
    decision_id = f"RAD-{asof}"
    state_sha = hashlib.sha256(Path(state_path).read_bytes()).hexdigest()
    warnings = list(result["warnings"]) + [str(w) for w in state.get("warnings") or []]
    stand = payload.get("mode") == "stand_down"
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    head = {"kind": "stand_down" if stand else "decision", "asof": asof, "date": asof,
            "decision_id": decision_id, "model": model, "effort": effort,
            "state_sha256": state_sha, "state_built_at": state.get("built_at"),
            "warnings": warnings, "book": result["book"], "written_at": now,
            "payload": payload}
    if stand:
        head["reason"] = payload.get("reason")
    recs = [head] + (lg.orders_to_records(result["orders"], asof, decision_id) if result["orders"] else [])
    failed = False
    try:
        lg.append(recs, journal_path, push=use_r2)
    except Exception as exc:  # noqa: BLE001
        print(f"JOURNAL WRITE/PUSH FAILED: {exc}")
        if not any(r.get("kind") in VERDICT_KINDS and record_asof(r) == asof
                   for r in lg.load(journal_path, pull=False)):
            return 1
        failed = True
    print(f"Journaled {len(recs)} record(s) for {asof}")

    after = lg.load(journal_path, pull=False)
    book_now = lg.replay(after)
    snapshot = book_snapshot(book_now)
    scoreboard = dict(state.get("scoreboard") or _read_json(DEFAULT_SCOREBOARD, {}) or {})
    if not scoreboard.get("nav_curve") and book_now.get("marks"):
        # replay marks are (date, nav) pairs
        scoreboard["nav_curve"] = [m[1] if isinstance(m, (list, tuple)) else m
                                   for m in book_now["marks"]]
    pos_by_id = {p.get("id"): p for p in payload.get("positions") or []}
    orders_views = [order_view(o, pos_by_id.get(o.get("position_id")))
                    for o in result["orders"] if o.get("type") == "open"]
    verdicts = verdict_views(payload, grammar_held(book))
    subject, html = render_email(payload, orders_views=orders_views, verdicts=verdicts,
                                 book_after=result["book"], snapshot=snapshot,
                                 scoreboard=scoreboard, warnings=warnings, asof=asof,
                                 model=model, effort=effort)

    if getattr(args, "html_out", None):
        Path(args.html_out).write_text(html, encoding="utf-8")
    if args.no_send:
        print(f"--no-send: skipping email ({subject})")
    else:
        rdir = Path(args.receipt_dir) if args.receipt_dir else RECEIPT_DIR
        path = rdir / f"{asof}.json"
        recipients = email_recipients()
        try:
            receipt, should_send = reserve_receipt(asof, decision_id, subject, html, recipients, path, use_r2)
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
                print(f"Email already confirmed sent for {asof}; skipping.")

    today = site_payload(payload, decision_id=decision_id, model=model, effort=effort, asof=asof,
                         orders_views=orders_views, verdicts=verdicts, snapshot=snapshot,
                         scoreboard=scoreboard, warnings=warnings, book_after=result["book"])
    tpath = Path(args.today_out)
    tpath.parent.mkdir(parents=True, exist_ok=True)
    tpath.write_text(json.dumps(today, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Wrote {tpath}")
    if use_r2:
        if r2_upload(tpath, TODAY_R2_KEY):
            print(f"Uploaded R2 {TODAY_R2_KEY}")
        else:
            print(f"R2 UPLOAD FAILED for {TODAY_R2_KEY}")
            failed = True
    return 1 if failed else 0


def grammar_held(book: dict) -> dict:
    lg = get_ledger()
    held = {}
    for pid, p in (book.get("positions") or {}).items():
        held[pid] = p
    try:
        for pid, v in lg.validator_positions(book).items():
            held.setdefault(pid, v)
    except Exception:  # noqa: BLE001
        pass
    return held


if __name__ == "__main__":
    raise SystemExit(main())
