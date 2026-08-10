"""Email the day's Daily Posts queue to McKinley - a convenience mirror of
content/queue/<date>.md, so the drafts are readable (and copyable) from a
phone. The QUEUE FILE stays the source of truth: Posted marks are flipped
in the repo file, never by replying to this email.

    python scripts/send_posts_email.py [--asof YYYY-MM-DD] [--dry-run]
                                       [--queue-dir PATH]

Env: EMAIL_USER / EMAIL_PASS (daily_pitch's smtp_credentials, .env
fallback included), POSTS_RECIPIENTS (comma-separated, defaults to
mckinleyslade@gmail.com). Best-effort by design: the queue on disk is the
delivery; this failing loudly must not fail the run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import smtplib
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from daily_pitch import smtp_credentials  # noqa: E402

DEFAULT_RECIPIENTS = "mckinleyslade@gmail.com"

CARD = """<div style="border:1px solid #ccc;border-radius:6px;margin:10px 0;
padding:10px 14px;font-family:Segoe UI,Arial,sans-serif">
<div style="font-size:12px;color:#666">{head}</div>
<pre style="white-space:pre-wrap;font-family:Consolas,Menlo,monospace;
font-size:14px;margin:8px 0">{text}</pre>
{evidence}
</div>"""


def build_html(payload: dict, md_note: str) -> str:
    cards = []
    for i, d in enumerate(payload.get("drafts") or [], 1):
        texts = d.get("texts") or [d.get("text") or ""]
        body = "\n\n---\n\n".join(str(t) for t in texts)
        ev = d.get("evidence") or {}
        ev_bits = [b for b in [ev.get("summary"),
                               f"N={ev['n']}" if ev.get("n") not in (None, "")
                               else None,
                               ev.get("script")] if b]
        ev_html = (f'<div style="font-size:12px;color:#888">'
                   f'{html.escape(" | ".join(ev_bits))}</div>' if ev_bits else "")
        idea = d.get("idea") or {}
        head = f"{i}. [{d.get('type')}] id={d.get('id')}"
        if idea:
            head += (f" &nbsp;&nbsp;{idea.get('side', '')} "
                     f"{idea.get('ticker', '')}, execute {idea.get('execute_on', '')},"
                     f" time stop {idea.get('time_td', '?')} td")
        cards.append(CARD.format(head=html.escape(head).replace("&amp;nbsp;", "&nbsp;"),
                                 text=html.escape(body), evidence=ev_html))
    return (f'<div style="font-family:Segoe UI,Arial,sans-serif">'
            f'<p style="color:#444">{html.escape(md_note)}</p>'
            f'{"".join(cards)}'
            f'<p style="font-size:12px;color:#888">Post by hand from the X app. '
            f'Then flip the <b>Posted:</b> line in content/queue/ - the file is '
            f'the record, this email is a mirror. Nothing auto-posts.</p></div>')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None)
    ap.add_argument("--queue-dir", default=str(ROOT / "content" / "queue"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    day = str(args.asof or dt.date.today())
    qdir = Path(args.queue_dir)
    jpath = qdir / f"{day}.json"
    if not jpath.exists():
        print(f"no queue json for {day}; nothing to email")
        return 1
    payload = json.loads(jpath.read_text(encoding="utf-8"))
    drafts = payload.get("drafts") or []
    note = (f"Queue for {day}: {len(drafts)} draft(s). "
            f"Types: {', '.join(d.get('type', '?') for d in drafts)}.")
    body = build_html(payload, note)
    subject = f"Daily Posts queue - {day} ({len(drafts)} drafts)"

    if args.dry_run:
        out = qdir / f"{day}.email.html"
        out.write_text(body, encoding="utf-8")
        print(f"[dry-run] wrote {out}")
        return 0

    sender, password = smtp_credentials()
    recipients = [a.strip() for a in os.environ.get(
        "POSTS_RECIPIENTS", DEFAULT_RECIPIENTS).split(",") if a.strip()]
    if not sender or not password:
        print("EMAIL_USER/EMAIL_PASS not set - queue email skipped "
              "(the queue file on disk is still the delivery)")
        return 1
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(body, "html"))
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender, password)
            server.sendmail(sender, recipients, msg.as_string())
    except smtplib.SMTPException as exc:
        print(f"QUEUE EMAIL FAILED ({exc}) - the queue file on disk is "
              f"still the delivery")
        return 1
    print(f"queue email sent to {', '.join(recipients)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
