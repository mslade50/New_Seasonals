"""One authorized email; preserve a receipt and do not silently resend.

Adapted from send_v3_report.py, which in turn came from the 17 September senders.
The run-once guard is keyed to the v3b receipt, so last night's superseded v3
receipt does not block this send and this one cannot be repeated.
Credentials come from the workspace .env or the environment, never from this file.
"""
from pathlib import Path
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import make_msgid, formatdate
import hashlib
import json
import os
import smtplib
import ssl
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
RECEIPT = ROOT / 'email_receipt_v3b.json'
DELIVERY = ROOT / 'delivery_receipt_v3b.json'
TO = 'mckinleyslade@gmail.com'
SUBJECT = ('Denali risk dial report v3b: rebuilt on the Sep 17 team introduction '
           '(5/10/21d), NYSE EMA5, dual filter outcome')

BODY = """McKinley,

This supersedes the v3 email I sent last night. That one was rebuilt from the wrong base: the 07:20 detailed report, not the 16:49 team introduction Denali actually has. Please ignore it.

v3b is rebuilt on the team introduction and keeps its structure, its tone and its 5/10/21-day framing exactly. Every table is recomputed on the production dial through 18 September, so the cohort moves from 1,792 dates to 1,790 and the numbers shift slightly.

Two things changed since that version. The NYSE component now arms and resets on the five-session EMA shipped on 18 September, which is edited into the existing NYSE section rather than added as a new one; that alone moves its 5- and 10-day averages from above the all-date comparison to below it, on 73 flagged dates instead of 111. And the dual filter we proposed alongside the introduction has been replayed and is not adopted, so the report carries a short section saying so in plain terms. I did not re-send the proposal document.

Attached: the detailed report and the one-page brief, each in Word and PDF.

Codex
"""

if RECEIPT.exists():
    raise SystemExit('Existing v3b email receipt found. Inspect status before any further send.')
verification = json.loads(DELIVERY.read_text())
config = dotenv_values(WORKSPACE / '.env')
sender = os.environ.get('EMAIL_USER') or config.get('EMAIL_USER')
password = os.environ.get('EMAIL_PASS') or config.get('EMAIL_PASS')
if not sender or not password:
    raise SystemExit('Configured email credentials are unavailable; no email sent.')

msg = EmailMessage()
msg['From'] = sender
msg['To'] = TO
msg['Subject'] = SUBJECT
msg['Date'] = formatdate(localtime=True)
msg['Message-ID'] = make_msgid()
msg.set_content(BODY)

attachments = {}
for name, expected in verification['files'].items():
    data = (ROOT / name).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected:
        raise SystemExit(f'Final-file verification failed for {name}; no email sent.')
    subtype = ('pdf' if name.endswith('.pdf')
               else 'vnd.openxmlformats-officedocument.wordprocessingml.document')
    msg.add_attachment(data, maintype='application', subtype=subtype, filename=name)
    attachments[name] = {'sha256': digest, 'bytes': len(data)}

receipt = {'recipient': TO, 'subject': SUBJECT, 'message_id': str(msg['Message-ID']),
           'status': 'prepared', 'supersedes_email_receipt': 'email_receipt.json',
           'attachments': attachments}


def save():
    RECEIPT.write_text(json.dumps(receipt, indent=2), encoding='utf-8')


save()
try:
    with smtplib.SMTP('smtp.gmail.com', 587, timeout=45) as smtp:
        smtp.ehlo()
        smtp.starttls(context=ssl.create_default_context())
        smtp.ehlo()
        smtp.login(sender, password)
        receipt['status'] = 'sending'
        save()
        refused = smtp.send_message(msg, from_addr=sender, to_addrs=[TO])
        if refused:
            receipt['status'] = 'recipient_refused'
            save()
            raise RuntimeError('Recipient was refused by SMTP')
        receipt['status'] = 'accepted_by_smtp'
        receipt['accepted_at_utc'] = datetime.now(timezone.utc).isoformat()
        save()
except Exception as exc:
    receipt['exception_type'] = type(exc).__name__
    save()
    raise SystemExit(f'Email attempt ended with {type(exc).__name__}; receipt status: '
                     f'{receipt["status"]}. Do not automatically resend.')

verification['email_status'] = 'Accepted by Gmail SMTP for ' + TO
verification['email_receipt'] = RECEIPT.name
DELIVERY.write_text(json.dumps(verification, indent=2), encoding='utf-8')
print(json.dumps({'status': receipt['status'], 'recipient': TO,
                  'attachments': len(attachments), 'message_id': receipt['message_id']}))
