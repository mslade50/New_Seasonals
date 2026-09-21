"""One authorized email; preserve a receipt and do not silently resend.

Adapted from artifacts/denali-risk-components-v2-20260917/send_revised_detailed_report.py.
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
RECEIPT = ROOT / 'email_receipt.json'
TO = 'mckinleyslade@gmail.com'
SUBJECT = 'Denali risk dial report v3: single main dial, NYSE breadth component'

BODY = """McKinley,

Version 3 of the Denali risk dial report is attached in Word and PDF, with the one-page brief rebuilt to match.

What changed since the 17 September version: the report now describes a single main dial, so the 5- and 21-session dial sections and their forward-return tables are gone. Pre FOMC Rally is out of the component set after its retirement on 17 September. Equity put/call stays on the signal board, and the text says plainly that it now contributes to no displayed dial rather than dropping it quietly.

A new section covers the NYSE net new highs recovery-reset component in the EMA5 form shipped on 18 September, including the smoothing evidence and the caveat that the change bought less flicker rather than forecasting power.

The dial-range tables are recomputed on the production main dial series at 5, 10, 21, 42 and 63 sessions, and every table in the report now shares one price source and one all-date baseline. Nothing strategy-specific appears in the document; it keeps the same vocabulary as the shared site.

Codex
"""

if RECEIPT.exists():
    raise SystemExit('Existing email receipt found. Inspect status before any further send.')
verification = json.loads((ROOT / 'delivery_receipt.json').read_text())
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
           'status': 'prepared', 'attachments': attachments}


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
(ROOT / 'delivery_receipt.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
print(json.dumps({'status': receipt['status'], 'recipient': TO,
                  'attachments': len(attachments), 'message_id': receipt['message_id']}))
