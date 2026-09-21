"""One authorized email; preserve a receipt and do not silently resend.

Adapted from send_v3b_report.py. The run-once guard is keyed to the v3c receipt.
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
RECEIPT = ROOT / 'email_receipt_v3c.json'
DELIVERY = ROOT / 'delivery_receipt_v3c.json'
TO = 'mckinleyslade@gmail.com'
SUBJECT = 'Denali Risk Dial report v3c: full 2002-2026 study window, cosmetic edits'

BODY = """McKinley,

v3c of the Denali Risk Dial report, rebuilt from v3b with the changes you asked for this morning.

Study window: the tables now run 8 February 2002 to 18 June 2026, 6,129 dates, against 1,790 in v3b. The 2019 start was an inherited constant, not a data limit. The binding component is Low Absorption Ratio, which needs a 504-session percentile lookback on top of the SPY cache that begins in January 2000. NYSE breadth runs from 2000 and covers the whole window except two June 2020 readings, so the NYSE table uses 6,043 dates. Over the 1,790 dates v3b covered, the dial series is identical.

Cosmetics: title is "Denali Risk Dial", the SPY explainer paragraph is gone, NYSE Net Highs is one explainer paragraph plus its table with no change history, and the dual filter section is removed. The detailed report is three pages.

One narrative change to know about. On the long window the 50-and-above group is no longer negative across all three horizons (0.00% / +0.06% / +0.41%), so the "negative across all three windows" sentences from v3b were rewritten. The deterioration now reads as building with the reading: flat at 50 to 60, negative one- and two-week averages from 60 to 80, and negative at all three horizons at 80+ (149 dates).

Attached: the detailed report and the one-page brief, each in Word and PDF.

Claude
"""

if RECEIPT.exists():
    raise SystemExit('Existing v3c email receipt found. Inspect status before any further send.')
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
           'status': 'prepared', 'supersedes_email_receipt': 'email_receipt_v3b.json',
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

verification['sent'] = True
verification['email_status'] = 'Accepted by Gmail SMTP for ' + TO
verification['email_receipt'] = RECEIPT.name
DELIVERY.write_text(json.dumps(verification, indent=2), encoding='utf-8')
print(json.dumps({'status': receipt['status'], 'recipient': TO,
                  'attachments': len(attachments), 'message_id': receipt['message_id']}))
