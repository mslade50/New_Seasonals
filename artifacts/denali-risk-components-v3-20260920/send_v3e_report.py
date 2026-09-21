"""One authorized email; preserve a receipt and do not silently resend.

Adapted from send_v3c_report.py. The run-once guard is keyed to the v3e receipt.
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
RECEIPT = ROOT / 'email_receipt_v3e.json'
DELIVERY = ROOT / 'delivery_receipt_v3e.json'
TO = 'mckinleyslade@gmail.com'
SUBJECT = 'Denali Risk Dial report v3e: 55 warning line, near-high validation section'

BODY = """McKinley,

v3e of the Denali Risk Dial report. This replaces v3c from earlier today.

Warning line moved from 50 to 55, the 85th percentile of readings over the 2002-2026 window (915 of 6,129 dates, 38 episodes). Bands are now 0-20, 20-40, 40-55, 55-65, 65-80 and 80+, so none straddles the line. The text says plainly that the 55 to 65 band still carries positive averages and that all three horizons turn negative from 65.

New section "When the dial matters most": the same tables restricted to the 3,032 dates where SPY closed within 2% of its trailing 252-session high, which is where the components are built to fire. On those dates the 55 split is negative at all three horizons (55 or above: -0.11%, -0.08%, -0.10% on 698 dates against +0.19%, +0.39%, +0.73% below), and the off-high contrast is stated in one sentence. Note the 80+ near-high band is open-ended; the dial runs above 100 on 23 of those dates and they mean-reverted hard, which is why that row reads -0.84% at 21 days rather than the -1.33% a closed 80-100 band gives.

Detailed report is four pages, one-pager one page. Attached in Word and PDF.

Claude
"""

if RECEIPT.exists():
    raise SystemExit('Existing v3e email receipt found. Inspect status before any further send.')
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
           'status': 'prepared', 'supersedes_email_receipt': 'email_receipt_v3c.json',
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
