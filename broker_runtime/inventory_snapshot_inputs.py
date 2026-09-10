"""Read-only provenance for the inventory feed; never changes TWS or orders."""
import csv
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo


def query_start(stamp, policy_path, account):
    """No requested lookback is treated as proof of a configured TWS setting.

    The reviewed policy pins the exact TWS settings file, timezone and broker
    account. With no seven-day proof, coverage is current-day only. An absent
    policy cannot attest even the account's midnight/timezone boundary.
    """
    import xml.etree.ElementTree as ET
    if stamp.tzinfo is None:raise ValueError('execution observation must include timezone')
    policy=json.loads(Path(policy_path).read_text(encoding='utf-8-sig'))
    if policy.get('broker_account')!=account or policy.get('review',{}).get('status')!='approved':
        raise ValueError('execution-history account/timezone policy is unreviewed')
    zone=ZoneInfo(policy['timezone'])
    days=1
    if policy.get('settings_path'):
        tree=ET.parse(policy['settings_path'])
        settings=[int(e.attrib['tradeLogShowLastNDays']) for e in tree.iter() if 'tradeLogShowLastNDays' in e.attrib]
        if not settings or min(settings)<int(policy.get('lookback_days',1)):
            raise ValueError('TWS execution lookback no longer matches reviewed settings')
        days=int(policy.get('lookback_days',1))
    if days not in range(1,8):raise ValueError('unsupported execution lookback')
    local=stamp.astimezone(zone)
    start=local.replace(hour=0,minute=0,second=0,microsecond=0)-dt.timedelta(days=days-1)
    return start.astimezone(dt.timezone.utc).isoformat()


def entry_metadata(path, account_snapshot):
    """Freeze OLV input metadata only when current broker exits match the CSV."""
    raw=Path(path).read_bytes()
    rows=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    orders=account_snapshot['orders']
    result={}
    for row in rows:
        if row.get('Strategy_Ref')!='Oversold Low Volume':continue
        symbol=row['Symbol'].upper();day=row['Staged_Date']
        dt.date.fromisoformat(day)
        ref=f'{symbol}|BUY|Oversold Low Volume|{day}'
        legs=[o for o in orders if o.get('order_ref')==ref and o.get('action')=='SELL'
              and o.get('account')==account_snapshot['broker_account']]
        if not legs:continue
        atr=float(row['Used_ATR']);target=float(row['Target_Price'])
        if not math.isfinite(atr) or atr<=0 or not math.isfinite(target) or target<=0:
            raise ValueError('OLV input ATR/target is invalid')
        targets=[o for o in legs if o.get('order_type')=='LMT' and float(o.get('lmt') or 0)==target]
        times=[o for o in legs if o.get('order_type')=='MKT' and o.get('good_after')]
        if (len(targets)!=1 or len(times)!=1 or not targets[0].get('oca_group')
                or targets[0].get('oca_group')!=times[0].get('oca_group')
                or targets[0].get('con_id')!=times[0].get('con_id')
                or any(o.get('sec_type')!='STK' or o.get('currency')!='USD' for o in (targets[0],times[0]))):
            raise ValueError('OLV input does not match one actual exit bracket')
        time=times[0]['good_after'].split()
        if len(time)!=3 or time[2] not in {'US/Eastern','America/New_York'}:
            raise ValueError('OLV broker deadline timezone is unavailable')
        deadline=dt.datetime.strptime(' '.join(time[:2]),'%Y%m%d %H:%M:%S').replace(tzinfo=ZoneInfo(time[2]))
        if str(deadline.date())!=row['Exit_Condition_Time'].split()[0]:
            raise ValueError('OLV staged and actual deadlines disagree')
        value=dict(atr=atr,exit_deadline_utc=deadline.astimezone(dt.timezone.utc).isoformat(),
                   exit_protocol='TIME',metadata_provenance='broker exits matched to staged execution CSV',
                   metadata_con_id=int(times[0]['con_id']),
                   metadata_source_sha256=hashlib.sha256(raw).hexdigest())
        if ref in result and result[ref]!=value:raise ValueError('conflicting OLV entry metadata')
        result[ref]=value
    return result
