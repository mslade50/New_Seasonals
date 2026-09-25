"""Point-in-time daily inputs; pinned legacy risk series and raw CME minutes."""
from datetime import datetime, timedelta, time
from pathlib import Path
import hashlib
import json
import math
import pandas as pd
import exchange_calendars as xcals
from .strategy import NY, aware

RISK_SERIES = 'legacy_63d_sma10'

def calendar_dates(day):
    day = pd.Timestamp(day).normalize()
    cal = xcals.get_calendar('XNYS', start=day-pd.Timedelta(days=40), end=day+pd.Timedelta(days=10))
    if not cal.is_session(day):
        raise ValueError('Not a cash trading session')
    if cal.session_close(day).tz_convert(NY).time() != time(16):
        raise ValueError('Early cash close: skip this session')
    previous = cal.previous_session(day)
    before = cal.previous_session(previous)
    # Fail closed around intervening weekday holidays: futures may have traded.
    for a, b in [(previous,day),(before,previous)]:
        if any(x.weekday() < 5 for x in pd.date_range(a+pd.Timedelta(days=1), b-pd.Timedelta(days=1))):
            raise ValueError('Holiday-adjacent futures session requires separate validation; skip')
    for d in [previous,before]:
        if cal.session_close(d).tz_convert(NY).time() != time(16):
            raise ValueError('Previous shortened session: skip')
    return previous.date().isoformat(), before.date().isoformat()

def legacy_score(frame, previous):
    if '63d' not in frame or frame.index.has_duplicates:
        raise ValueError('Unique dated legacy 63d risk history required')
    series = frame['63d'].sort_index()
    series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
    if series.index.has_duplicates:
        raise ValueError('Duplicate risk dates')
    series = series.loc[:previous].dropna()
    if len(series) < 10 or str(series.index[-1].date()) != previous:
        raise ValueError('Prior cash-session legacy risk score is missing/stale')
    score = float(series.iloc[-10:].mean())
    if not math.isfinite(score) or not 0 <= score <= 100:
        raise ValueError('Invalid risk score')
    return score

def session_range(frame, day):
    """Raw same-contract minute bars; known legacy 16:15 halt may be absent."""
    start = pd.Timestamp(day, tz=NY)-pd.Timedelta(days=1)+pd.Timedelta(hours=18)
    end = pd.Timestamp(day, tz=NY)+pd.Timedelta(hours=17)
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError('Timezone-aware minute bars required')
    f = frame.sort_index().loc[lambda x: (x.index>=start)&(x.index<end)]
    if f.index.has_duplicates or not len(f):
        raise ValueError('Empty/duplicate futures minutes')
    expected = pd.date_range(start,end-pd.Timedelta(minutes=1),freq='min')
    missing = expected.difference(f.index)
    if any(not time(16,15)<=x.tz_convert(NY).time()<time(16,30) for x in missing):
        raise ValueError(f'Incomplete prior CME session {day}: {len(missing)} missing minutes')
    vals = f[['open','high','low','close']]
    if not vals.map(lambda v: math.isfinite(v) and v>0).all().all():
        raise ValueError('Invalid historical OHLC')
    if ((f.high < f[['open','close','low']].max(axis=1)) | (f.low > f[['open','close','high']].min(axis=1))).any():
        raise ValueError('Inconsistent historical OHLC')
    return float(f.high.max()), float(f.low.min()), float(f.close.iloc[-1])

def build_manifest(config, day, risk_frame, bars, *, now, roll_verified=False):
    previous, before = calendar_dates(day)
    now = aware(now)
    cutoff = datetime.combine(datetime.fromisoformat(day).date(),time(9,30),NY)
    if now >= cutoff or now < datetime.combine(datetime.fromisoformat(day).date(),time(0),NY):
        raise ValueError('Prepare inputs on session date before 09:30 ET')
    if not roll_verified:
        raise ValueError('Explicit same-contract/front-contract review required')
    score = legacy_score(risk_frame,previous)
    markets = {}
    for m in config.markets:
        if m.signal.expiry <= day.replace('-',''):
            raise ValueError('Expired/expiring signal contract')
        high,low,_ = session_range(bars[m.name],previous)
        _,_,close = session_range(bars[m.name],before)
        markets[m.name] = {'prior_tr':max(high-low,abs(high-close),abs(low-close)),
                           'signal_con_id':m.signal.con_id,'execution_con_id':m.execution.con_id}
    body = dict(day=day,prepared_at=now.isoformat(),previous_session=previous,
                config_hash=config.fingerprint,risk_series=RISK_SERIES,score=score,
                roll_verified=True,markets=markets)
    body['hash'] = digest(body)
    return body

def digest(body):
    return hashlib.sha256(json.dumps(body,sort_keys=True,allow_nan=False).encode()).hexdigest()

def load_manifest(path,config):
    body = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    claimed = body.pop('hash')
    if claimed != digest(body) or body['config_hash'] != config.fingerprint:
        raise ValueError('Manifest hash/config mismatch')
    previous,_ = calendar_dates(body['day'])
    if body['risk_series'] != RISK_SERIES or body['previous_session'] != previous or body['roll_verified'] is not True:
        raise ValueError('Wrong risk series, vintage or unreviewed roll')
    prepared = aware(body['prepared_at']).astimezone(NY)
    if prepared.date().isoformat()!=body['day'] or prepared.time()>=time(9,30):
        raise ValueError('Manifest not prepared before this session open')
    if not math.isfinite(body['score']) or not 0<=body['score']<=100:
        raise ValueError('Invalid risk score')
    if set(body['markets']) != {m.name for m in config.markets}:
        raise ValueError('Manifest markets differ')
    for m in config.markets:
        item = body['markets'][m.name]
        if item['signal_con_id']!=m.signal.con_id or item['execution_con_id']!=m.execution.con_id or not math.isfinite(item['prior_tr']) or item['prior_tr']<=0:
            raise ValueError('Invalid market input')
    body['hash']=claimed
    return body
