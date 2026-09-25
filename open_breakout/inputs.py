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

def session_window(day):
    """Full CME session for trade date `day`: 18:00 ET the evening before to 17:00 ET."""
    return (pd.Timestamp(day, tz=NY)-pd.Timedelta(days=1)+pd.Timedelta(hours=18),
            pd.Timestamp(day, tz=NY)+pd.Timedelta(hours=17))

def missing_bars(index, day, bar_minutes=1):
    """Expected bar starts of the full session absent from `index`. A bar lying wholly inside the
    known legacy 16:15-16:30 ET halt may be absent (so every hourly bar is required)."""
    start, end = session_window(day)
    step = pd.Timedelta(minutes=bar_minutes)
    halt = end-pd.Timedelta(minutes=45), end-pd.Timedelta(minutes=30)   # 16:15, 16:30 ET
    missing = pd.date_range(start,end-step,freq=step).difference(index)
    return [x for x in missing if not (halt[0]<=x and x+step<=halt[1])]

def session_range(frame, day):
    """Raw same-contract minute bars; known legacy 16:15 halt may be absent."""
    start, end = session_window(day)
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError('Timezone-aware minute bars required')
    f = frame.sort_index().loc[lambda x: (x.index>=start)&(x.index<end)]
    if f.index.has_duplicates or not len(f):
        raise ValueError('Empty/duplicate futures minutes')
    missing = missing_bars(f.index, day, 1)
    if missing:
        raise ValueError(f'Incomplete prior CME session {day}: {len(missing)} missing minutes')
    vals = f[['open','high','low','close']]
    if not vals.map(lambda v: math.isfinite(v) and v>0).all().all():
        raise ValueError('Invalid historical OHLC')
    if ((f.high < f[['open','close','low']].max(axis=1)) | (f.low > f[['open','close','high']].min(axis=1))).any():
        raise ValueError('Inconsistent historical OHLC')
    return float(f.high.max()), float(f.low.min()), float(f.close.iloc[-1])

# ---- Prior-range filter inputs (prereg docs/prereg_open_breakout_range_filter_2026-09-25.md) ----
ATR_SESSIONS = 20
# Research continuous series (Databento *.v.0): the front for UTC calendar date D is the volume
# leader of the second-most-recent CME trade date strictly before D, so the switch lands at
# 00:00 UTC inside a trade date. Calibrated on IB trade-date volumes for the Sep-2025..Jun-2026
# NQ and ES rolls (10 of 10 switch times reproduced). As in the research, the strictly greater
# volume decides with no tie band, and a lead that moves back to the earlier expiry is followed
# (the instrument change makes those sessions roll sessions with no TR). Only a deciding trade
# date with no strict leader (equal or zero volumes) is ambiguous.
ROLL_LAG_TRADE_DATES = 2
RANGE_SOURCE = 'IB TRADES 1-hour bars, useRTH=False; previous + signal expiry; volume-front continuous, lag 2 trade dates'

def cme_session_dates(index):
    """CME trade date of each bar start: 18:00 ET opens the next date. 17:00-18:00 bars are excluded by the caller."""
    local = index.tz_convert(NY)
    return (local.normalize().tz_localize(None) + pd.to_timedelta((local.hour >= 18).astype(int), unit='D')).date

def _clean_bars(frame):
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError('Timezone-aware bars required')
    f = frame.sort_index()
    if f.index.has_duplicates:
        raise ValueError('Duplicate bars')
    local = f.index.tz_convert(NY)
    f = f[~((local.hour == 17))]
    vals = f[['open','high','low','close']]
    if len(f) and not vals.map(lambda v: math.isfinite(v) and v > 0).all().all():
        raise ValueError('Invalid range-history OHLC')
    return f

def continuous_sessions(contracts, bar_minutes=60):
    """Research-style full-session table (18:00-17:00 ET) from a volume-front continuous series.
    contracts: [{'expiry': 'YYYYMMDD', 'bars': frame}] with open/high/low/close/volume.
    TR is NaN on a session whose instrument differs from the prior session's or that spans two
    instruments (engine.make_sessions roll rule). 'ambiguous' marks sessions whose front came
    from a deciding trade date with no strict volume leader, or from before the first one;
    'missing' counts absent bars under the same rule as the 1-minute prior-session check."""
    contracts = sorted(contracts, key=lambda c: c['expiry'])
    frames, volumes = {}, {}
    for c in contracts:
        f = _clean_bars(c['bars'])
        if not len(f):
            continue
        f = f.assign(session=cme_session_dates(f.index), utc_date=f.index.tz_convert('UTC').date, instrument=c['expiry'])
        frames[c['expiry']] = f
        volumes[c['expiry']] = f.groupby('session').volume.sum()
    if not frames:
        raise ValueError('No range history')
    vol = pd.DataFrame(volumes).fillna(0.).sort_index()
    order = list(vol.columns)
    leader = vol.idxmax(axis=1)
    top = vol.max(axis=1)
    second = vol.apply(lambda r: r.nlargest(2).iloc[-1] if len(r) > 1 else 0., axis=1)
    # Strictly greater volume decides; an exact tie or an all-zero date has no leader.
    tie = (top <= 0) | (top <= second)
    trade_dates = list(vol.index)
    decide = {}
    for d in sorted({x for f in frames.values() for x in f.utc_date}):
        n = sum(1 for t in trade_dates if t < d)
        decide[d] = trade_dates[n - ROLL_LAG_TRADE_DATES] if n >= ROLL_LAG_TRADE_DATES else None
    parts = []
    for expiry, f in frames.items():
        deciding = f.utc_date.map(decide)
        keep = deciding.map(lambda t: t is not None and leader[t] == expiry)
        unknown = deciding.isna()
        g = f[keep | unknown].copy()
        g['unknown'] = unknown[keep | unknown]
        g['ambiguous'] = deciding[keep | unknown].map(lambda t: t is None or bool(tie[t]))
        if expiry != order[0]:
            g = g[~g.unknown]
        parts.append(g)
    bars = pd.concat(parts).sort_index()
    if bars.index.has_duplicates:
        raise ValueError('Overlapping front-contract bars')
    table = bars.groupby('session').agg(high=('high','max'),low=('low','min'),close=('close','last'),
                                        instrument=('instrument','first'),instrument_count=('instrument','nunique'),
                                        bars=('close','size'),ambiguous=('ambiguous','any'),unknown=('unknown','any'))
    missing = {d: len(missing_bars(g.index, str(d), bar_minutes)) for d, g in bars.groupby('session')}
    table['missing'] = [missing[d] for d in table.index]
    table.index = pd.to_datetime(table.index)
    prev_close = table.close.shift()
    tr = pd.concat([table.high-table.low,(table.high-prev_close).abs(),(table.low-prev_close).abs()],axis=1).max(axis=1)
    roll = table.instrument.ne(table.instrument.shift()) | table.instrument_count.ne(1) | table.unknown
    table['tr'] = tr.where(~roll)
    table['roll'] = roll
    return table

def regular_sessions(start, end):
    """XNYS sessions with a 16:00 close: the full 18:00-17:00 CME session must be complete on these."""
    cal = xcals.get_calendar('XNYS', start=pd.Timestamp(start)-pd.Timedelta(days=5), end=pd.Timestamp(end)+pd.Timedelta(days=5))
    days = cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
    return [d.tz_localize(None) if d.tz is not None else d for d in days
            if cal.session_close(d).tz_convert(NY).time() == time(16)]

def prior_range(entry, previous, before, prior_tr, tick):
    """atr20 = mean of the 20 most recent valid full-session TRs strictly before `previous`.
    Raises ValueError on any fail-closed condition."""
    if entry.get('error'):
        raise ValueError(entry['error'])
    signal = max(entry['contracts'], key=lambda c: c['expiry'])
    if not len(signal['bars']):
        raise ValueError('signal-contract range history empty')
    # An earlier expiry that last traded inside the history window must have bars: an empty
    # response there is a failed request, not a contract with zero volume.
    first = signal['bars'].index.min()
    for c in entry['contracts']:
        if c is not signal and not len(c['bars']) and pd.Timestamp(c['expiry'], tz=NY) >= first:
            raise ValueError(f'no range history for expiry {c["expiry"]}, which traded inside the window')
    table = continuous_sessions(entry['contracts'], entry.get('bar_minutes', 60))
    atr20, window, span = atr20_from_table(table, previous)
    prev = pd.Timestamp(previous)
    # Hourly same-contract TR of the previous session must equal the 1-minute prior_tr within a tick.
    same = _clean_bars(signal['bars'])
    same = same.groupby(cme_session_dates(same.index)).agg(high=('high','max'),low=('low','min'),close=('close','last'))
    same.index = pd.to_datetime(same.index)
    if prev not in same.index or pd.Timestamp(before) not in same.index:
        raise ValueError('previous sessions missing from signal-contract range history')
    p, b = same.loc[prev], same.loc[pd.Timestamp(before)]
    hourly_tr = max(p.high-p.low, abs(p.high-b.close), abs(p.low-b.close))
    if abs(hourly_tr - prior_tr) > tick + 1e-9:
        raise ValueError(f'hourly TR {hourly_tr} differs from 1-minute prior_tr {prior_tr} by more than one tick')
    return dict(atr20=atr20, ratio=float(prior_tr/atr20),
                atr20_window=[str(window.index[0].date()), str(window.index[-1].date())],
                roll_excluded=[str(d.date()) for d in span.index[span.tr.isna()]])

def atr20_from_table(table, previous):
    """Mean of the 20 most recent valid TRs strictly before `previous`, with completeness and roll checks."""
    prev = pd.Timestamp(previous)
    before = table[table.index < prev]
    valid = before.tr.dropna()
    if len(valid) < ATR_SESSIONS:
        raise ValueError(f'only {len(valid)} valid prior TRs (need {ATR_SESSIONS})')
    window = valid.iloc[-ATR_SESSIONS:]
    # Sessions feeding the window, including the one supplying the first prev_close.
    first = table.index[table.index.get_loc(window.index[0]) - 1] if table.index.get_loc(window.index[0]) else window.index[0]
    span = before.loc[first:]
    if (span.high < span.low).any():
        raise ValueError('Inconsistent range-history OHLC')
    # Same completeness rule as session_range, on every regular (16:00 close) cash session;
    # shortened holiday/early-close sessions count as they are, as in the research.
    for d in regular_sessions(first, before.index[-1]):
        if d not in table.index:
            raise ValueError(f'missing session {d.date()} in range history')
        if int(table.loc[d, 'missing']):
            raise ValueError(f'incomplete session {d.date()}: {int(table.loc[d, "missing"])} missing bars')
    if span.ambiguous.any():
        raise ValueError(f'ambiguous roll: {[str(d.date()) for d in span.index[span.ambiguous]]}')
    atr20 = float(window.mean())
    if not math.isfinite(atr20) or atr20 <= 0:
        raise ValueError('Invalid atr20')
    return atr20, window, span

def range_decision(filt, info):
    """Manifest fields for one market. filt: RangeFilter or None; info: prior_range() dict or an error string."""
    out = dict(prior_range_status='UNAVAILABLE', atr20=None, ratio=None, skip_prior_range=False,
               half_prior_range=False, prior_range_reason=None)
    if isinstance(info, dict):
        out.update(prior_range_status='OK', atr20=info['atr20'], ratio=info['ratio'],
                   atr20_window=info['atr20_window'], roll_excluded=info['roll_excluded'])
    else:
        out['prior_range_reason'] = f'UNAVAILABLE: {info}'
    if filt is None or not filt.enabled:
        return out
    if out['prior_range_status'] != 'OK':
        out['skip_prior_range'] = True   # fail closed in both modes
    elif out['ratio'] >= filt.threshold:
        key = 'skip_prior_range' if filt.mode == 'skip' else 'half_prior_range'
        out[key] = True
        out['prior_range_reason'] = f'ratio {out["ratio"]:.4f} >= threshold {filt.threshold}'
    return out

def build_manifest(config, day, risk_frame, bars, *, now, roll_verified=False, range_bars=None, range_error=None):
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
        prior_tr = max(high-low,abs(high-close),abs(low-close))
        markets[m.name] = {'prior_tr':prior_tr,
                           'signal_con_id':m.signal.con_id,'execution_con_id':m.execution.con_id}
        if range_bars is None or m.name not in range_bars:
            info = f'no range history ({range_error or "not fetched"})'
        else:
            try:
                info = prior_range(range_bars[m.name],previous,before,prior_tr,m.signal.tick)
            except Exception as exc:
                # Fail closed per market: recorded, and skipped when the filter is enabled.
                info = f'{type(exc).__name__}: {exc}'
        markets[m.name].update(range_decision(config.prior_range_filter,info))
    filt = config.prior_range_filter
    body = dict(day=day,prepared_at=now.isoformat(),previous_session=previous,
                config_hash=config.fingerprint,risk_series=RISK_SERIES,score=score,
                roll_verified=True,markets=markets,
                prior_range_filter=None if filt is None else dict(enabled=filt.enabled,threshold=filt.threshold,mode=filt.mode),
                range_source=RANGE_SOURCE)
    body['hash'] = digest(body)
    return body

def digest(body):
    return hashlib.sha256(json.dumps(body,sort_keys=True,allow_nan=False).encode()).hexdigest()

def check_range_fields(config,item):
    """A disabled filter never skips or halves; an enabled one needs a consistent, fail-closed decision."""
    skip,half = item.get('skip_prior_range',False),item.get('half_prior_range',False)
    if not isinstance(skip,bool) or not isinstance(half,bool):
        raise ValueError('Invalid prior-range flags')
    filt = config.prior_range_filter
    if filt is None or not filt.enabled:
        if skip or half:
            raise ValueError('Prior-range action in a manifest for a config with the filter disabled')
        return
    status = item.get('prior_range_status')
    if status == 'OK':
        ratio = item.get('ratio')
        if not isinstance(ratio,(int,float)) or not math.isfinite(ratio) or ratio <= 0:
            raise ValueError('Invalid prior-range ratio')
        hit = ratio >= filt.threshold
        if (skip,half) != ((hit,False) if filt.mode=='skip' else (False,hit)):
            raise ValueError('Prior-range flags inconsistent with ratio and threshold')
    elif status == 'UNAVAILABLE':
        if not skip:
            raise ValueError('Unavailable prior range must skip the market when the filter is enabled')
    else:
        raise ValueError('Prior-range status missing with the filter enabled')

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
        check_range_fields(config,item)
    body['hash']=claimed
    return body
