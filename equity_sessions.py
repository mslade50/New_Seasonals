"""NYSE cash-equity session times shared by scanner and broker candidates."""
from functools import lru_cache
import pandas as pd


@lru_cache(maxsize=1)
def calendar():
    import exchange_calendars as xcals
    return xcals.get_calendar('XNYS',start='1990-01-01',end='2040-12-31')


def session_close(day):
    """Aware ET close. Closed/non-supported dates raise; never guess 16:00."""
    stamp=pd.Timestamp(day)
    if stamp.tzinfo is not None:
        stamp=stamp.tz_convert('America/New_York').tz_localize(None)
    return calendar().session_close(stamp.normalize()).tz_convert('America/New_York')


def near_close(day, *, seconds=60):
    if not isinstance(seconds,int) or seconds<0 or seconds>=60*60:
        raise ValueError('near-close offset must be 0..3599 seconds')
    return session_close(day)-pd.Timedelta(seconds=seconds)


def last_settled_session(now):
    stamp=pd.Timestamp(now)
    if stamp.tzinfo is None:
        raise ValueError('valuation time must include timezone')
    local=stamp.tz_convert('America/New_York')
    day=local.tz_localize(None).normalize()
    cal=calendar()
    prior=cal.date_to_session(day,direction='previous')
    if cal.session_close(prior)>stamp:
        prior=cal.previous_session(prior)
    return pd.Timestamp(prior).tz_localize(None).normalize()


def primary_olv_deadlines(frame):
    """Correct only Primary OLV deadlines after the PA frame was separated."""
    result=frame.copy(deep=True)
    for index,row in result.iterrows():
        if row.get('Strategy_Ref')!='Oversold Low Volume':
            continue
        for field in ('Exit_Condition_Time','Entry_Expire_Time'):
            value=str(row.get(field) or '').strip()
            if value in {'','ERROR','MANUAL_EXIT','nan'}:
                raise ValueError(f'OLV {field} is unavailable')
            result.at[index,field]=near_close(value.split()[0]).strftime('%Y-%m-%d %H:%M:%S')
    return result
