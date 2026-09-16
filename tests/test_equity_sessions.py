import pandas as pd
import pytest
from equity_sessions import near_close,last_settled_session,primary_olv_deadlines


@pytest.mark.parametrize('day,expected',[
    ('2026-09-09','15:59:00'),('2026-11-27','12:59:00'),
    ('2026-12-24','12:59:00'),('2027-11-26','12:59:00'),
    ('2028-07-03','12:59:00')])
def test_exchange_deadlines(day,expected):
    assert near_close(day).strftime('%H:%M:%S')==expected


def test_closed_day_cannot_become_a_market_deadline():
    with pytest.raises(ValueError):near_close('2026-11-26')


def test_early_close_bar_is_settled_after_13_et_not_before():
    assert str(last_settled_session('2026-11-27T12:59:00-05:00').date())=='2026-11-25'
    assert str(last_settled_session('2026-11-27T13:01:00-05:00').date())=='2026-11-27'


def test_primary_olv_only_and_input_frame_preserved():
    frame=pd.DataFrame([
        dict(Strategy_Ref='Oversold Low Volume',Exit_Condition_Time='2026-11-27 15:59:00',Entry_Expire_Time='2026-11-25 15:59:00'),
        dict(Strategy_Ref='Other',Exit_Condition_Time='2026-11-27 15:59:00',Entry_Expire_Time='2026-11-25 15:59:00')])
    result=primary_olv_deadlines(frame)
    assert result.loc[0,'Exit_Condition_Time']=='2026-11-27 12:59:00'
    assert result.loc[1,'Exit_Condition_Time']==frame.loc[1,'Exit_Condition_Time']
    assert frame.loc[0,'Exit_Condition_Time']=='2026-11-27 15:59:00'
