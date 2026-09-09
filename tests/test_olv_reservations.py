"""OLV submitted-order contracts, with the real engine and synthetic bars."""
import pytest
from tests.test_olv_stop_and_cap import _frame, _run, _olv_strategy
from pages.strat_backtester import _apply_daily_risk_scale
import pandas as pd


def strategy(**overrides):
    return _olv_strategy(target_anchor='submitted_limit', **overrides)


def test_gap_fill_keeps_submitted_target_and_fill_based_loss_threshold():
    dates, df = _frame(highs={2:103.0, 3:104.6})
    df.loc[dates[1], ['Open','Low','Close']] = [97.0,96.5,98.0]
    row = _run(df, strategy()).iloc[0]
    assert row['Price'] == 97
    assert row['Exit Price'] == 104.5
    assert row['Exit Date'] == dates[3]
    # A close above the actual-fill loss threshold cannot confirm a stop.
    df.loc[dates[2], ['High','Low','Close','Volume']] = [100,95,96,2_000_000]
    assert _run(df, strategy()).iloc[0]['Exit Type'] == 'Target'


def test_pending_limits_reserve_capacity_even_when_not_yet_filled():
    dates, df = _frame(n=18)
    df['Low'] = 100
    df.loc[dates[3], 'Low'] = 99
    rows = _run(df, strategy(risk_bps=50,
        ticker_notional_cap={'pct_nav':.2,'exempt':[], 'include_pending':True}), extra_candidates=[1,2])
    assert (rows['Shares'] * rows['Price']).sum() <= 20_000
    assert rows.Shares.sum() == 201


def test_expired_unfilled_limit_releases_capacity():
    dates, df = _frame(n=20)
    df['Low'] = 100
    df.loc[dates[5], 'Low'] = 99
    rows = _run(df, strategy(risk_bps=50,
        ticker_notional_cap={'pct_nav':.2,'exempt':[], 'include_pending':True}), extra_candidates=[4])
    assert len(rows) == 1 and rows.iloc[0]['Shares'] == 200


def test_nonbinding_capacity_preserves_staged_risk_remainder():
    _, df = _frame()
    row = _run(df, strategy(risk_bps=35.1,
        ticker_notional_cap={'pct_nav':.5,'exempt':[], 'include_pending':True})).iloc[0]
    assert row['Shares'] == 140
    assert row['Risk $'] == pytest.approx(351)


def test_etf_exemption_keeps_all_pending_orders():
    dates, df = _frame(n=18)
    df['Low'] = 100
    df.loc[dates[3], 'Low'] = 99
    rows = _run(df, strategy(risk_bps=50,
        ticker_notional_cap={'pct_nav':.2,'exempt':['TEST'], 'include_pending':True}), extra_candidates=[1,2])
    assert len(rows)==3 and rows.Shares.sum()==600


def test_no_same_day_target_credit_after_anchor_change():
    dates, df=_frame(highs={1:110,3:105})
    rows=_run(df,strategy())
    assert rows.iloc[0]['Exit Date']==dates[3]


def test_live_floor_and_pnl_from_final_quantity():
    rows = pd.DataFrame([{'Strategy':'Oversold Low Volume','Shares':203,
        'PnL':1015.,'Risk $':507.5,'Price':99.5,'Exit Price':104.5,'Action':'BUY','_Sizing ID':'fixture'}])
    _apply_daily_risk_scale(rows, rows.index, .625)
    assert rows.iloc[0]['Shares'] == 126
    assert rows.iloc[0]['PnL'] == 630


def test_open_position_payload_uses_engine_target_after_gap(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    from scripts import build_site
    from scripts.build_trade_ledger import shape_flat_trades
    dates, df = _frame()
    df.loc[dates[1], ['Open','Low','Close']] = [97.,96.5,98.]
    rows = _run(df, strategy())
    monkeypatch.setattr(build_site, 'open_mask', lambda frame: pd.Series(True,index=frame.index))
    monkeypatch.setattr(build_site, 'load_sector_map', lambda: {})
    monkeypatch.setattr(build_site, 'strategy_exec_map', lambda: {})
    position = build_site.build_positions(shape_flat_trades(rows), {'TEST':df})['positions'][0]
    assert position['Tgt_Price'] == 104.5
    assert position['Stop_Price'] == 94.5
    from scripts.signal_chart_common import trade_geometry, chart_relpath
    assert trade_geometry(shape_flat_trades(rows).iloc[0], df)['tgt_px'] == 104.5
    assert chart_relpath('Oversold Low Volume','TEST',dates[0]) == 'signals/Oversold_Low_Volume_submitted_target_v1/TEST_20240102.png'
    assert chart_relpath('Other','TEST',dates[0]) == 'signals/Other/TEST_20240102.png'
    with pytest.raises(ValueError, match='OLV target missing'):
        build_site.build_positions(shape_flat_trades(rows).drop(columns='Target Price'), {'TEST':df})
