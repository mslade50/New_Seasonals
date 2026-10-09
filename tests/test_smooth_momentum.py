import numpy as np
import pandas as pd

from indicators import calculate_indicators, smooth_momentum_features
from scripts.backtest_smooth_momentum import definitions, signal_selection, marks, research_bars


def prices(n=400):
    index=pd.bdate_range('2005-01-03',periods=n)
    market_returns=.002+.001*np.sin(np.arange(n)/4)
    market=pd.Series(100*np.cumprod(1+market_returns),index=index)
    close=pd.Series(100*np.cumprod(1+2*market_returns),index=index)
    raw=pd.DataFrame({'Open':close,'High':close+1,'Low':close-1,'Close':close,'Volume':1_000_000.},index=index)
    return raw,market,pd.Series(6.5,index=index)


def test_beta_and_path_efficiency_have_independent_known_values():
    raw,market,hours=prices()
    f=smooth_momentum_features(calculate_indicators(raw,{},'X'),market,hours)
    np.testing.assert_allclose(f.beta126.iloc[126:],2.,atol=1e-8)
    np.testing.assert_allclose(f.efficiency63.iloc[63:],1.,atol=1e-12)
    assert f.beta126.iloc[:126].isna().all()


def test_smooth_features_are_prefix_causal_and_missing_prices_are_unavailable():
    raw,market,hours=prices()
    full=smooth_momentum_features(calculate_indicators(raw,{},'X'),market,hours)
    prefix=smooth_momentum_features(calculate_indicators(raw.iloc[:300],{},'X'),market.iloc[:300],hours.iloc[:300])
    pd.testing.assert_frame_equal(prefix,full.iloc[:300])
    raw.loc[raw.index[280],'Close']=np.nan
    bad=smooth_momentum_features(calculate_indicators(raw,{},'X'),market,hours)
    assert np.isnan(bad.beta126.iloc[281]) and np.isnan(bad.efficiency63.iloc[281])


def test_weekly_election_is_current_date_only_and_breaks_ties_by_ticker():
    f=pd.DataFrame({'ticker':['B','A','C'],'signal_date':pd.to_datetime(['2020-01-03']*3),
                    'trend_stack':True,'beta126':1.5,'efficiency63':[.3,.3,.4],
                    'largest_step_share63':.1,'momentum126_skip21':[.3,.3,.15],
                    'relative_return126':.1,'dollar_volume63':30_000_000.,'signal_close':100.,
                    'market_above200':True,'weekly':True})
    p={'family':'weekly','top':1,'efficiency':.2,'momentum':.15,'market':False}
    assert signal_selection(f,p).tolist()==[False,True,False]
    later=f.iloc[:1].copy(); later.signal_date=pd.Timestamp('2020-01-10'); later.efficiency63=.99
    combined=pd.concat([f,later],ignore_index=True)
    assert signal_selection(combined,p)[:3].tolist()==[False,True,False]
    assert len(definitions())==64


def test_variable_exit_marks_include_exit_price_and_both_costs():
    sessions=pd.bdate_range('2026-01-05',periods=3)
    m=pd.DataFrame({'entry':[100.],'atr':[2.]})
    p={'Close':np.array([[101.,102.,103.]]),'date':np.array([sessions.to_numpy()])}
    r=pd.DataFrame({'exit':[104.],'exit_day':[1],'net_atr':[(104*.9995-100*1.0005)/2]})
    pnl,live=marks(m,r,p,sessions)
    np.testing.assert_allclose(pnl,[.475,1.474,0.],atol=1e-10)
    assert live.tolist()==[True,True,False]


def test_missing_session_remains_unavailable_and_closure_is_not_a_bar():
    # Jan 9, 2025 is the Carter mourning closure. Jan 10 is a missing feed bar.
    sessions=pd.to_datetime(['2025-01-08','2025-01-10','2025-01-13'])
    g=pd.DataFrame({'date':pd.to_datetime(['2025-01-08','2025-01-09','2025-01-13']),
                    'ticker':'X','Open':100.,'High':101.,'Low':99.,'Close':100.,'Volume':1000.})
    df,valid=research_bars(g,pd.Series(6.5,index=sessions))
    assert df.index.tolist()==sessions.tolist()
    assert valid.tolist()==[True,False,True]
    assert df.loc['2025-01-10',['Open','High','Low','Close']].isna().all()
