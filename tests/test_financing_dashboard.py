import pytest
from fundamental.financing_dashboard import classify,price_exclusions

AS_OF='2026-09-24T17:00:00+00:00'


def price():
    return dict(ticker='ABC',cik=1,price_status='complete',current_identity_verified=True,
        listing_warning='',split_recent=False,history_sessions=260,close=10,
        dollar_volume_20=20e6,median_dollar_volume_20=15e6,return_20=.2,return_60=.5,
        relative_60=.3,distance_high_252=-.1,above_sma50=True,above_sma200=True,price_as_of='2026-09-23')


def financial():
    return dict(status='calculated',balance_date='2026-06-30',filing_accepted_at='2026-08-01T12:00:00Z',
        monthly_burn_6m=10e6,runway_6m=10,reported_liquidity=100e6)


def review():
    return dict(cik=1,as_of=AS_OF,balance_date='2026-06-30',liquidity_verified=True,
        newer_releases_checked=True,financing_checked=True,reported_liquidity=100e6,
        sources=[dict(url='https://www.sec.gov/example')])


def run(p=None,f=None,r=None):
    return classify(p or price(),f if f is not None else financial(),r if r is not None else review(),as_of=AS_OF,price_session='2026-09-23')


def test_verified_healthy_name_is_watch_only_without_probability():
    result=run()
    assert result['bucket']=='watchlist'
    assert 'signal' not in result and 'offering_probability' not in result


@pytest.mark.parametrize('updates',[
    {'return_60':-.74},{'above_sma200':False},{'distance_high_252':-.3},
    {'dollar_volume_20':9e6},{'median_dollar_volume_20':2e6},
    {'split_recent':True},{'current_identity_verified':False},
    {'return_60':8.3},{'history_sessions':120},{'price_status':'unavailable'},
    {'price_as_of':'2026-09-22'},{'relative_60':None}])
def test_damaged_illiquid_extreme_or_stale_names_cannot_surface(updates):
    p=price();p.update(updates)
    assert run(p=p)['bucket']=='excluded'


def test_fresh_rally_flag_cannot_bypass_sustained_strength():
    p=price();p.update(setups=['Fresh rally'],return_60=-.1,return_20=.05)
    assert run(p=p)['bucket']=='excluded'


@pytest.mark.parametrize('update',[
    {'as_of':'2026-09-22T12:00:00Z'},{'cik':2},{'balance_date':'2026-03-31'},
    {'liquidity_verified':False},{'financing_checked':False},{'newer_releases_checked':False},
    {'reported_liquidity':50e6},{'sources':[]},{'post_balance_financing':True},
    {'newer_liquidity_disclosure':True},{'unresolved':['Cash restricted']}])
def test_missing_or_conflicting_reviews_stay_out_of_main_list(update):
    r=review();r.update(update)
    assert run(r=r)['bucket']=='review'


def test_cash_only_calculation_needs_a_new_review_not_an_assumed_zero():
    assert run(r={})['bucket']=='review'


def test_current_investments_can_eliminate_funding_flag():
    f=financial();f.update(reported_liquidity=400e6,runway_6m=40)
    assert run(f=f)['bucket']=='excluded'


def test_capex_only_burn_does_not_qualify():
    f=financial();f.update(monthly_burn_6m=0,runway_6m=None,runway_with_capex_6m=5)
    assert run(f=f)['bucket']=='excluded'


@pytest.mark.parametrize('update',[{'status':'unavailable'},{'balance_date':'2025-12-31'},{'filing_accepted_at':'2026-09-25T01:00:00Z'}])
def test_unknown_stale_or_future_financials_do_not_surface(update):
    f=financial();f.update(update)
    assert run(f=f)['bucket']=='review'


def test_embedded_payload_cannot_close_script_element():
    import json
    from scripts.build_financing_dashboard import encode_payload
    value={'name':'</script><script>alert(1)</script>','note':'A & B'}
    encoded=encode_payload(value)
    assert '</script>' not in encoded
    assert json.loads(encoded)==value
