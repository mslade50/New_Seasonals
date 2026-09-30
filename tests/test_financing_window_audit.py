import pytest
from fundamental.financing_window_audit import audit_outcome, validate_window


def window():
    return dict(window_id="1-2025-06-30",cik=1,session="2025-06-30",as_of="2025-06-30T20:30:00Z",outcome_end="2025-08-29",filings=["one"])


def review():
    return dict(sec_inventory_reviewed=True,exhibits_reviewed=True,financial_notes_reviewed=True,ir_inventory_reviewed=True,ir_sources=["https://issuer/news"],evidence_sources=["https://sec/one"])


def docs():
    return [dict(window_ids=["1-2025-06-30"],accession="one",file_type="8-K")]


def event(**kw):
    return dict(dict(cik=1,event_id="new",status="verified",announcement_date="2025-07-01",endpoint_class="primary_equity",sources=["https://issuer/new"]),**kw)


def test_no_event_requires_complete_review_and_capture():
    assert audit_outcome(window(),review(),docs(),[])["outcome"]==0
    for key in ["sec_inventory_reviewed","exhibits_reviewed","financial_notes_reviewed","ir_inventory_reviewed"]:
        r=review();r[key]=False
        assert audit_outcome(window(),r,docs(),[])["outcome"] is None
    assert audit_outcome(window(),review(),[],[])["outcome"] is None
    bad=docs();bad.append(dict(window_ids=[window()["window_id"]],accession="one",file_type="linked_exhibit",error="failed"))
    assert audit_outcome(window(),review(),bad,[])["outcome"] is None


def test_verified_positive_does_not_need_complete_negative_search():
    assert audit_outcome(window(),{},[],[event()])["outcome"]==1
    assert audit_outcome(window(),{},[],[event(status="provisional")])["outcome"] is None


def test_boundary_and_duplicate_stages_are_not_extra_events():
    assert audit_outcome(window(),review(),docs(),[event(announcement_date="2025-06-30",announcement_at="2025-06-30T20:02:00Z")])["outcome"]==0
    assert audit_outcome(window(),review(),docs(),[event(announcement_date="2025-06-30")])["outcome"] is None
    assert audit_outcome(window(),review(),docs(),[event(announcement_date="2025-08-29")])["outcome"]==1
    assert audit_outcome(window(),review(),docs(),[event(announcement_date="2025-08-30")])["outcome"]==0
    with pytest.raises(ValueError,match="Duplicate"):
        audit_outcome(window(),review(),docs(),[event(),event()])


def test_mixed_financing_is_visible_without_silently_changing_endpoint():
    got=audit_outcome(window(),review(),docs(),[event(endpoint_class="mixed_debt_equity")])
    assert got["outcome"] is None and got["outcome_status"]=="mixed_financing_boundary"
    assert got["inclusive_cash_equity_outcome"]==1 and got["strict_standalone_outcome"]==0
    assert audit_outcome(window(),{},docs(),[event(endpoint_class="mixed_debt_equity")])["strict_standalone_outcome"] is None


def test_atm_shelf_other_issuer_and_unverified_evidence_never_primary_positive():
    for e in [event(endpoint_class="atm"),event(endpoint_class="shelf"),event(cik=2)]:
        assert audit_outcome(window(),review(),docs(),[e])["outcome"]==0
    with pytest.raises(ValueError,match="missing evidence"):
        audit_outcome(window(),review(),docs(),[event(sources=[])])


def test_horizon_and_inventory_integrity():
    validate_window(window())
    with pytest.raises(ValueError,match="horizon"):
        validate_window(dict(window(),outcome_end="2025-09-01"))
    with pytest.raises(ValueError,match="Empty"):
        validate_window(dict(window(),filings=[]))


def test_unresolved_plausible_primary_event_blocks_a_negative():
    got=audit_outcome(window(),review(),docs(),[event(status="provisional")])
    assert got["outcome"] is None


def test_excluded_atm_on_signal_date_does_not_create_timing_ambiguity():
    got=audit_outcome(window(),review(),docs(),[event(endpoint_class="atm",announcement_date="2025-06-30")])
    assert got["outcome"]==0


def test_omitted_linked_exhibit_blocks_negative_until_captured():
    parent=docs()[0]
    parent['linked_exhibits']=['https://sec/important-exhibit']
    got=audit_outcome(window(),review(),[parent],[])
    assert got['outcome'] is None
    assert 'missing_linked_exhibits' in got['gaps']
    child=dict(window_ids=[window()['window_id']],accession='one',file_type='linked_exhibit',url='https://sec/important-exhibit')
    assert audit_outcome(window(),review(),[parent,child],[])['outcome']==0


def test_same_day_unknown_time_excludes_observation_despite_later_positive():
    got=audit_outcome(window(),review(),docs(),[
        event(event_id='ambiguous',announcement_date='2025-06-30'),
        event(event_id='later',announcement_date='2025-07-02')])
    assert got['primary_events']==['later']
    assert got['outcome'] is None
    assert got['inclusive_cash_equity_outcome'] is None
    assert got['strict_standalone_outcome'] is None
