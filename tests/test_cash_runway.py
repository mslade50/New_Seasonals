from copy import deepcopy
import pytest

from fundamental.cash_runway import Facts, apply_manual_review, calculate_runway, financing_excerpts


def fixture():
    rows = [
        ("a", "2026-03-31", "2026-05-08T12:00:00Z"),
        ("b", "2026-06-30", "2026-08-08T12:00:00Z"),
        ("c", "2026-06-30", "2026-10-08T12:00:00Z"),
    ]
    submissions = {"name": "Pre-revenue Test", "sic": "2836", "filings": {"recent": {
        "accessionNumber": [r[0] for r in rows], "reportDate": [r[1] for r in rows],
        "acceptanceDateTime": [r[2] for r in rows], "form": ["10-Q"] * 3,
        "primaryDocument": ["report.htm"] * 3}}}
    payload = {"cik": 1, "facts": {"us-gaap": {}}}
    def add(tag, val, start=None, end="2026-06-30", accession="b"):
        obs = dict(val=val, end=end, accn=accession, form="10-Q")
        if start: obs["start"] = start
        payload["facts"]["us-gaap"].setdefault(tag, {"units": {"USD": []}})["units"]["USD"].append(obs)
    add("CashAndCashEquivalentsAtCarryingValue", 30_000_000)
    add("ShortTermInvestments", 6_000_000)
    add("NetCashProvidedByUsedInOperatingActivities", -24_000_000, "2026-01-01")
    add("NetCashProvidedByUsedInOperatingActivities", -9_000_000, "2026-01-01", "2026-03-31", "a")
    add("PaymentsToAcquirePropertyPlantAndEquipment", 3_000_000, "2026-01-01")
    add("PaymentsToAcquirePropertyPlantAndEquipment", 1_000_000, "2026-01-01", "2026-03-31", "a")
    return payload, submissions, add


def calc(p, s):
    return calculate_runway(p, s, ticker="TEST", as_of="2026-09-22T16:00:00Z")


def test_ytd_difference_and_prerevenue():
    p, s, _ = fixture()
    r = calc(p, s)
    assert r["status"] == "calculated"  # no revenue tag required
    assert r["ocf_3m"] == -15_000_000
    assert r["capex_3m"] == 2_000_000
    assert r["monthly_burn_3m"] == 5_000_000
    assert r["runway_6m"] == 9
    assert r["runway_with_capex_6m"] == 8
    assert r["flow_sources"]["ocf_3m"]["method"] == "YTD difference"
    assert len(r["flow_sources"]["ocf_3m"]["inputs"]) == 2


def test_future_restatement_and_unknown_acceptance_excluded():
    p, s, add = fixture()
    add("CashAndCashEquivalentsAtCarryingValue", 900_000_000, accession="c")
    add("CashAndCashEquivalentsAtCarryingValue", 500_000_000, accession="unknown")
    assert calc(p, s)["cash"] == 30_000_000


def test_combined_cash_not_double_counted_and_restricted_excluded():
    p, s, add = fixture()
    add("CashCashEquivalentsAndShortTermInvestments", 36_000_000)
    add("CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", 80_000_000)
    assert calc(p, s)["reported_liquidity"] == 36_000_000


def test_missing_investment_is_explicit_lower_bound():
    p, s, _ = fixture()
    del p["facts"]["us-gaap"]["ShortTermInvestments"]
    r = calc(p, s)
    assert r["current_investments"] is None
    assert r["reported_liquidity"] == 30_000_000
    assert "unverified" in r["liquidity_basis"]


def test_missing_capex_does_not_become_zero():
    p, s, _ = fixture()
    del p["facts"]["us-gaap"]["PaymentsToAcquirePropertyPlantAndEquipment"]
    r = calc(p, s)
    assert r["runway_with_capex_6m"] is None
    assert r["runway_6m"] == 9


def test_cash_positive_operations_can_burn_after_capex():
    p, s, _ = fixture()
    p["facts"]["us-gaap"]["NetCashProvidedByUsedInOperatingActivities"]["units"]["USD"][0]["val"] = 1_000_000
    r = calc(p, s)
    assert r["bucket"] == "No observed burn"
    assert r["runway_6m"] is None
    assert r["runway_with_capex_6m"] == 108


def test_conflicting_fact_fails_closed():
    p, s, add = fixture()
    add("CashAndCashEquivalentsAtCarryingValue", 5_000_000)
    r = calc(p, s)
    assert r["reported_liquidity"] is None
    assert any("Conflicting" in w for w in r["warnings"])


def test_bad_combined_cash_does_not_mask_conflict():
    p, s, add = fixture()
    add("CashCashEquivalentsAndShortTermInvestments", 20_000_000)
    assert calc(p, s)["reported_liquidity"] is None


def test_foreign_and_financial_entities_do_not_get_generic_runway():
    p, s, _ = fixture()
    s["sic"] = "6021"
    assert calc(p, s)["status"] == "unavailable"
    s["sic"] = "2836"
    p["facts"] = {"ifrs-full": {}}
    assert calc(p, s)["status"] == "unavailable"


def test_stale_balance_excluded_from_priority():
    p, s, _ = fixture()
    r = calculate_runway(p, s, ticker="TEST", as_of="2027-02-01T16:00:00Z")
    assert r["bucket"] == "Stale"


def test_negative_unsigned_capex_is_unavailable():
    p, s, _ = fixture()
    p["facts"]["us-gaap"]["PaymentsToAcquirePropertyPlantAndEquipment"]["units"]["USD"][0]["val"] = -100
    assert calc(p, s)["runway_with_capex_6m"] is None


def test_financing_keywords_never_book_cash():
    p, s, _ = fixture()
    text = "The company may sell up to $100 million under its at-the-market offering."
    assert financing_excerpts(text)
    assert calc(p, s)["financing_status"] == "Not reviewed"


def test_timezone_is_required():
    p, s, _ = fixture()
    with pytest.raises(ValueError, match="timezone"):
        calculate_runway(p, s, ticker="TEST", as_of="2026-09-22")


def test_six_months_from_fiscal_ytd_difference():
    p, s, add = fixture()
    s["filings"]["recent"]["accessionNumber"] += ["d", "e"]
    s["filings"]["recent"]["reportDate"] += ["2026-01-31", "2026-07-31"]
    s["filings"]["recent"]["acceptanceDateTime"] += ["2026-03-01T12:00:00Z", "2026-09-01T12:00:00Z"]
    s["filings"]["recent"]["form"] += ["10-Q", "10-Q"]
    add("NetCashProvidedByUsedInOperatingActivities", -73_437_000, "2025-11-01", "2026-07-31", "e")
    add("NetCashProvidedByUsedInOperatingActivities", -33_937_000, "2025-11-01", "2026-01-31", "d")
    result = Facts(p, s, "2026-09-22T16:00:00Z").flow(("NetCashProvidedByUsedInOperatingActivities",), "2026-07-31", 6)
    assert result["value"] == -39_500_000
    assert result["start"] == "2026-02-01"


@pytest.mark.parametrize("investment_tag", ["AvailableForSaleSecuritiesDebtSecuritiesCurrent",
                                          "DebtSecuritiesAvailableForSaleExcludingAccruedInterestCurrent"])
def test_alternate_current_debt_securities_and_productive_asset_tags(investment_tag):
    p, s, _ = fixture()
    concepts = p["facts"]["us-gaap"]
    concepts[investment_tag] = concepts.pop("ShortTermInvestments")
    concepts["PaymentsToAcquireProductiveAssets"] = concepts.pop("PaymentsToAcquirePropertyPlantAndEquipment")
    r = calc(p, s)
    assert r["reported_liquidity"] == 36_000_000
    assert r["capex_6m"] == 3_000_000


def test_manual_audit_mismatch_blocks_and_preserves_calculated_values():
    p, s, _ = fixture()
    r = calc(p, s)
    review = {"ticker":"TEST", "balance_date":"2026-06-30", "as_of":"2026-09-22",
              "sources":[{"url":"https://www.sec.gov/example"}], "expected":{"cash":1}}
    a = apply_manual_review(r, review)
    assert a["check_status"] == "MISMATCH"
    assert r["status"] == "audit_mismatch"
    assert r["cash"] == 30_000_000
    review["balance_date"] = "2026-03-31"
    with pytest.raises(ValueError, match="period mismatch"):
        apply_manual_review(r, review)


def test_cross_year_six_month_flow_uses_contiguous_quarters():
    p, s, add = fixture()
    recent = s["filings"]["recent"]
    recent["accessionNumber"] += ["annual", "nine"]
    recent["reportDate"] += ["2025-12-31", "2025-09-30"]
    recent["acceptanceDateTime"] += ["2026-02-01T12:00:00Z", "2025-11-01T12:00:00Z"]
    recent["form"] += ["10-K", "10-Q"]
    add("NetCashProvidedByUsedInOperatingActivities", -30_000_000, "2025-01-01", "2025-12-31", "annual")
    add("NetCashProvidedByUsedInOperatingActivities", -24_000_000, "2025-01-01", "2025-09-30", "nine")
    result = Facts(p, s, "2026-09-22T16:00:00Z").flow(("NetCashProvidedByUsedInOperatingActivities",), "2026-03-31", 6)
    assert result["value"] == -15_000_000
    assert result["start"] == "2025-10-01"


def test_replay_detects_capture_tampering(tmp_path):
    import hashlib
    import json
    from scripts.build_cash_runway_watchlist import Capture
    source = tmp_path / "source"
    (source / "raw").mkdir(parents=True)
    (source / "captures.json").write_text(json.dumps([dict(name="x.json", source="https://data.sec.gov/x",
            sha256=hashlib.sha256(b"original").hexdigest(), fetched_at="2026-09-22T16:00:00Z")]))
    (source / "raw" / "x.json").write_bytes(b"changed")
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(ValueError, match="provenance mismatch"):
        Capture(output, source).get("x.json", "https://data.sec.gov/x")
