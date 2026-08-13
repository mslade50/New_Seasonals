from scripts.pull_scan_caches import SETS


def test_site_requires_fundamental_research_inputs():
    required, optional = SETS["site"]
    required_keys = {key for key, _ in required}
    optional_keys = {key for key, _ in optional}
    fundamental_keys = {
        "fundamental/current/daily_report_latest.json",
        "fundamental/current/company_maps_latest.json",
    }

    assert fundamental_keys <= required_keys
    assert fundamental_keys.isdisjoint(optional_keys)
