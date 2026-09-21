import datetime as dt

import pytest

from scripts import build_discretionary_focus as builder


@pytest.mark.parametrize("form", ["10-Q", "10-K", "20-F", "40-F", "20-F/A", "40-F/A"])
def test_periodic_financial_reports_include_foreign_issuers(monkeypatch, form):
    client = builder.FMPNewsClient("fixture", retries=1)
    monkeypatch.setattr(client, "_fetch_list", lambda *a, **k: [
        {"formType": "6-K", "filingDate": "2026-09-08",
         "finalLink": "https://www.sec.gov/Archives/monthly-metrics.htm"},
        {"formType": form, "filingDate": "2026-03-10", "cik": "0001872195",
         "finalLink": "https://www.sec.gov/Archives/annual.htm"},
    ])
    filing = client.fetch_latest_sec_filing("BLSH", as_of=dt.date(2026, 9, 16))
    assert filing["form"] == form
    assert filing["url"].endswith("annual.htm")
    assert filing["issuer_cik"] == "0001872195"


@pytest.mark.parametrize("form,date,url", [
    ("6-K", "2026-09-08", "https://www.sec.gov/Archives/current.htm"),
    ("8-K", "2026-09-08", "https://www.sec.gov/Archives/current.htm"),
    ("20-F", "2026-09-17", "https://www.sec.gov/Archives/annual.htm"),
    ("20-F", "2026-03-10", "https://sec.gov.example.com/annual.htm"),
    ("20-F", "2025-03-10", "https://www.sec.gov/Archives/annual.htm"),
])
def test_foreign_report_support_preserves_source_and_freshness_gates(monkeypatch, form, date, url):
    client = builder.FMPNewsClient("fixture", retries=1)
    monkeypatch.setattr(client, "_fetch_list", lambda *a, **k: [
        {"formType": form, "filingDate": date, "finalLink": url},
    ])
    with pytest.raises(builder.FocusBuildError):
        client.fetch_latest_sec_filing("BLSH", as_of=dt.date(2026, 9, 16))
