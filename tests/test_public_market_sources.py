import pandas as pd
import pytest
from public_market_sources import (parse_listing_directory, symbol_candidate,
    share_structure, normalize_yahoo_news, reported_earnings_dates)

NOW = pd.Timestamp("2026-09-22T15:00:00Z")


def directory():
    return "\n".join(["Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares",
        "NA|Nano Common Stock|G|N|N|100|N|N", "GOOG|Alphabet Class C Capital Stock|Q|N|N|100|N|N",
        "TEST|Test Common Stock|Q|Y|N|100|N|N", "SPY|SPY ETF|Q|N|N|100|Y|N",
        "PREF|Example Preferred Stock|Q|N|N|100|N|N", "File Creation Time: 0922202607:00|||||||"])


def test_directory_na_ticker_test_etf_preferred():
    rows = parse_listing_directory(directory(), nasdaq=True).set_index("ticker")
    assert rows.loc["NA", "listing_eligible"]
    assert rows.loc["GOOG", "listing_eligible"]
    assert not rows.loc[["TEST", "SPY", "PREF"], "listing_eligible"].any()
    with pytest.raises(ValueError): parse_listing_directory(directory().split("File Creation")[0], nasdaq=True)


def info():
    return dict(symbol="NA", country="United States", quoteType="EQUITY", marketCap=1e9,
                regularMarketPrice=10, averageVolume=400000, regularMarketTime=NOW.timestamp())


def test_listing_needs_fresh_metadata_and_liquidity():
    listing = parse_listing_directory(directory(), nasdaq=True).iloc[0].to_dict()
    assert symbol_candidate(listing, info(), fetched_at=NOW)["eligible"]
    for changes in [{"country": None}, {"symbol": "OTHER"}, {"regularMarketTime": 1},
                    {"averageVolume": None}, {"regularMarketPrice": 3}, {"quoteType": "ETF"}]:
        assert not symbol_candidate(listing, info() | changes, fetched_at=NOW)["eligible"]
    assert not symbol_candidate(listing, info(), fetched_at=NOW+pd.Timedelta(days=10))["eligible"]


def test_public_float_is_not_share_float():
    with pytest.raises(ValueError): share_structure({"EntityPublicFloat": 1000000, "sharesOutstanding": 500})
    with pytest.raises(ValueError): share_structure({"floatShares": 1000, "sharesOutstanding": 500})
    assert share_structure({"floatShares": 400, "sharesOutstanding": 500})["float_shares"] == 400


def test_news_feed_membership_does_not_imply_issuer_match():
    def item(title, published="2026-09-22T12:00:00Z"):
        return {"content": {"title": title, "summary": "", "pubDate": published,
                "canonicalUrl": {"url": "https://example.com/article"}, "provider": {"displayName": "Publisher"}}}
    rows = normalize_yahoo_news([item("Microsoft earnings"), item("Meta earnings"), item("MSFT earnings", "2026-09-23T12:00:00Z")],
                               ticker="MSFT", company_name="Microsoft Corporation", as_of=NOW)
    assert len(rows) == 1 and rows[0]["title"] == "Microsoft earnings"


def test_reported_dates_zero_eps_counts_estimates_do_not():
    frame = pd.DataFrame({"Reported EPS": [0, None, 1], "EPS Estimate": [1, 1, 1]},
        index=pd.to_datetime(["2026-08-01T20:00:00Z", "2026-09-01T20:00:00Z", "2026-10-01T20:00:00Z"]))
    rows = reported_earnings_dates(frame, as_of=NOW)
    assert len(rows) == 1 and rows[0]["eps_actual"] == 0
    assert "requires_primary_confirmation" in rows[0]["evidence_status"]
