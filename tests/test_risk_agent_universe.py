"""Guard tests for risk_agent_universe: the R2 blindness allowlist and sleeve constants.

The Risk Agent must never read book objects (fills, positions, sizing state,
other sleeves' ideas). r2_key_allowed() is the only gate, so these tests pin
both the allow and deny sides and the deny-first ordering.
"""
import pytest

import risk_agent_universe as ru

ALLOWED_KEYS = [
    "master_prices.parquet",
    "options/positioning_history.parquet",
    "shared/site_risk.json",
    "intraday/15min/SPY.parquet",
    "risk_agent/journal.jsonl",
]

DENIED_KEYS = [
    "live_fills.parquet",
    "live_fills/generations/x.parquet",
    "exposure_state.json",
    "pitch_journal.jsonl",
    "pitch_today.json",
    "backtest_trades_full.parquet",
    "event_sleeve_state.json",
    "trend_sleeve_state.json",
    "dial_sleeve_paper.json",
    "morning_orders.json",
    "radar_recs.json",
    "seasonal_agent_journal.jsonl",
    "rd2_environment.json",
    "site/builds/x/site_risk.json",
    "ops/sleeve_runtime_status.json",
    "review_inbox/v1/pitch/x.json",
    "discretionary_focus/current.json",
    "unknown_new_file.parquet",  # default deny
]


@pytest.mark.parametrize("key", ALLOWED_KEYS)
def test_market_data_and_own_namespace_allowed(key):
    assert ru.r2_key_allowed(key) is True


@pytest.mark.parametrize("key", DENIED_KEYS)
def test_book_and_unknown_keys_denied(key):
    assert ru.r2_key_allowed(key) is False


def test_no_denied_prefix_is_itself_allowed():
    for denied in ru.DENIED_R2_PREFIXES:
        assert ru.r2_key_allowed(denied) is False, denied
        assert ru.r2_key_allowed(denied + "x") is False, denied
    for allowed in ru.ALLOWED_R2_PREFIXES:
        assert not any(allowed.startswith(d) for d in ru.DENIED_R2_PREFIXES), allowed


def test_sleeve_capital():
    assert ru.SLEEVE_CAPITAL == 200_000


def test_futures_roots_are_well_formed():
    for key, fut in ru.FUTURES.items():
        assert fut.root == key
        assert fut.multiplier > 0, key
        assert fut.series.endswith("=F"), key


@pytest.mark.parametrize("micro,parent", [("MES", "ES"), ("MNQ", "NQ"),
                                          ("MGC", "GC"), ("MCL", "CL")])
def test_micro_is_one_tenth_of_parent(micro, parent):
    assert ru.FUTURES[micro].multiplier == pytest.approx(ru.FUTURES[parent].multiplier / 10)
    assert ru.FUTURES[micro].series == ru.FUTURES[parent].series


def test_etfs_unique_and_sized():
    assert len(set(ru.ETFS)) == len(ru.ETFS), "duplicate ETF symbol"
    assert len(ru.ETFS) >= 100


def test_optionable_unique():
    assert len(set(ru.OPTIONABLE)) == len(ru.OPTIONABLE), "duplicate optionable symbol"
