"""Guards for daily_scan's per-ticker staleness drop (2026-08-07).

This is the ONLY thing blocking a delisted or renamed symbol from forward
signalling. Before it existed, nothing dropped a ticker whose newest bar was
old: the per-ticker validation only trimmed bars that were too NEW. A dead
symbol therefore kept scoring on frozen bars, and a dead ticker does not go
quiet, it goes WRONG. BK's five most recent bars spanned three weeks after it
became BNY, so it ranked as the hottest 5-day name in the entire pitch tape.

The companion rule these tests also pin: the universe still CONTAINS dead
names so the backtest can trade them over the period they were alive.
Excluding them from the universe instead deleted 25 historical trades.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import strategy_config as sc  # noqa: E402


def test_abort_fraction_frozen():
    import daily_scan
    assert daily_scan.STALE_TICKER_ABORT_FRAC == 0.05


def test_delisted_names_stay_in_the_universe_for_backtest_history():
    """The whole point of the redesign: history is preserved."""
    retained = sc.UNIVERSE_DELISTED & set(sc.CSV_UNIVERSE)
    with_data = sc.UNIVERSE_DELISTED - sc.UNIVERSE_NO_DATA
    assert retained == with_data, (
        "every delisted name WITH price history must remain in the universe; "
        "removing them deletes real historical trades and deepens survivorship "
        "bias")


def test_no_data_names_are_excluded_because_there_is_nothing_to_preserve():
    """The one case where universe exclusion IS correct."""
    assert sc.UNIVERSE_NO_DATA
    assert not (sc.UNIVERSE_NO_DATA & set(sc.CSV_UNIVERSE))


def test_no_data_set_is_a_subset_of_the_delisted_catalogue():
    assert sc.UNIVERSE_NO_DATA <= sc.UNIVERSE_DELISTED


def test_corp_action_exclusions_still_filter():
    """Deal-pinned LIVE names are a different case and stay excluded outright:
    their price action is an artifact in history too."""
    assert sc.UNIVERSE_CORP_ACTION_EXCLUSIONS
    assert not (sc.UNIVERSE_CORP_ACTION_EXCLUSIONS & set(sc.CSV_UNIVERSE))


def test_delisted_catalogue_is_not_used_as_a_universe_filter():
    """Regression: UNIVERSE_DELISTED was briefly a CSV_UNIVERSE filter and
    that erased 25 booked trades. It must stay documentation only."""
    src = (ROOT / "strategy_config.py").read_text(encoding="utf-8")
    # the comprehension ends at the closing paren on its own line; splitting on
    # a bare ")" would cut at read_csv(...) and pass vacuously
    csv_block = src.split("CSV_UNIVERSE = sorted(")[1].split("\n    )")[0]
    assert "UNIVERSE_DELISTED" not in csv_block, (
        "UNIVERSE_DELISTED must not filter CSV_UNIVERSE — the universe feeds "
        "the backtest as well as the live scan")
    assert "UNIVERSE_NO_DATA" in csv_block
