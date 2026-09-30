"""Guards for scripts/build_seasonal_state.py (Daily Seasonal stage A).

Outlier detection on a synthetic rank frame, entry-lag cycle stats on a
synthetic price path whose move sits exactly on the anchor-to-entry day, and
fingerprints from both journals.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_seasonal_state as bss  # noqa: E402
import seasonal_edge as se  # noqa: E402

TODAY = pd.Timestamp("2026-09-30")


def vec(r5, r10, r21, r63, r126=50.0, r252=50.0):
    return {5: r5, 10: r10, 21: r21, 63: r63, 126: r126, 252: r252}


# --- classification -----------------------------------------------------------
def test_two_adjacent_horizons_in_the_same_tail_qualify():
    got = bss.classify_ranks(vec(50, 92, 94, 60))
    assert got["side"] == "long" and got["agree_horizons"] == [10, 21]
    got = bss.classify_ranks(vec(8, 6, 40, 50))
    assert got["side"] == "short" and got["agree_horizons"] == [5, 10]


def test_non_adjacent_agreement_does_not_qualify():
    assert bss.classify_ranks(vec(92, 50, 93, 50)) is None
    assert bss.classify_ranks(vec(91, 89, 50, 50)) is None


def test_a_single_extreme_horizon_qualifies():
    got = bss.classify_ranks(vec(50, 50, 96, 50))
    assert got["side"] == "long" and got["agree_horizons"] == [21]
    assert got["adjacent"] is False
    got = bss.classify_ranks(vec(50, 50, 50, 4))
    assert got["side"] == "short" and got["agree_horizons"] == [63]


def test_126_and_252_never_flag_on_their_own():
    assert bss.classify_ranks(vec(50, 50, 50, 50, 99, 99)) is None


def test_both_sides_pick_the_more_extreme_and_flag_the_conflict():
    got = bss.classify_ranks(vec(91, 92, 50, 3))
    assert got["side"] == "short" and got["conflict"] is True


def test_the_longest_agreeing_run_is_reported():
    got = bss.classify_ranks(vec(95, 93, 91, 90))
    assert got["agree_horizons"] == [5, 10, 21, 63]


def test_asset_class_map():
    assert bss.asset_class("TLT") == "rates"
    assert bss.asset_class("^VIX") == "volatility"
    assert bss.asset_class("AAPL") == "equity"
    assert not bss.has_volume("^GSPC") and not bss.has_volume("EURUSD=X")
    assert bss.has_volume("SPY")


# --- synthetic prices -----------------------------------------------------------
def stepped_frame(step_lag: int = 1, step: float = 0.10) -> pd.DataFrame:
    """Flat 100 with a permanent +step jump every year, `step_lag` sessions
    after the bar whose trading-day-of-year matches the last bar's."""
    idx = pd.bdate_range("2012-01-02", "2026-09-29")
    doy = se._trading_doy(idx).to_numpy()
    target = int(doy[-1])
    close = np.full(len(idx), 100.0)
    for year in range(2012, 2026):
        pos = np.flatnonzero((idx.year == year) & (doy == target))
        if pos.size:
            close[pos[0] + step_lag:] *= 1 + step
    return pd.DataFrame({"Open": close, "High": close * 1.005,
                         "Low": close * 0.995, "Close": close,
                         "Volume": 1e6}, index=idx)


def test_entry_lag_excludes_the_anchor_to_entry_move():
    frame = stepped_frame(step_lag=1)
    lag0 = bss.window_stats(frame, TODAY, 5, "long", None, entry_lag=0)
    lag1 = bss.window_stats(frame, TODAY, 5, "long", None, entry_lag=1)
    assert lag0["n"] >= 10 and lag0["k"] == lag0["n"]
    assert lag0["mean_pct"] == pytest.approx(10.0, abs=0.01)
    # measured from the T+1 entry close the jump is already in the price
    assert lag1["mean_pct"] == pytest.approx(0.0, abs=0.01)
    assert lag1["k"] == 0


def test_entry_lag_keeps_a_move_that_happens_after_entry():
    frame = stepped_frame(step_lag=3)
    lag1 = bss.window_stats(frame, TODAY, 5, "long", None, entry_lag=1)
    assert lag1["mean_pct"] == pytest.approx(10.0, abs=0.01)
    assert lag1["mean_atr"] > 5
    short = bss.window_stats(frame, TODAY, 5, "short", None, entry_lag=1)
    assert short["k"] == 0


def test_cycle_filter_counts_only_same_phase_years():
    frame = stepped_frame(step_lag=3)
    cyc = bss.window_stats(frame, TODAY, 5, "long", se.cycle_phase(2026))
    assert set(y % 4 for y in cyc["years"]) == {2}
    assert cyc["n"] == len([y for y in range(2012, 2026) if y % 4 == 2])


def test_path_turn():
    frame = stepped_frame(step_lag=3)
    assert bss.path_turn(frame, TODAY, "long") == 0     # never adverse
    dip = stepped_frame(step_lag=3, step=-0.05)
    turn = bss.path_turn(dip, TODAY, "long")
    assert turn is None or 1 <= turn <= bss.PATH_TURN_TD


def test_build_ranks_on_a_synthetic_frame():
    tickers = {"AAA": vec(92, 95, 60, 50), "BBB": vec(50, 50, 50, 50),
               "CCC": vec(40, 7, 9, 50), "DDD": vec(99, 50, 50, 50),
               "^GSPC": vec(50, 50, 91, 93)}
    ranks = pd.DataFrame([{"Date": TODAY, "ticker": t,
                           **{f"atr_sznl_{h}d": v for h, v in r.items()}}
                          for t, r in tickers.items()])
    frame = stepped_frame(step_lag=3).reset_index(names="date")
    prices = pd.concat([frame.assign(ticker=t) for t in tickers])
    prices.loc[prices["ticker"] == "DDD", "Volume"] = 10.0   # illiquid
    warnings: list[str] = []
    out = bss.build_ranks(prices, TODAY, warnings, ranks=ranks,
                          sectors={"AAA": "Utilities"})
    got = {o["ticker"]: o for o in out["outliers"]}
    assert set(got) == {"AAA", "CCC", "^GSPC"}
    assert out["n_flagged"] == 4 and out["n_dropped_illiquid"] == 1
    assert got["AAA"]["side"] == "long" and got["AAA"]["sector"] == "Utilities"
    assert got["AAA"]["stats_horizon_td"] == 5
    assert got["CCC"]["side"] == "short" and got["CCC"]["agree_horizons"] == [10, 21]
    assert got["^GSPC"]["class"] == "us_large"
    assert got["^GSPC"]["adv_usd_21d"] is not None     # indices skip the floor only
    assert out["by_class"] == {"equity": 2, "us_large": 1}
    for key in ("cycle", "all_years", "ext", "path_turn_td", "atr14", "close"):
        assert key in got["AAA"]
    capped = bss.build_ranks(prices, TODAY, [], ranks=ranks, sectors={}, cap=1)
    assert len(capped["outliers"]) == 1 and capped["n_outliers"] == 3


# --- history ------------------------------------------------------------------
def _write(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def test_fingerprints_from_both_journals(tmp_path):
    seasonal = _write(tmp_path / "s.jsonl", [
        {"kind": "idea", "idea_id": "2026-09-28-S1", "date": "2026-09-28", "rank": 1,
         "fingerprint": "sss1"},
        {"kind": "idea", "idea_id": "2026-08-01-S1", "date": "2026-08-01", "rank": 1,
         "fingerprint": "old"}])
    pitch = _write(tmp_path / "p.jsonl", [
        {"kind": "idea", "idea_id": "2026-09-29-1", "date": "2026-09-29", "rank": 1,
         "fingerprint": "ppp1"},
        {"kind": "killed", "date": "2026-09-29", "title": "x", "reason": "y"}])
    warnings: list[str] = []
    out = bss.build_history(TODAY, warnings, seasonal_path=seasonal, pitch_path=pitch)
    assert out["recent_fingerprints"] == {"sss1": "2026-09-28"}
    assert out["pitch_recent_fingerprints"] == {"ppp1": "2026-09-29"}
    assert warnings == []


def test_missing_pitch_journal_is_empty_not_fatal(tmp_path):
    seasonal = _write(tmp_path / "s.jsonl", [])
    out = bss.build_history(TODAY, [], seasonal_path=seasonal,
                            pitch_path=tmp_path / "absent.jsonl")
    assert out["pitch_recent_fingerprints"] == {}


def test_state_carries_only_the_seasonal_registry():
    # Owner decision 2026-09-30: the pitch registry is never read or inlined.
    src = (ROOT / "scripts" / "build_seasonal_state.py").read_text(encoding="utf-8")
    assert "pitch_negative_registry" not in src
    assert "PITCH.negative_registry_path" not in src
    assert '"negative_registry": registry_block(SEASONAL.negative_registry_path' \
        in " ".join(src.split()).replace("( ", "(")
    assert bss.SEASONAL.negative_registry_path.name == \
        "seasonal_agent_negative_registry.md"
    warnings: list[str] = []
    block = bss.registry_block(bss.SEASONAL.negative_registry_path, warnings,
                               "negative_registry")
    assert block["path"] == "data/seasonal_agent_negative_registry.md"
    assert warnings == [] and "text" in block
