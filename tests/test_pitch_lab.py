"""Guards for pitch_lab.py — the shared Daily Pitch research library.

The point of promoting the check machinery out of the per-day scratch folders
is that the statistics deciding kills are identical every morning. These tests
freeze the conventions: fractions in / percent out, lag-aware entry alignment,
declustering, controls that exclude triggers, and the exact sign test that
replaces t-stats at small N. All synthetic data — no dependence on the price
cache.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pitch_lab as pl  # noqa: E402


@pytest.fixture()
def dates():
    return pd.bdate_range("2020-01-01", periods=400)


@pytest.fixture()
def panel(dates):
    rng = np.random.default_rng(7)
    a = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, len(dates))),
                  index=dates)
    b = pd.Series(50 * np.cumprod(1 + rng.normal(0, 0.01, len(dates))),
                  index=dates)
    return pd.DataFrame({"AAA": a, "BBB": b})


# ---------------------------------------------------------------------------
# units: the 2026-08-07 double-scaling bug must stay impossible to reintroduce
# ---------------------------------------------------------------------------
def test_summarize_units_fractions_in_percent_out():
    r = pl.summarize(np.array([0.01, 0.03]))
    assert r["n"] == 2
    assert r["mean_pct"] == pytest.approx(2.0)
    assert r["hit"] == pytest.approx(100.0)
    assert r["worst_pct"] == pytest.approx(1.0)


def test_summarize_empty_and_nan():
    assert pl.summarize(np.array([]))["n"] == 0
    r = pl.summarize(np.array([np.nan, 0.02]))
    assert r["n"] == 1 and r["mean_pct"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# forward-return alignment (the load-bearing lag convention)
# ---------------------------------------------------------------------------
def test_fwd_lag_alignment():
    s = pd.Series([1.0, 2.0, 4.0, 8.0, 16.0],
                  index=pd.bdate_range("2024-01-01", periods=5))
    # signal D=index0: enter close D+1 (2.0), exit h=1 later (4.0) -> +100%
    assert pl.fwd_lag(s, h=1, lag=1).iloc[0] == pytest.approx(1.0)
    # lag=0 reproduces the naive close-to-close look
    assert pl.fwd_lag(s, h=1, lag=0).iloc[0] == pytest.approx(1.0)
    assert pl.fwd_ret(s, 2).iloc[0] == pytest.approx(3.0)
    # off-the-end windows are NaN, never wrapped
    assert np.isnan(pl.fwd_lag(s, h=2, lag=1).iloc[2])


def test_vehicle_ret_pair_is_leg_difference(panel):
    h = 5
    pair = pl.vehicle_ret(panel, [("AAA", 1.0), ("BBB", -1.0)], h)
    manual = pl.fwd_lag(panel["AAA"], h) - pl.fwd_lag(panel["BBB"], h)
    pd.testing.assert_series_equal(pair, manual)


# ---------------------------------------------------------------------------
# episodes and controls
# ---------------------------------------------------------------------------
def test_declusters_keeps_first_of_cluster(dates):
    trig = pd.DatetimeIndex([dates[10], dates[12], dates[14], dates[40]])
    kept = pl.declusters(trig, min_gap_td=5, all_dates=dates)
    assert list(kept) == [dates[10], dates[40]]


def test_local_control_excludes_triggers_and_far_days(dates):
    trig = pd.DatetimeIndex([dates[200]])
    ctl = pl.local_control(dates, trig, win=10)
    assert dates[200] not in ctl
    assert dates[195] in ctl and dates[210] in ctl
    assert dates[189] not in ctl and dates[211] not in ctl


def test_era_split_partitions_everything(dates):
    vals = np.ones(len(dates)) * 0.01
    rows = pl.era_split(dates, vals, cut="2021-01-01")
    assert rows[0]["n"] + rows[1]["n"] == len(dates)


# ---------------------------------------------------------------------------
# the small-N statistic
# ---------------------------------------------------------------------------
def test_sign_test_exact_values():
    assert pl.sign_test(6, 6) == pytest.approx(0.5**6)          # 0.015625
    assert pl.sign_test(8, 9) == pytest.approx(10 / 512)        # 8-1 record
    assert pl.sign_test(0, 6) == pytest.approx(1.0)
    assert np.isnan(pl.sign_test(3, 0))


def test_bootstrap_p_small_on_clean_positive():
    v = np.array([0.01, 0.02, 0.015, 0.03, 0.012, 0.02])
    assert pl.bootstrap_p_le0(v) < 0.01
    assert np.isnan(pl.bootstrap_p_le0(np.array([0.01, 0.02])))  # N<3 guard


# ---------------------------------------------------------------------------
# battery + C2 helpers run end to end on synthetic data
# ---------------------------------------------------------------------------
def test_battery_smoke(panel, capsys):
    mask = pd.Series(False, index=panel.index)
    mask.iloc[50::60] = True
    pl.battery(panel, mask, [("AAA", 1.0)], h=5, title="synthetic",
               cost_bps=4.0, variants={"looser": mask})
    out = capsys.readouterr().out
    assert "conditional vs controls" in out
    assert "sign p" in out
    assert "cost:" in out


def test_battery_no_triggers_is_loud(panel, capsys):
    mask = pd.Series(False, index=panel.index)
    pl.battery(panel, mask, [("AAA", 1.0)], h=5, title="empty", cost_bps=4.0)
    assert "NO TRIGGERS EVER" in capsys.readouterr().out


def test_horizon_scan_shapes(panel):
    dates = pd.DatetimeIndex([panel.index[30], panel.index[90]])
    rows = pl.horizon_scan(panel, dates, [("AAA", 1.0)], hs=(1, 3, 5))
    assert [r["label"] for r in rows] == ["h=1", "h=3", "h=5"]
    assert all(r["n"] == 2 for r in rows)
    assert "edge_pct" in rows[0]


def test_episode_paths_end_matches_fwd_lag(panel):
    d = panel.index[100]
    h = 5
    paths = pl.episode_paths(panel, pd.DatetimeIndex([d]),
                             [("AAA", 1.0)], h=h)
    assert paths.shape == (1, h)
    expected = pl.fwd_lag(panel["AAA"], h).loc[d]
    assert paths.loc[d, h] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# watchlist round trip
# ---------------------------------------------------------------------------
def test_watchlist_roundtrip(tmp_path):
    p = tmp_path / "wl.json"
    assert pl.load_watchlist(p) == {"entries": []}
    d = {"entries": [{"added": "2026-08-07", "title": "x",
                      "trigger": "y", "expires": "2026-09-01"}]}
    pl.save_watchlist(d, p)
    assert pl.load_watchlist(p) == d


def test_read_only_contract():
    """pitch_lab must never import book modules or write into data/ paths
    other than the watchlist (which only save_watchlist touches)."""
    src = (ROOT / "pitch_lab.py").read_text(encoding="utf-8")
    for banned in ("strategy_config", "daily_scan", "strat_backtester",
                   "order_staging"):
        assert banned not in src, f"pitch_lab imports book module {banned}"


# --- union-calendar hazard (found 2026-08-19) -------------------------------
# close_panel returns a UNION calendar, so a ticker on a different exchange
# calendar (CL=F trades some equity holidays) leaves NaN rows in every other
# column. Two distinct failures came out of that on a live check: a raw
# rolling(252).max() blanked 1,237 of 6,704 windows and cut XLE's 52-week-high
# day count from 420 to 343, and pct_change's default pad fill shifted pct_rank
# by up to 29.4 percentile points. Both are silent. These freeze the fix.

@pytest.fixture()
def holed(dates):
    """A clean series and the same series with foreign-calendar holes."""
    rng = np.random.default_rng(11)
    s = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates)))),
                  index=dates)
    holes = dates[[37, 120, 121, 300]]
    union = s.copy()
    union.loc[holes] = np.nan
    return s.drop(holes), union


def test_rolling_on_valid_survives_calendar_holes(holed):
    clean, union = holed
    raw = union.rolling(60).max()
    fixed = pl.rolling_on_valid(union, lambda x: x.rolling(60).max())
    # the raw form loses whole windows; the fixed form loses only the holes
    assert raw.notna().sum() < fixed.notna().sum()
    # only the 59-day warmup is lost, not whole windows around each hole
    assert fixed.notna().sum() == len(clean) - 59
    # and it agrees with computing on the hole-free series directly
    want = clean.rolling(60).max()
    got = fixed.reindex(clean.index)
    pd.testing.assert_series_equal(got, want, check_names=False)


def test_pct_rank_is_calendar_invariant(holed):
    """The same sessions must rank identically whether or not the caller's
    index carries another instrument's trading days."""
    clean, union = holed
    a = pl.pct_rank(clean, 21).dropna()
    b = pl.pct_rank(union, 21).reindex(clean.index).dropna()
    assert len(a) > 100
    pd.testing.assert_series_equal(a, b, check_names=False)


def test_zscore_is_calendar_invariant(holed):
    clean, union = holed
    a = pl.zscore(clean).dropna()
    b = pl.zscore(union).reindex(clean.index).dropna()
    assert len(a) > 100
    pd.testing.assert_series_equal(a, b, check_names=False)


def test_pct_rank_does_not_pad_across_holes(holed):
    """A hole must not become a synthetic zero-return session. With a 1-day
    lookback the session after a hole has no defined return at all."""
    clean, union = holed
    r = pl.pct_rank(union, 1)
    hole = union.index[union.isna()][0]
    nxt = union.index[union.index.get_loc(hole) + 1]
    assert pd.isna(r.loc[hole])
    assert pd.isna(r.loc[nxt])
