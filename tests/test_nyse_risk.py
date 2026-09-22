from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nyse_risk import (BASIS_V2_START, MODEL_VERSION, append_main_scores,
                       compute_nyse_main, main_dial_from_frame,
                       smooth_nyse_net, warning_severity)

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "scratch" / "nyse_smoothing_study"

STATS = {"signals": {"Low Absorption Ratio": {"horizons": {"63d": {"diff_mean": -3.5}}},
                     "Defensive Leadership": {"horizons": {"63d": {"diff_mean": -4.0}}}}}


def inputs():
    dates = pd.bdate_range("2024-01-02", periods=500)
    return (pd.Series(20., index=dates), pd.Series(100., index=dates),
            pd.Series(-1., index=dates))


def test_basis_string_names_the_ema_trigger():
    assert MODEL_VERSION == "nyse-reset-floor-v2-ema5"


def test_ema_definition_matches_the_study_variant():
    net = pd.Series([3., -8., 11., -2., -40., np.nan, 5., -6., -7., -9., 12., 4.])
    ok = net.notna().astype(int)
    expected = (net.ewm(span=5, adjust=False).mean()
                .where(ok.rolling(5, min_periods=5).min().eq(1)))
    pd.testing.assert_series_equal(smooth_nyse_net(net), expected)
    # Warm-up and the post-gap blackout are both five sessions of unknown.
    assert smooth_nyse_net(net).iloc[:4].isna().all()
    assert smooth_nyse_net(net).iloc[5:10].isna().all()
    assert pd.notna(smooth_nyse_net(net).iloc[10])


def test_boundaries_and_missing():
    # Constant -1 keeps the EMA at -1, so the tiers can be read straight off
    # `distance`; position 8's +20 print lifts the EMA above zero and position
    # 9 shows that a negative RAW print no longer fires while the EMA is up.
    dd = pd.Series([.01, .01, .01, .01, .01999, .02, .03, .03001, .01, .01, .01])
    nh = pd.Series([-1., -1., -1., -1., -1., -1., -1., -1., 20., -1., np.nan])
    assert smooth_nyse_net(nh).iloc[8] > 0 and smooth_nyse_net(nh).iloc[9] > 0
    np.testing.assert_allclose(
        warning_severity(nh, dd),
        [np.nan, np.nan, np.nan, np.nan, 1, .6, .6, 0, 0, 0, np.nan],
        equal_nan=True)


def test_single_nonnegative_print_cannot_reset_but_an_ema_crossing_does():
    base, spy, net = inputs()

    blip = net.copy()
    blip.iloc[450] = 0.                       # one healthy print, EMA still red
    assert smooth_nyse_net(blip).iloc[450] < 0
    held = compute_nyse_main(base, spy, blip, STATS)
    assert not held.nyse_reset.iloc[450]
    assert held.nyse_contribution.iloc[450] > 0
    assert held.nyse_effective.iloc[450] > 0

    real = net.copy()
    real.iloc[450] = 10.                      # enough to lift the EMA to >= 0
    assert smooth_nyse_net(real).iloc[450] >= 0
    cleared = compute_nyse_main(base, spy, real, STATS)
    assert cleared.nyse_reset.iloc[450]
    assert cleared.nyse_contribution.iloc[450] == 0


def test_recovery_erases_both_smoothing_queues_and_requires_rearm():
    base, spy, net = inputs()
    net.iloc[450] = 10.
    result = compute_nyse_main(base, spy, net, STATS)
    # The warning stays cleared for as long as the EMA holds at or above zero.
    ema = smooth_nyse_net(net)
    rearm = 450 + int(np.flatnonzero(ema.iloc[450:].lt(0).to_numpy())[0])
    assert rearm > 450
    assert result.nyse_contribution.iloc[449] > 0
    assert (result.nyse_contribution.iloc[450:rearm] == 0).all()
    assert (result.main_score.iloc[450:rearm] == base.iloc[450:rearm]).all()
    assert result.nyse_contribution.iloc[rearm] > 0
    assert (result.main_score >= base).all()
    stop = rearm + 2
    pd.testing.assert_frame_equal(
        result.iloc[:stop],
        compute_nyse_main(base.iloc[:stop], spy.iloc[:stop], net.iloc[:stop], STATS))


def test_unknown_cannot_clear_and_uses_base_until_gap_has_expired():
    base, spy, net = inputs()
    net.iloc[400] = np.nan
    result = compute_nyse_main(base, spy, net, STATS)
    assert not result.nyse_reset.iloc[400]
    assert result.nyse_effective.iloc[400] > 0
    # One unknown reading blanks the five-session EMA window, then the 64/5/10
    # influence chain needs 77 more complete sessions: 400..404 + 77 = 481.
    assert not result.nyse_available.iloc[400:481].any()
    assert result.nyse_available.iloc[481]
    assert result.main_score.iloc[400] == base.iloc[400]


def test_saved_decisions_and_am_refresh_migration():
    dates = pd.bdate_range("2026-09-01", "2026-09-21")
    old = pd.DataFrame({"63d": np.arange(len(dates), dtype=float)}, index=dates)
    old["main_score"] = np.nan
    old.loc["2026-09-17", "main_score"] = 55
    history = main_dial_from_frame(old)
    assert history.loc["2026-09-17"] == 55
    pd.testing.assert_series_equal(history.loc[:"2026-09-16"], old["63d"].rolling(10,min_periods=1).mean().loc[:"2026-09-16"], check_names=False)
    spy = pd.Series(100., index=pd.bdate_range("2024-01-02", "2026-09-21"))
    frame = old.copy()
    frame.loc["2026-09-18":, "main_score"] = np.nan
    out, _ = append_main_scores(frame, old, spy, pd.Series(-1., index=spy.index), STATS, "2026-09-18")
    assert out.loc["2026-09-17", "main_score"] == 55
    assert out.loc["2026-09-18":, "main_score"].notna().all()
    pd.testing.assert_series_equal(out["63d"], old["63d"])


def test_am_refresh_cannot_rescore_a_saved_raw_trigger_row():
    """--refresh-last reopens the previous session; a v1 row still never moves."""
    assert BASIS_V2_START == pd.Timestamp("2026-09-18")
    dates = pd.bdate_range("2026-09-01", "2026-09-18")
    old = pd.DataFrame({"63d": np.arange(len(dates), dtype=float)}, index=dates)
    old["main_score"] = np.nan
    old.loc["2026-09-17", "main_score"] = 85.039406
    spy = pd.Series(100., index=pd.bdate_range("2024-01-02", "2026-09-18"))
    net = pd.Series(-1., index=spy.index)
    # run_date = the refreshed session itself, the worst case the AM job builds.
    out, _ = append_main_scores(old.copy(), old, spy, net, STATS, "2026-09-17")
    assert out.loc["2026-09-17", "main_score"] == 85.039406
    assert pd.notna(out.loc["2026-09-18", "main_score"])
    # A v2 row, by contrast, is still correctable by the AM run.
    saved = out.copy()
    saved.loc["2026-09-18", "main_score"] = 1.0
    again, _ = append_main_scores(saved.copy(), saved, spy, net, STATS, "2026-09-18")
    assert again.loc["2026-09-17", "main_score"] == 85.039406
    assert again.loc["2026-09-18", "main_score"] != 1.0


def test_display_respects_recovery_and_unknown_data(tmp_path):
    from nyse_risk import load_nyse_signal
    from fragility_core import _compute_decay_metadata
    from daily_risk_report import _status_badge
    _, spy, net = inputs()
    path = tmp_path / "breadth.parquet"
    net.iloc[-2] = 10.
    spy.iloc[-1] = 96
    pd.DataFrame({"nyse_net": net}).to_parquet(path)
    sig = load_nyse_signal(spy, path)
    assert sig['recovery_cleared']
    assert _compute_decay_metadata(sig, .04) is None
    assert _status_badge(sig, {'drawdown': -.04})[0] == 'OFF'
    # The card prints the trigger value with the raw print beside it, and the
    # site chart serializes the EMA series the trigger actually reads.
    assert "5d EMA" in sig['detail'] and "raw" in sig['detail']
    assert sig['net_highs_ema5'].iloc[-1] == smooth_nyse_net(net).iloc[-1]
    assert sig['raw_net'].iloc[-1] == net.iloc[-1]
    net.iloc[-1] = np.nan
    pd.DataFrame({"nyse_net": net}).to_parquet(path)
    sig = load_nyse_signal(spy, path)
    assert _status_badge(sig, {})[0] == 'UNAVAILABLE'


def test_one_day_blip_no_longer_clears_the_live_august_2026_episode(tmp_path):
    """The flicker the change was made for: 08-19 and 08-25 printed positive."""
    from nyse_risk import load_nyse_signal
    dates = pd.bdate_range("2025-01-01", "2026-08-31")
    spy = pd.Series(100., index=dates)
    net = pd.Series(30., index=dates)
    net.iloc[-21:] = [61., 103., 91., 31., 39., 0., 15., 37., 98., 13., -61.,
                      -117., 27., -20., -8., -5., 13., 14., 7., 4., -102.]
    path = tmp_path / "breadth.parquet"
    pd.DataFrame({"nyse_net": net}).to_parquet(path)
    hist = load_nyse_signal(spy, path)["signal_history"]
    assert not bool(hist.loc["2026-08-17"])   # raw -61, EMA not yet negative
    assert bool(hist.loc["2026-08-18"])
    assert bool(hist.loc["2026-08-19"])       # raw +27, EMA still negative
    assert bool(hist.loc["2026-08-25"])       # raw +13, EMA still negative
    assert not bool(hist.loc["2026-08-26"])   # EMA finally back above zero
    assert bool(hist.loc["2026-08-31"])


@pytest.mark.skipif(
    not (ROOT / "data" / "market_breadth.parquet").exists()
    or not (ROOT / "data" / "master_prices.parquet").exists()
    or not (STUDY / "aug_sep_2026.csv").exists(),
    reason="production breadth/price caches or the study CSVs are absent")
def test_production_fire_set_replicates_the_smoothing_study():
    """The shipped trigger must reproduce the ema5 variant that was measured.

    Headline numbers from `scratch/nyse_smoothing_study/`: 244 fire days, 34
    declustered episodes, and a first Aug-2026 fire on 2026-08-18.
    """
    mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                         filters=[("ticker", "==", "SPY")])
    mp["date"] = pd.to_datetime(mp["date"])
    spy = mp.sort_values("date").set_index("date")["Close"].astype(float)
    spy = spy[~spy.index.duplicated(keep="last")].dropna()
    net = pd.read_parquet(ROOT / "data" / "market_breadth.parquet")[
        "nyse_net"].astype(float).reindex(spy.index)
    distance = (1 - spy / spy.rolling(252).max()).clip(lower=0)
    fired = warning_severity(net, distance).fillna(0).gt(0)

    # The study scored every variant on one calendar; the binding constraint is
    # the longest lookback in its grid (21 sessions).
    eligible = (distance.notna() & spy.notna()
                & net.notna().astype(int).rolling(21, min_periods=21).min().eq(1))
    # ... and on its own sample, which ends at the last session it scored.
    # Breadth is now collected twice a trading day, so counting sessions the
    # study never saw would turn a frozen replication into a moving target
    # that goes red on every new fire day (first hit 2026-09-18, 245 vs 244).
    study_end = pd.read_csv(STUDY / "last30_sessions.csv", parse_dates=[0],
                            index_col=0).index.max()
    eligible &= spy.index <= study_end
    pos = np.flatnonzero((fired & eligible).to_numpy())
    assert pos.size == 244
    episodes = 1 + sum(1 for j in range(1, pos.size) if pos[j] - pos[j - 1] >= 10)
    assert episodes == 34

    study = pd.read_csv(STUDY / "aug_sep_2026.csv",
                        parse_dates=["date"]).set_index("date")
    np.testing.assert_array_equal(fired.reindex(study.index).to_numpy(),
                                  study["fire_ema5"].to_numpy())
    august = study.index[study["fire_ema5"].to_numpy()][0]
    assert august == pd.Timestamp("2026-08-18")

    last30 = pd.read_csv(STUDY / "last30_sessions.csv",
                         parse_dates=[0], index_col=0)
    np.testing.assert_allclose(
        smooth_nyse_net(net).reindex(last30.index).round(1).to_numpy(),
        last30["ema5"].to_numpy())


def test_refuses_to_publish_main_score_when_spy_lags_latest_risk_row():
    dates = pd.bdate_range("2025-01-01", "2026-09-21")
    frame = pd.DataFrame({"63d": 60.}, index=dates)
    spy = pd.Series(100., index=dates[:-1])  # stale Friday Yahoo response
    net = pd.Series(-100., index=dates)
    with pytest.raises(ValueError, match="SPY.*latest risk session"):
        append_main_scores(frame, None, spy, net, STATS, pd.Timestamp("2026-09-21"))
