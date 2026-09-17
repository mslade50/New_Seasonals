import numpy as np
import pandas as pd

from nyse_risk import append_main_scores, compute_nyse_main, main_dial_from_frame, warning_severity

STATS = {"signals": {"Low Absorption Ratio": {"horizons": {"63d": {"diff_mean": -3.5}}},
                     "Defensive Leadership": {"horizons": {"63d": {"diff_mean": -4.0}}}}}


def inputs():
    dates = pd.bdate_range("2024-01-02", periods=500)
    return (pd.Series(20., index=dates), pd.Series(100., index=dates),
            pd.Series(-1., index=dates))


def test_boundaries_and_missing():
    dd = pd.Series([.01999, .02, .03, .03001, .01, .01])
    nh = pd.Series([-1., -1., -1., -1., 0., np.nan])
    np.testing.assert_allclose(warning_severity(nh, dd), [1, .6, .6, 0, 0, np.nan], equal_nan=True)


def test_recovery_erases_both_smoothing_queues_and_requires_rearm():
    base, spy, net = inputs()
    net.iloc[450] = 0
    spy.iloc[451:453] = 96.
    result = compute_nyse_main(base, spy, net, STATS)
    assert result.nyse_contribution.iloc[449] > 0
    assert (result.nyse_contribution.iloc[450:453] == 0).all()
    assert (result.main_score.iloc[450:453] == base.iloc[450:453]).all()
    assert result.nyse_contribution.iloc[453] > 0
    assert (result.main_score >= base).all()
    pd.testing.assert_frame_equal(result.iloc[:455], compute_nyse_main(base.iloc[:455], spy.iloc[:455], net.iloc[:455], STATS))


def test_unknown_cannot_clear_and_uses_base_until_gap_has_expired():
    base, spy, net = inputs()
    net.iloc[420] = np.nan
    result = compute_nyse_main(base, spy, net, STATS)
    assert not result.nyse_reset.iloc[420]
    assert result.nyse_effective.iloc[420] > 0
    assert not result.nyse_available.iloc[420:497].any()
    assert result.nyse_available.iloc[497]
    assert result.main_score.iloc[420] == base.iloc[420]


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
