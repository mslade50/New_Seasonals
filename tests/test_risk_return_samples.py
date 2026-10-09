import copy

import numpy as np
import pandas as pd
import pytest

from scripts.risk_return_samples import build_return_samples
from scripts.build_risk_json import assert_shared_payload_clean, redact_for_shared


def inputs():
    dates = pd.bdate_range("2026-01-02", periods=60)
    main = pd.Series(50., index=dates)
    price = pd.Series(100. + np.arange(60), index=dates)
    reduced = {"current_score": 50., "band_low": 45., "band_high": 55.,
               "episode_dates": list(dates[::11])}
    return main, price, reduced


def test_full_sample_counts_each_date_and_reduced_keeps_greedy_anchors():
    main, price, reduced = inputs()
    original = copy.deepcopy(reduced)
    result = build_return_samples(main, price, reduced)
    assert len(result["all"]["episode_dates"]) == 60
    assert len(result["reduced"]["episode_dates"]) == 6
    assert result["all"]["sample_counts"] == {"5": 55, "10": 50, "21": 39}
    assert result["reduced"]["sample_counts"] == {"5": 5, "10": 5, "21": 4}
    assert result["reduced"]["returns"]["21"] is None
    assert result["all"]["returns"]["21"]["baseline_n"] == 39
    assert reduced == original
    assert result["asof"] == result["score_asof"]


def test_raw_scores_are_used_at_band_boundaries_and_zero_is_valid():
    main, price, _ = inputs()
    main[:] = 100
    main.iloc[0] = 44.999
    main.iloc[1] = 45
    main.iloc[2] = 55
    main.iloc[3] = 55.001
    main.iloc[-1] = 50
    reduced = {"current_score": 50., "band_low": 45., "band_high": 55.,
               "episode_dates": [main.index[1], main.index[-1]]}
    result = build_return_samples(main, price, reduced)
    assert result["all"]["episode_dates"] == [main.index[i].strftime("%Y-%m-%d") for i in (1,2,59)]
    main[:] = 0
    reduced = {"current_score": 0., "band_low": 0., "band_high": 5., "episode_dates": list(main.index[::11])}
    assert len(build_return_samples(main, price, reduced)["all"]["episode_dates"]) == 60


def test_outcomes_and_statistics_exclude_incomplete_observations():
    main, price, reduced = inputs()
    sample = build_return_samples(main, price, reduced)["all"]
    for window in (5,10,21):
        rows = sample["outcomes"][str(window)]
        completed = [r for r in rows if r["status"] == "complete"]
        pending = [r for r in rows if r["status"] == "incomplete"]
        assert len(pending) == window
        assert pending[-1]["available"] == 0
        assert all(r["value"] is None for r in pending)
        assert sample["returns"][str(window)]["mean"] == pytest.approx(np.mean([r["value"] for r in completed]))
        assert completed[0]["endDate"] == main.index[window].strftime("%Y-%m-%d")


def test_mismatched_saved_vintage_or_anchors_fail_closed():
    main, price, reduced = inputs()
    for patch in ({"current_score": 51}, {"episode_dates": []}, {"sample_counts": {5: 999}},
                  {"returns": {5: {"mean": .99}}}):
        with pytest.raises(ValueError):
            build_return_samples(main, price, {**reduced, **patch})
    assert build_return_samples(None, price, reduced) is None
    price.iloc[0] = 0
    with pytest.raises(ValueError):
        build_return_samples(main, price, reduced)


def test_research_sample_survives_shared_redaction_without_book_data():
    main, price, reduced = inputs()
    sample = build_return_samples(main, price, reduced)
    shared = redact_for_shared({"return_samples": sample, "sizing_state": {"score": 50., "threshold": 50., "throttle_on": True}})
    assert shared["return_samples"] == sample
    assert_shared_payload_clean(shared)


def downside_inputs():
    from scripts.risk_return_samples import build_downside_samples
    dates = pd.bdate_range("2026-01-02", periods=60)
    main = pd.Series(100., index=dates)
    main.iloc[[14, 15, 25, 59]] = 50.
    spy = pd.DataFrame({"High": 101., "Low": 99., "Close": 100.}, index=dates)
    spy.iloc[16, spy.columns.get_loc("Low")] = 94.
    reduced = {"current_score": 50., "band_low": 45., "band_high": 55.,
               "episode_dates": list(dates[[14, 25, 59]])}
    samples = build_return_samples(main, spy.Close, reduced)
    return dates, spy, samples, build_downside_samples


def test_downside_denominator_includes_non_breaches_and_tracks_exact_cohorts():
    dates, spy, samples, build = downside_inputs()
    result = build(samples, spy)
    full = result["all"]["windows"]["5"]
    reduced = result["reduced"]["windows"]["5"]
    assert result["all"]["episode_dates"] == samples["all"]["episode_dates"]
    assert result["reduced"]["episode_dates"] == samples["reduced"]["episode_dates"]
    assert (full["n_selected"], full["n_complete"], full["n_incomplete"]) == (4, 3, 1)
    assert full["hits"]["3"] == 2
    assert full["rates"]["3"] == pytest.approx(200/3)
    assert (reduced["n_complete"], reduced["hits"]["3"]) == (2, 1)
    assert reduced["rates"]["3"] == 50
    assert any(r.get("max_drawdown_atr", 99) < 1 for r in full["outcomes"])
    assert all(r["iv_change_points"] is None for r in full["outcomes"] if r["status"] == "complete")


def test_downside_thresholds_use_unrounded_values_and_zero_drawdown_is_eligible():
    _, spy, samples, build = downside_inputs()
    spy.iloc[16, spy.columns.get_loc("Low")] = 98.0008
    full = build(samples, spy)["all"]["windows"]["5"]
    first = full["outcomes"][0]
    assert first["max_drawdown_atr"] == pytest.approx(.9996)
    assert first["breaches"]["1"] is False
    assert full["hits"]["1"] == 0
    spy.loc[spy.index[15:20], "Low"] = 100.5
    assert build(samples, spy)["all"]["windows"]["5"]["outcomes"][0]["max_drawdown_atr"] == 0


def test_missing_future_low_is_unavailable_without_shifting_the_window():
    _, spy, samples, build = downside_inputs()
    spy.iloc[16, spy.columns.get_loc("Low")] = np.nan
    full = build(samples, spy)["all"]["windows"]["5"]
    assert full["outcomes"][0]["status"] == "unavailable"
    assert full["outcomes"][1]["status"] == "unavailable"
    assert full["n_complete"] + full["n_incomplete"] + full["n_unavailable"] == 4
    assert samples["all"]["sample_counts"]["5"] == 3


def test_missing_vix_does_not_change_downside_denominators():
    dates, spy, samples, build = downside_inputs()
    vix = pd.Series(15., index=dates)
    with_vix = build(samples, spy, vix)
    without_vix = build(samples, spy)
    for name in ("all", "reduced", "nonoverlap"):
        for window in ("5", "10", "21"):
            a, b = with_vix[name]["windows"][window], without_vix[name]["windows"][window]
            assert (a["n_complete"], a["hits"], a["rates"]) == (b["n_complete"], b["hits"], b["rates"])
    assert with_vix["all"]["windows"]["5"]["outcomes"][0]["iv_change_points"] == 0


def test_downside_vintage_and_redaction_are_enforced():
    _, spy, samples, build = downside_inputs()
    result = build(samples, spy)
    shared = redact_for_shared({"downside_samples": result})
    assert shared["downside_samples"] == result
    assert_shared_payload_clean(shared)
    with pytest.raises(ValueError, match="market dates"):
        build({**samples, "asof": "1999-01-01"}, spy)
    assert build(samples, None) is None


def test_nonoverlap_sample_uses_fixed_disjoint_windows_before_outcomes():
    main, price, reduced = inputs()
    result = build_return_samples(main, price, reduced)
    cohort = result["nonoverlap"]
    expected = [main.index[i].strftime("%Y-%m-%d") for i in (0, 22, 44)]
    assert result["nonoverlap_gap"] == 21
    assert cohort["episode_dates"] == expected
    assert cohort["sample_counts"] == {"5": 3, "10": 3, "21": 2}
    for window in (5, 10, 21):
        rows = cohort["outcomes"][str(window)]
        assert [r["date"] for r in rows] == expected
        assert all(a["endDate"] < b["date"] for a, b in zip(rows, rows[1:]))
    # Selection is unaffected by realized prices, including the unfinished match.
    changed = price.copy()
    changed.iloc[1:] *= .5
    assert build_return_samples(main, changed, reduced)["nonoverlap"]["episode_dates"] == expected
    assert cohort["outcomes"]["21"][-1]["status"] == "incomplete"


def test_nonoverlap_downside_uses_exact_return_anchors_including_pending():
    _, spy, samples, build = downside_inputs()
    downside = build(samples, spy)["nonoverlap"]
    assert downside["episode_dates"] == samples["nonoverlap"]["episode_dates"]
    for window in ("5", "10", "21"):
        rows = downside["windows"][window]["outcomes"]
        assert [r["date"] for r in rows] == downside["episode_dates"]
        assert [r["status"] for r in rows] == [r["status"] for r in samples["nonoverlap"]["outcomes"][window]]
