from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import research.intraday.gap_reversal_daily as daily_gap
from research.intraday.gap_reversal_daily import (
    LONG_ARM_ID,
    SHORT_ARM_ID,
    build_material_gap_views,
    freeze_universe,
    normalize_daily_prices,
    prepare_daily_candidates,
    run_daily_gap_reversal_research,
    write_daily_gap_research_artifacts,
)
from trading_calendar import TRADING_DAY


def _daily_rows(ticker: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": dates,
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.0,
            "Volume": 300_000.0,
        }
    )


def _two_arm_prices() -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=23, freq=TRADING_DAY)
    long = _daily_rows("AAA", dates)
    short = _daily_rows("BBB", dates)
    final = dates[-1]
    long.loc[long["date"].eq(final), ["Open", "High", "Low", "Close", "Volume"]] = [
        99.0,
        150.0,
        98.4,
        100.0,
        0.0,
    ]
    short.loc[short["date"].eq(final), ["Open", "High", "Low", "Close", "Volume"]] = [
        102.0,
        103.6,
        99.0,
        102.5,
        0.0,
    ]
    return pd.concat([long, short], ignore_index=True)


def test_atr_and_volume_gates_are_lagged_and_limits_fill_exactly() -> None:
    raw = _two_arm_prices()
    normalized, _ = normalize_daily_prices(raw, as_of=raw["date"].max())
    candidates, _ = prepare_daily_candidates(normalized)

    assert set(candidates["arm_id"]) == {LONG_ARM_ID, SHORT_ARM_ID}
    long = candidates.loc[candidates["arm_id"].eq(LONG_ARM_ID)].iloc[0]
    short = candidates.loc[candidates["arm_id"].eq(SHORT_ARM_ID)].iloc[0]

    # Current-day ranges are deliberately huge and current volume is zero;
    # neither can leak into the T-1 ATR or liquidity gate.
    assert long["atr14_lagged"] == pytest.approx(2.0)
    assert short["atr14_lagged"] == pytest.approx(2.0)
    assert long["prior_median_dollar_volume20"] == pytest.approx(30_000_000.0)
    assert short["prior_median_dollar_volume20"] == pytest.approx(30_000_000.0)

    assert long["limit_price"] == pytest.approx(98.5)
    assert long["filled"]
    assert long["fill_price"] == pytest.approx(98.5)
    assert long["gross_return"] == pytest.approx(100.0 / 98.5 - 1.0)

    assert short["limit_price"] == pytest.approx(103.5)
    assert short["filled"]
    assert short["fill_price"] == pytest.approx(103.5)
    assert short["gross_return"] == pytest.approx(-(102.5 / 103.5 - 1.0))


def test_current_day_changes_do_not_change_same_day_atr() -> None:
    raw = _two_arm_prices()
    normalized, _ = normalize_daily_prices(raw, as_of=raw["date"].max())
    baseline, _ = prepare_daily_candidates(normalized)
    changed = normalized.copy()
    mask = changed["date"].eq(changed["date"].max()) & changed["ticker"].eq("AAA")
    changed.loc[mask, "high"] = 500.0
    rerun, _ = prepare_daily_candidates(changed)
    before = baseline.loc[baseline["ticker"].eq("AAA"), "atr14_lagged"].iloc[0]
    after = rerun.loc[rerun["ticker"].eq("AAA"), "atr14_lagged"].iloc[0]
    assert after == before == pytest.approx(2.0)


def test_top_three_ranks_before_fill_and_unfilled_slots_stay_cash() -> None:
    day = pd.Timestamp("2026-02-02")
    candidates = pd.DataFrame(
        [
            {"arm_id": LONG_ARM_ID, "ticker": "A", "date": day, "gap_atr": 4.0, "filled": True, "gross_return": 0.010},
            {"arm_id": LONG_ARM_ID, "ticker": "B", "date": day, "gap_atr": 3.0, "filled": False, "gross_return": 0.000},
            {"arm_id": LONG_ARM_ID, "ticker": "C", "date": day, "gap_atr": 2.0, "filled": True, "gross_return": -0.005},
            # This attractive filled return is fourth-ranked and must not leak into selection.
            {"arm_id": LONG_ARM_ID, "ticker": "D", "date": day, "gap_atr": 1.0, "filled": True, "gross_return": 0.500},
            {"arm_id": SHORT_ARM_ID, "ticker": "S", "date": day, "gap_atr": 1.0, "filled": False, "gross_return": 0.000},
        ]
    )
    daily, _, _, _, selected = build_material_gap_views(
        candidates,
        pd.DatetimeIndex([day]),
        bootstrap_reps=100,
    )
    primary_long = daily.loc[
        daily["arm_id"].eq(LONG_ARM_ID)
        & daily["cost_bps"].eq(10.0)
        & daily["material_gap_threshold_atr"].eq(0.0)
    ].iloc[0]
    # Two fills pay 10 bps each; the unused/unfilled slot returns zero.
    assert primary_long["slot_portfolio_return"] == pytest.approx(
        (0.010 - 0.005 - 2 * 0.001) / 3.0
    )
    chosen = selected.loc[selected["arm_id"].eq(LONG_ARM_ID), "ticker"].tolist()
    assert chosen == ["A", "B", "C"]


def test_split_like_discontinuity_is_filtered() -> None:
    raw = _daily_rows("AAA", pd.date_range("2026-01-02", periods=23, freq=TRADING_DAY))
    raw.loc[raw.index[-1], ["Open", "High", "Low", "Close"]] = [50.0, 51.0, 49.0, 50.0]
    normalized, _ = normalize_daily_prices(raw, as_of=raw["date"].max())
    candidates, eligibility = prepare_daily_candidates(normalized)
    assert candidates.empty
    assert eligibility.loc[0, "n_discontinuity_filtered"] == 1


def test_universe_dedupes_and_applies_frozen_exclusions() -> None:
    source = ["aaa", "AAA", "CBZ", "THS", "ES=F", "BTC-USD", "^VIX", "^GSPC", "^NDX", "MISS"]
    available = ["AAA", "CBZ", "THS", "ES=F", "BTC-USD", "^VIX", "^GSPC", "^NDX"]
    frozen = freeze_universe(source, available, source_row_count=len(source))
    assert frozen.tickers == ("AAA", "^GSPC", "^NDX")
    assert frozen.source_row_count == len(source)
    assert frozen.unique_pre_filter_count == 9
    assert frozen.post_filter_count == 4
    assert frozen.available_count == 3
    assert frozen.missing_from_prices_count == 1
    assert frozen.pre_filter_sha256 != frozen.post_filter_sha256


def test_as_of_cutoff_excludes_future_and_rejects_duplicate_rows() -> None:
    raw = _daily_rows("AAA", pd.DatetimeIndex(["2026-08-27", "2026-08-28"]))
    normalized, original = normalize_daily_prices(raw, as_of="2026-08-27")
    assert original == 2
    assert normalized["date"].tolist() == [pd.Timestamp("2026-08-27")]
    duplicate = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate ticker/date"):
        normalize_daily_prices(duplicate, as_of="2026-08-28")


def test_writer_is_fresh_manifest_last_and_research_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _two_arm_prices()
    normalized, original = normalize_daily_prices(raw, as_of=raw["date"].max())
    frozen = freeze_universe(["AAA", "BBB"], normalized["ticker"].unique())
    result = run_daily_gap_reversal_research(
        normalized,
        frozen,
        as_of=raw["date"].max(),
        original_row_count=original,
        bootstrap_reps=100,
    )
    fake_module = tmp_path / "research" / "intraday" / "gap_reversal_daily.py"
    monkeypatch.setattr(daily_gap, "__file__", str(fake_module))
    output = tmp_path / "artifacts" / "bundle"
    input_path = tmp_path / "prices.parquet"
    raw.to_parquet(input_path, index=False)
    write_daily_gap_research_artifacts(
        result,
        output,
        input_provenance={"price_input_path": str(input_path), "price_input_sha256": "test"},
    )
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["research_only"] is True
    assert manifest["no_order"] is True
    assert manifest["production_writes"] is False
    assert manifest["automatic_promotion"] is False
    assert manifest["manifest_written_last"] is True
    assert manifest["daily_ohlc_range_touch_is_optimistic"] is True
    assert manifest["cannot_override_15m_causal_test"] is True
    assert manifest["as_of_completed_session"] == str(raw["date"].max().date())
    assert (output / "report.html").is_file()
    with pytest.raises(FileExistsError):
        write_daily_gap_research_artifacts(
            result,
            output,
            input_provenance={"price_input_path": str(input_path)},
        )


def test_normalizer_rejects_impossible_ohlc_and_nonfinite_values() -> None:
    raw = _daily_rows("AAA", pd.DatetimeIndex(["2026-01-02"]))
    raw.loc[0, "High"] = 98.0
    with pytest.raises(ValueError, match="impossible OHLC"):
        normalize_daily_prices(raw, as_of="2026-01-02")
    raw = _daily_rows("AAA", pd.DatetimeIndex(["2026-01-02"]))
    raw.loc[0, "Volume"] = np.inf
    with pytest.raises(ValueError, match="non-finite volume"):
        normalize_daily_prices(raw, as_of="2026-01-02")


def test_broad_cache_mode_drops_and_audits_only_malformed_rows() -> None:
    raw = _daily_rows("AAA", pd.DatetimeIndex(["2026-01-02", "2026-01-05"]))
    raw.loc[1, "Open"] = 0.0
    normalized, original = normalize_daily_prices(
        raw,
        as_of="2026-01-05",
        drop_invalid_rows=True,
    )
    assert original == 2
    assert normalized[["ticker", "date"]].to_dict("records") == [
        {"ticker": "AAA", "date": pd.Timestamp("2026-01-02")}
    ]
    assert normalized.attrs["normalization_rejections"][0]["rejection_reason"] == (
        "nonpositive_ohlc"
    )


def test_as_of_precedes_duplicate_and_invalid_row_audits() -> None:
    base = _daily_rows("AAA", pd.DatetimeIndex(["2026-08-27"]))
    future = _daily_rows("AAA", pd.DatetimeIndex(["2026-08-28"]))
    future.loc[0, "Open"] = 0.0
    raw = pd.concat([base, future, future], ignore_index=True)
    normalized, original = normalize_daily_prices(
        raw,
        as_of="2026-08-27",
        drop_invalid_rows=True,
    )
    assert original == 3
    assert len(normalized) == 1
    assert normalized.attrs["normalization_rejections"] == []

    admitted_duplicate = pd.concat([base, base.assign(Open=0.0)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate ticker/date"):
        normalize_daily_prices(
            admitted_duplicate,
            as_of="2026-08-27",
            drop_invalid_rows=True,
        )


def test_missing_canonical_predecessor_excludes_the_next_row() -> None:
    dates = pd.date_range("2026-01-02", periods=40, freq=TRADING_DAY)
    raw = _daily_rows("AAA", dates)
    raw.loc[raw.index[-1], ["Open", "High", "Low", "Close"]] = [99.0, 101.0, 98.0, 100.0]
    raw = raw.loc[~raw["date"].eq(dates[-2])].copy()
    normalized, _ = normalize_daily_prices(raw, as_of=dates[-1])
    candidates, eligibility = prepare_daily_candidates(normalized)
    assert not candidates["date"].eq(dates[-1]).any()
    assert eligibility.loc[0, "n_predecessor_adjacency_fail"] >= 2


def test_malformed_predecessor_is_audited_and_taints_successor() -> None:
    dates = pd.date_range("2026-01-02", periods=40, freq=TRADING_DAY)
    raw = _daily_rows("AAA", dates)
    raw.loc[raw.index[-2], "Open"] = 0.0
    raw.loc[raw.index[-1], ["Open", "High", "Low", "Close"]] = [99.0, 101.0, 98.0, 100.0]
    normalized, _ = normalize_daily_prices(
        raw,
        as_of=dates[-1],
        drop_invalid_rows=True,
    )
    assert normalized.attrs["normalization_rejections"][0]["date"] == dates[-2]
    candidates, eligibility = prepare_daily_candidates(normalized)
    assert not candidates["date"].eq(dates[-1]).any()
    assert eligibility.loc[0, "n_predecessor_adjacency_fail"] >= 2


def test_hac_primary_inference_is_present_and_deterministic() -> None:
    days = pd.date_range("2025-01-02", periods=80, freq=TRADING_DAY)
    rows = []
    for index, day in enumerate(days):
        rows.append(
            {
                "arm_id": LONG_ARM_ID,
                "ticker": f"L{index % 4}",
                "date": day,
                "gap_atr": 1.0,
                "filled": True,
                "gross_return": 0.002 + 0.001 * np.sin(index / 5.0),
            }
        )
        rows.append(
            {
                "arm_id": SHORT_ARM_ID,
                "ticker": f"S{index % 4}",
                "date": day,
                "gap_atr": 1.0,
                "filled": True,
                "gross_return": 0.001 + 0.001 * np.cos(index / 5.0),
            }
        )
    candidates = pd.DataFrame(rows)
    first = build_material_gap_views(candidates, days, bootstrap_reps=100)[2]
    second = build_material_gap_views(candidates, days, bootstrap_reps=100)[2]
    assert first["hac_lag"].ge(1).all()
    assert first["hac_standard_error"].gt(0).all()
    assert first["holm_hac_p_value_primary"].notna().all()
    pd.testing.assert_series_equal(
        first["holm_hac_p_value_primary"], second["holm_hac_p_value_primary"]
    )


def test_evaluation_start_keeps_pre_start_warmup_and_report_limitations() -> None:
    raw = _two_arm_prices()
    normalized, original = normalize_daily_prices(raw, as_of=raw["date"].max())
    frozen = freeze_universe(["AAA", "BBB"], normalized["ticker"].unique())
    start = raw["date"].max()
    result = run_daily_gap_reversal_research(
        normalized,
        frozen,
        as_of=start,
        evaluation_start=start,
        original_row_count=original,
        bootstrap_reps=100,
    )
    assert result.evaluation_start == start
    assert len(result.study_sessions) == 1
    assert set(result.selected_orders["date"]) == {start}
    assert result.eligibility_summary["n_rows"].eq(23).all()
    report = daily_gap.build_daily_gap_html(result)
    assert "Mean @20bps" in report
    assert "Calendar-year diagnostics" in report
    assert "Leave-one-year-out diagnostics" in report
    assert "Largest ticker concentrations" in report
    assert "Current-universe and price-vintage limitations" in report
    assert "Optimistic range-touch screen" in report
