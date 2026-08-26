from __future__ import annotations

import pandas as pd
import pytest

from scripts.analyze_episodic_pivot_process import (
    cadence_summary,
    main,
    winner_analysis,
)


def _events() -> pd.DataFrame:
    n = 20
    frame = pd.DataFrame(
        {
            "ticker": [f"T{i:02d}" for i in range(n)],
            "date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-16"]
                + [f"2024-02-{day:02d}" for day in range(1, 18)]
            ),
            "excess_next_open_to_close_20d_pct": list(range(n)),
            "excess_next_open_to_close_60d_pct": list(range(n)),
            "gap_pct": [10.0 + i for i in range(n)],
            "event_rvol_20": [2.0 + i / 10 for i in range(n)],
            "prior_atr_pct_14": [4.1 + i / 10 for i in range(n)],
            "prior_63d_return_pct": [-float(i) for i in range(n)],
            "prior_addv_63": [5_000_000.0 + i * 1_000_000 for i in range(n)],
            "event_date_cluster_size": [1] * n,
            "unique_source_clusters": [2] * n,
            "earnings_date_match": [i % 2 == 0 for i in range(n)],
            "preopen_sec_event_type": ["EARNINGS_GUIDANCE"] * n,
            "secondary_context_event_type": ["EARNINGS_GUIDANCE"] * n,
        }
    )
    frame["evidence_posture"] = "TEST"
    return frame


def test_cadence_summary_includes_zero_calendar_periods():
    cadence, by_year = cadence_summary(_events())
    full_weeks = cadence[
        cadence["window"].eq("Full history") & cadence["period"].eq("week")
    ].iloc[0]

    assert full_weeks["events"] == 20
    assert full_weeks["total_periods"] > full_weeks["active_periods"]
    assert full_weeks["zero_rate_pct"] > 0
    assert by_year.to_dict("records") == [{"year": 2024, "events": 20}]


def test_winner_analysis_separates_fast_and_durable_cohorts():
    facts, traits, composition, atr_quartiles, top_events = winner_analysis(_events())

    assert facts["eligible_20_n"] == 20
    assert facts["eligible_60_n"] == 20
    assert facts["top_20_n"] == 2
    assert facts["top_60_n"] == 2
    assert facts["durable_n"] == 2
    assert facts["top_decile_overlap_n"] == 2
    assert set(traits["cohort"]) == {
        "Balanced baseline",
        "Top-decile 20d",
        "Top-decile 60d",
        "Durable",
    }
    assert set(composition["cohort"]) == set(traits["cohort"])
    assert len(atr_quartiles) == 12
    assert set(top_events["top_list"]) == {"TOP_20D", "DURABLE_60D"}


def test_cli_rejects_incomplete_evidence_before_creating_output(tmp_path):
    candidates = _events().assign(basis_review_cleared=True)
    evidence = candidates[["ticker", "date"]].assign(
        event_id=[f"EVENT-{index}" for index in range(len(candidates))]
    )
    candidates_path = tmp_path / "candidates.parquet"
    evidence_path = tmp_path / "evidence.parquet"
    output_dir = tmp_path / "report"
    candidates.to_parquet(candidates_path, index=False)
    evidence.to_parquet(evidence_path, index=False)

    with pytest.raises(SystemExit, match="evidence input is missing required columns"):
        main(
            [
                "--candidates",
                str(candidates_path),
                "--evidence",
                str(evidence_path),
                "--output-dir",
                str(output_dir),
            ]
        )

    assert not output_dir.exists()
