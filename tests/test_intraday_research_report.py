from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from research.intraday.report import evaluate_bundle, write_report
from research.intraday.templates import (
    GAP_FIRST_HOUR_TEMPLATE_ID,
    INTRADAY_SHOCK_TEMPLATE_ID,
)

TEMPLATES = (GAP_FIRST_HOUR_TEMPLATE_ID, INTRADAY_SHOCK_TEMPLATE_ID)


def _bundle(root: Path) -> Path:
    bundle = root / "bundle"
    bundle.mkdir(parents=True)
    manifest = {
        "research_only": True,
        "no_order": True,
        "production_writes": False,
        "automatic_promotion": False,
        "requested_tickers": ["AAA"],
        "n_primary_trades": 2,
        "n_exact_full_sessions": 100,
        "raw_source_provenance": {
            "snapshot_ticker_count": 197,
            "snapshot_source_meta_sha256": "a" * 64,
            "universe_file_sha256": "b" * 64,
            "sector_map_file_sha256": "c" * 64,
        },
    }
    (bundle / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    stats = []
    for template in TEMPLATES:
        for cost, mean in ((5.0, -0.0002), (10.0, -0.0007), (20.0, -0.0017)):
            stats.append(
                {
                    "template_id": template,
                    "cost_bps": cost,
                    "mean_daily_return": mean,
                    "bootstrap_mean_ci_2_5": mean - 0.0001,
                    "bootstrap_mean_ci_97_5": mean + 0.0001,
                    "holm_p_value_primary": 0.01 if cost == 10 else None,
                }
            )
    pd.DataFrame(stats).to_csv(bundle / "day_cluster_stats.csv", index=False)
    pd.DataFrame(
        [
            {
                "template_id": template,
                "cost_bps": 10.0,
                "test_year": 2025,
                "test_mean_return": -0.0005,
                "eligible_for_stability_gate": True,
            }
            for template in TEMPLATES
        ]
    ).to_csv(bundle / "rolling_5y_train_1y_test.csv", index=False)
    pd.DataFrame(
        [
            {
                "template_id": template,
                "cost_bps": 10.0,
                "capacity_slots": 3,
                "mean_session_return": -0.0004,
            }
            for template in TEMPLATES
        ]
    ).to_csv(bundle / "capacity_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "template_id": template,
                "cost_bps": 10.0,
                "omitted_year": 2025,
                "remaining_mean_return": -0.0006,
            }
            for template in TEMPLATES
        ]
    ).to_csv(bundle / "leave_one_year_out.csv", index=False)
    concentration = [
        {
            "template_id": template,
            "group": "AAA" if kind == "ticker" else "Technology",
            "share_of_template_absolute_endpoint_contribution": 0.25,
        }
        for template in TEMPLATES
        for kind in ("ticker",)
    ]
    pd.DataFrame(concentration).to_csv(bundle / "ticker_summary.csv", index=False)
    for row in concentration:
        row["group"] = "Technology"
    pd.DataFrame(concentration).to_csv(bundle / "sector_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "template_id": template,
                "cost_bps": 10.0,
                "year": 2025,
                "mean_active_day_return": -0.0005,
            }
            for template in TEMPLATES
        ]
    ).to_csv(bundle / "annual_stats.csv", index=False)
    pd.DataFrame(
        [
            {
                "template_id": template,
                "cost_bps": 10.0,
                "direction": direction,
                "n_trades": 1,
                "mean_daily_equal_notional_return": -0.0005,
                "win_rate_daily": 0.4,
            }
            for template in TEMPLATES
            for direction in ("long", "short")
        ]
    ).to_csv(bundle / "side_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "view": "preregistered_filter_on",
                "n_signals": 1,
                "mean_daily_equal_notional_gross_return": 0.0,
                "n_additional_signals_vs_filtered": 0,
            },
            {
                "view": "audit_filter_off",
                "n_signals": 2,
                "mean_daily_equal_notional_gross_return": -0.0001,
                "n_additional_signals_vs_filtered": 1,
            },
        ]
    ).to_csv(bundle / "discontinuity_sensitivity.csv", index=False)
    pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "role": "candidate",
                "status": "loaded",
                "exclusion_reason": "",
            }
        ]
    ).to_csv(bundle / "coverage_audit.csv", index=False)
    pd.DataFrame(
        [{"input_rejection_reasons": "market_session_status:half_day"}]
    ).to_parquet(bundle / "signal_input_rejections.parquet", index=False)
    pd.DataFrame([{"execution_status": "missing_scheduled_entry_bar"}]).to_parquet(
        bundle / "execution_rejections.parquet", index=False
    )
    return bundle


def test_negative_primary_means_fail_closed_to_reject(tmp_path: Path):
    bundle = _bundle(tmp_path)
    verdicts = evaluate_bundle(bundle)
    assert [item.status for item in verdicts] == ["Reject v0", "Reject v0"]


def test_report_is_confined_refuses_overwrite_and_manifests_sha(tmp_path: Path):
    artifact_root = tmp_path / "artifacts"
    bundle = _bundle(artifact_root)
    report = write_report(bundle, artifact_root=artifact_root)
    assert report.is_file()
    text = report.read_text(encoding="utf-8")
    assert "both templates fail" in text
    manifest = json.loads((bundle / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["report"]["filename"] == "report.html"
    assert len(manifest["report"]["sha256"]) == 64
    assert manifest["report"]["manifest_rewritten_after_report"] is True
    with pytest.raises(FileExistsError):
        write_report(bundle, artifact_root=artifact_root)


def test_report_rejects_bundle_outside_artifact_root(tmp_path: Path):
    allowed = tmp_path / "allowed"
    bundle = _bundle(tmp_path / "outside")
    with pytest.raises(ValueError, match="artifact root"):
        write_report(bundle, artifact_root=allowed)

