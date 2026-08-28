import pandas as pd

from scripts import build_pitch_state as pitch


def _receipt(job_id, day, status="success"):
    return {
        "schema_version": pitch.AUTOMATION_RECEIPT_SCHEMA,
        "job_id": job_id,
        "run_date_et": day.isoformat(),
        "status": status,
        "source": "local",
    }


def test_pitch_pipeline_uses_today_am_and_previous_postclose_receipts():
    calls = []

    def fetch(job_id, day):
        calls.append((job_id, day))
        return _receipt(job_id, day)

    today = pd.Timestamp("2026-08-27")
    result = pitch.build_pipeline(
        today,
        tape={"freshest_bar": "2026-08-26"},
        risk={"fragility": {"as_of": "2026-08-26"}, "pc_fear": {}},
        warnings=[],
        fetch=fetch,
    )

    assert result["ok"] is True
    assert ("scan_am", today.date()) in calls
    assert ("portfolio_report", pd.Timestamp("2026-08-26").date()) in calls


def test_pitch_pipeline_surfaces_a_failed_local_job():
    def fetch(job_id, day):
        return _receipt(job_id, day, "failure" if job_id == "risk_am" else "success")

    result = pitch.build_pipeline(
        pd.Timestamp("2026-08-27"),
        tape={"freshest_bar": "2026-08-26"},
        risk={"fragility": {"as_of": "2026-08-26"}, "pc_fear": {}},
        warnings=[],
        fetch=fetch,
    )

    assert result["ok"] is False
    assert any(row["job_id"] == "risk_am" for row in result["missing"])
