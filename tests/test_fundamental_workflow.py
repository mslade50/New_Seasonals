from pathlib import Path

from scripts.pull_fundamental_inputs import OPTIONAL


ROOT = Path(__file__).resolve().parents[1]


def test_cloud_research_includes_specialist_lanes() -> None:
    workflow = (ROOT / ".github" / "workflows" / "fundamental_sleeve_research.yml").read_text(
        encoding="utf-8"
    )
    orchestrator = (ROOT / "scripts" / "run_fundamental_sleeve.py").read_text(encoding="utf-8")
    assert "run_fundamental_sleeve.py" in workflow
    assert "--include-specialists" in orchestrator
    assert "--upload" not in workflow


def test_cloud_inputs_preserve_underwrite_decisions() -> None:
    key = "fundamental/current/underwrite_decisions_latest.json"
    assert key in OPTIONAL
    assert OPTIONAL[key].name == "underwrite_decisions_latest.json"
