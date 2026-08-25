import json
from pathlib import Path

import pytest

from scripts.stage_private_site_cloud_build import is_runtime_path, stage_source


def test_stage_copies_tracked_code_but_never_runtime_data(tmp_path):
    source = tmp_path / "source"
    dest = tmp_path / "assembler"
    files = {
        "app.py": "print('ok')",
        "site/index.html": "<h1>site</h1>",
        "functions/chart.js": "export default {}",
        "data/local.json": "LOCAL",
        "data/sp500_risk_classification.csv": "ticker,beta_2y,label\nSPY,1.0,neutral\n",
        "data/signal_horizon_stats.json": '{"signals": {}}',
        "data/macro_events.csv": "date,event\n2026-08-21,opex\n",
        "dist/data/meta.json": "LOCAL DIST",
        "scratch/probe.py": "LOCAL SCRATCH",
        "credentials.json": "SECRET",
        "atr_seasonal_ranks.parquet": "LOCAL PARQUET",
        "wrangler.toml": 'pages_build_output_dir = "dist"',
    }
    for rel, value in files.items():
        path = source / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")

    marker = stage_source(
        source,
        dest,
        tracked_files=files,
        source_sha="a" * 40,
    )

    assert (dest / "app.py").is_file()
    assert (dest / "site/index.html").is_file()
    assert (dest / "reference/sp500_risk_classification.csv").read_text(
        encoding="utf-8"
    ).startswith("ticker,beta_2y,label")
    assert json.loads((dest / "reference/signal_horizon_stats.json").read_text(
        encoding="utf-8"
    )) == {"signals": {}}
    assert (dest / "reference/macro_events.csv").read_text(
        encoding="utf-8"
    ).startswith("date,event")
    assert not (dest / "data").exists()
    assert not (dest / "dist").exists()
    assert not (dest / "scratch").exists()
    assert not (dest / "credentials.json").exists()
    assert not (dest / "atr_seasonal_ranks.parquet").exists()
    wrangler = (dest / "wrangler.toml").read_text(encoding="utf-8")
    assert 'name = "seasonals-mslade"' in wrangler
    assert 'pages_build_output_dir = "dist"' in wrangler
    assert marker["source_sha"] == "a" * 40
    saved = json.loads((dest / ".private-site-cloud-stage.json").read_text())
    assert saved["mode"] == "private-site-cloud-source"


def test_stage_refuses_nonempty_destination(tmp_path):
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    source.mkdir()
    dest.mkdir()
    (dest / "leftover.txt").write_text("old", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not empty"):
        stage_source(source, dest, tracked_files=[], source_sha="abc")


@pytest.mark.parametrize(
    "path",
    [
        "data/a.json",
        "dist/index.html",
        "charts/a.png",
        "scratch/a.py",
        ".env",
        "credentials.json",
        "atr_seasonal_ranks.parquet",
        "wrangler.toml",
    ],
)
def test_runtime_paths_are_excluded(path):
    assert is_runtime_path(path)
