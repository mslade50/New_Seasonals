"""Cross-lane guards for the local, research-only operating system."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESEARCH_SOURCES = (
    ROOT / "research" / "experiment_registry.py",
    *(ROOT / "research" / "idea_miner").glob("*.py"),
    *(ROOT / "research" / "intraday").glob("*.py"),
    *(ROOT / "research" / "opportunity_book").glob("*.py"),
    *(ROOT / "research" / "trend_v2").glob("*.py"),
    ROOT / "scripts" / "build_wide_opportunity_book.py",
    ROOT / "scripts" / "run_intraday_research.py",
    ROOT / "scripts" / "run_trend_v2_research.py",
    ROOT / "scripts" / "run_weekly_idea_miner.py",
)

BANNED_IMPORT_ROOTS = {
    "alpaca",
    "boto3",
    "cache_io",
    "daily_scan",
    "googleapiclient",
    "gspread",
    "httpx",
    "ib_insync",
    "ibapi",
    "order_staging",
    "requests",
    "smtplib",
    "socket",
    "streamlit",
    "subprocess",
    "yfinance",
}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


@pytest.mark.parametrize("source", RESEARCH_SOURCES, ids=lambda path: path.name)
def test_research_os_has_no_network_broker_or_production_imports(source: Path):
    imported = _imports(source)
    forbidden = sorted(
        name
        for name in imported
        if name.split(".", 1)[0] in BANNED_IMPORT_ROOTS
        or name == "urllib.request"
    )
    assert forbidden == [], f"{source.relative_to(ROOT)} imports {forbidden}"


def test_strategy_config_is_only_a_read_only_universe_source():
    users = [
        source.relative_to(ROOT).as_posix()
        for source in RESEARCH_SOURCES
        if "strategy_config" in _imports(source)
    ]
    assert users == ["research/opportunity_book/core.py"]


def test_research_os_does_not_add_production_entrypoints():
    paths = {path.relative_to(ROOT).as_posix() for path in RESEARCH_SOURCES}
    assert all(
        not path.startswith((".github/", "functions/", "site/", "pages/", "data/"))
        for path in paths
    )
