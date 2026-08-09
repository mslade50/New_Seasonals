"""Shared test fixtures. Currently the Daily Pitch survey scaffolding.

Since 2026-08-08 the publisher refuses ideas whose morning cannot be found on
disk: the day's `scratch/pitch_checks/<asof>/` folder must exist, hold stage
B1's surface map and real check scripts, and every evidence path must resolve
inside it. That means every payload fixture needs a real surveyed morning
behind it. Building one here keeps the two pitch test modules from drifting
into two different notions of what a surveyed morning looks like, and keeps
both of them off the repo's actual scratch directory.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def checks_root(tmp_path, monkeypatch):
    """An isolated checks root, installed as pitch_grammar's default so a test
    never passes or fails on what happens to be sitting in scratch/."""
    import pitch_grammar as pg
    root = tmp_path / "pitch_checks"
    root.mkdir()
    monkeypatch.setattr(pg, "CHECKS_ROOT", root)
    return root


@pytest.fixture()
def survey(checks_root):
    """Give a payload a surveyed morning: the day folder, the surface map, a
    check and a development script per idea, evidence pointed at them all.

    Returns the day directory. Pass `root=` to write into a different (e.g.
    yesterday's) folder, or `dev=False` to leave out the round-3 scripts.
    """
    import pitch_grammar as pg

    def _survey(payload, root=None, dev=True):
        day = Path(root or checks_root) / str(payload.get("asof", ""))
        day.mkdir(parents=True, exist_ok=True)
        (day / pg.SURFACE_MAP_NAME).write_text("# surface map\n",
                                               encoding="utf-8")
        fields = ("script", "dev_script") if dev else ("script",)
        for i, idea in enumerate(payload.get("ideas") or [], 1):
            evidence = idea.setdefault("evidence", {})
            for field in fields:
                script = day / f"idea{i}_{field}.py"
                script.write_text("# a real check\n", encoding="utf-8")
                evidence[field] = str(script)
        return day

    return _survey
