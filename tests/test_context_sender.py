"""Guards for the Market Context publish path (scripts/send_context_slack.py).

The gates here are the only thing standing between a half-finished evening and
Scott's phone, and every one of them fails open if it is subtly wrong: a
parser that silently finds zero nuggets posts a header with nothing under it,
and a linter whose regex never matches passes every brief.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import send_context_slack as snd  # noqa: E402


GOOD_BRIEF = """# Market Context — Monday 2026-08-10

**Headline:** August payroll sessions run against the full-sample drift, mean -0.36% versus +0.05%.

## Tomorrow's tape
1. **^VIX — payroll sessions** [solid]
   VIX closed lower on 208 of 317 payroll sessions, mean -1.14% (t=-2.6).
   Era-stable across the 2018 split.

2. **SPY — August payroll sessions** [suggestive]
   Restricted to August the cell inverts: n=26, mean -0.36%, hit 34.6%.

## Today in context
1. **DIA — down 50bp after a 52w high** [suggestive]
   91 prior instances, next-session mean +0.07%, record 54-37, sign p 0.047.

2. **FXI — five consecutive down closes** [anecdote]
   12 prior streaks, next-session mean +0.11%.

## Calendar
- Wed 2026-08-12 08:30 ET, CPI

---
*Cells scanned: 1273. Conventions: close-to-close forward returns, 1999+.*
"""

GOOD_SIDECAR = {
    "asof": "2026-08-06", "quiet": False, "prices_stale": False,
    "headline": "August payroll sessions run against the full-sample drift.",
    "cells_scanned": 1273,
    "nuggets": [
        {"lane": "tomorrow", "fingerprint": "E:nfp|^VIX|k1", "subject": "^VIX",
         "cell": "payroll sessions", "tag": "solid", "n": 317,
         "mean_pct": -1.141},
        {"lane": "tomorrow", "fingerprint": "E:nfp|SPY|k1", "subject": "SPY",
         "cell": "August payroll sessions", "tag": "suggestive", "n": 26,
         "mean_pct": -0.358},
        {"lane": "today", "fingerprint": "P3:drop50_after_high|DIA",
         "subject": "DIA", "cell": "reversal", "tag": "suggestive", "n": 91,
         "mean_pct": 0.066},
        {"lane": "today", "fingerprint": "P7b:down_streak|FXI",
         "subject": "FXI", "cell": "streak", "tag": "anecdote", "n": 12,
         "mean_pct": 0.111},
    ],
}


@pytest.fixture
def cell_map(tmp_path, monkeypatch):
    """A cell map on disk for the date under test, in an isolated tree."""
    def _make(run_date: str = "2026-08-09", exists: bool = True):
        monkeypatch.setattr(snd, "CELL_MAP_DIR", tmp_path / "checks")
        if exists:
            d = tmp_path / "checks" / run_date
            d.mkdir(parents=True)
            (d / "00_cell_map.md").write_text("map", encoding="utf-8")
        return run_date
    return _make


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------
def test_parse_finds_title_headline_lanes_and_footer():
    brief = snd.parse_brief(GOOD_BRIEF)
    assert brief["title"] == "Market Context — Monday 2026-08-10"
    assert brief["headline"].startswith("August payroll sessions")
    assert len(brief["items"]["Tomorrow's tape"]) == 2
    assert len(brief["items"]["Today in context"]) == 2
    assert brief["items"]["Tomorrow's tape"][0]["tag"] == "solid"
    assert "Cells scanned: 1273" in brief["footer"]


def test_parse_keeps_multi_line_bodies_together():
    brief = snd.parse_brief(GOOD_BRIEF)
    body = brief["items"]["Tomorrow's tape"][0]["body"]
    assert "208 of 317" in body and "Era-stable" in body


def test_parse_does_not_mistake_a_calendar_bullet_for_a_nugget():
    brief = snd.parse_brief(GOOD_BRIEF)
    assert brief["items"]["Calendar"] == []


def test_to_mrkdwn_bold_italic_link():
    assert snd.to_mrkdwn("**bold**") == "*bold*"
    assert snd.to_mrkdwn("*it*") == "_it_"
    assert snd.to_mrkdwn("[t](u)") == "<u|t>"
    # A single '*' left behind by the bold pass must not become italic.
    assert snd.to_mrkdwn("**a** and **b**") == "*a* and *b*"


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
def test_good_brief_has_no_hard_issues(cell_map):
    run_date = cell_map()
    hard, _ = snd.lint_brief(snd.parse_brief(GOOD_BRIEF), GOOD_SIDECAR, run_date)
    assert hard == []


def test_missing_cell_map_blocks_publication(cell_map):
    run_date = cell_map(exists=False)
    hard, _ = snd.lint_brief(snd.parse_brief(GOOD_BRIEF), GOOD_SIDECAR, run_date)
    assert any("cell map" in h for h in hard)


def test_missing_sidecar_blocks_publication(cell_map):
    run_date = cell_map()
    hard, _ = snd.lint_brief(snd.parse_brief(GOOD_BRIEF), {}, run_date)
    assert any("sidecar" in h for h in hard)


def test_anecdote_budget_is_enforced(cell_map):
    run_date = cell_map()
    text = GOOD_BRIEF.replace("[suggestive]", "[anecdote]")
    sidecar = json.loads(json.dumps(GOOD_SIDECAR))
    for n in sidecar["nuggets"]:
        if n["tag"] == "suggestive":
            n["tag"] = "anecdote"
    hard, _ = snd.lint_brief(snd.parse_brief(text), sidecar, run_date)
    assert any("anecdote" in h and "budget" in h for h in hard)


def test_an_anecdote_may_not_lead(cell_map):
    run_date = cell_map()
    text = GOOD_BRIEF.replace("[solid]", "[anecdote]")
    hard, _ = snd.lint_brief(snd.parse_brief(text), GOOD_SIDECAR, run_date)
    assert any("never be the headline" in h for h in hard)


def test_unknown_tag_is_hard(cell_map):
    run_date = cell_map()
    text = GOOD_BRIEF.replace("[solid]", "[strong]")
    hard, _ = snd.lint_brief(snd.parse_brief(text), GOOD_SIDECAR, run_date)
    assert any("must be one of" in h for h in hard)


def test_advisory_imperative_is_hard(cell_map):
    run_date = cell_map()
    text = GOOD_BRIEF.replace("Era-stable across the 2018 split.",
                              "Fade the move into the print.")
    hard, _ = snd.lint_brief(snd.parse_brief(text), GOOD_SIDECAR, run_date)
    assert any("advisory verb" in h for h in hard)


def test_a_rate_cut_is_not_an_imperative(cell_map):
    """The ban is on instructions, not on the word. 'the Fed cut rates' and
    'short-dated vol' have to survive or the linter is unusable."""
    run_date = cell_map()
    text = GOOD_BRIEF.replace("Era-stable across the 2018 split.",
                              "The cell spans the 2024 cut cycle and short "
                              "dated vol was bid throughout.")
    hard, soft = snd.lint_brief(snd.parse_brief(text), GOOD_SIDECAR, run_date)
    assert not any("advisory verb" in h for h in hard)
    assert not any("opens a sentence" in s for s in soft)


def test_an_ambiguous_opener_warns_without_blocking(cell_map):
    """'Cut cycles have been kind to duration' is a legal sentence; the author
    gets told, the post still goes."""
    run_date = cell_map()
    text = GOOD_BRIEF.replace("Era-stable across the 2018 split.",
                              "Cut cycles have been kind to duration here.")
    hard, soft = snd.lint_brief(snd.parse_brief(text), GOOD_SIDECAR, run_date)
    assert hard == []
    assert any("opens a sentence" in s for s in soft)


def test_nugget_count_floor_and_ceiling(cell_map):
    run_date = cell_map()
    text = GOOD_BRIEF.split("2. **SPY")[0] + "\n## Calendar\n- x\n"
    sidecar = {**GOOD_SIDECAR, "nuggets": GOOD_SIDECAR["nuggets"][:1]}
    hard, _ = snd.lint_brief(snd.parse_brief(text), sidecar, run_date)
    assert any("ships 4 to 8" in h for h in hard)


def test_quiet_brief_skips_the_nugget_floor(cell_map):
    run_date = cell_map()
    quiet_text = ("# Market Context — Monday 2026-08-10\n\n"
                  "## QUIET TAPE\n- Nothing scheduled inside 3 sessions.\n"
                  "- No price-state trigger fired.\n\n## Calendar\n- CPI Wed\n"
                  "\n---\n*Cells scanned: 1204.*\n")
    hard, _ = snd.lint_brief(snd.parse_brief(quiet_text),
                             {"asof": "2026-08-06", "quiet": True,
                              "nuggets": []}, run_date)
    assert hard == []


def test_markdown_and_sidecar_must_agree_on_the_count(cell_map):
    run_date = cell_map()
    sidecar = {**GOOD_SIDECAR, "nuggets": GOOD_SIDECAR["nuggets"][:2]}
    hard, _ = snd.lint_brief(snd.parse_brief(GOOD_BRIEF), sidecar, run_date)
    assert any("same brief" in h for h in hard)


def test_quotes_n_accepts_the_ways_a_nugget_states_it():
    for good in ["n=317", "208 of 317 payroll sessions",
                 "317 payroll sessions since 2000", "91 prior instances",
                 "12 prior streaks", "26 August prints"]:
        assert snd.QUOTES_N.search(good), good
    assert not snd.QUOTES_N.search("One hundred and twenty six prior streaks")


# ---------------------------------------------------------------------------
# blocks
# ---------------------------------------------------------------------------
def test_blocks_carry_the_standing_colour_and_every_nugget():
    blocks, color = snd.build_blocks(snd.parse_brief(GOOD_BRIEF), GOOD_SIDECAR)
    assert color == snd.BRAND_COLOR
    assert blocks[0]["type"] == "header"
    text = json.dumps(blocks)
    for subject in ("^VIX", "SPY", "DIA", "FXI"):
        assert subject in text
    assert "Cells scanned" in text


def test_stale_prices_change_the_colour_and_add_a_banner():
    sidecar = {**GOOD_SIDECAR, "prices_stale": True, "last_bar": "2026-08-05"}
    blocks, color = snd.build_blocks(snd.parse_brief(GOOD_BRIEF), sidecar)
    assert color == snd.STALE_COLOR
    assert "PRICES STALE" in json.dumps(blocks)


def test_quiet_section_survives_into_the_blocks():
    quiet_text = ("# Market Context — Monday 2026-08-10\n\n"
                  "## QUIET TAPE\n- Nothing scheduled inside 3 sessions.\n")
    blocks, _ = snd.build_blocks(snd.parse_brief(quiet_text), {"quiet": True})
    assert "Nothing scheduled" in json.dumps(blocks)


# ---------------------------------------------------------------------------
# after publish
# ---------------------------------------------------------------------------
def test_flag_state_advances_only_what_published(tmp_path, monkeypatch):
    path = tmp_path / "flags.json"
    monkeypatch.setattr(snd, "FLAG_STATE_PATH", path)
    snd.advance_flag_state(GOOD_SIDECAR, "2026-08-09")
    flags = json.loads(path.read_text(encoding="utf-8"))["flags"]
    assert set(flags) == {n["fingerprint"] for n in GOOD_SIDECAR["nuggets"]}
    assert flags["E:nfp|^VIX|k1"]["last_published"] == "2026-08-09"
    assert flags["E:nfp|^VIX|k1"]["count"] == 1
    # A second publish increments rather than resetting.
    snd.advance_flag_state(GOOD_SIDECAR, "2026-08-10")
    flags = json.loads(path.read_text(encoding="utf-8"))["flags"]
    assert flags["E:nfp|^VIX|k1"]["count"] == 2
    assert flags["E:nfp|^VIX|k1"]["last_published"] == "2026-08-10"


def test_corrupt_flag_state_is_rebuilt_not_fatal(tmp_path, monkeypatch):
    path = tmp_path / "flags.json"
    path.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(snd, "FLAG_STATE_PATH", path)
    snd.advance_flag_state(GOOD_SIDECAR, "2026-08-09")
    assert json.loads(path.read_text(encoding="utf-8"))["flags"]


def test_journal_appends_one_record_per_nugget(tmp_path, monkeypatch):
    path = tmp_path / "journal.jsonl"
    monkeypatch.setattr(snd, "JOURNAL_PATH", path)
    snd.append_journal(GOOD_SIDECAR, "2026-08-09", Path("2026-08-09.md"))
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(GOOD_SIDECAR["nuggets"])
    assert {r["kind"] for r in rows} == {"nugget"}
    assert rows[0]["run_date"] == "2026-08-09"


def test_quiet_evening_still_journals_one_record(tmp_path, monkeypatch):
    path = tmp_path / "journal.jsonl"
    monkeypatch.setattr(snd, "JOURNAL_PATH", path)
    snd.append_journal({"asof": "2026-08-06", "quiet": True, "nuggets": []},
                       "2026-08-09", Path("2026-08-09.md"))
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1 and rows[0]["kind"] == "quiet"
