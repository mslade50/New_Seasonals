"""Guards for the Daily Posts pipeline: grammar lint (disclosure + opsec),
queue md parsing + mark ingest, journal folding, replay-row derivation and
its compatibility with the pitch grader's replay."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import posts_journal  # noqa: E402
from posts_grammar import (  # noqa: E402
    _universe_sets, derive_order_row, fingerprint, validate_queue,
)
from build_posts_state import ingest_queue_marks, parse_queue_md  # noqa: E402


def _stat(text="SPY up 1.2% on the day, 34 of 41 Augusts closed higher",
          **kw):
    return {"id": kw.pop("id", "x20260810-1"), "type": "stat",
            "text": text, **kw}


def _idea(**kw):
    idea = {"ticker": "IWM", "side": "long",
            "entry": {"type": "LIMIT", "anchor": "open", "offset_atr": -0.25},
            "time_td": 2, "atr": 4.2, "ref_close": 224.5,
            "execute_on": "2026-08-11"}
    idea.update(kw.pop("idea", {}))
    draft = {"id": kw.pop("id", "x20260810-2"), "type": "idea",
             "text": "long IWM tomorrow, open -0.25 ATR, 2 day hold. "
                     "23 of 31 comparable setups closed higher.",
             "evidence": {"summary": "post-opex week cell", "n": 31},
             "idea": idea}
    draft.update(kw)
    return draft


def _queue(*drafts):
    return {"asof": "2026-08-10", "drafts": list(drafts)}


class TestGrammar:
    def test_clean_queue_passes(self):
        hard, _ = validate_queue(_queue(_stat(), _idea()))
        assert hard == []

    def test_identity_strings_block(self):
        hard, _ = validate_queue(_queue(_stat(
            "backtests live in new_seasonals, 5 of 6 up")))
        assert any("identity" in h for h in hard)

    def test_soft_identity_warns_not_blocks(self):
        hard, soft = validate_queue(_queue(_stat(
            "Scott Bessent speaks at 10am, 7 of 9 auction days green")))
        assert hard == []
        assert any("scott" in s for s in soft)

    def test_journal_post_may_not_name_tickers(self):
        hard, _ = validate_queue(_queue(
            {"id": "j1", "type": "journal",
             "text": "added to the $SPY dip book today, third leg"}))
        assert any("journal" in h and "SPY" in h for h in hard)

    def test_dollar_sizes_block_prices_pass(self):
        hard, _ = validate_queue(_queue(_stat(
            "we sized it at $15k of risk, 6 of 8 worked")))
        assert any("dollar size" in h for h in hard)
        hard, _ = validate_queue(_queue(_stat(
            "the $224 level held 6 of 8 times")))
        assert hard == []

    def test_stat_needs_a_number(self):
        hard, _ = validate_queue(_queue(_stat("seasonality is favorable")))
        assert any("carry its number" in h for h in hard)

    def test_overlong_needs_long_flag(self):
        hard, _ = validate_queue(_queue(_stat("7 " + "x" * 300)))
        assert any("chars" in h for h in hard)
        hard, _ = validate_queue(_queue(_stat("7 " + "x" * 300, long=True)))
        assert hard == []

    def test_idea_requires_time_stop_and_frozen_atr(self):
        bad = _idea()
        del bad["idea"]["time_td"]
        hard, _ = validate_queue(_queue(bad))
        assert any("time_td" in h for h in hard)
        bad = _idea()
        bad["idea"]["atr"] = 0
        hard, _ = validate_queue(_queue(bad))
        assert any("atr" in h for h in hard)

    def test_overflow_ticker_blocks_everywhere(self):
        liquid, overflow = _universe_sets()
        if not overflow:
            pytest.skip("strategy_config unavailable")
        name = sorted(t for t in overflow if t.isalpha() and len(t) >= 4)[0]
        hard, _ = validate_queue(_queue(_stat(
            f"{name} up 8 days straight")))
        assert any("overflow" in h for h in hard)
        hard, _ = validate_queue(_queue(_idea(idea={"ticker": name})))
        assert any("overflow" in h for h in hard)

    def test_repetition_needs_changed_since(self):
        draft = _idea()
        fp = fingerprint(draft["idea"])
        hard, _ = validate_queue(_queue(draft), {fp: "2026-08-05"})
        assert any("changed_since" in h for h in hard)
        draft["changed_since"] = "post-opex cell refreshed, N 27->31"
        hard, _ = validate_queue(_queue(draft), {fp: "2026-08-05"})
        assert hard == []


class TestQueueIngest:
    MD = """# Daily Posts queue - 2026-08-08

## 1. [stat] id=q1
Posted: yes

SPY stat text as posted

## 2. [idea] id=q2
Posted: https://x.com/acct/status/123

idea text, edited by hand

## 3. [take] id=q3
Posted: no

never went out
"""

    def test_parse_queue_md(self):
        blocks = parse_queue_md(self.MD)
        assert [b["id"] for b in blocks] == ["q1", "q2", "q3"]
        assert blocks[0]["mark"] == "yes"
        assert blocks[1]["mark"].startswith("https://")
        assert blocks[2]["mark"] == "no"
        assert blocks[1]["text"] == "idea text, edited by hand"

    def test_ingest_is_idempotent_and_captures_url(self, tmp_path):
        qdir = tmp_path / "queue"
        qdir.mkdir()
        (qdir / "2026-08-08.md").write_text(self.MD, encoding="utf-8")
        journal = tmp_path / "journal.jsonl"
        posts_journal.append(
            [{"kind": "draft", "draft_id": i, "date": "2026-08-08",
              "type": t} for i, t in
             [("q1", "stat"), ("q2", "idea"), ("q3", "take")]],
            journal)
        warnings: list[str] = []
        out = ingest_queue_marks(qdir, journal, pd.Timestamp("2026-08-10"),
                                 warnings)
        assert out["posted"] == 2 and out["unmarked"] == 1
        again = ingest_queue_marks(qdir, journal, pd.Timestamp("2026-08-10"),
                                   warnings)
        assert again["posted"] == 0
        folded = {d["draft_id"]: d
                  for d in posts_journal.fold_drafts(posts_journal.load(
                      journal, pull=False))}
        assert folded["q1"]["posted"] and folded["q2"]["posted"]
        assert folded["q2"]["posted_url"] == "https://x.com/acct/status/123"
        assert not folded["q3"].get("posted")

    def test_recent_fingerprints_counts_posted_only(self, tmp_path):
        journal = tmp_path / "j.jsonl"
        posts_journal.append(
            [{"kind": "draft", "draft_id": "a", "date": "2026-08-08",
              "type": "idea", "fingerprint": "IWM|long|LIMIT(open)|2-5d"},
             {"kind": "draft", "draft_id": "b", "date": "2026-08-08",
              "type": "idea", "fingerprint": "SPY|short|MOO|1d"},
             {"kind": "posted", "draft_id": "a", "date": "2026-08-08"}],
            journal)
        fps = posts_journal.recent_fingerprints(
            posts_journal.load(journal, pull=False), "2026-08-01")
        assert "IWM|long|LIMIT(open)|2-5d" in fps
        assert "SPY|short|MOO|1d" not in fps


class TestReplayRow:
    def test_close_anchored_limit_and_dates(self):
        draft = _idea(idea={"entry": {"type": "LIMIT", "anchor": "close",
                                      "offset_atr": -0.5}})
        row = derive_order_row(draft)
        assert row["Limit_Price"] == pytest.approx(224.5 - 0.5 * 4.2)
        assert row["Execute_On"] == "2026-08-11"
        assert row["Entry_Expire_Date"] == "2026-08-11"  # window_td default 1
        assert row["Time_Exit_Date"] == "2026-08-12"     # time_td 2
        assert row["Action"] == "BUY" and row["Risk_Amt"] == 100 * 4.2

    def test_open_anchored_leaves_level_to_replay(self):
        row = derive_order_row(_idea())
        assert row["Limit_Price"] == "" and row["Entry_Offset_ATR"] == -0.25

    def test_replay_leg_accepts_derived_row(self):
        # Integration with the pitch grader's replay: MOO long, 2-day hold,
        # no stop/target -> time MOC exit on the second session.
        from grade_pitch_journal import replay_leg
        idx = pd.DatetimeIndex(pd.bdate_range("2026-08-11", periods=4))
        bars = pd.DataFrame(
            {"Open": [100.0, 101.0, 102.0, 103.0],
             "High": [101.0, 102.0, 103.0, 104.0],
             "Low": [99.0, 100.0, 101.0, 102.0],
             "Close": [100.5, 101.5, 102.5, 103.5]}, index=idx)
        draft = _idea(idea={"entry": {"type": "MOO"}, "ticker": "TEST"})
        leg = replay_leg(bars, derive_order_row(draft))
        assert leg["status"] == "closed"
        assert leg["entry_price"] == pytest.approx(100.0)
        assert leg["exit_kind"] == "time_moc"
        assert leg["exit_price"] == pytest.approx(101.5)


class TestScoreboard:
    def test_posted_unposted_split(self):
        from posts_scoreboard import build_scoreboard
        drafts = [
            {"draft_id": "a", "date": "2026-08-05", "type": "idea",
             "posted": True,
             "outcome": {"status": "closed", "r_multiple": 1.0}},
            {"draft_id": "b", "date": "2026-08-05", "type": "idea",
             "outcome": {"status": "closed", "r_multiple": -0.5}},
            {"draft_id": "c", "date": "2026-08-06", "type": "stat",
             "posted": True},
        ]
        board = build_scoreboard(drafts, pd.Timestamp("2026-08-10"))
        roll = board["rolling_60d"]
        assert roll["n"] == 2  # stat posts are not graded material
        assert roll["posted"]["avg_r"] == pytest.approx(1.0)
        assert roll["unposted"]["avg_r"] == pytest.approx(-0.5)
