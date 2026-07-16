"""Pilot promotion review (2026-07-16): a pilot strategy must carry terms.

The 'B (Pilot)' grade was inert metadata — nothing consumed it, and pilot
sizing decisions ('consider 40 bps after clean quarters') lived in prose.
Every pilot now carries execution['pilot'] with a start date, review
deadline, and explicit promote criteria, so the size-up review can't be
forgotten. NO kill criteria by McKinley's call (2026-07-16, "we aren't
gonna kill those") — pilots stay regardless; the review is size-up only.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _NoOp:
    def __getattr__(self, name): return self
    def __call__(self, *a, **k): return self
    def __bool__(self): return False
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def cache_data(self, *a, **k):
        def deco(fn): return fn
        return deco
    cache_resource = cache_data


sys.modules['streamlit'] = _NoOp()

from strategy_config import STRATEGY_BOOK

REQUIRED_KEYS = {"start", "review_by", "promote_if"}


def _pilots():
    return [s for s in STRATEGY_BOOK
            if 'pilot' in str(s.get('stats', {}).get('grade', '')).lower()]


def test_pilots_exist():
    assert len(_pilots()) >= 2  # bear fade + leader gap fade as of 2026-07


def test_every_pilot_has_governance_block():
    for s in _pilots():
        blk = s.get('execution', {}).get('pilot')
        assert isinstance(blk, dict), (
            f"{s['name']}: grade says Pilot but execution has no 'pilot' block "
            f"— add start/review_by/promote_if/kill_if"
        )
        missing = REQUIRED_KEYS - set(blk)
        assert not missing, f"{s['name']}: pilot block missing {sorted(missing)}"
        for k in REQUIRED_KEYS:
            assert str(blk[k]).strip(), f"{s['name']}: pilot['{k}'] is empty"


def test_governance_block_implies_pilot_grade():
    # inverse direction: a strategy that keeps a pilot block after promotion
    # should have been cleaned up (or re-graded) — flag the mismatch
    for s in STRATEGY_BOOK:
        if s.get('execution', {}).get('pilot'):
            assert 'pilot' in str(s.get('stats', {}).get('grade', '')).lower(), (
                f"{s['name']}: has execution['pilot'] but grade is not Pilot — "
                f"promote/kill happened? Remove or update the block."
            )
