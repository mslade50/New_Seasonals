"""Guards for the negative-registry parser that feeds the Daily Pitch state.

Why this file exists (2026-08-13): the registry's bullet style changed on
2026-08-10 from a one-line `- **slug** - body` to a sentence-shaped
`- **A whole sentence.**` with the body on indented continuation lines. The
parser only matched the old form, so four mornings of entries parsed to
NOTHING and never reached `research.negative_registry` in the state file --
which is exactly the copy stage B is told to check candidates against, and it
was silently missing the most recent and most relevant lessons. Nothing failed
loudly; the count just sat at 54 while the file grew.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_pitch_research_index import parse_registry  # noqa: E402

MIXED = """# Registry

## Old style

- **book-wide throttle** - aggregate shows no degradation (p = .47). (foo.md)
- **another slug**: a colon separator also counts. (bar.py)

## New style

- **Adding confirming legs to a momentum state does not create a state.**
  Synchronized 52w highs add +0.036pp at h=10 and are NEGATIVE at h=2, 3 and 5.
  (c9_sync_52w_high.py)
- **A single-ticker result has to be priced against its reference class, not
  just against its own bootstrap.** Cochran Q 24.56 on 26 df (p 0.544), so the
  cross-section is homogeneous at zero. (r1b.py)

Prose paragraphs are not entries.

**Correction owed to a published number.** A bold lead-in with no bullet is
not an entry either. (a2.py)
"""


@pytest.fixture()
def mixed(tmp_path):
    path = tmp_path / "registry.md"
    path.write_text(MIXED, encoding="utf-8")
    return parse_registry(path)


def test_both_bullet_styles_parse(mixed):
    assert len(mixed) == 4, [e["key"] for e in mixed]


def test_old_style_keeps_key_and_body(mixed):
    entry = mixed[0]
    assert entry["key"] == "book-wide throttle"
    assert entry["why_dead"].startswith("aggregate shows no degradation")
    assert entry["section"] == "Old style"


def test_wrapped_bold_key_is_joined_before_matching(mixed):
    """The regression that caused this file: a bold key spanning two source
    lines leaves `**` unclosed on the line a per-line regex sees."""
    entry = mixed[3]
    assert entry["key"] == (
        "A single-ticker result has to be priced against its reference class, "
        "not just against its own bootstrap")
    assert "Cochran Q 24.56" in entry["why_dead"]
    assert entry["section"] == "New style"


def test_continuation_lines_join_into_the_body(mixed):
    entry = mixed[2]
    assert entry["key"] == (
        "Adding confirming legs to a momentum state does not create a state")
    assert "+0.036pp at h=10" in entry["why_dead"]
    assert "c9_sync_52w_high.py" in entry["why_dead"]


def test_prose_and_bold_lead_ins_are_not_entries(mixed):
    keys = " ".join(e["key"] for e in mixed)
    assert "Correction owed" not in keys


def test_every_entry_has_a_body_and_a_clean_key(mixed):
    for entry in mixed:
        assert entry["why_dead"], entry["key"]
        assert "*" not in entry["key"]
        assert not entry["key"].endswith(".")


def test_live_registry_parses_every_section():
    """The live file, so a future style drift fails here instead of quietly
    shrinking the state's inlined copy."""
    entries = parse_registry()
    assert len(entries) > 100, f"only {len(entries)} parsed; style drift?"
    assert not [e for e in entries if not e["why_dead"]]
    sections = {e["section"] for e in entries}
    # every dated sweep section since the style changed must be represented
    for stamp in ("2026-08-10", "2026-08-11", "2026-08-12", "2026-08-13"):
        assert any(stamp in s for s in sections), f"no entries for {stamp}"
