"""Which agent product a pitch-grammar publish belongs to (2026-09-30).

The Daily Pitch and the Daily Seasonal share one grammar, one publisher, one
delivery-receipt protocol and one grader. What differs is WHERE each writes:
its journal, scoreboard, watchlist, negative registry, checks folder, Sheets
tab, email subject and R2 receipt namespace. That table lives here and only
here, so a new product is one entry rather than a hunt through five modules.

`pitch` is the default everywhere and its values are the historical paths,
byte for byte (guard: tests/test_seasonal_agent_publish_paths.py). This
module imports nothing from the pitch modules so any of them can import it.

Aligned sites - change together:
- daily_pitch.py (--product), pitch_grammar.py (product-aware validation),
  pitch_delivery.py (receipt namespace), pitch_journal.py (R2 journal keys)
- scripts/grade_pitch_journal.py, scripts/check_pitch_delivered.py,
  scripts/build_seasonal_state.py
- docs/claude_ref/daily_seasonal.md
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


@dataclass(frozen=True)
class Product:
    name: str
    label: str                    # email heading and subject prefix
    tab_name: str                 # Sheets tab in Trade_Signals_Log
    checks_root: Path             # scratch/<x>_checks, one folder per date
    journal_path: Path
    journal_r2_key: str
    scoreboard_path: Path
    watchlist_path: Path
    negative_registry_path: Path
    state_path: Path
    default_ideas: Path
    receipt_dir: Path
    receipt_r2_prefix: str
    receipt_download_dir: Path
    idea_id_prefix: str           # "" for the pitch: ids stay YYYY-MM-DD-N
    scan_source: str              # the Scan_Source column on every order row
    site_payload: bool            # only the pitch feeds the site Pitch tab
    fills_approvals: bool         # Pitch-{idea_id} fills are pitch-only


PITCH = Product(
    name="pitch",
    label="Daily Pitch",
    tab_name="Pitch",
    checks_root=ROOT / "scratch" / "pitch_checks",
    journal_path=DATA / "pitch_journal.jsonl",
    journal_r2_key="pitch_journal.jsonl",
    scoreboard_path=DATA / "pitch_scoreboard.json",
    watchlist_path=DATA / "pitch_watchlist.json",
    negative_registry_path=DATA / "pitch_negative_registry.md",
    state_path=DATA / "pitch_state.json",
    default_ideas=DATA / "pitch_ideas.json",
    receipt_dir=DATA / "pitch_delivery_receipts",
    receipt_r2_prefix="pitch_delivery_receipts",
    receipt_download_dir=ROOT / "artifacts" / "pitch_delivery_receipts",
    idea_id_prefix="",
    scan_source="Pitch",
    site_payload=True,
    fills_approvals=True,
)

SEASONAL = Product(
    name="seasonal",
    label="Daily Seasonal",
    tab_name="Seasonal Agent",
    checks_root=ROOT / "scratch" / "seasonal_checks",
    journal_path=DATA / "seasonal_agent_journal.jsonl",
    journal_r2_key="seasonal_agent_journal.jsonl",
    scoreboard_path=DATA / "seasonal_agent_scoreboard.json",
    watchlist_path=DATA / "seasonal_agent_watchlist.json",
    negative_registry_path=DATA / "seasonal_agent_negative_registry.md",
    state_path=DATA / "seasonal_state.json",
    default_ideas=DATA / "seasonal_agent_ideas.json",
    receipt_dir=DATA / "seasonal_agent_delivery_receipts",
    receipt_r2_prefix="seasonal_agent_delivery_receipts",
    receipt_download_dir=ROOT / "artifacts" / "seasonal_agent_delivery_receipts",
    # "S" keeps a seasonal idea id from ever matching a pitch id, so a
    # Pitch-{idea_id} fill tag or a merged view can never cross the products.
    idea_id_prefix="S",
    scan_source="Seasonal_Agent",
    site_payload=False,
    fills_approvals=False,
)

PRODUCTS: dict[str, Product] = {p.name: p for p in (PITCH, SEASONAL)}


def get_product(name: str | None = None) -> Product:
    key = str(name or "pitch").strip().lower()
    if key not in PRODUCTS:
        raise ValueError(f"unknown product {name!r}; expected one of "
                         f"{sorted(PRODUCTS)}")
    return PRODUCTS[key]


def journal_r2_key(path: Path) -> str | None:
    """The R2 mirror key for a PRODUCTION journal path, else None. A test or
    dev journal at any other path never touches R2."""
    for product in PRODUCTS.values():
        if Path(path) == product.journal_path:
            return product.journal_r2_key
    return None
