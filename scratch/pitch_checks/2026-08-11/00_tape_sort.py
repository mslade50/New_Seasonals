"""Sort the whole 218-name tape by every extreme axis, grouped by asset class.

Feeds stage B1's surface map. No thesis here, just the picture.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

TAPE = json.loads((Path(__file__).resolve().parents[3] / "data" / "pitch_tape.json").read_text())
rows = TAPE["tickers"] if isinstance(TAPE, dict) and "tickers" in TAPE else TAPE

if isinstance(rows, dict):
    rows = [dict(ticker=k, **v) for k, v in rows.items()]

print(f"n={len(rows)}  keys={sorted(rows[0].keys())}\n")


def top(field, n=15, rev=True):
    ok = [r for r in rows if r.get(field) is not None]
    ok.sort(key=lambda r: r[field], reverse=rev)
    return ok[:n]


for field, rev in [
    ("rank_5d", True), ("rank_5d", False),
    ("rank_21d", True), ("rank_21d", False),
    ("rank_63d", True), ("rank_63d", False),
    ("z10", True), ("z10", False),
    ("dist_52w_high_pct", True), ("dist_52w_low_pct", False),
    ("dist_sma200_pct", True), ("dist_sma200_pct", False),
    ("ret_5d", True), ("ret_5d", False),
]:
    label = f"{field} {'HIGH' if rev else 'LOW'}"
    vals = top(field, 15, rev)
    print(f"--- {label} ---")
    print("  " + "  ".join(f"{r['ticker']}:{r[field]:.1f}" for r in vals))
    print()
