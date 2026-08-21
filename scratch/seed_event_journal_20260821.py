"""One-time seed of the event-sleeve journal (2026-08-21).

The journal shipped the same day the sleeve's first trade (V4 SVXY) staged,
so the entry record predates the machinery. Mint it from the state via the
same backfill path every future run uses, and push to R2.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from event_sleeve import (append_journal, backfill_entry_records,
                          load_journal, load_state)

state = load_state()
records = load_journal()
missing = backfill_entry_records(state, records)
if not missing:
    print("Nothing to backfill — journal already covers all open positions.")
else:
    n = append_journal(missing)
    print(f"Appended {n} backfilled entry record(s):")
    for r in missing:
        print(f"  {r['trade']}: {r['action']} {r['qty']} {r['ticker']} "
              f"@ ref {r['ref_close']} on {r['date']} (exit {r['exit_on']})")
