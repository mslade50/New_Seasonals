"""Did this week's PM Weekly deliver? Exit non-zero when it did not.

Needs a `brief` or `stand_down` record for the state's ISO week in the
journal and a `sent` receipt for the same decision id. `--require-r2` also
compares the R2 journal's verdict record for the week with the local one.

    python scripts/check_pm_agent_delivered.py [--week 2026-W41] [--require-r2]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pm_agent_journal as J  # noqa: E402
import pm_agent_universe as U  # noqa: E402
import weekly_pm_agent as W  # noqa: E402


def _digest(records: list[dict]) -> list[str]:
    return sorted(json.dumps(r, sort_keys=True, separators=(",", ":"), default=str) for r in records)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", default=None)
    ap.add_argument("--state", default=None)
    ap.add_argument("--journal", default=None)
    ap.add_argument("--receipt-dir", default=None)
    ap.add_argument("--require-r2", action="store_true")
    a = ap.parse_args(argv)

    week = a.week
    if not week:
        st = W._read_json(Path(a.state or U.state_path())) or {}
        week = st.get("week")
    if not week:
        print("FAILED: no week given and no usable state. The PM Weekly did not deliver.")
        return 1
    journal = Path(a.journal or U.journal_path())
    verdicts = J.verdicts_for(J.load(journal), week)
    if len(verdicts) != 1:
        print(f"FAILED: {len(verdicts)} brief/stand_down record(s) for {week}, expected 1.")
        return 1
    if a.require_r2:
        try:
            import cache_io
            if not cache_io.is_configured():
                raise RuntimeError("R2 is not configured")
            target = U.home() / "r2_check" / "journal.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            if not cache_io.download_to_local(J.R2_JOURNAL_KEY, str(target)):
                raise RuntimeError("R2 journal download failed")
            cloud = J.verdicts_for(J.load(target), week)
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED: {exc}. The PM Weekly did not deliver.")
            return 1
        if _digest(cloud) != _digest(verdicts):
            print(f"FAILED: local and R2 journals disagree for {week}.")
            return 1
        print("OK: R2 journal matches the local verdict record")
    rdir = Path(a.receipt_dir or U.receipt_dir())
    try:
        receipt = W.read_receipt(rdir / f"{week}.json")
    except W.ReceiptError as exc:
        print(f"FAILED: {exc}")
        return 1
    if receipt is None or receipt.get("status") != "sent":
        print(f"FAILED: no sent delivery receipt for {week} "
              f"({None if receipt is None else receipt.get('status')}).")
        return 1
    if receipt.get("decision_id") != verdicts[0].get("decision_id"):
        print("FAILED: receipt decision id does not match the journal.")
        return 1
    print(f"OK: {verdicts[0]['kind']} {verdicts[0].get('decision_id')} journaled and sent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
