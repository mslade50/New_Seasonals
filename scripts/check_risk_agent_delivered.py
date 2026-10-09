"""Did tonight's Risk Agent actually deliver? Exit non-zero when it did not.

Delivery needs two pieces of durable evidence for the asof: a `decision` or
`stand_down` record in the journal, and a `sent` delivery receipt for the same
decision id. `--require-r2` also downloads the R2 journal copy and compares its
verdict records for the asof with the local ones.

    python scripts/check_risk_agent_delivered.py [--asof YYYY-MM-DD] [--require-r2]
                                                 [--state PATH] [--journal PATH]
                                                 [--receipt-dir DIR]

The default asof is the state's asof (the completed session the run was built
for), falling back to today's date.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import daily_risk_agent as dra  # noqa: E402


def _default_asof(state_path: Path) -> str:
    try:
        asof = json.loads(state_path.read_text(encoding="utf-8")).get("asof")
        if asof:
            return str(asof)
    except (OSError, json.JSONDecodeError):
        pass
    return str(dt.date.today())


def _digest(records: list[dict]) -> list[str]:
    return sorted(json.dumps({k: v for k, v in r.items() if k != "written_at"},
                             sort_keys=True, separators=(",", ":"), default=str)
                  for r in records)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None)
    ap.add_argument("--state", default=str(dra.DEFAULT_STATE))
    ap.add_argument("--journal", default=None)
    ap.add_argument("--receipt-dir", default=None)
    ap.add_argument("--require-r2", action="store_true")
    args = ap.parse_args(argv)

    lg = dra.get_ledger()
    asof = args.asof or _default_asof(Path(args.state))
    journal = Path(args.journal) if args.journal else Path(lg.JOURNAL_PATH)
    records = lg.load(journal, pull=False)
    verdicts = dra.verdict_records(records, asof)
    if len(verdicts) != 1:
        print(f"FAILED: {len(verdicts)} decision/stand_down record(s) journaled for {asof}, "
              "expected exactly 1. The Risk Agent did not deliver.")
        return 1

    if args.require_r2:
        try:
            import cache_io
            if not cache_io.is_configured():
                raise RuntimeError("R2 is not configured")
            target = ROOT / "artifacts" / "risk_agent_journal" / lg.R2_JOURNAL_KEY
            if not cache_io.download_to_local(lg.R2_JOURNAL_KEY, str(target)):
                raise RuntimeError(f"R2 journal download failed: {cache_io.last_download_error()}")
            cloud = dra.verdict_records(lg.load(target, pull=False), asof)
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED: {exc}. The Risk Agent did not deliver.")
            return 1
        if _digest(cloud) != _digest(verdicts):
            print(f"FAILED: local and R2 journals disagree for {asof}.")
            return 1
        print("OK: R2 journal matches the local verdict record")

    rdir = Path(args.receipt_dir) if args.receipt_dir else dra.RECEIPT_DIR
    try:
        receipt = dra.read_receipt(asof, rdir / f"{asof}.json")
    except dra.ReceiptError as exc:
        print(f"FAILED: {exc}")
        return 1
    if receipt is None:
        print(f"FAILED: no delivery receipt for {asof}. The Risk Agent did not deliver.")
        return 1
    if receipt.get("status") != "sent":
        print(f"FAILED: delivery receipt for {asof} is {receipt.get('status')}, not sent.")
        return 1
    if receipt.get("decision_id") != verdicts[0].get("decision_id"):
        print(f"FAILED: receipt decision id {receipt.get('decision_id')!r} does not match "
              f"the journal's {verdicts[0].get('decision_id')!r}.")
        return 1
    print(f"OK: {verdicts[0]['kind']} {verdicts[0].get('decision_id')} journaled and sent "
          f"(receipt {receipt.get('delivery_id')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
