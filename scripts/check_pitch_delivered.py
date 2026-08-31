"""Did today's Daily Pitch actually deliver? Exit non-zero when it did not.

The headless pitch run is a long agent session, and an agent that gives up
politely still exits 0. Task Scheduler would then show green on a morning with
no email. Delivery now requires two matching pieces of durable evidence: a
valid verdict dated today in the journal and a ``sent`` delivery receipt whose
digest matches those exact verdict records. Production postflight also pulls
that receipt from R2, so a local-only success cannot turn the task green.

A stand-down counts as delivery because it IS a delivered verdict: the email
goes out, the tab is cleared, and the journal records what was swept. What
this check still catches is the failure it was written for, an agent that
finished without publishing anything at all.

One or two ideas count too, as of 2026-08-10. The floor used to be exactly
three, which meant the 08-10 run (17 candidates, 16 killed, one survivor)
had no legal way to ship its survivor and shipped nothing instead. The
grammar makes a short slate expensive; it is not this check's job to make it
impossible.

    python scripts/check_pitch_delivered.py [--asof YYYY-MM-DD] [--require-r2]
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pitch_journal  # noqa: E402
import pitch_delivery  # noqa: E402
from pitch_grammar import IDEA_COUNT, MIN_IDEA_COUNT  # noqa: E402


def _receipt_path(args) -> Path:
    if args.delivery_receipt:
        return Path(args.delivery_receipt)
    journal = Path(args.journal)
    if journal != pitch_journal.JOURNAL_PATH:
        name = f"{journal.stem}.delivery.{args.asof}.json"
        return journal.with_name(name)
    return pitch_delivery.default_receipt_path(args.asof)


def _confirm_receipt(args, records: list[dict]) -> bool:
    path = _receipt_path(args)
    try:
        receipt = pitch_delivery.load_receipt(
            path, args.asof, use_r2=args.require_r2,
            require_remote=args.require_r2)
        if receipt is None:
            raise pitch_delivery.DeliveryReceiptError(
                f"no delivery receipt exists at {path}")
        pitch_delivery.verify_sent_receipt(receipt, records, args.asof)
    except pitch_delivery.DeliveryReceiptError as exc:
        print(f"FAILED: {exc}. The pitch did not deliver.")
        return False
    print(f"OK: sent receipt {receipt['delivery_id']} matches the journal")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=str(dt.date.today()))
    ap.add_argument("--journal", default=str(pitch_journal.JOURNAL_PATH),
                    help="journal path; matches daily_pitch.py's flag so a "
                         "test or dev run never reads the shared trail")
    ap.add_argument("--delivery-receipt", default=None,
                    help="explicit local receipt path (tests/dev only)")
    ap.add_argument("--require-r2", action="store_true",
                    help="require and verify the production R2 receipt")
    args = ap.parse_args()

    all_records = pitch_journal.load(Path(args.journal), pull=False)
    if args.require_r2:
        try:
            cloud_records = pitch_delivery.load_cloud_journal()
            local_today = pitch_delivery.verdict_records(
                all_records, args.asof)
            cloud_today = pitch_delivery.verdict_records(
                cloud_records, args.asof)
            if (pitch_delivery.verdict_digest(local_today)
                    != pitch_delivery.verdict_digest(cloud_today)
                    or len(local_today) != len(cloud_today)):
                raise pitch_delivery.DeliveryReceiptError(
                    f"local and R2 journals disagree for {args.asof}")
        except pitch_delivery.DeliveryReceiptError as exc:
            print(f"FAILED: {exc}. The pitch did not deliver.")
            return 1
        print("OK: R2 journal matches the local verdict records")
    today = [r for r in all_records if str(r.get("date")) == args.asof]
    ideas = [r for r in today if r.get("kind") == "idea"]
    stand_down = [r for r in today if r.get("kind") == "stand_down"]

    # The stand-down cases are checked FIRST, because a stand-down sitting
    # beside composed ideas means the run published twice or crashed part way
    # through, and the short-slate rule below must never launder that into a
    # pass. A stand-down amended by a DIRECTED idea is the one legitimate
    # mixture: the sweep found nothing, McKinley overruled a specific kill.
    if stand_down:
        if not ideas:
            killed = sum(1 for r in today if r.get("kind") == "killed")
            if not _confirm_receipt(args, all_records):
                return 1
            print(f"OK: stand-down journaled for {args.asof} "
                  f"({killed} kill record(s)). Nothing shipped, by verdict.")
            return 0
        if all(str(r.get("directed_by", "")).strip() for r in ideas):
            if not _confirm_receipt(args, all_records):
                return 1
            print(f"OK: stand-down for {args.asof} amended by {len(ideas)} "
                  f"directed idea(s).")
            return 0
        print(f"FAILED: {len(ideas)} idea record(s) AND a stand-down for "
              f"{args.asof}. That is a half-published run, not a verdict.")
        return 1

    if MIN_IDEA_COUNT <= len(ideas) <= IDEA_COUNT:
        if not _confirm_receipt(args, all_records):
            return 1
        short = "" if len(ideas) == IDEA_COUNT else " (short slate)"
        print(f"OK: {len(ideas)} ideas journaled for {args.asof}{short}")
        return 0

    print(f"FAILED: {len(ideas)} idea record(s) journaled for {args.asof}, "
          f"expected {MIN_IDEA_COUNT} to {IDEA_COUNT} or a stand-down. "
          f"The pitch did not deliver.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
