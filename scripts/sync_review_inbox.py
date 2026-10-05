"""Cloud bridge from already-delivered R2 evidence to review-only snapshots.

No agent/publisher run, SMTP, Sheets, broker, or Scheduler operation. This lets
the private-site inbox use the existing pinned publishers without changing their
runtime. Reuses existing R2 credentials/conditional writes; no settings changes.
"""
from __future__ import annotations
import argparse
import datetime as dt
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pitch_delivery
import review_publish

def delivered_slate(records: list[dict], receipt: dict, asof: str) -> tuple[list[dict],dict|None,str]:
    pitch_delivery.verify_sent_receipt(receipt,records,asof)
    selected=pitch_delivery.verdict_records(records,asof)
    ideas={}
    stand_down=None
    for row in selected:
        if row['kind']=='idea':
            if not row.get('orders') or not row.get('idea_id'):
                raise ValueError('Delivered idea has no immutable prepared order details')
            ideas[row['idea_id']]=row
        elif row['kind']=='stand_down':
            stand_down=row
    if ideas:
        return list(ideas.values()),None,pitch_delivery.verdict_digest(selected)
    if stand_down:
        return [],stand_down,pitch_delivery.verdict_digest(selected)
    raise ValueError('Confirmed delivery has no idea or stand-down verdict')

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--asof',default=dt.datetime.now(ZoneInfo('America/New_York')).date().isoformat())
    args=ap.parse_args();dt.date.fromisoformat(args.asof)
    import pitch_products
    from strategy_config import ACCOUNT_VALUE
    failures=0
    for product in ['pitch','seasonal']:
        spec=pitch_products.get_product(product)
        try:
            path=ROOT/'artifacts'/'review_inbox_sync'/product/f'{args.asof}.receipt.json'
            receipt=pitch_delivery.load_receipt(path,args.asof,use_r2=True,product=product)
            if receipt is None:
                print(f'{product}: no receipt yet for {args.asof}; feed remains explicitly unavailable')
                continue
            records=pitch_delivery.load_cloud_journal(spec.journal_r2_key)
            ideas,stand_down,digest=delivered_slate(records,receipt,args.asof)
            review_publish.publish(product=product,asof=args.asof,ideas=ideas,receipt=receipt,
                verdict_digest=digest,account_value=ACCOUNT_VALUE,stand_down=stand_down,
                path=ROOT/'artifacts'/'review_inbox_sync'/product/f'{args.asof}.json',use_r2=True)
            print(f'{product}: reconciled {len(ideas)} delivered proposal(s), human review only')
        except Exception as exc:
            failures+=1;print(f'{product}: review reconciliation failed: {exc}')
    return 1 if failures else 0

if __name__=='__main__':raise SystemExit(main())
