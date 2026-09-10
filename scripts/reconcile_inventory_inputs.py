"""Write a review report from captured broker evidence; no network or activation."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from inventory_reconciliation import inventory_reconciliation


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--book',type=Path,required=True)
    parser.add_argument('--fills',type=Path,required=True)
    parser.add_argument('--canonical',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    book=json.loads(args.book.read_text(encoding='utf-8-sig'))
    incoming=json.loads(args.fills.read_text(encoding='utf-8-sig'))
    fills=incoming['fills']
    if args.canonical:
        import pandas as pd
        previous=pd.read_parquet(args.canonical).rename(columns={'time_utc':'time'})
        fills=previous.astype(object).where(previous.notna(),None).to_dict('records')+fills
    from strategy_config import STRATEGY_BOOK
    report=inventory_reconciliation(book,fills,strategies={s['name'] for s in STRATEGY_BOOK})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
    print(f'Review only: {len(report["tranches"])} tagged exit claims; {len(report["issues"])} structural issues. No inventory activated.')
    for row in report['tranches']:
        print(row['symbol'],row['strategy'],row['ref_date'],row['claimed_qty'],row['quantity_basis'])
    for row in report['balances']:
        if row['residual']:
            print('Unassigned balance:',row['symbol'],row['residual'])
    return 0


if __name__=='__main__':
    raise SystemExit(main())
