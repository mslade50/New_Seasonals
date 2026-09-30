"""Build a highly filtered, local financing research dashboard from captured sources."""
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from collections import Counter
from datetime import datetime,timedelta,timezone
from pathlib import Path
import pandas as pd
import exchange_calendars as xc
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fundamental.financing_dashboard import classify,price_exclusions,DashboardPolicy
from fundamental.cash_runway import filing_rows,filing_url,utc


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def encode_payload(data):
    # HTML parses script terminators even inside application/json blocks.
    return json.dumps(data,ensure_ascii=False,allow_nan=False).replace('<',chr(92)+'u003c').replace('>',chr(92)+'u003e').replace('&',chr(92)+'u0026')


def build(output):
    if (ROOT/'artifacts').resolve() not in output.resolve().parents:
        raise ValueError('Dashboard output must be under artifacts/')
    manifest=read(output/'snapshot_manifest.json')
    source=Path(manifest['source_snapshot'])
    if sha(source/'watchlist.json')!=manifest['source_sha256']:
        raise ValueError('Discovery snapshot digest mismatch')
    prices=read(output/'price_rows.json');financials=read(output/'financials.json')
    reviews=read(output/'reviews.json') if (output/'reviews.json').exists() else {}
    inputs=[output/p for p in ['snapshot_manifest.json','price_rows.json','financials.json','price_captures.json','price_repairs.json']]
    if (output/'reviews.json').exists():inputs.append(output/'reviews.json')
    announcements=read(output/'announcements.json') if (output/'announcements.json').exists() else []
    if announcements:inputs.append(output/'announcements.json')
    if len({a['event_id'] for a in announcements})!=len(announcements):raise ValueError('Duplicate announcement identity')
    for event in announcements:
        if not event.get('sources') or utc(event['published_at'])>utc(manifest['as_of']):raise ValueError('Announcement evidence or cutoff invalid')
    captured=0
    for folder in ['sources','sec','review_sources']:
        index=output/folder/'captures.json'
        if index.exists():
            inputs.append(index)
            for item in read(index):
                if sha(output/folder/'raw'/item['name'])!=item['sha256']:raise ValueError('Source capture changed: '+item['name'])
                captured+=1
    rows=[]
    filings=[]
    cutoff=utc(manifest['as_of'])
    for price in prices:
        if price.get('bars_path') and sha(Path(price['bars_path']))!=price['bars_sha256']:
            raise ValueError('Price bars changed: '+price['ticker'])
        row=classify(price,financials.get(price['ticker']),reviews.get(price['ticker']),as_of=manifest['as_of'],price_session=manifest['price_session'])
        rows.append(row)
        subpath=output/'sec/raw'/(price['ticker']+'_submissions.json')
        if not price_exclusions(price) and subpath.exists():
            for filing in filing_rows(read(subpath)):
                if not filing.get('acceptanceDateTime'):continue
                stamp=utc(filing['acceptanceDateTime']);form=filing['form'];items=filing.get('items','')
                if not cutoff-timedelta(days=30)<=stamp<=cutoff:continue
                if form.startswith(('S-3','S-1','F-3','F-1')):label='Registration capacity; not a completed raise'
                elif form.startswith('424'):label='Prospectus; distinguish ATM, resale and primary issuance'
                elif form=='FWP':label='Offering-related material; terms need review'
                elif form in ['8-K','8-K/A'] and any(i in items for i in ['1.01','3.02']):label='Agreement / securities update; may concern debt or noncash equity'
                else:continue
                if price['ticker']=='FTK' and filing['accessionNumber']=='0000928054-26-000090':
                    label='Reviewed: $75m term-loan funding at closing; partly repays existing debt. Not equity.'
                filings.append(dict(ticker=price['ticker'],cik=price['cik'],form=form,accepted_at=stamp.isoformat(),url=filing_url(price['cik'],filing['accessionNumber'],filing.get('primaryDocument')),label=label))
    for filing in filings:
        for event in announcements:
            if filing['url'] in {s['url'] for s in event['sources']}:
                filing['label']='Related to the reviewed '+event['ticker']+' offering above; final pricing and cash receipt unverified'
                filing['event_id']=event['event_id']
    filings.sort(key=lambda r:r['accepted_at'],reverse=True)
    status=Counter(r['bucket'] for r in rows)
    calendar=xc.get_calendar('XNYS')
    prior_close=calendar.schedule.loc[manifest['price_session'],'close'].isoformat()
    data=dict(as_of=manifest['as_of'],price_session=manifest['price_session'],discovery_as_of=manifest['discovery_as_of'],prior_close=prior_close,
        generated_at=datetime.now(timezone.utc).isoformat(),policy=DashboardPolicy().to_dict(),
        counts=dict(discovery_universe=manifest['discovery_universe'],refreshed=len(rows),price_qualified=sum(not price_exclusions(r) for r in prices),financial_checked=len(financials),repaired_prices=len(read(output/'price_repairs.json')),**status),
        rows=[r for r in rows if r['bucket']!='excluded'],filings=filings,announcements=announcements,source_scope='Existing September 22 setup candidates refreshed; not a new whole-market scan',
        research_only=True,offering_probability=None)
    # Keep detailed financial evidence in its captured inputs, not duplicated in every UI row.
    keep=['status','balance_date','balance_age_days','filing_accepted_at','filing_url','cash','current_investments','reported_liquidity','liquidity_basis','runway_6m','monthly_burn_6m','monthly_burn_3m','warnings']
    for row in data['rows']:row['financial']={k:v for k,v in row['financial'].items() if k in keep}
    write(output/'dashboard_data.json',data)
    fields=['ticker','company_name','bucket','price_as_of','close','return_20','return_60','distance_high_252','dollar_volume_20','median_dollar_volume_20','reported_liquidity','balance_date','runway_6m','review_items','exclusions']
    with (output/'dashboard_rows.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for r in rows:
            flat=dict(r,**{k:r['financial'].get(k) for k in ['reported_liquidity','balance_date','runway_6m']})
            writer.writerow({k:'; '.join(flat[k]) if isinstance(flat.get(k),list) else flat.get(k) for k in fields})
    payload=encode_payload(data)
    template=ROOT/'fundamental/financing_dashboard.html'
    (output/'financing_dashboard.html').write_text(template.read_text(encoding='utf-8').replace('__DATA__',payload),encoding='utf-8')
    source_paths=[ROOT/p for p in ['fundamental/financing_dashboard.py','fundamental/financing_dashboard.html','scripts/build_financing_dashboard.py']]
    result=dict(run_id=output.name,completion_status='RENDERED_PENDING_QA',as_of=manifest['as_of'],counts=data['counts'],captured_source_hashes_verified=captured,
        input_hashes={str(p.resolve()):sha(p) for p in inputs},source_hashes={str(p.resolve()):sha(p) for p in source_paths},
        outputs={'report':{'path':str(output/'financing_dashboard.html'),'sha256':sha(output/'financing_dashboard.html')}},
        artifact_hashes={name:sha(output/name) for name in ['dashboard_data.json','dashboard_rows.csv','financing_dashboard.html']})
    write(output/'manifest.json',result)
    return data['counts']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(build(args.output_dir.resolve()),indent=2))


if __name__=='__main__':main()
