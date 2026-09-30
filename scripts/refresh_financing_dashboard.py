"""Refresh a prior financing setup cohort into a NEW local dashboard snapshot.

Curated reviews are deliberately not copied: each new cutoff needs a new review.
"""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def refresh_prices(output, source):
    import sys,json,hashlib,time
    from datetime import datetime,timezone,timedelta
    from pathlib import Path
    import pandas as pd
    import yfinance as yf
    from dotenv import load_dotenv
    from scripts.build_financing_watchlist import last_completed_session
    from scripts.build_cash_runway_watchlist import Capture,dump
    from fundamental.financing_opportunity import price_metrics,ticker_bars
    from fundamental.cash_runway import calculate_runway
    from public_market_sources import parse_listing_directory
    load_dotenv(ROOT/'.env')
    OUT=output
    SOURCE=source
    old=json.loads((SOURCE/'watchlist.json').read_text(encoding='utf-8'))
    seed=[r for r in old if r.get('setups')]
    manifest=dict(as_of=datetime.now(timezone.utc).isoformat(),discovery_as_of=json.loads((SOURCE/'manifest.json').read_text(encoding='utf-8'))['as_of'],discovery_universe=len(old),refresh_universe=len(seed),source_snapshot=str(SOURCE),source_sha256=hashlib.sha256((SOURCE/'watchlist.json').read_bytes()).hexdigest(),research_only=True)
    manifest['price_session']=last_completed_session(manifest['as_of'])
    dump(OUT/'snapshot_manifest.json',manifest)
    folder=OUT/'sources';folder.mkdir(exist_ok=True)
    capture=Capture(folder)
    exchange=capture.get_json('exchange.json','https://www.sec.gov/files/company_tickers_exchange.json')
    secmap={r['ticker'].replace('.','-'):int(r['cik']) for r in (dict(zip(exchange['fields'],v)) for v in exchange['data'])}
    listing={}
    for name,nasdaq in [('nasdaqlisted.txt',True),('otherlisted.txt',False)]:
     raw=capture.get(name,'https://www.nasdaqtrader.com/dynamic/SymDir/'+name)
     listing.update({r['ticker']:r for r in parse_listing_directory(raw.decode(),nasdaq=nasdaq).to_dict('records')})
    for r in seed:
     l=listing.get(r['ticker'])
     r['current_identity_verified']=bool(l and not l['listing_reasons'] and secmap.get(r['ticker'])==r['cik'])
     r['listing_warning']=l['listing_reasons'] if l else 'Current listing absent'
    dump(OUT/'seed_universe.json',seed)
    prices=OUT/'prices';prices.mkdir(exist_ok=True)
    yf.set_tz_cache_location(str(OUT/'yahoo_cache'))
    tickers=['SPY']+[r['ticker'] for r in seed]
    end=pd.Timestamp(manifest['price_session'])+pd.Timedelta(days=1)
    start=end-pd.Timedelta(days=400)
    entries={}
    for pos in range(0,len(tickers),65):
     group=tickers[pos:pos+65]
     raw=yf.download(group,start=str(start.date()),end=str(end.date()),auto_adjust=False,actions=True,threads=4,progress=False,timeout=20)
     for ticker in group:
      frame=ticker_bars(raw,ticker)
      if frame.empty or 'Close' not in frame or frame.Close.dropna().empty:
       entries[ticker]={'status':'unavailable'};continue
      path=prices/(ticker+'.parquet');frame.to_parquet(path)
      entries[ticker]=dict(status='captured',path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),fetched_at=datetime.now(timezone.utc).isoformat(),source='https://finance.yahoo.com/quote/'+ticker+'/history/')
     dump(OUT/'price_captures.json',entries)
     print('Refreshed prices',min(pos+65,len(tickers)),len(tickers),flush=True)
     time.sleep(.5)
    benchmark=pd.read_parquet(prices/'SPY.parquet').dropna(subset=['Adj Close'])
    rows=[]
    for r in seed:
     r={k:v for k,v in r.items() if k not in ['financial','candidate','funding_status']}
     frame=pd.read_parquet(prices/(r['ticker']+'.parquet')) if entries.get(r['ticker'],{}).get('status')=='captured' else pd.DataFrame()
     r.update(price_metrics(frame,benchmark,session=manifest['price_session']))
     r['price_source']=entries.get(r['ticker'],{}).get('source')
     r['price_fetched_at']=entries.get(r['ticker'],{}).get('fetched_at')
     rows.append(r)
    dump(OUT/'price_rows.json',rows)
    print('PRICE STAGE COMPLETE',flush=True)

def reconcile_prices(output, source):
    import json,sys,hashlib
    from pathlib import Path
    import pandas as pd
    from fundamental.financing_opportunity import price_metrics
    from fundamental.financing_dashboard import price_exclusions
    OUT=output
    SOURCE=source
    def read(p):return json.loads(p.read_text(encoding='utf-8'))
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=read(OUT/'snapshot_manifest.json');seed=read(OUT/'seed_universe.json');new=read(OUT/'price_captures.json');old=read(SOURCE/'price_captures.json')
    benchmark=pd.read_parquet(OUT/'prices/SPY.parquet').dropna(subset=['Adj Close'])
    folder=OUT/'reconciled_prices';folder.mkdir(exist_ok=True)
    rows=[];provenance=[]
    for source_row in seed:
     ticker=source_row['ticker']
     row={k:source_row[k] for k in ['ticker','company_name','cik','exchange','listing_warning','listing_as_of','current_identity_verified']}
     entry=new.get(ticker,{})
     frame=pd.DataFrame()
     if entry.get('status')=='captured':
      fresh=Path(entry['path']);assert sha(fresh)==entry['sha256'];frame=pd.read_parquet(fresh)
      prior=SOURCE/'prices'/(ticker+'.parquet')
      required=['Open','Close','Adj Close','Volume'];valid=frame.dropna(subset=required)
      missing=benchmark.index[-252:].difference(valid.index)
      if len(missing) and prior.exists() and old.get(ticker,{}).get('status')=='captured':
       assert sha(prior)==old[ticker]['sha256'];previous=pd.read_parquet(prior).dropna(subset=required)
       common=valid.index.intersection(previous.index)[-20:]
       # Fill only when both captures agree on the recent price/adjustment basis.
       compatible=len(common)>=10 and all(((valid.loc[common,c]/previous.loc[common,c]-1).abs()<.00001).all() for c in ['Open','Close','Adj Close'])
       repair=missing.intersection(previous.index)
       if compatible and len(repair):
        frame=pd.concat([valid,previous.loc[repair]]).sort_index()
        provenance.append(dict(ticker=ticker,sessions=[str(d.date()) for d in repair],fresh_path=str(fresh),fresh_sha256=sha(fresh),prior_path=str(prior),prior_sha256=sha(prior),reason='Fresh capture omitted sessions; reused hash-verified observations with matching price basis'))
      frame=frame.loc[frame.index<=pd.Timestamp(manifest['price_session'])]
      path=folder/(ticker+'.parquet');frame.to_parquet(path)
      row['bars_path']=str(path.resolve());row['bars_sha256']=sha(path)
     row.update(price_metrics(frame,benchmark,session=manifest['price_session']))
     if not frame.empty:
      b=frame.dropna(subset=['Adj Close','Close','Volume'])
      row['median_dollar_volume_20']=float((b.Close*b.Volume).iloc[-20:].median())
      row['above_sma200']=bool(len(b)>=200 and b['Adj Close'].iloc[-1]>=b['Adj Close'].iloc[-200:].mean())
      row['sparkline']=[round(float(v),3) for v in b['Adj Close'].iloc[-61:]]
     row['price_source']=entry.get('source');row['price_fetched_at']=entry.get('fetched_at')
     rows.append(row)
    for name,value in [('price_rows.json',rows),('price_repairs.json',provenance)]: (OUT/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    strong=[r for r in rows if not price_exclusions(r)]
    print('Price-qualified',len(strong),[r['ticker'] for r in strong]);print('Reused prior captured bars for',len(provenance),'names')

def refresh_financials(output, source):
    import json,sys
    from pathlib import Path
    from dotenv import load_dotenv
    from scripts.build_financing_watchlist import ResumableCapture
    from scripts.build_cash_runway_watchlist import dump,scan_financing
    from fundamental.financing_dashboard import price_exclusions
    from fundamental.cash_runway import calculate_runway
    load_dotenv(ROOT/'.env')
    OUT=output
    manifest=json.loads((OUT/'snapshot_manifest.json').read_text(encoding='utf-8'))
    rows=json.loads((OUT/'price_rows.json').read_text(encoding='utf-8'))
    folder=OUT/'sec';folder.mkdir(exist_ok=True)
    capture=ResumableCapture(folder)
    financials={}
    for row in [r for r in rows if not price_exclusions(r)]:
     ticker=row['ticker'];cik=row['cik']
     try:
      sub=capture.get_json(ticker+'_submissions.json',f'https://data.sec.gov/submissions/CIK{cik:010d}.json')
      if ticker not in {t.replace('.','-') for t in sub.get('tickers',[])}:raise ValueError('SEC identity mismatch')
      facts=capture.get_json(ticker+'_companyfacts.json',f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json')
      f=calculate_runway(facts,sub,ticker=ticker,as_of=manifest['as_of'])
      if f.get('runway_6m') is not None and f['runway_6m']<=24:
       scan_financing(capture,f,sub,manifest['as_of'],max_filings=24)
     except Exception as e:f=dict(ticker=ticker,status='unavailable',warnings=[str(e)])
     financials[ticker]=f;dump(OUT/'financials.json',financials)
     print(ticker,f.get('status'),f.get('runway_6m'),flush=True)
    print('FINANCIAL STAGE COMPLETE',flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path,required=True,help='Prior financing-watchlist snapshot with watchlist.json and prices/')
    parser.add_argument('--output-dir',type=Path,required=True,help='New directory under artifacts/')
    args=parser.parse_args()
    output=args.output_dir.resolve();source=args.source_dir.resolve()
    if (ROOT/'artifacts').resolve() not in output.parents:parser.error('Output must be under artifacts/')
    if output.exists():parser.error('Use a new output directory; retained snapshots are not overwritten')
    output.mkdir(parents=True)
    refresh_prices(output,source)
    reconcile_prices(output,source)
    refresh_financials(output,source)
    from scripts.build_financing_dashboard import build
    print(build(output))

if __name__=='__main__':main()
