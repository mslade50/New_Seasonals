"""Frozen-data research on higher-beta, smooth short/medium-term momentum.

No live strategy, scanner, broker or production-data writes. Select early-era
family leaders before inspecting recent results. The current stock universe
is explicitly fixed, not claimed point-in-time. All output stays in artifacts.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import itertools
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import strategy_config as config
from indicators import calculate_indicators, smooth_momentum_features
from scripts.backtest_inside_day_breakout import ExitSpec, simulate, one_position_mask, metrics, valid_ohlc, buy_stop_fill
from scripts.backtest_inside_day_filter_audit import market_open_hours, sharpe
from trading_calendar import TRADING_DAY

START=pd.Timestamp('2005-01-01')
MAX_HOLD=42

def research_bars(g,hours):
    """Keep missing NYSE sessions as unavailable bars, never compress a hold."""
    df=g.sort_values('date').set_index('date').drop(columns='ticker')
    sessions=hours.index[(hours.index>=df.index.min())&(hours.index<=df.index.max())]
    df=df.reindex(sessions)
    valid=valid_ohlc(df)
    df.loc[~valid,['Open','High','Low','Close']]=np.nan
    return df,valid

def definitions():
    out=[]
    for efficiency,momentum,market in itertools.product((.2,.3),(.15,.25),(False,True)):
        context={'efficiency':efficiency,'momentum':momentum,'market':market}
        for quiet,box in itertools.product((.4,.7),(5,10)):
            out.append({**context,'family':'box','quiet':quiet,'box':box,'entry':'stop'})
        for entry in ('open','stop'):
            out.append({**context,'family':'reclaim','entry':entry})
        for top in (1,2):
            out.append({**context,'family':'weekly','entry':'open','top':top})
    return [{'name':f'M{i+1:03d}',**p} for i,p in enumerate(out)]

def context_mask(f,p):
    mask=(f.trend_stack & f.beta126.between(1.2,3.) & (f.efficiency63>=p['efficiency'])
          & (f.largest_step_share63<=.15) & (f.momentum126_skip21>=p['momentum'])
          & (f.relative_return126>0) & (f.dollar_volume63>=20_000_000)
          & (f.signal_close>=10))
    if p['market']:
        mask &= f.market_above200
    return mask.fillna(False).to_numpy()

def signal_selection(f,p):
    mask=context_mask(f,p)
    if p['family']=='box':
        mask &= ((f.quiet_volume_rate_ratio<=p['quiet']) & (f.range5_atr<=3)
                 & (f.range10_atr<=4) & (f.distance_high252_pct<=10)).fillna(False).to_numpy()
    elif p['family']=='reclaim':
        mask &= (f.ema8_reclaim & f.touched_ema21_recent3 & f.above_ema21
                 & (f.day_range_atr<=1.5)).fillna(False).to_numpy()
    else:
        mask &= f.weekly.to_numpy(bool)
        ranked=f.loc[mask].assign(score=lambda x:x.momentum126_skip21*x.efficiency63)
        ranked=ranked.sort_values(['signal_date','score','ticker'],ascending=[True,False,True])
        chosen=ranked.groupby('signal_date',sort=False).head(p['top']).index
        mask[:]=False; mask[chosen]=True
    return mask

def prepare(out):
    snapshot=(ROOT/'data/master_prices.parquet').read_bytes()
    raw=pd.read_parquet(io.BytesIO(snapshot)); raw.date=pd.to_datetime(raw.date).dt.normalize()
    hours=market_open_hours()
    # The hours provider may lag newly announced mourning/disaster closures.
    # The repo calendar carries those closures; intersect before session math.
    hours=hours.reindex(pd.date_range(hours.index.min(),hours.index.max(),freq=TRADING_DAY))
    end=min(pd.Timestamp(raw.date.max()),hours.index.max())
    raw=raw[raw.date<=end]
    if raw.duplicated(['ticker','date']).any():
        raise ValueError('Duplicate ticker dates in master prices')
    # Current identity metadata is used only to identify a fixed stock universe;
    # its present-day price, capitalization and volume are never historical gates.
    symbols_bytes=(ROOT/'data/symbol_master.parquet').read_bytes()
    symbols=pd.read_parquet(io.BytesIO(symbols_bytes))
    # This file is produced by a common-stock screener with isEtf/isFund=false.
    # Supplement its incomplete overlap with the configured book using known
    # equity-industry classifications, never present-day cap/volume thresholds.
    stock_names=set(symbols.ticker)
    sectors_bytes=(ROOT/'data/sector_map.parquet').read_bytes()
    sectors=pd.read_parquet(io.BytesIO(sectors_bytes))
    equity_sectors={'Technology','Industrials','Financial Services','Consumer Cyclical',
                    'Healthcare','Energy','Basic Materials','Consumer Defensive',
                    'Communication Services','Utilities','Real Estate'}
    stock_names |= set(sectors.loc[sectors.sector.isin(equity_sectors),'ticker'])
    excluded=set(config.OLV_CAP_EXEMPT_ETFS)|set(config.LEV3X_ALL)|{'PALL','PPLT'}
    wanted=(set(config.CSV_UNIVERSE)|set(config.LIQUID_PLUS_COMMODITIES))&stock_names-excluded
    spy=raw[raw.ticker=='SPY'].sort_values('date').set_index('date').Close.reindex(hours.index)
    assert not spy.empty
    weekly=set(hours.groupby(hours.index.to_period('W-FRI')).tail(1).index)
    rules=definitions(); frames=[]; parts={k:[] for k in ['Open','High','Low','Close','date']}
    offset=0; coverage=[]
    for i,(ticker,g) in enumerate(raw[raw.ticker.isin(wanted)].groupby('ticker',sort=True)):
        df,valid=research_bars(g,hours)
        df=calculate_indicators(df,{},ticker)
        f=smooth_momentum_features(df,spy,hours)
        f['ticker'],f['signal_date'],f['signal_idx']=ticker,df.index,np.arange(len(df))
        f['signal_close'],f['signal_high'],f['signal_low'],f['atr']=df.Close,df.High,df.Low,df.ATR
        f['above_ema21']=df.Close>df.EMA21; f['weekly']=df.index.isin(weekly)
        # Pool all contexts/triggers before any cross-sectional ranking so no
        # stricter candidate accidentally determines another candidate's pool.
        union=np.zeros(len(f),bool)
        for p in rules:
            q=context_mask(f,p)
            if p['family']=='weekly': q &= f.weekly.to_numpy(bool)
            elif p['family']=='reclaim': q &= (f.ema8_reclaim&f.touched_ema21_recent3&f.above_ema21&(f.day_range_atr<=1.5)).fillna(False).to_numpy()
            else: q &= ((f.quiet_volume_rate_ratio<=p['quiet'])&(f.range5_atr<=3)&(f.range10_atr<=4)&(f.distance_high252_pct<=10)).fillna(False).to_numpy()
            union |= q
        union &= df.index>=START
        if ticker=='RAMP': union &= df.index<pd.Timestamp('2026-05-18')
        if ticker=='CBZ': union &= df.index<pd.Timestamp('2026-07-29')
        idx=np.flatnonzero(union); chosen=idx[idx+1+MAX_HOLD<len(df)]
        chosen=chosen[np.array([valid.iloc[j-1:j+MAX_HOLD+2].all() for j in chosen],bool)]
        clean=np.isfinite(df.ATR.to_numpy()[chosen])&(df.ATR.to_numpy()[chosen]>0)
        chosen=chosen[clean]
        ff=f.iloc[idx].copy(); ff['event_idx']=-1
        ff.loc[df.index[chosen],'event_idx']=offset+np.arange(len(chosen))
        position=chosen[:,None]+1+np.arange(MAX_HOLD+1)[None,:]
        for k in parts:
            values=df.index.to_numpy() if k=='date' else df[k].to_numpy(float)
            parts[k].append(values[position])
        frames.append(ff.reset_index(drop=True)); offset+=len(chosen)
        coverage.append({'ticker':ticker,'bars':len(df),'raw_pool':len(ff),'clean_paths':len(chosen)})
        if i%100==0: print(f'Prepared {i+1}/{len(wanted)}: {ticker}',flush=True)
    f=pd.concat(frames,ignore_index=True)
    paths={k:np.concatenate(v) for k,v in parts.items()}
    f.to_parquet(out/'features.parquet',index=False); np.savez_compressed(out/'paths.npz',**paths)
    pd.DataFrame(coverage).to_csv(out/'coverage.csv',index=False)
    # Keep immutable inputs for subsequent portfolio-fit and robustness work.
    (out/'prices.parquet').write_bytes(snapshot)
    ledger=(ROOT/'data/backtest_trades_full.parquet').read_bytes()
    (out/'book_ledger.parquet').write_bytes(ledger)
    manifest={'price_sha256':hashlib.sha256(snapshot).hexdigest(),'ledger_sha256':hashlib.sha256(ledger).hexdigest(),
              'symbol_identity_sha256':hashlib.sha256(symbols_bytes).hexdigest(),
              'sector_identity_sha256':hashlib.sha256(sectors_bytes).hexdigest(),'end':str(end.date()),
              'fixed_universe':sorted(wanted),'covered':len(coverage),'rules':rules,
              'indicator_source_sha256':hashlib.sha256((ROOT/'indicators.py').read_bytes()).hexdigest(),
              'research_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'notes':['Fixed current stock universe, NOT point-in-time membership','Beta, liquidity and smoothness evaluated on each historical signal date',
                       'RAMP/CBZ post-announcement periods excluded; wider acquisition audit remains incomplete',
                       'All exits use common mature clean 42-session cohort; no live strategy changes']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return f,paths,manifest

def marks(m,r,paths,sessions):
    if m.empty: return np.zeros(len(sessions)),np.zeros(len(sessions),bool)
    n=len(m); width=int(r.exit_day.max())+1; day=r.exit_day.to_numpy(int)
    active=np.arange(width)[None,:]<=day[:,None]
    c=paths['Close'][:,:width].astype(float).copy(); c[np.arange(n),day]=r.exit
    atr=m.atr.to_numpy(float); entry=m.entry.to_numpy(float)
    change=np.empty_like(c); change[:,0]=(c[:,0]-entry*1.0005)/atr
    change[:,1:]=np.diff(c,axis=1)/atr[:,None]; change[~active]=0
    change[np.arange(n),day]-=r.exit.to_numpy(float)*.0005/atr
    np.testing.assert_allclose(change.sum(axis=1),r.net_atr,atol=1e-8)
    loc=sessions.get_indexer(pd.to_datetime(paths['date'][:,:width].ravel())).reshape(n,width)
    assert (loc[active]>=0).all()
    pnl=np.zeros(len(sessions)); live=np.zeros(len(sessions),bool)
    np.add.at(pnl,loc[active],change[active]); live[loc[active]]=True
    np.testing.assert_allclose(pnl.sum(),r.net_atr.sum(),atol=1e-8)
    return pnl,live

def evaluate(f,paths,manifest,out):
    sessions=pd.date_range(START,pd.Timestamp(manifest['end']),freq=TRADING_DAY)
    locations=sessions.get_indexer(pd.to_datetime(paths['date'].ravel())).reshape(paths['date'].shape)
    assert (locations>=0).all() and (np.diff(locations,axis=1)==1).all(), 'Nonconsecutive forward NYSE sessions'
    rows=[]; annual=[]; daily={}; trades=[]
    for p in definitions():
        chosen=signal_selection(f,p); raw=f.loc[chosen]
        clean=raw[raw.event_idx>=0]; ii=clean.event_idx.to_numpy(int)
        m=clean.reset_index(drop=True).copy(); pp={k:v[ii] for k,v in paths.items()}
        m['entry_idx']=m.signal_idx+1; m['entry_date']=pp['date'][:,0]
        if p['entry']=='open': fill=pp['Open'][:,0]
        else:
            trigger=m[f'box_high{p["box"]}'].to_numpy() if p['family']=='box' else m.signal_high.to_numpy()
            fill=buy_stop_fill(trigger,pp['Open'][:,0],pp['High'][:,0])
        usable=np.isfinite(fill)&(fill>0)
        m=m[usable].reset_index(drop=True); m['entry']=fill[usable].astype(float)
        pp={k:v[usable] for k,v in pp.items()}
        for hold,stop in itertools.product((5,10,21,42),(None,2.)):
            spec=ExitSpec(hold,stop,None); r=simulate(m,pp,spec)
            accepted=one_position_mask(m,r); mm=m[accepted].reset_index(drop=True); rr=r[accepted].reset_index(drop=True)
            ap={k:v[accepted] for k,v in pp.items()}; key=p['name']+'_'+spec.name
            pnl,live=marks(mm,rr,ap,sessions); daily[key]=(pnl,live)
            row={'candidate':p['name'],'family':p['family'],'exit':spec.name,'key':key,'raw_signals':len(raw),**metrics(rr)}
            for era,mask,lo,hi in [('train',mm.signal_date<'2018-01-01',2005,2017),('recent',mm.signal_date>='2018-01-01',2018,2025),('full',pd.Series(True,index=mm.index),2005,2025)]:
                tt=mm.loc[mask].reset_index(drop=True); rt=rr.loc[mask].reset_index(drop=True)
                ep,el=marks(tt,rt,{k:v[mask.to_numpy()] for k,v in ap.items()},sessions)
                period=(sessions.year>=lo)&(sessions.year<=(2026 if era in ('recent','full') else hi))
                count=tt.signal_date.dt.year.value_counts().reindex(range(lo,hi+1),fill_value=0)
                row.update({era+'_tim':sharpe(ep[el]),era+'_all':sharpe(ep[period]),era+'_pf':metrics(rt).get('pf_atr',np.nan),
                            era+'_trades':len(tt),era+'_mean_trades_year':float(count.mean()),era+'_min_year':int(count.min()),
                            era+'_max_year':int(count.max()),era+'_mean_beta':tt.beta126.mean(),era+'_mean_efficiency':tt.efficiency63.mean()})
            row['frequency_ok']=(row['recent_min_year']>10 and row['full_max_year']<=200)
            rows.append(row)
            for year in range(2005,2027):
                mask=mm.signal_date.dt.year==year
                annual.append({'key':key,'year':year,'trades':int(mask.sum()),'raw_signals':int((raw.signal_date.dt.year==year).sum()),
                               'net_atr':float(rr.loc[mask,'net_atr'].sum()),'calendar_pnl_atr':float(pnl[sessions.year==year].sum())})
            trades.append(pd.concat([mm[['ticker','signal_date','entry_date','entry','atr','beta126','efficiency63']],rr],axis=1).assign(key=key,exit_date=ap['date'][np.arange(len(rr)),rr.exit_day]))
        print(f'Evaluated {p["name"]} {p["family"]}: {len(raw)} raw signals',flush=True)
    result=pd.DataFrame(rows); result.to_csv(out/'results.csv',index=False)
    pd.DataFrame(annual).to_csv(out/'annual.csv',index=False)
    pd.concat(trades,ignore_index=True).to_parquet(out/'trades.parquet',index=False)
    np.savez_compressed(out/'daily.npz',dates=sessions.to_numpy(),**{k:np.array(v) for k,v in daily.items()})
    early=result[(result.train_mean_trades_year>10)&(result.train_max_year<=200)]
    leaders=early.sort_values('train_tim',ascending=False).groupby('family',sort=False).head(1)
    leaders.to_csv(out/'early_selected_family_leaders.csv',index=False)
    cols=['key','family','train_tim','recent_tim','full_tim','recent_pf','recent_trades','recent_mean_trades_year','recent_min_year','full_max_year','frequency_ok']
    print('EARLY SELECTED LEADERS\n'+leaders[cols].to_string(index=False),flush=True)
    print('EXPLORATORY RECENT LEADERS\n'+result[result.frequency_ok].sort_values('recent_tim',ascending=False).head(10)[cols].to_string(index=False),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'artifacts/smooth-momentum-expanded')
    parser.add_argument('--reuse',action='store_true'); args=parser.parse_args(); out=args.out.resolve()
    if not out.is_relative_to(ROOT/'artifacts'): parser.error('Output must be under artifacts/')
    out.mkdir(parents=True,exist_ok=True)
    if args.reuse:
        manifest=json.loads((out/'manifest.json').read_text())
        assert manifest['rules']==definitions()
        assert manifest['indicator_source_sha256']==hashlib.sha256((ROOT/'indicators.py').read_bytes()).hexdigest()
        assert manifest['research_source_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        f=pd.read_parquet(out/'features.parquet')
        with np.load(out/'paths.npz') as z: paths={k:z[k] for k in z.files}
    else: f,paths,manifest=prepare(out)
    evaluate(f,paths,manifest,out)

if __name__=='__main__': main()
