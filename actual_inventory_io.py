"""Read a reviewed Primary seed and its matching canonical fill generation.

The absence of a reviewed start/continuous history remains unknown; it never
reads the modeled Portfolio sheet. A stale live feed may trigger a read-only
broker observation, published separately from the site's command-agent book.
"""
from __future__ import annotations
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pandas as pd
from tagged_inventory import TaggedInventory, build_tagged_inventory

ROOT = Path(__file__).resolve().parent


def _coverage_start(primary, algorithms):
    # Gateway's closed-session proof is specific to the existing OLV stock
    # route. It must never attest other algorithms or whole-account history.
    if (set(algorithms)=={'Oversold Low Volume'}
            and (primary.get('olv_coverage') or {}).get('scope')=='OLV_US_STK_NON_OVERNIGHT'
            and primary.get('olv_continuous_from')):
        return primary['olv_continuous_from']
    return primary.get('continuous_from')

def load_reviewed_seed(seed_path=None):
    """Read an explicit local review or the shared R2 review, never modeled data.

    Both scheduled scanners use the same R2 object by default. An explicit
    local path is useful for validation and must not silently fall back.
    """
    configured = seed_path or os.environ.get('TAGGED_INVENTORY_SEED')
    if configured:
        return json.loads(Path(configured).read_text(encoding='utf-8-sig'))
    from cache_io import _client, _r2_creds
    client, creds = _client(), _r2_creds()
    if client is None or creds is None:
        raise FileNotFoundError('reviewed starting inventory is not configured')
    key = os.environ.get('TAGGED_INVENTORY_SEED_R2_KEY', 'ops/tagged_inventory_seed.json')
    return json.loads(client.get_object(Bucket=creds['R2_BUCKET'], Key=key)['Body'].read())


def _load_canonical_inventory(*, asof=None, algo_strategies=None, seed_path=None, max_age_seconds=300, reviewed_seed=None):
    try:
        seed=reviewed_seed if reviewed_seed is not None else load_reviewed_seed(seed_path)
        from cache_io import _client, _r2_creds
        client,creds=_client(),_r2_creds()
        if client is None or creds is None:
            raise ValueError("canonical fill storage unavailable")
        def read(key):
            return client.get_object(Bucket=creds["R2_BUCKET"],Key=key)["Body"].read()
        status=json.loads(read("live_fills_status.json"))
        body=read("live_fills.parquet")
        if status.get("canonical_sha256")!=hashlib.sha256(body).hexdigest():
            raise ValueError("fill generation and status do not match")
        # Harvester summary.complete means no historical gap after Primary
        # validation; broker aggregate completeness lives one level below.
        if status.get("complete") is not True or status.get("gap",{}).get("gap"):
            raise ValueError("canonical fill history contains unresolved gaps")
        requested=pd.Timestamp(asof or pd.Timestamp.now(tz="UTC"))
        through=pd.Timestamp((status.get("completeness") or {}).get("accounts",{}).get("primary",{}).get("complete_through"))
        if requested.tzinfo is None or pd.isna(through) or through.tzinfo is None:
            raise ValueError("inventory observation time is unavailable")
        now=pd.Timestamp.now(tz="UTC")
        if through>now+pd.Timedelta(seconds=5) or requested-through>pd.Timedelta(seconds=max_age_seconds):
            raise ValueError("inventory generation is stale or future-dated")
        observed=min(requested,through)
        if algo_strategies is None:
            from strategy_config import STRATEGY_BOOK
            algo_strategies={s["name"] for s in STRATEGY_BOOK}
        coverage=json.loads(json.dumps(status.get("completeness") or {}))
        primary=coverage['accounts']['primary']
        primary['continuous_from']=_coverage_start(primary,algo_strategies)
        return build_tagged_inventory(seed,pd.read_parquet(io.BytesIO(body)).to_dict("records"),
                                      coverage,
                                      asof=observed.isoformat(),
                                      algo_strategies=algo_strategies,
                                      entry_metadata=seed.get("entry_metadata"))
    except FileNotFoundError:
        return TaggedInventory(reasons=["reviewed starting inventory is not configured"])
    except Exception as exc:
        # Surface the boundary without credential values or broker response text.
        return TaggedInventory(reasons=[f"actual inventory could not be verified ({type(exc).__name__})"])


def load_actual_inventory(*, asof=None, algo_strategies=None, seed_path=None,
                          max_age_seconds=300, fills_loader=None, canonical_loader=None):
    """Read a coherent live observation, extending canonical history if needed.

    A stale feed can publish one read-only local Gateway observation.
    A newly reviewed seed inside the live coverage window
    does not depend on pre-seed history. Older seeds require overlapping,
    digest-verified canonical coverage. Source failure never means flat.
    """
    try:
        seed=load_reviewed_seed(seed_path)
    except Exception as exc:
        return TaggedInventory(reasons=[f'reviewed starting inventory is unavailable ({type(exc).__name__})'])
    token=os.environ.get('STATUS_TOKEN','').strip()
    if not token and fills_loader is None:
        return _load_canonical_inventory(asof=asof,algo_strategies=algo_strategies,
            reviewed_seed=seed,max_age_seconds=max_age_seconds)
    stage = 'loading inventory dependencies'
    try:
        from scripts.harvest_fills import fetch_fills,normalize,merge_fills,validate_source_completeness,DEFAULT_BROKER_URL
        url=os.environ.get('EXEC_BROKER_URL',DEFAULT_BROKER_URL)
        stage = 'reading live execution coverage'
        try:
            payload=(fills_loader or fetch_fills)(url,token)
            if fills_loader is None:
                validate_source_completeness(payload)
        except Exception:
            if fills_loader is None:
                from scripts.refresh_inventory_observation import refresh_local_inventory
                stage = 'refreshing the local Primary observation'
                refresh_local_inventory(url)
                payload=fetch_fills(url,token)
            else:
                raise
        stage = 'validating live execution coverage'
        validate_source_completeness(payload)
        if algo_strategies is None:
            from strategy_config import STRATEGY_BOOK
            algo_strategies={s['name'] for s in STRATEGY_BOOK}
        coverage=json.loads(json.dumps(payload['completeness']))
        primary=coverage['accounts']['primary']
        start=pd.Timestamp(seed['asof_utc'])
        primary['continuous_from']=_coverage_start(primary,algo_strategies)
        live_start=pd.Timestamp(primary['continuous_from'])
        through=pd.Timestamp(primary['complete_through'])
        requested=pd.Timestamp(asof or pd.Timestamp.now(tz='UTC'))
        if any(pd.isna(t) or t.tzinfo is None for t in (start,live_start,through,requested)):
            raise ValueError('inventory coverage timestamps are invalid')
        if requested-through>pd.Timedelta(seconds=max_age_seconds):
            raise ValueError('live inventory is stale')
        frame=normalize(payload['fills'])
        if live_start>start:
            stage = 'bridging reviewed inventory to current executions'
            if canonical_loader is None:
                from cache_io import _client,_r2_creds
                client,creds=_client(),_r2_creds()
                if client is None or creds is None:raise ValueError('canonical fill storage unavailable')
                def canonical_loader():
                    def read(key):return client.get_object(Bucket=creds['R2_BUCKET'],Key=key)['Body'].read()
                    return json.loads(read('live_fills_status.json')),read('live_fills.parquet')
            status,body=canonical_loader()
            if status.get('canonical_sha256')!=hashlib.sha256(body).hexdigest():
                raise ValueError('canonical fill generation mismatch')
            old=(status.get('completeness') or {}).get('accounts',{}).get('primary',{})
            old=dict(old,continuous_from=_coverage_start(old,algo_strategies))
            prior_coverage=status.get('completeness') or {}
            if (status.get('complete') is not True or old.get('complete') is not True
                    or status.get('gap',{}).get('gap') or prior_coverage.get('truncated')
                    or prior_coverage.get('merge_error') or prior_coverage.get('incomplete_days')
                    or old.get('broker_account')!=primary['broker_account']
                    or pd.Timestamp(old['continuous_from'])>start
                    or pd.Timestamp(old['complete_through'])<live_start):
                raise ValueError('verified fill history does not bridge seed to current executions')
            frame,_=merge_fills(pd.read_parquet(io.BytesIO(body)),frame)
            primary['continuous_from']=old['continuous_from']
        stage = 'matching Primary orders and frozen entry metadata'
        book=payload.get('book')
        accounts=[a for a in (book or {}).get('accounts',[]) if a.get('key')=='primary']
        if len(accounts)!=1 or accounts[0].get('error') or accounts[0].get('broker_account')!=seed['broker_account']:
            raise ValueError('matching Primary order observation unavailable')
        observation=pd.Timestamp(round(float(accounts[0]['orders_source_at'])*1000),unit='ms',tz='UTC')
        if observation>through:raise ValueError('book observation is newer than execution coverage')
        metadata=dict(seed.get('entry_metadata') or {})
        for ref,value in (accounts[0].get('entry_metadata') or {}).items():
            if ref in metadata and any(metadata[ref].get(k)!=value.get(k) for k in ('atr','exit_deadline_utc','exit_protocol')):
                raise ValueError('frozen entry metadata changed')
            metadata[ref]=value
        if algo_strategies is None:
            from strategy_config import STRATEGY_BOOK
            algo_strategies={s['name'] for s in STRATEGY_BOOK}
        stage = 'reconciling algorithm inventory with Primary positions'
        result=build_tagged_inventory(seed,frame.to_dict('records'),coverage,
            asof=min(requested,through).isoformat(),algo_strategies=algo_strategies,entry_metadata=metadata)
        if result.status=='known':
            # An unassigned discretionary trade can change the net stock
            # holding without changing the algorithm ledger. Do not turn that
            # discrepancy into an automated over-close or guessed allocation.
            actual={}
            for row in accounts[0].get('positions',[]):
                if row.get('account')!=seed['broker_account']:
                    raise ValueError('position account mismatch')
                contract=int(row['con_id'])
                if contract in actual:raise ValueError('duplicate broker contract')
                actual[contract]=float(row['position'])
            owned={}
            for row in result.tranches:
                owned[row['con_id']]=owned.get(row['con_id'],0)+row['signed_qty']
            for contract,qty in owned.items():
                net=actual.get(contract,0)
                if qty*net<=0 or abs(qty)>abs(net):
                    raise ValueError('algorithm holdings exceed the reconciled broker position; allocation review required')
            result.observed_book=book
            result.source_evidence = dict(seed=seed, fills=json.loads(frame.to_json(orient='records', date_format='iso')),
                                          coverage=coverage, book=book, entry_metadata=metadata)
        return result
    except Exception as exc:
        from scripts.refresh_inventory_observation import InventoryRefreshError
        reason = str(exc) if isinstance(exc, InventoryRefreshError) else f'{stage} failed ({type(exc).__name__})'
        return TaggedInventory(reasons=[f'live inventory could not be verified: {reason}'])


def load_raw_exit_bars(ticker, *, now=None, download=None):
    """Settled raw OHLCV for a frozen actual entry/stop; never adjusted-cache fallback."""
    import numpy as np
    stamp=pd.Timestamp(now or pd.Timestamp.now(tz="America/New_York"))
    if stamp.tzinfo is None:
        raise ValueError("raw exit valuation time must be timezone aware")
    local=stamp.tz_convert("America/New_York")
    day=local.tz_localize(None).normalize()
    from equity_sessions import last_settled_session
    end=last_settled_session(stamp)+pd.Timedelta(days=1)
    if download is None:
        import yfinance as yf
        download=yf.download
    raw=download(ticker,start=str((day-pd.Timedelta(days=120)).date()),
                 end=str(end.date()),auto_adjust=False,back_adjust=False,
                 progress=False,threads=False,timeout=20)
    if raw is None or raw.empty:
        raise ValueError("raw exit bars unavailable")
    if isinstance(raw.columns,pd.MultiIndex):
        raw=raw.xs(ticker,level="Ticker",axis=1)
    raw=raw.copy()
    if isinstance(raw.columns,pd.MultiIndex):
        raw.columns=raw.columns.get_level_values(0)
    raw.columns=[str(c).capitalize() for c in raw.columns]
    cols=["Open","High","Low","Close","Volume"]
    if not set(cols)<=set(raw):
        raise ValueError("raw OHLCV incomplete")
    raw=raw[cols].apply(pd.to_numeric,errors="coerce")
    raw.index=pd.to_datetime(raw.index).tz_localize(None).normalize()
    raw=raw[raw.index<end].sort_index()
    raw=raw.loc[~raw.index.duplicated(keep="last")]
    if raw.empty or not np.isfinite(raw.to_numpy()).all() or (raw[cols[:4]]<=0).any().any() or (raw.Volume<0).any():
        raise ValueError("raw OHLCV contains invalid values")
    raw.attrs["price_basis"]="raw"
    return raw


def load_pending_entry_notionals(inventory, *, asof=None, book_loader=None):
    """Read existing book endpoint only. Raise on unknown capacity inputs."""
    from daily_execution_report import fetch_book, DEFAULT_BROKER_URL
    from olv_sizing import pending_entry_notionals
    if inventory.status != 'known' or not inventory.broker_account:
        raise ValueError('actual Primary inventory is unverified')
    token = os.environ.get('STATUS_TOKEN', '').strip()
    if not token and book_loader is None and inventory.observed_book is None:
        raise ValueError('broker read access is unavailable')
    book = inventory.observed_book if book_loader is None and inventory.observed_book is not None else (book_loader or fetch_book)(os.environ.get('EXEC_BROKER_URL',DEFAULT_BROKER_URL), token)
    # Require the fill generation to include everything already reflected in
    # this book. Otherwise a partial fill can disappear from remaining orders
    # before it appears in held inventory, understating combined exposure.
    primary = [a for a in book['accounts'] if a.get('key') == 'primary']
    if len(primary) != 1:
        raise ValueError('ambiguous Primary order snapshot')
    # Both broker observations and the relay's ISO timestamps have millisecond
    # resolution. Avoid float-seconds conversion inventing later nanoseconds.
    source = pd.Timestamp(round(float(primary[0]['orders_source_at']) * 1000), unit='ms', tz='UTC')
    if source > pd.Timestamp(inventory.asof_utc):
        raise ValueError('fill inventory has not caught up with pending-order snapshot')
    return pending_entry_notionals(book, inventory.broker_account,
        asof=asof or pd.Timestamp.now(tz='UTC').isoformat())


def load_primary_nav(inventory, *, asof=None, max_age_seconds=90):
    """Primary NetLiquidation from the same verified observation as capacity.

    Strategy risk sizing keeps its configured account value. Live concentration
    limits use actual Primary NAV, never PA or a stale/static substitute.
    """
    if inventory.status != 'known' or not inventory.broker_account:
        raise ValueError('actual Primary inventory is unverified')
    accounts = [a for a in (inventory.observed_book or {}).get('accounts', [])
                if a.get('key') == 'primary']
    if (len(accounts) != 1 or accounts[0].get('error')
            or accounts[0].get('broker_account') != inventory.broker_account):
        raise ValueError('matching Primary NAV observation unavailable')
    primary = accounts[0]
    now = pd.Timestamp(asof or pd.Timestamp.now(tz='UTC'))
    source = pd.Timestamp(round(float(primary['orders_source_at']) * 1000), unit='ms', tz='UTC')
    if (now.tzinfo is None or not pd.Timedelta(0) <= now-source <= pd.Timedelta(seconds=max_age_seconds)
            or source > pd.Timestamp(inventory.asof_utc)):
        raise ValueError('Primary NAV observation is stale or incoherent')
    nav = float(primary['nlv'])
    if not math.isfinite(nav) or nav <= 0:
        raise ValueError('Primary NetLiquidation is invalid')
    return nav


def olv_positions_from_inventory(inventory):
    """Adapt exact raw tranches for the existing volume-confirmed exit calculation."""
    if inventory.status!="known":
        raise ValueError("actual Primary inventory is unverified")
    positions=[]
    for row in inventory.tranches:
        if row["strategy"]!="Oversold Low Volume":
            continue
        if row.get("price_basis")!="raw" or not row.get("entry_order_ref"):
            raise ValueError("actual OLV entry price/tag provenance is unavailable")
        if row["signed_qty"]<=0:
            raise ValueError("OLV allocation has an unexpected short position")
        deadline=pd.Timestamp(row["exit_deadline_utc"])
        if deadline.tzinfo is None or any(not math.isfinite(float(row[key])) or float(row[key])<=0 for key in ("atr","entry_price")):
            raise ValueError("actual OLV exit deadline/ATR is unavailable")
        positions.append({"ticker":row["symbol"],"shares":int(row["signed_qty"]),
                          "entry":float(row["entry_price"]),"atr":float(row["atr"]),
                          "entry_date":pd.Timestamp(row["entry_date"]).normalize(),
                          "time_exit":str(deadline.tz_convert("America/New_York").date()),
                          "account_key":"primary","broker_account":row["account"],
                          "con_id":row["con_id"],"tranche_id":row["tranche_id"],
                          "ref_date":row["ref_date"],"entry_order_ref":row["entry_order_ref"]})
    return positions
