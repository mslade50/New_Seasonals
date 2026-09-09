"""Read a reviewed Primary seed and its matching canonical fill generation.

The absence of a reviewed start/continuous history remains unknown; it never
reads the modeled Portfolio sheet. No writes, orders, or source repair occurs.
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

def load_actual_inventory(*, asof=None, algo_strategies=None, seed_path=None, max_age_seconds=300):
    path=Path(seed_path or os.environ.get("TAGGED_INVENTORY_SEED", ROOT/".local/tagged_inventory_seed.json"))
    try:
        seed=json.loads(path.read_text(encoding="utf-8-sig"))
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
        return build_tagged_inventory(seed,pd.read_parquet(io.BytesIO(body)).to_dict("records"),
                                      status.get("completeness") or {},
                                      asof=observed.isoformat(),
                                      algo_strategies=algo_strategies,
                                      entry_metadata=seed.get("entry_metadata"))
    except FileNotFoundError:
        return TaggedInventory(reasons=["reviewed starting inventory is not configured"])
    except Exception as exc:
        # Surface the boundary without credential values or broker response text.
        return TaggedInventory(reasons=[f"actual inventory could not be verified ({type(exc).__name__})"])


def load_raw_exit_bars(ticker, *, now=None, download=None):
    """Settled raw OHLCV for a frozen actual entry/stop; never adjusted-cache fallback."""
    import numpy as np
    stamp=pd.Timestamp(now or pd.Timestamp.now(tz="America/New_York"))
    if stamp.tzinfo is None:
        raise ValueError("raw exit valuation time must be timezone aware")
    local=stamp.tz_convert("America/New_York")
    day=local.tz_localize(None).normalize()
    end=day+pd.Timedelta(days=1) if local.hour>=16 else day
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
    if not token and book_loader is None:
        raise ValueError('broker read access is unavailable')
    book = (book_loader or fetch_book)(os.environ.get('EXEC_BROKER_URL',DEFAULT_BROKER_URL), token)
    # Require the fill generation to include everything already reflected in
    # this book. Otherwise a partial fill can disappear from remaining orders
    # before it appears in held inventory, understating combined exposure.
    primary = [a for a in book['accounts'] if a.get('key') == 'primary']
    if len(primary) != 1:
        raise ValueError('ambiguous Primary order snapshot')
    source = pd.Timestamp(primary[0]['orders_source_at'], unit='s', tz='UTC')
    if source > pd.Timestamp(inventory.asof_utc):
        raise ValueError('fill inventory has not caught up with pending-order snapshot')
    return pending_entry_notionals(book, inventory.broker_account,
        asof=asof or pd.Timestamp.now(tz='UTC').isoformat())


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
