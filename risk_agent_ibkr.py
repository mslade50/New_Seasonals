"""Real-time IBKR option pricing for the Risk Agent (quotes only, blind by design).

    python risk_agent_ibkr.py chain SPY --max-dte 14 [--min-dte 0] [--out PATH]
    python risk_agent_ibkr.py quote <conid> [<conid> ...]

BLINDNESS: the Risk Agent sleeve must never see the real book. This module
requests ONLY contract details, option chain parameters, market data (quotes
and model greeks) and historical bars. It never reads account state or
places anything. It skips the high-level connect helper, because that synchronises
account state into the IB object at startup; instead it opens the bare API
socket (client.connectAsync) so nothing but market data flows. The IB object
is private to this module and is never returned or printed; callers get plain
quote dicts. tests/test_risk_agent_ibkr.py greps this source for forbidden
method names, so keep prose in here free of them too.

Env: RISK_AGENT_IB_HOST (127.0.0.1), RISK_AGENT_IB_PORT (7496),
RISK_AGENT_IB_CLIENT_ID (77).

IB BID_ASK historical bar semantics relied on (TWS API docs): for
whatToShow="BID_ASK" the bar's open is the time-weighted average bid over the
bar, close is the time-weighted average ask, low is the minimum bid and high
the maximum ask. historical_bid_ask averages open (bid) and close (ask) over
the bars in the window.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import math
import os
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from risk_agent_grammar import chain_quote_key  # noqa: E402

ET = ZoneInfo("America/New_York")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7496
DEFAULT_CLIENT_ID = 77
LIVE_CHAINS_PATH = ROOT / "data" / "risk_agent_chains_live.json"
BATCH = 90                 # concurrent market data lines per batch (default limit is 100)
HIST_TIMEOUT = 15.0

_SHARED = {"ib": None, "failed": False}


def _log(msg: str) -> None:
    print(f"[risk_agent_ibkr] {msg}", flush=True)


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _pos(v):
    x = _num(v)
    return x if x is not None and x > 0 else None


def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def connect(timeout: float = 8):
    """Open a market-data-only API session. Returns the private IB or None. Never raises."""
    try:
        from ib_insync import IB, util
    except ImportError:
        _log("ib_insync is not installed; IBKR pricing unavailable")
        return None
    host = os.environ.get("RISK_AGENT_IB_HOST", DEFAULT_HOST)
    port = int(os.environ.get("RISK_AGENT_IB_PORT", DEFAULT_PORT))
    cid = int(os.environ.get("RISK_AGENT_IB_CLIENT_ID", DEFAULT_CLIENT_ID))
    ib = IB()
    try:
        ib.wrapper.clientId = cid
        util.run(ib.client.connectAsync(host, port, cid, timeout))
        if not ib.client.isReady():
            raise ConnectionError("API socket not ready")
        return ib
    except BaseException as exc:  # noqa: BLE001 - never raise to the caller
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        _log(f"TWS/Gateway not reachable at {host}:{port} ({type(exc).__name__}: {exc})")
        try:
            ib.disconnect()
        except Exception:  # noqa: BLE001
            pass
        return None


def _shared(timeout: float = 8):
    if _SHARED["ib"] is not None:
        return _SHARED["ib"]
    if _SHARED["failed"]:
        return None
    ib = connect(timeout)
    if ib is None:
        _SHARED["failed"] = True
    _SHARED["ib"] = ib
    return ib


def is_available() -> bool:
    return _shared() is not None


def disconnect() -> None:
    ib = _SHARED["ib"]
    _SHARED["ib"] = None
    if ib is not None:
        try:
            ib.disconnect()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Market data helpers
# ---------------------------------------------------------------------------

def _greeks(t):
    g = getattr(t, "modelGreeks", None)
    if not g:
        return {}
    return {"iv": _pos(g.impliedVol), "delta": _num(g.delta), "gamma": _num(g.gamma),
            "theta": _num(g.theta), "vega": _num(g.vega)}


def _tick_ts(t) -> str:
    ts = getattr(t, "time", None)
    if isinstance(ts, dt.datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=dt.timezone.utc)
        return ts.astimezone(dt.timezone.utc).isoformat(timespec="seconds")
    return _now_utc().isoformat(timespec="seconds")


def _quote_of(t, con_id=None) -> dict | None:
    bid = _num(t.bid)
    bid = bid if bid is not None and bid >= 0 else None
    ask = _pos(t.ask)
    if bid is None and ask is None:
        return None
    mid = (bid + ask) / 2.0 if bid and ask else None
    g = _greeks(t)
    return {"bid": bid, "ask": ask, "mid": mid,
            "iv": g.get("iv"), "delta": g.get("delta"), "gamma": g.get("gamma"),
            "theta": g.get("theta"), "vega": g.get("vega"),
            "con_id": con_id if con_id is not None else getattr(t.contract, "conId", None),
            "bid_size": _num(getattr(t, "bidSize", None)),
            "ask_size": _num(getattr(t, "askSize", None)),
            "quote_ts": _tick_ts(t)}


def _stream(ib, contracts, max_seconds=6.0, want=0.85):
    """Subscribe, wait until most lines carry a quote, cancel. Returns tickers."""
    tickers = [ib.reqMktData(c, "", False, False) for c in contracts]
    t0 = time.time()
    while time.time() - t0 < max_seconds:
        n = max(1, len(tickers))
        have = sum(1 for t in tickers if _num(t.ask) not in (None, -1.0) and _num(t.bid) is not None
                   and (_greeks(t) or _num(t.ask) <= 0))
        if have >= want * n:
            break
        ib.sleep(0.4)
    for c in contracts:
        try:
            ib.cancelMktData(c)
        except Exception:  # noqa: BLE001
            pass
    return tickers


def _set_data_type_and_spot(ib, stock, wait=4.0):
    """Ask for live data; fall back to frozen/delayed. Returns (spot, spot_ts, type)."""
    for mdt in (1, 2, 3, 4):
        ib.reqMarketDataType(mdt)
        t = ib.reqMktData(stock, "", False, False)
        t0 = time.time()
        while time.time() - t0 < wait:
            if _pos(t.marketPrice()) or _pos(t.last) or _pos(t.close):
                break
            ib.sleep(0.25)
        spot = _pos(t.marketPrice()) or _pos(t.last) or _pos(t.close)
        got = int(getattr(t, "marketDataType", 0) or mdt)
        ts = _tick_ts(t)
        try:
            ib.cancelMktData(stock)
        except Exception:  # noqa: BLE001
            pass
        if spot:
            return float(spot), ts, got
    return None, None, 0


def _pick_strikes(strikes, spot, band, max_n):
    inband = [k for k in strikes if abs(k / spot - 1.0) <= band]
    if not inband:
        inband = sorted(strikes, key=lambda k: abs(k - spot))[:4]
    inband = sorted(inband)
    if len(inband) <= max_n:
        return inband
    atm = sorted(inband, key=lambda k: abs(k - spot))[:4]
    n_rest = max(0, max_n - len(atm))
    idx = ({round(i * (len(inband) - 1) / max(1, n_rest - 1)) for i in range(n_rest)}
           if n_rest else set())
    return sorted(set(atm) | {inband[i] for i in idx})


def _iso(expiry8: str) -> str:
    return f"{expiry8[:4]}-{expiry8[4:6]}-{expiry8[6:8]}"


def option_chain(underlying, *, min_dte=0, max_dte=60, expiries=None, strike_band_pct=None,
                 max_strikes_per_expiry=40, time_budget=55.0) -> dict:
    """Live quotes for every listed expiry in [min_dte, max_dte] (weeklies and 0DTE included).

    Returns {"underlying","spot","spot_ts","market_data_type","asof_utc","asof",
    "quotes": {chain_quote_key(...): {...}}, "expiries": [...], "truncated": bool}.
    On any failure returns the same shape with empty quotes and an "error" string.
    """
    underlying = underlying.upper()
    out = {"underlying": underlying, "spot": None, "spot_ts": None, "market_data_type": 0,
           "asof_utc": _now_utc().isoformat(timespec="seconds"),
           "asof": dt.datetime.now(ET).date().isoformat(), "quotes": {}, "expiries": [],
           "truncated": False}
    ib = _shared()
    if ib is None:
        out["error"] = "ibkr_unavailable"
        return out
    t_start = time.time()
    try:
        from ib_insync import Option, Stock
        stock = Stock(underlying, "SMART", "USD")
        if not ib.qualifyContracts(stock):
            out["error"] = "could not qualify underlying"
            return out
        spot, spot_ts, mdt = _set_data_type_and_spot(ib, stock)
        out.update(spot=spot, spot_ts=spot_ts, market_data_type=mdt)
        if not spot:
            out["error"] = "no underlying price"
            return out
        params = ib.reqSecDefOptParams(stock.symbol, "", stock.secType, stock.conId)
        smart = [p for p in params if p.exchange == "SMART"] or list(params)
        std = [p for p in smart if p.tradingClass == underlying] or smart[:1]
        if not std:
            out["error"] = "no option parameters"
            return out
        tclass = std[0].tradingClass
        all_exp = sorted(set().union(*(set(p.expirations) for p in std)))
        all_k = sorted(set().union(*(set(p.strikes) for p in std)))
        today = dt.datetime.now(ET).date()
        if expiries:
            want = {e.replace("-", "") for e in expiries}
            chosen = [e for e in all_exp if e in want]
        else:
            chosen = [e for e in all_exp
                      if min_dte <= (dt.date(int(e[:4]), int(e[4:6]), int(e[6:8])) - today).days <= max_dte]
        out["expiries"] = [_iso(e) for e in chosen]
        contracts = []
        for e in chosen:
            dte = max((dt.date(int(e[:4]), int(e[4:6]), int(e[6:8])) - today).days, 0)
            band = strike_band_pct if strike_band_pct is not None else max(0.03, 0.025 * math.sqrt(dte))
            for k in _pick_strikes(all_k, spot, band, max_strikes_per_expiry):
                for right in ("C", "P"):
                    contracts.append(Option(underlying, e, k, right, "SMART", tradingClass=tclass))
        for i in range(0, len(contracts), 50):
            if time.time() - t_start > time_budget:
                out["truncated"] = True
                break
            ib.qualifyContracts(*contracts[i:i + 50])
        live = [c for c in contracts if c.conId]
        types = []
        for i in range(0, len(live), BATCH):
            if time.time() - t_start > time_budget:
                out["truncated"] = True
                break
            for t in _stream(ib, live[i:i + BATCH]):
                q = _quote_of(t)
                if q is None:
                    continue
                c = t.contract
                out["quotes"][chain_quote_key(_iso(c.lastTradeDateOrContractMonth[:8]),
                                              c.strike, c.right)] = q
                mt = int(getattr(t, "marketDataType", 0) or 0)
                if mt:
                    types.append(mt)
        if types:
            out["market_data_type"] = max(set(types), key=types.count)
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def quote_contracts(con_ids) -> dict:
    """Refresh specific legs. Returns {con_id: quote dict + symbol/expiry/strike/right}."""
    ib = _shared()
    ids = [int(c) for c in con_ids if c]
    if ib is None or not ids:
        return {}
    out = {}
    try:
        from ib_insync import Contract
        ib.reqMarketDataType(1)
        contracts = [Contract(conId=c, exchange="SMART") for c in ids]
        ib.qualifyContracts(*contracts)
        live = [c for c in contracts if c.conId]
        for i in range(0, len(live), BATCH):
            for t in _stream(ib, live[i:i + BATCH], max_seconds=5.0):
                q = _quote_of(t)
                if q is None:
                    continue
                c = t.contract
                q.update(symbol=c.symbol, right=c.right, strike=c.strike,
                         expiry=_iso(c.lastTradeDateOrContractMonth[:8]),
                         market_data_type=int(getattr(t, "marketDataType", 0) or 0))
                out[int(c.conId)] = q
    except Exception as exc:  # noqa: BLE001
        _log(f"quote_contracts failed: {type(exc).__name__}: {exc}")
    return out


def resolve_con_id(underlying, expiry, strike, right):
    """conId of one option contract (expiry YYYY-MM-DD), or None."""
    ib = _shared()
    if ib is None:
        return None
    try:
        from ib_insync import Option
        c = Option(underlying, expiry.replace("-", ""), float(strike), right, "SMART",
                   tradingClass=underlying)
        ib.qualifyContracts(c)
        return int(c.conId) or None
    except Exception as exc:  # noqa: BLE001
        _log(f"resolve_con_id failed: {type(exc).__name__}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Historical bid/ask
# ---------------------------------------------------------------------------

def _hist_bars(ib, con_id, day, end_et, seconds):
    from ib_insync import Contract
    c = Contract(conId=int(con_id), exchange="SMART")
    ib.qualifyContracts(c)
    if not c.conId:
        return []
    end = f"{day.replace('-', '')} {end_et}:00 US/Eastern"
    coro = ib.reqHistoricalDataAsync(c, endDateTime=end, durationStr=f"{seconds} S",
                                     barSizeSetting="1 min", whatToShow="BID_ASK",
                                     useRTH=True, formatDate=1)
    from ib_insync import util
    return util.run(asyncio.wait_for(coro, timeout=HIST_TIMEOUT)) or []


def _avg_bid_ask(bars, ts):
    bids = [b.open for b in bars if _num(b.open) is not None and b.open >= 0]
    asks = [b.close for b in bars if _pos(b.close)]
    if not bids or not asks:
        return None
    return {"bid": sum(bids) / len(bids), "ask": sum(asks) / len(asks), "ts": ts}


def historical_bid_ask(con_id, date, start_et="09:30", end_et="09:35"):
    """Average bid/ask over [start_et, end_et] ET on `date` (YYYY-MM-DD), or None."""
    ib = _shared()
    if ib is None or not con_id:
        return None
    try:
        h0, m0 = map(int, start_et.split(":"))
        h1, m1 = map(int, end_et.split(":"))
        seconds = max(60, ((h1 * 60 + m1) - (h0 * 60 + m0)) * 60)
        bars = _hist_bars(ib, con_id, date, end_et, seconds)
        return _avg_bid_ask(bars, f"{date}T{end_et}:00-ET")
    except Exception as exc:  # noqa: BLE001
        _log(f"historical_bid_ask failed: {type(exc).__name__}: {exc}")
        return None


def historical_close_bid_ask(con_id, date):
    """Bid/ask of the last regular-hours minute on `date`, or None."""
    ib = _shared()
    if ib is None or not con_id:
        return None
    try:
        bars = _hist_bars(ib, con_id, date, "16:00", 60)
        return _avg_bid_ask(bars[-1:], f"{date}T16:00:00-ET")
    except Exception as exc:  # noqa: BLE001
        _log(f"historical_close_bid_ask failed: {type(exc).__name__}: {exc}")
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def merge_live_file(path: Path, block: dict) -> None:
    cur = {}
    if path.exists():
        try:
            cur = json.loads(path.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            cur = {}
    if isinstance(cur.get("chains"), dict):
        cur = cur["chains"]
    cur[block["underlying"]] = block
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cur, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("chain")
    c.add_argument("underlying")
    c.add_argument("--min-dte", type=int, default=0)
    c.add_argument("--max-dte", type=int, default=60)
    c.add_argument("--max-strikes", type=int, default=40)
    c.add_argument("--band-pct", type=float, default=None)
    c.add_argument("--out", default=str(LIVE_CHAINS_PATH))
    q = sub.add_parser("quote")
    q.add_argument("con_ids", nargs="+", type=int)
    args = ap.parse_args(argv)
    try:
        if not is_available():
            print("IBKR not reachable; nothing written.")
            return 1
        if args.cmd == "chain":
            blk = option_chain(args.underlying, min_dte=args.min_dte, max_dte=args.max_dte,
                               strike_band_pct=args.band_pct, max_strikes_per_expiry=args.max_strikes)
            if blk.get("error") or not blk["quotes"]:
                print(f"{blk['underlying']}: no live chain ({blk.get('error', 'no quotes')})")
                return 1
            merge_live_file(Path(args.out), blk)
            print(f"{blk['underlying']}: spot {blk['spot']} data_type {blk['market_data_type']} "
                  f"expiries {len(blk['expiries'])} ({blk['expiries'][0]}..{blk['expiries'][-1]}) "
                  f"quotes {len(blk['quotes'])}{' TRUNCATED' if blk['truncated'] else ''} -> {args.out}")
            return 0
        res = quote_contracts(args.con_ids)
        for cid, v in res.items():
            print(f"{cid} {v['symbol']} {v['expiry']} {v['strike']:g}{v['right']} "
                  f"bid {v['bid']} ask {v['ask']} type {v['market_data_type']}")
        return 0 if res else 1
    finally:
        disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
