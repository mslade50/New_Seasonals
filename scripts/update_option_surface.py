"""Record nightly ETF/index option surfaces from IBKR and publish them to R2.

This is a read-only, prospective recorder. IBKR does not provide expired-chain
history, so each successful run becomes one new observation for term, skew,
and positioning percentiles used by the private Options page.

Suggested run time: weekdays around 5:20 PM ET while TWS is open.
Frozen market data is requested so the run is consistent after the close.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import cache_io
from options_surface import (
    CHAIN_TENORS,
    OPTIONS_MACRO_ETFS,
    POSITIONING_HISTORY_R2_KEY,
    SURFACE_HISTORY_R2_KEY,
    summarize_surface,
)

SURFACE_PATH = os.path.join(ROOT, "data", "option_surface_history.parquet")
POSITIONING_PATH = os.path.join(ROOT, "data", "option_positioning_history.parquet")
TERM_TARGETS = (7, 10, 20, 30, 45, 60, 90, 120, 180, 270, 365)
CHAIN_STRIKE_CAP = 18


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _num(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _mid(ticker):
    bid, ask = _num(ticker.bid), _num(ticker.ask)
    return (bid + ask) / 2.0 if bid and ask and bid > 0 and ask > 0 else None


def _iv(ticker):
    greeks = ticker.modelGreeks
    value = _num(greeks.impliedVol) if greeks else None
    return value if value and value > 0 else None


def _wait_quotes(ib, tickers, max_seconds=12.0, want_fraction=0.75, require_oi=False):
    started = time.time()
    while time.time() - started < max_seconds:
        n = max(1, len(tickers))
        priced = sum(_mid(t) is not None and _iv(t) is not None for t in tickers)
        oi = sum(_option_oi(t) is not None for t in tickers)
        if priced >= want_fraction * n and (not require_oi or oi >= 0.40 * n):
            break
        ib.sleep(0.5)


def _subscribe(ib, contracts, generic_ticks="", max_seconds=12.0):
    tickers = [ib.reqMktData(contract, generic_ticks, False, False) for contract in contracts]
    _wait_quotes(ib, tickers, max_seconds=max_seconds, require_oi=bool(generic_ticks))
    for contract in contracts:
        try:
            ib.cancelMktData(contract)
        except Exception:  # noqa: BLE001
            pass
    return tickers


def _option_oi(ticker):
    field = "callOpenInterest" if ticker.contract.right == "C" else "putOpenInterest"
    value = _num(getattr(ticker, field, None))
    return value if value is not None and value >= 0 else None


def _option_volume(ticker):
    field = "callVolume" if ticker.contract.right == "C" else "putVolume"
    values = [_num(getattr(ticker, field, None)), _num(getattr(ticker, "volume", None))]
    values = [v for v in values if v is not None and v >= 0]
    return max(values) if values else None


def _dte(expiry, today):
    return (dt.date(int(expiry[:4]), int(expiry[4:6]), int(expiry[6:8])) - today).days


def _nearest_strike(strikes, spot):
    return min(strikes, key=lambda strike: abs(strike - spot))


def _representative_expiries(expiries, today):
    valid = [expiry for expiry in expiries if _dte(expiry, today) > 0]
    selected = []
    for target in TERM_TARGETS:
        if not valid:
            break
        hit = min(valid, key=lambda expiry: abs(_dte(expiry, today) - target))
        if hit not in selected:
            selected.append(hit)
    return sorted(selected)


def _quote_term(ib, Option, symbol, trading_class, expiries, strikes, spot, today):
    contracts = []
    atm = _nearest_strike(strikes, spot)
    for expiry in _representative_expiries(expiries, today):
        contracts.extend([
            Option(symbol, expiry, atm, "C", "SMART", tradingClass=trading_class),
            Option(symbol, expiry, atm, "P", "SMART", tradingClass=trading_class),
        ])
    if not contracts:
        return []
    ib.qualifyContracts(*contracts)
    live = [contract for contract in contracts if contract.conId]
    ticks = _subscribe(ib, live, max_seconds=12.0)
    grouped = {}
    for ticker in ticks:
        expiry = ticker.contract.lastTradeDateOrContractMonth[:8]
        grouped.setdefault(expiry, []).append(ticker)
    rows = []
    for expiry, pair in grouped.items():
        ivs = [_iv(ticker) for ticker in pair]
        ivs = [value for value in ivs if value is not None]
        call = next((ticker for ticker in pair if ticker.contract.right == "C"), None)
        put = next((ticker for ticker in pair if ticker.contract.right == "P"), None)
        call_mid, put_mid = _mid(call) if call else None, _mid(put) if put else None
        dte = _dte(expiry, today)
        atm_iv = sum(ivs) / len(ivs) if ivs else None
        rows.append({
            "date": expiry, "dte": dte, "atm_strike": atm,
            "atm_iv": atm_iv,
            "straddle_mid": call_mid + put_mid if call_mid and put_mid else None,
            "em_1s_pct": atm_iv * math.sqrt(dte / 365.0) if atm_iv and dte > 0 else None,
        })
    return sorted(rows, key=lambda row: row["dte"])


def _chain_expiries(expiries, today):
    valid = [expiry for expiry in expiries if _dte(expiry, today) > 0]
    selected = []
    for target in CHAIN_TENORS:
        if not valid:
            break
        hit = min(valid, key=lambda expiry: abs(_dte(expiry, today) - target))
        if hit not in selected:
            selected.append(hit)
    return selected


def _quote_chain(ib, Option, symbol, trading_class, expiry, strikes, spot, today):
    dte = _dte(expiry, today)
    band_pct = max(0.08, 0.025 * math.sqrt(max(1, dte)))
    band = [strike for strike in strikes if spot * (1 - band_pct) <= strike <= spot * (1 + band_pct)]
    if len(band) > CHAIN_STRIKE_CAP:
        band = [band[round(i * (len(band) - 1) / (CHAIN_STRIKE_CAP - 1))] for i in range(CHAIN_STRIKE_CAP)]
    contracts = [
        Option(symbol, expiry, strike, right, "SMART", tradingClass=trading_class)
        for strike in band for right in ("C", "P")
    ]
    if not contracts:
        return []
    ib.qualifyContracts(*contracts)
    live = [contract for contract in contracts if contract.conId]
    ticks = _subscribe(ib, live, generic_ticks="100,101", max_seconds=15.0)
    rows = []
    for ticker in ticks:
        greeks = ticker.modelGreeks
        rows.append({
            "expiry": expiry, "dte": dte, "strike": float(ticker.contract.strike),
            "right": ticker.contract.right, "con_id": int(ticker.contract.conId or 0),
            "bid": _num(ticker.bid), "ask": _num(ticker.ask), "mid": _mid(ticker),
            "iv": _iv(ticker), "delta": _num(greeks.delta) if greeks else None,
            "gamma": _num(greeks.gamma) if greeks else None,
            "theta": _num(greeks.theta) if greeks else None,
            "vega": _num(greeks.vega) if greeks else None,
            "oi": _option_oi(ticker), "volume": _option_volume(ticker),
        })
    return rows


def _read_history(path, r2_key):
    cache_io.download_to_local(r2_key, path)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        log(f"could not read {os.path.basename(path)} ({exc}); starting a new local frame")
        return pd.DataFrame()


def _same_day_replace(prior, new, keys):
    if prior.empty:
        return new.copy()
    prior_keys = pd.MultiIndex.from_frame(prior[keys].astype(str))
    new_keys = pd.MultiIndex.from_frame(new[keys].astype(str))
    return pd.concat([prior.loc[~prior_keys.isin(new_keys)], new], ignore_index=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", help="comma-separated subset; default is the full ETF/index-proxy universe")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7496)
    parser.add_argument("--client-id", type=int, default=135)
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args(argv)
    symbols = [s.strip().upper() for s in args.tickers.split(",")] if args.tickers else list(OPTIONS_MACRO_ETFS)

    try:
        from ib_insync import IB, Option, Stock
    except ImportError:
        log("ib_insync is required on the TWS machine")
        return 1

    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    surface_prior = _read_history(SURFACE_PATH, SURFACE_HISTORY_R2_KEY)
    positioning_prior = _read_history(POSITIONING_PATH, POSITIONING_HISTORY_R2_KEY)
    ib = IB()
    try:
        ib.connect(args.host, args.port, clientId=args.client_id, timeout=8, readonly=True)
    except Exception as exc:  # noqa: BLE001
        log(f"TWS not reachable ({type(exc).__name__}: {exc})")
        return 1

    today = dt.date.today()
    pulled_at = time.time()
    summaries, positions, errors = [], [], {}
    try:
        ib.reqMarketDataType(2)  # frozen after close; live while the market is open
        for i, symbol in enumerate(symbols, 1):
            try:
                stock = Stock(symbol, "SMART", "USD")
                if not ib.qualifyContracts(stock):
                    raise RuntimeError("could not qualify underlying")
                [underlying] = ib.reqTickers(stock)
                spot = _num(underlying.marketPrice()) or _num(underlying.close)
                if not spot:
                    raise RuntimeError("no underlying mark")
                params = ib.reqSecDefOptParams(stock.symbol, "", stock.secType, stock.conId)
                if not params:
                    raise RuntimeError("no option parameters")
                standard = [p for p in params if p.tradingClass == symbol] or params
                expiries = sorted(set().union(*(set(p.expirations) for p in standard)))
                strikes = sorted(set().union(*(set(p.strikes) for p in standard)))
                trading_class = symbol if standard is not params else params[0].tradingClass
                term = _quote_term(ib, Option, symbol, trading_class, expiries, strikes, spot, today)
                chain = []
                for expiry in _chain_expiries(expiries, today):
                    chain.extend(_quote_chain(ib, Option, symbol, trading_class, expiry, strikes, spot, today))
                summary = summarize_surface(
                    symbol, today.isoformat(), spot, term, chain,
                    market_data_type=int(getattr(underlying, "marketDataType", 0) or 0), pulled_at=pulled_at,
                )
                summaries.append(summary)
                for row in chain:
                    positions.append({"date": today.isoformat(), "ticker": symbol, "spot": spot,
                                      "pulled_at": pulled_at, **row})
                log(f"{i}/{len(symbols)} {symbol}: {len(term)} term points, {len(chain)} chain contracts")
            except Exception as exc:  # noqa: BLE001
                errors[symbol] = f"{type(exc).__name__}: {exc}"
                log(f"{i}/{len(symbols)} {symbol}: {errors[symbol]}")
            ib.sleep(0.25)
    finally:
        try:
            ib.disconnect()
        except Exception:  # noqa: BLE001
            pass

    if not summaries:
        log(f"no snapshots recorded ({len(errors)} errors)")
        return 1
    surface_new = pd.DataFrame(summaries)
    surface_out = _same_day_replace(surface_prior, surface_new, ["date", "ticker"])
    surface_out = surface_out.sort_values(["ticker", "date"])
    surface_out.to_parquet(SURFACE_PATH, index=False)
    if positions:
        positioning_new = pd.DataFrame(positions)
        positioning_out = _same_day_replace(
            positioning_prior, positioning_new, ["date", "ticker", "expiry", "strike", "right"]
        )
        positioning_out = positioning_out.sort_values(["ticker", "date", "expiry", "strike", "right"])
        positioning_out.to_parquet(POSITIONING_PATH, index=False)
    log(f"saved {len(surface_new)} surfaces; {len(positions)} positioning rows; {len(errors)} errors")
    if not args.no_upload:
        cache_io.upload_from_local(SURFACE_PATH, SURFACE_HISTORY_R2_KEY)
        if positions:
            cache_io.upload_from_local(POSITIONING_PATH, POSITIONING_HISTORY_R2_KEY)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
