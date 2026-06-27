"""Regression test for seasonal_order_staging.build_seasonal_rows.

Locks the staging contract: entry-type classification (limit vs MOO), nostage
routing (equity shorts + non-tradeable signals), midterm sizing, 1% daily cap.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from seasonal_order_staging import (
    build_seasonal_rows, classify_entry, is_tradeable_stk,
    SEASONAL_RISK_BPS, SEASONAL_MIDTERM_RISK_BPS,
)

ACCT = 750_000.0


def _ticket(verb, entry, stop, stop_atr, target, tsd=21, rr=2.0):
    return f"{verb} ~{entry:.2f} | stop {stop:.2f} ({stop_atr} ATR) | target {target:.2f} | time-stop {tsd}td | R/R {rr}"


def _cand(ticker, channel, verb, entry, stop, stop_atr, target):
    return {"ticker": ticker, "channel": channel, "horizon": "21d", "conviction": "A",
            "direction": "long" if verb == "BUY" else "short",
            "evidence": {"TICKET": _ticket(verb, entry, stop, stop_atr, target)}}


def _payload(asof, cands):
    return {"meta": {"asof": asof}, "candidates": cands}


def test_classify_and_tradeable():
    assert is_tradeable_stk("V") and is_tradeable_stk("LQD")
    assert not is_tradeable_stk("SB=F")
    assert not is_tradeable_stk("^MXX")
    assert not is_tradeable_stk("BTC-USD")
    # equity channel -> limit; macro non-US-ETF -> MOO; macro US ETF -> limit
    assert classify_entry("V", "Equity Seasonal Tickets") == "REL_OPEN"
    assert classify_entry("TLT", "Macro / Cross-Asset Tickets") == "MOO"
    assert classify_entry("SPY", "Macro / Cross-Asset Tickets") == "REL_OPEN"


def test_routing_and_entry_types():
    # 2025 = non-midterm -> 20 bps. entry=100 stop=95 -> risk_ps=5, atr=5.
    cands = [
        _cand("V", "Equity Seasonal Tickets", "BUY", 100, 95, 1.0, 110),    # eq long -> Seasonal, REL_OPEN
        _cand("NVDA", "Equity Seasonal Tickets", "SELL", 100, 105, 1.0, 90),  # eq short -> nostage
        _cand("TLT", "Macro / Cross-Asset Tickets", "BUY", 100, 95, 1.0, 110),  # macro ETF -> Seasonal, MOO
        _cand("SPY", "Macro / Cross-Asset Tickets", "BUY", 100, 95, 1.0, 110),  # US-session ETF -> Seasonal, REL_OPEN
        _cand("CL=F", "Macro / Cross-Asset Tickets", "BUY", 100, 95, 1.0, 110),  # future -> nostage (deferred)
    ]
    seasonal, nostage = build_seasonal_rows(_payload("2025-06-24", cands), ACCT)

    syms = {r["Symbol"]: r for r in seasonal}
    assert set(syms) == {"V", "TLT", "SPY"}
    assert syms["V"]["Order_Type"] == "REL_OPEN" and syms["V"]["TIF"] == "DAY"
    assert syms["TLT"]["Order_Type"] == "MOO" and syms["TLT"]["TIF"] == "OPG"
    assert syms["SPY"]["Order_Type"] == "REL_OPEN"

    nos = {r["Symbol"]: r for r in nostage}
    assert set(nos) == {"NVDA", "CL=F"}
    # equity short: sized, segregated, tagged, separate scan source
    assert nos["NVDA"]["Action"] == "SELL" and nos["NVDA"]["Trade_Direction"] == "Short"
    assert nos["NVDA"]["Quantity"] > 0 and nos["NVDA"]["Scan_Source"] == "Seasonal_NoStage"
    assert "[eq-short]" in nos["NVDA"]["Strategy_Ref"]
    # non-tradeable future: not staged (Quantity 0), SecType inferred, flagged
    assert nos["CL=F"]["Quantity"] == 0 and nos["CL=F"]["SecType"] == "FUT"
    assert nos["CL=F"]["Order_Type"] == "NONE" and "[need-proxy]" in nos["CL=F"]["Strategy_Ref"]

    # 20 bps non-midterm: risk$ = 750k * 20bps = 1500; risk_ps 5 -> 300 shares
    assert abs(syms["V"]["Risk_Amt"] - 1500.0) < 1e-6
    assert syms["V"]["Quantity"] == 300
    assert syms["V"]["Risk_Bps"] == SEASONAL_RISK_BPS
    # bracket metadata backed out of the ticket
    assert abs(syms["V"]["Frozen_ATR"] - 5.0) < 1e-6
    assert abs(syms["V"]["Stop_ATR_Mult"] - 1.0) < 1e-6
    assert abs(syms["V"]["Tgt_ATR_Mult"] - 2.0) < 1e-6


def test_midterm_downsize():
    cands = [_cand("V", "Equity Seasonal Tickets", "BUY", 100, 95, 1.0, 110)]
    seasonal, _ = build_seasonal_rows(_payload("2026-06-24", cands), ACCT)  # 2026 midterm
    assert seasonal[0]["Risk_Bps"] == SEASONAL_MIDTERM_RISK_BPS
    assert abs(seasonal[0]["Risk_Amt"] - ACCT * SEASONAL_MIDTERM_RISK_BPS / 10000.0) < 1e-6


def test_daily_cap_prorata():
    # 6 equity longs @ 20 bps ($1500 each) = $9000 > 1% cap ($7500) -> scale 0.8333
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    cands = [_cand(t, "Equity Seasonal Tickets", "BUY", 100, 95, 1.0, 110) for t in tickers]
    seasonal, _ = build_seasonal_rows(_payload("2025-06-24", cands), ACCT)
    total = sum(r["Risk_Amt"] for r in seasonal)
    assert abs(total - 7500.0) < 1.0          # capped at 1% of account
    assert all(r["Risk_Bps"] < SEASONAL_RISK_BPS for r in seasonal)  # scaled down


def main():
    test_classify_and_tradeable()
    test_routing_and_entry_types()
    test_midterm_downsize()
    test_daily_cap_prorata()
    print("All seasonal_order_staging cases passed.")


if __name__ == "__main__":
    main()
