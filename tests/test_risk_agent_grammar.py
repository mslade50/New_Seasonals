"""Guard tests for risk_agent_grammar: option loss gate, sizing and decision validation.

Pure logic. No network. checks_dir lives under tmp_path.
"""
import pytest

import risk_agent_grammar as rg

ASOF = "2026-10-09"
EXPIRY = "2026-11-20"
NAV = 200_000.0
UP = {"5": 0.06, "10": 0.09, "21": 0.13, "42": 0.18, "63": 0.22, "126": 0.30}


def leg(right, strike, qty, bid, ask, expiry=EXPIRY):
    return {"right": right, "strike": strike, "qty": qty, "bid": bid,
            "ask": ask, "expiry": expiry}


def _gate(legs, qty=1, **kw):
    args = dict(spot=100.0, asof=ASOF, up_q999=UP, per_contract_cost=0.0)
    args.update(kw)
    return rg.option_structure_gate(legs, qty, 200_000.0, **args)


# ===========================================================================
# Option structure loss gate
# ===========================================================================

def test_long_straddle_pass_terminal():
    g = _gate([leg("C", 100, 1, 5, 5), leg("P", 100, 1, 5, 5)])
    assert g["status"] == "PASS_TERMINAL"
    assert g["loss_bound"] == "FINITE"
    assert g["max_loss"] == pytest.approx(1000.0)


# The spec for this case says "loss 300", but long 100P at ask 4 and short 95P
# at bid 2 is a 200 net debit, and a put spread's terminal minimum is 0, so the
# max loss is 200 per unit. The grammar's arithmetic is right here; the spec's
# number is not. Kept as 200 below, with a 300 boundary case using ask 5.
def test_put_spread_loss_is_net_debit_200_per_unit():
    g = _gate([leg("P", 100, 1, 3.9, 4), leg("P", 95, -1, 2, 2.1)])
    assert g["status"] == "PASS_TERMINAL"
    assert g["max_loss"] == pytest.approx(200.0)
    assert g["max_structure_qty"] == 50


def test_put_spread_300_per_unit_cap_boundary():
    legs = [leg("P", 100, 1, 4.9, 5), leg("P", 95, -1, 2, 2.1)]
    g33 = _gate(legs, qty=33)
    assert g33["status"] == "PASS_TERMINAL"
    assert g33["max_loss"] == pytest.approx(9900.0)
    assert g33["max_structure_qty"] == 33
    g34 = _gate(legs, qty=34)
    assert g34["status"] == "REJECT"
    assert g34["reasons"] == ["LOSS_EXCEEDS_5_PERCENT"]
    assert g34["max_loss"] == pytest.approx(10200.0)


def test_iron_condor_loss_300_before_dividend_reserve():
    legs = [leg("P", 90, 1, 0.9, 1.0), leg("P", 95, -1, 2.0, 2.1),
            leg("C", 105, -1, 2.0, 2.1), leg("C", 110, 1, 0.9, 1.0)]
    # spot=None disables the American dividend reserve; the spec's 300 is pre-reserve.
    g = _gate(legs, spot=None)
    assert g["status"] == "PASS_TERMINAL"
    assert g["max_loss"] == pytest.approx(300.0)


def test_iron_condor_with_spot_adds_dividend_reserve_on_short_call():
    legs = [leg("P", 90, 1, 0.9, 1.0), leg("P", 95, -1, 2.0, 2.1),
            leg("C", 105, -1, 2.0, 2.1), leg("C", 110, 1, 0.9, 1.0)]
    g = _gate(legs)
    reserve = float(g["dividend_reserve_per_unit"])
    assert reserve > 0
    assert g["status"] == "PASS_TERMINAL"
    assert g["max_loss"] == pytest.approx(300.0 + reserve)


def test_naked_short_put_100_loss_9700_pass():
    g = _gate([leg("P", 100, -1, 3.0, 3.2)])
    assert g["status"] == "PASS_TERMINAL"
    assert g["max_loss"] == pytest.approx(9700.0)


def test_naked_short_put_110_loss_10700_reject():
    g = _gate([leg("P", 110, -1, 3.0, 3.2)])
    assert g["status"] == "REJECT"
    assert g["max_loss"] == pytest.approx(10700.0)
    assert g["reasons"] == ["LOSS_EXCEEDS_5_PERCENT"]


def test_short_straddle_is_stress_budgeted():
    legs = [leg("C", 100, -1, 5, 5.2), leg("P", 100, -1, 5, 5.2)]
    g = _gate(legs)
    assert g["status"] == "PASS_STRESS"
    assert g["loss_bound"] == "STRESS"
    # 42 calendar days -> 29 TD -> horizon 42 -> 0.18 x 3
    assert g["stress_move"] == pytest.approx(0.54)


def test_short_straddle_without_stress_inputs_rejected():
    legs = [leg("C", 100, -1, 5, 5.2), leg("P", 100, -1, 5, 5.2)]
    g = _gate(legs, up_q999=None)
    assert g["status"] == "REJECT"
    assert g["reasons"] == ["UNBOUNDED_NEEDS_STRESS_INPUTS"]
    assert g["loss_bound"] == "UNBOUNDED"


def test_stress_floor_applies_when_history_is_small():
    legs = [leg("C", 100, -1, 5, 5.2), leg("P", 100, -1, 5, 5.2)]
    low = {h: 0.01 for h in (5, 10, 21, 42, 63, 126)}
    g = _gate(legs, up_q999=low)
    assert g["stress_move"] == pytest.approx(0.25)


def test_calendar_is_unknown():
    legs = [leg("C", 100, 1, 5, 5), leg("C", 100, -1, 4, 4.1, expiry="2026-12-18")]
    g = _gate(legs)
    assert g["status"] == "UNKNOWN"
    assert g["reasons"] == ["MIXED_EXPIRY_OR_MULTIPLIER"]


def test_crossed_quote_is_invalid():
    g = _gate([leg("P", 100, 1, 6.0, 5.0)])
    assert g["status"] == "INVALID"


def test_zero_leg_qty_is_invalid():
    g = _gate([leg("P", 100, 0, 5, 5)])
    assert g["status"] == "INVALID"


def test_zero_structure_qty_is_invalid():
    g = _gate([leg("P", 100, 1, 5, 5)], qty=0)
    assert g["status"] == "INVALID"


def test_dividend_reserve_on_short_call_in_call_spread():
    legs = [leg("C", 100, -1, 5, 5.2), leg("C", 110, 1, 1.9, 2.0)]
    g = _gate(legs)
    assert float(g["dividend_reserve_per_unit"]) > 0
    assert g["status"] == "PASS_TERMINAL"


def test_no_dividend_reserve_for_put_spread():
    legs = [leg("P", 100, 1, 3.9, 4), leg("P", 95, -1, 2, 2.1)]
    g = _gate(legs)
    assert float(g["dividend_reserve_per_unit"]) == 0


def test_reference_capital():
    assert rg.reference_capital(250_000) == 200_000
    assert rg.reference_capital(150_000) == 150_000
    assert rg.reference_capital(None) == 200_000


def test_default_stop_atr():
    assert rg.default_stop_atr(10, False) == 3.0
    assert rg.default_stop_atr(5, True) == 1.0
    assert rg.default_stop_atr(10, True) == 1.3
    assert rg.default_stop_atr(20, True) == 1.6


# ===========================================================================
# Decision validation
# ===========================================================================

THESIS = ("Energy has lagged crude for three weeks while inventories drew; "
          "a long XLE rides the catch-up above the September base.")


def _ctx(tmp_path, **over):
    (tmp_path / "00_surface_map.md").write_text("# surface map\n", encoding="utf-8")
    (tmp_path / "chk.py").write_text("print('check')\n", encoding="utf-8")
    ctx = {
        "asof": ASOF,
        "nav": NAV,
        "positions": {},
        "quotes": {"XLE": {"close": 90.0, "atr": 2.0},
                   "ES=F": {"close": 6000.0, "atr": 60.0}},
        "chains": {"SPY": {"spot": 600.0, "asof": ASOF, "quotes": {
            "2026-11-20|600|P": {"bid": 9.0, "ask": 9.2, "con_id": 1},
            "2026-11-20|580|P": {"bid": 5.0, "ask": 5.1, "con_id": 2},
        }}},
        "stress": {"SPY": {"21": 0.1, "42": 0.15}},
        "checks_dir": tmp_path,
    }
    ctx.update(over)
    return ctx


def _open_etf(tmp_path, pid="RA-2026-10-09-1", symbol="XLE", **over):
    pos = {
        "id": pid,
        "action": "open",
        "instrument": {"type": "etf", "symbol": symbol},
        "side": "long",
        "risk_bps": 50,
        "entry": {"type": "MOO"},
        "exit": {"time_td": 10, "stop": 86.0},
        "thesis": THESIS,
        "what_kills_it": "A close below 86 or crude breaking its 50-day average.",
        "survived": "Held through the August drawdown without a stop.",
        "evidence": {"summary": "Pre-registered catch-up study over 40 episodes.",
                     "n": 40, "script": str(tmp_path / "chk.py")},
        "forecast": {"horizon_td": 10, "expected_return_pct": 2.0, "p_win": 0.55},
    }
    pos.update(over)
    return pos


def _decision(positions, **over):
    payload = {
        "schema_version": "risk_agent.v2",
        "asof": ASOF,
        "mode": "decision",
        "posture": {"summary": "Cash-heavy with a small energy long",
                    "net_beta": 0.3, "cash_pct": 60},
        "forecasts": [
            {"horizon_td": 5, "p_up": 0.55, "q10_pct": -2.0, "q90_pct": 3.0},
            {"horizon_td": 21, "p_up": 0.52, "q10_pct": -5.0, "q90_pct": 6.0},
        ],
        "considered_and_rejected": [{"idea": "Short SPY vol",
                                     "reason": "no edge after costs"}],
        "positions": positions,
    }
    payload.update(over)
    return payload


def _spy_put_spread(tmp_path, long_strike=600, short_strike=580, qty=2, risk_bps=120):
    legs = [
        {"right": "P", "strike": long_strike, "expiry": EXPIRY, "qty": 1},
        {"right": "P", "strike": short_strike, "expiry": EXPIRY, "qty": -1},
    ]
    return {
        "id": "RA-2026-10-09-1",
        "action": "open",
        "side": "long",
        "instrument": {"type": "option_structure", "underlying": "SPY",
                       "structure_qty": qty, "legs": legs},
        "risk_bps": risk_bps,
        "entry": {"type": "CHAIN"},
        "exit": {"time_td": 10},
        "thesis": THESIS,
        "what_kills_it": "SPY closing through 580 before expiry.",
        "survived": "Survived the July dip without a roll.",
        "evidence": {"summary": "Put spread payoff study on SPY 21d windows.",
                     "n": 30, "script": str(tmp_path / "chk.py")},
        "forecast": {"horizon_td": 10, "expected_return_pct": 1.0, "p_win": 0.6},
    }


def _errors(result):
    return result["errors"]


def test_valid_etf_open_sizes_to_risk_budget(tmp_path):
    res = rg.validate_decision(_decision([_open_etf(tmp_path)]), _ctx(tmp_path))
    assert res["errors"] == []
    assert len(res["orders"]) == 1
    order = res["orders"][0]
    assert order["kind"] == "etf"
    # floor(50 bps x 200k / 1e4 / (4.0 x 1)) = 250
    assert order["qty"] == 250


def test_mes_future_sizes_to_risk_budget(tmp_path):
    pos = _open_etf(tmp_path, symbol=None, side="long", exit={"time_td": 10, "stop": 5900.0})
    pos["instrument"] = {"type": "future", "root": "MES", "contract_month": "2026-12"}
    res = rg.validate_decision(_decision([pos]), _ctx(tmp_path))
    assert res["errors"] == []
    order = res["orders"][0]
    assert order["kind"] == "future"
    assert order["series"] == "ES=F"
    # stop distance 100, multiplier 5: floor(50 x 20 / (100 x 5)) = 2
    assert order["qty"] == 2


def test_missing_surface_map_is_error(tmp_path):
    ctx = _ctx(tmp_path)
    (tmp_path / "00_surface_map.md").unlink()
    res = rg.validate_decision(_decision([_open_etf(tmp_path)]), ctx)
    assert any("00_surface_map.md" in e for e in _errors(res))


def test_evidence_script_outside_checks_dir_is_error(tmp_path, tmp_path_factory):
    outside_dir = tmp_path_factory.mktemp("outside")
    outside = outside_dir / "chk.py"
    outside.write_text("print('elsewhere')\n", encoding="utf-8")
    pos = _open_etf(tmp_path, evidence={"summary": "Pre-registered catch-up study over 40 episodes.",
                                        "n": 40, "script": str(outside)})
    res = rg.validate_decision(_decision([pos]), _ctx(tmp_path))
    assert any("evidence.script" in e for e in _errors(res))


def test_stop_on_wrong_side_is_error(tmp_path):
    pos = _open_etf(tmp_path, exit={"time_td": 10, "stop": 91.0})
    res = rg.validate_decision(_decision([pos]), _ctx(tmp_path))
    assert any("wrong side" in e for e in _errors(res))


def test_stop_inside_half_atr_is_error(tmp_path):
    pos = _open_etf(tmp_path, exit={"time_td": 10, "stop": 89.5})  # 0.5 away, ATR 2
    res = rg.validate_decision(_decision([pos]), _ctx(tmp_path))
    assert any("0.5 ATR" in e for e in _errors(res))


def test_etf_risk_above_200_bps_is_error(tmp_path):
    pos = _open_etf(tmp_path, risk_bps=250)
    res = rg.validate_decision(_decision([pos]), _ctx(tmp_path))
    assert any("risk_bps" in e for e in _errors(res))


def test_unknown_etf_symbol_is_error(tmp_path):
    pos = _open_etf(tmp_path, symbol="ZZZZ")
    res = rg.validate_decision(_decision([pos]), _ctx(tmp_path))
    assert any("not a tradeable etf" in e for e in _errors(res))


def test_held_position_without_verdict_is_error(tmp_path):
    held = {"RA-2026-10-08-1": {"kind": "etf", "symbol": "SPY", "side": "long",
                                "qty": 10, "risk_bps": 40, "notional": 6000,
                                "multiplier": 1.0}}
    res = rg.validate_decision(_decision([]), _ctx(tmp_path, positions=held))
    assert any("RA-2026-10-08-1" in e for e in _errors(res))


def test_close_verdict_on_held_position_produces_close_order(tmp_path):
    held = {"RA-2026-10-08-1": {"kind": "etf", "symbol": "SPY", "side": "long",
                                "qty": 10, "risk_bps": 40, "notional": 6000,
                                "multiplier": 1.0}}
    pos = {"id": "RA-2026-10-08-1", "action": "close",
           "reason": "target reached, take the gain and free the risk budget",
           "entry": {"type": "MOO"}}
    res = rg.validate_decision(_decision([pos]), _ctx(tmp_path, positions=held))
    assert res["errors"] == []
    assert len(res["orders"]) == 1
    assert res["orders"][0]["type"] == "close"
    assert res["orders"][0]["position_id"] == "RA-2026-10-08-1"


def test_open_during_stand_down_is_error(tmp_path):
    payload = _decision([_open_etf(tmp_path)], mode="stand_down",
                        reason="Data hold: the shared export is a day stale.")
    res = rg.validate_decision(payload, _ctx(tmp_path))
    assert any("stand-down" in e for e in _errors(res))


def test_missing_21d_forecast_is_error(tmp_path):
    payload = _decision([_open_etf(tmp_path)])
    payload["forecasts"] = [f for f in payload["forecasts"] if f["horizon_td"] != 21]
    res = rg.validate_decision(payload, _ctx(tmp_path))
    assert any("21" in e and "forecasts missing" in e for e in _errors(res))


def test_em_dash_in_thesis_is_error(tmp_path):
    pos = _open_etf(tmp_path, thesis=THESIS + " " + rg.EM_DASH + " the base holds.")
    res = rg.validate_decision(_decision([pos]), _ctx(tmp_path))
    assert any("em dash" in e for e in _errors(res))


def test_valid_spy_put_spread_passes_gate(tmp_path):
    res = rg.validate_decision(_decision([_spy_put_spread(tmp_path)]), _ctx(tmp_path))
    assert res["errors"] == []
    assert len(res["orders"]) == 1
    order = res["orders"][0]
    assert order["kind"] == "option"
    assert order["gate"]["status"] == "PASS_TERMINAL"


def test_option_leg_not_in_chain_is_error(tmp_path):
    pos = _spy_put_spread(tmp_path, long_strike=590)
    res = rg.validate_decision(_decision([pos]), _ctx(tmp_path))
    assert any("no executable quote" in e for e in _errors(res))


def test_book_risk_cap_is_error(tmp_path):
    held = {}
    holds = []
    for i, sym in enumerate(["SPY", "QQQ", "IWM", "DIA"], 1):
        pid = f"RA-2026-10-08-{i}"
        held[pid] = {"kind": "etf", "symbol": sym, "side": "long", "qty": 10,
                     "risk_bps": 370, "notional": 1000, "multiplier": 1.0}
        holds.append({"id": pid, "action": "hold",
                      "reason": "stop still well below the close and the thesis is intact"})
    res = rg.validate_decision(_decision(holds + [_open_etf(tmp_path)]),
                               _ctx(tmp_path, positions=held))
    assert any("book risk" in e for e in _errors(res))


def test_six_opens_breach_per_day_cap(tmp_path):
    symbols = ["XLE", "XLK", "XLF", "XLI", "XLV", "XLU"]
    quotes = {s: {"close": 90.0, "atr": 2.0} for s in symbols}
    positions = [
        _open_etf(tmp_path, pid=f"RA-2026-10-09-{i}", symbol=s, risk_bps=10)
        for i, s in enumerate(symbols, 1)
    ]
    ctx = _ctx(tmp_path)
    ctx["quotes"].update(quotes)
    res = rg.validate_decision(_decision(positions), ctx)
    assert any("per day" in e for e in _errors(res))
