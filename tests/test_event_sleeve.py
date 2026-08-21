"""Guards for event_sleeve.py — prereg spec invariants + action logic.

Price data is synthetic; the macro calendar is the committed CSV. Dates
used: FOMC 2027-01-27 (non-midterm), 2026-09-16 (midterm), Sep opex
2026-09-18, Dec opex 2026-12-18.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import event_sleeve as es


def synth_px(last_date: str, rank21: float = 30.0,
             z10: float = 0.0) -> pd.DataFrame:
    idx = pd.bdate_range(end=last_date, periods=300)
    df = pd.DataFrame({"Close": np.linspace(90, 100, len(idx))}, index=idx)
    df["rank21"] = rank21
    df["z10"] = z10
    return df


def px_for(day: str, **kw) -> dict:
    prior = (pd.Timestamp(day) - es.NYSE_BDAY).normalize()
    return {t: synth_px(str(prior.date()), **kw)
            for t in ("SPY", "IWM", "SVXY")}


def run(day: str, state=None, **kw):
    state = state if state is not None else {"positions": {}}
    rows, log = es.compute_actions(pd.Timestamp(day), px_for(day, **kw), state)
    return rows, log, state


def test_prereg_config_frozen():
    s = es.EVENT_SLEEVE
    assert s["T1_FOMC_DRIFT"] == {"ticker": "SPY", "side": "LONG",
                                  "nav_frac": 0.25}
    assert s["T2_FOMC_MIDTERM_SHORT"]["rank21_max"] == 50
    assert s["T3_SEP_POSTQUAD_SHORT"]["z10_skip_below"] == -1.0
    assert s["T4_DEC_POSTOPEX_LONG"] == {"ticker": "IWM", "side": "LONG",
                                         "nav_frac": 0.25}
    assert s["V2_NOVDEC_VOL"] == {"ticker": "SVXY", "side": "LONG",
                                  "nav_frac": 0.05}
    assert s["V4_POSTOPEX_VOL"] == {"ticker": "SVXY", "side": "LONG",
                                    "nav_frac": 0.10}
    assert es.FOMC_ENTRY_TD_BEFORE == 4
    assert es.V4_EXIT_TD_AFTER == 3


def test_nyse_calendar():
    assert not es.is_session(pd.Timestamp("2026-04-03"))   # Good Friday
    assert es.is_session(pd.Timestamp("2026-10-12"))       # Columbus Day
    assert es.is_session(pd.Timestamp("2026-11-11"))       # Veterans Day
    assert not es.is_session(pd.Timestamp("2026-11-26"))   # Thanksgiving


def test_t1_entry_and_exit_nonmidterm():
    # FOMC 2027-01-27 (Wed). td-4 = Thursday 2027-01-21.
    rows, log, state = run("2027-01-21")
    assert len(rows) == 1 and rows[0]["Trade"] == "T1_FOMC_DRIFT"
    assert rows[0]["Action"] == "BUY" and rows[0]["Order_Type"] == "MOC"
    assert "T1_FOMC_DRIFT" in state["positions"]
    rows2, _ = es.compute_actions(
        pd.Timestamp("2027-01-27"), px_for("2027-01-27"), state)
    assert len(rows2) == 1 and rows2[0]["Action"] == "SELL"
    assert rows2[0]["Order_Type"] == "MOO" and rows2[0]["TIF"] == "OPG"
    assert not state["positions"]


def test_t1_idempotent_rerun():
    rows, _, state = run("2027-01-21")
    rows2, log2 = es.compute_actions(
        pd.Timestamp("2027-01-21"), px_for("2027-01-21"), state)
    assert rows2 == [] and any("idempotent" in l for l in log2)


def test_t2_midterm_weak_tape_shorts():
    # FOMC 2026-09-16 (midterm). td-4 = Thursday 2026-09-10.
    rows, _, state = run("2026-09-10", rank21=30)
    assert len(rows) == 1 and rows[0]["Trade"] == "T2_FOMC_MIDTERM_SHORT"
    assert rows[0]["Action"] == "SELL_SHORT"
    rows2, _ = es.compute_actions(
        pd.Timestamp("2026-09-16"), px_for("2026-09-16"), state)
    assert rows2[0]["Action"] == "BUY_TO_COVER"
    assert rows2[0]["Order_Type"] == "MOO"


def test_t2_overbought_excluded_and_no_t1_in_midterm():
    rows, log, state = run("2026-09-10", rank21=75)
    assert rows == []
    assert any("overbought" in l for l in log)
    assert not any(r["Trade"] == "T1_FOMC_DRIFT" for r in rows)


def test_t3_sep_opex_entry_and_washout_skip():
    rows, _, state = run("2026-09-18", z10=0.5)
    assert len(rows) == 1 and rows[0]["Trade"] == "T3_SEP_POSTQUAD_SHORT"
    assert rows[0]["Ticker"] == "IWM" and rows[0]["Action"] == "SELL_SHORT"
    rows_skip, log_skip, _ = run("2026-09-18", z10=-1.4)
    assert rows_skip == [] and any("washout" in l for l in log_skip)


def test_t3_cover_on_sep_last_session():
    _, _, state = run("2026-09-18", z10=0.5)
    rows, _ = es.compute_actions(
        pd.Timestamp("2026-09-30"), px_for("2026-09-30"), state)
    assert len(rows) == 1 and rows[0]["Action"] == "BUY_TO_COVER"
    assert rows[0]["Order_Type"] == "MOC"
    assert not state["positions"]


def test_t4_dec_window():
    rows, _, state = run("2026-12-18")
    t4 = [r for r in rows if r["Trade"] == "T4_DEC_POSTOPEX_LONG"]
    assert len(t4) == 1 and t4[0]["Action"] == "BUY"
    rows2, _ = es.compute_actions(
        pd.Timestamp("2026-12-31"), px_for("2026-12-31"), state)
    assert any(r["Trade"] == "T4_DEC_POSTOPEX_LONG"
               and r["Action"] == "SELL" for r in rows2)


def test_exit_without_position_stages_nothing():
    rows, log, _ = run("2026-09-30")
    assert all(r["Trade"] != "T3_SEP_POSTQUAD_SHORT" for r in rows)


def test_quantity_sizing():
    rows, _, _ = run("2027-01-21")
    from strategy_config import ACCOUNT_VALUE
    r = rows[0]
    assert r["Quantity"] == int(0.25 * ACCOUNT_VALUE / r["Ref_Close"])


def test_no_action_random_day():
    rows, _, _ = run("2026-08-07")
    assert rows == []


def test_missed_exit_self_heals():
    # T3 covered nothing on Sep 30 (runner down); Oct 2 run stages it LATE.
    _, _, state = run("2026-09-18", z10=0.5)
    rows, log = es.compute_actions(
        pd.Timestamp("2026-10-02"), px_for("2026-10-02"), state)
    assert len(rows) == 1 and rows[0]["Action"] == "BUY_TO_COVER"
    assert "LATE" in rows[0]["Note"]
    assert not state["positions"]


def test_v4_normal_opex_entry_and_exit():
    # August 2026 opex = Fri Aug 21; +3 sessions -> Wed Aug 26.
    rows, _, state = run("2026-08-21")
    assert len(rows) == 1 and rows[0]["Trade"] == "V4_POSTOPEX_VOL"
    assert rows[0]["Ticker"] == "SVXY" and rows[0]["Action"] == "BUY"
    assert state["positions"]["V4_POSTOPEX_VOL"]["exit_on"] == "2026-08-26"
    rows2, _ = es.compute_actions(
        pd.Timestamp("2026-08-26"), px_for("2026-08-26"), state)
    assert len(rows2) == 1 and rows2[0]["Action"] == "SELL"
    assert rows2[0]["Order_Type"] == "MOC"


def test_v4_skips_september_opex():
    rows, log, _ = run("2026-09-18", z10=0.5)
    assert all(r["Trade"] != "V4_POSTOPEX_VOL" for r in rows)
    assert any("V4: September opex" in l for l in log)


def test_v4_stands_down_while_v2_open():
    state = {"positions": {"V2_NOVDEC_VOL": {
        "shares": 100, "entry_date": "2027-11-01", "ref_close": 50.0,
        "exit_on": "2027-12-31", "exit_order_type": "MOC"}}}
    # Nov 2027 opex = Fri Nov 19.
    rows, log, state = run("2027-11-19", state=state)
    assert rows == []
    assert any("V2 already holds" in l for l in log)


def test_v2_entry_nonmidterm_and_midterm_skip():
    rows, _, state = run("2027-11-01")
    assert len(rows) == 1 and rows[0]["Trade"] == "V2_NOVDEC_VOL"
    assert state["positions"]["V2_NOVDEC_VOL"]["exit_on"] == "2027-12-31"
    rows2, log2, _ = run("2026-11-02")   # first Nov session of midterm 2026
    assert all(r["Trade"] != "V2_NOVDEC_VOL" for r in rows2)
    assert any("midterm year" in l for l in log2)


def test_dec_opex_midterm_stages_t4_and_v4():
    # 2026 is midterm: V2 idle, so Dec opex stages BOTH T4 and V4.
    rows, _, _ = run("2026-12-18")
    trades = {r["Trade"] for r in rows}
    assert trades == {"T4_DEC_POSTOPEX_LONG", "V4_POSTOPEX_VOL"}


def test_backtest_evidence_frozen():
    """The Events tab renders these numbers as-is (freeze policy: prereg
    transcriptions, never recomputed). An edit here is a new study."""
    ev = es.BACKTEST_EVIDENCE
    assert set(ev) == set(es.EVENT_SLEEVE)
    for trade, e in ev.items():
        for key in ("n", "avg_bps", "hit", "span", "cross_check",
                    "expected", "worst", "kill", "conviction"):
            assert key in e, f"{trade} missing {key}"
    assert ev["T1_FOMC_DRIFT"]["n"] == 150
    assert ev["T1_FOMC_DRIFT"]["t"] == 2.51
    assert ev["T2_FOMC_MIDTERM_SHORT"]["conviction"] == "pilot"
    assert ev["T3_SEP_POSTQUAD_SHORT"]["avg_bps"] == 185.0
    assert ev["T4_DEC_POSTOPEX_LONG"]["hit"] == 0.65
    assert ev["V2_NOVDEC_VOL"]["t"] is None   # hit-rate cell, no t in prereg
    assert ev["V4_POSTOPEX_VOL"]["n"] == 164
    assert ev["V4_POSTOPEX_VOL"]["t"] == 3.55
    assert len(es.REJECTED_STUDIES) == 7
    for r in es.REJECTED_STUDIES:
        assert r["verdict"] in {"rejected", "dropped", "excluded", "parked"}
        assert r["name"] and r["reason"]


def _px_frame(dates, opens, closes):
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    return pd.DataFrame({"Open": opens, "Close": closes}, index=idx)


def test_journal_records_from_entry_and_exit():
    rows, _, state = run("2026-08-21")   # V4 entry
    recs = es.journal_records_from(rows, state)
    assert len(recs) == 1 and recs[0]["kind"] == "entry"
    assert recs[0]["trade"] == "V4_POSTOPEX_VOL"
    assert recs[0]["exit_on"] == "2026-08-26"
    assert recs[0]["exit_order_type"] == "MOC"
    rows2, _ = es.compute_actions(
        pd.Timestamp("2026-08-26"), px_for("2026-08-26"), state)
    recs2 = es.journal_records_from(rows2, state)
    assert recs2[0]["kind"] == "exit" and recs2[0]["late"] is False


def test_journal_backfill_from_state():
    _, _, state = run("2026-08-21")
    minted = es.backfill_entry_records(state, [])
    assert len(minted) == 1 and minted[0]["trade"] == "V4_POSTOPEX_VOL"
    assert minted[0]["kind"] == "entry" and minted[0]["action"] == "BUY"
    assert minted[0]["order_type"] == "MOC"
    assert minted[0]["exit_on"] == "2026-08-26"
    # already journaled -> nothing minted again
    assert es.backfill_entry_records(state, minted) == []


def test_journal_roundtrip_nondefault_path(tmp_path):
    p = tmp_path / "j.jsonl"
    n = es.append_journal([{"kind": "entry", "trade": "X", "ticker": "SPY",
                            "action": "BUY", "qty": 1, "order_type": "MOC",
                            "date": "2026-01-02", "ref_close": 100.0}],
                          path=p)
    assert n == 1
    recs = es.load_journal(p, pull=False)
    assert len(recs) == 1
    assert recs[0]["trade"] == "X" and "written_at" in recs[0]


def test_realized_history_moc_and_moo_legs():
    px = {"SVXY": _px_frame(["2026-08-21", "2026-08-26"],
                            [60.0, 62.0], [61.0, 63.0]),
          "SPY": _px_frame(["2027-01-21", "2027-01-27"],
                           [500.0, 512.0], [505.0, 515.0])}
    records = [
        {"kind": "entry", "trade": "V4_POSTOPEX_VOL", "ticker": "SVXY",
         "action": "BUY", "qty": 100, "order_type": "MOC",
         "date": "2026-08-21"},
        {"kind": "exit", "trade": "V4_POSTOPEX_VOL", "ticker": "SVXY",
         "action": "SELL", "qty": 100, "order_type": "MOC",
         "date": "2026-08-26"},
        {"kind": "entry", "trade": "T1_FOMC_DRIFT", "ticker": "SPY",
         "action": "BUY", "qty": 10, "order_type": "MOC",
         "date": "2027-01-21"},
        {"kind": "exit", "trade": "T1_FOMC_DRIFT", "ticker": "SPY",
         "action": "SELL", "qty": 10, "order_type": "MOO",
         "date": "2027-01-27"},
    ]
    hist = es.realized_history(records, px)
    assert len(hist["closed"]) == 2 and hist["open"] == []
    v4 = next(r for r in hist["closed"] if r["trade"] == "V4_POSTOPEX_VOL")
    assert v4["entry_px"] == 61.0 and v4["exit_px"] == 63.0   # MOC = Close
    assert v4["pnl"] == pytest.approx(200.0)
    t1 = next(r for r in hist["closed"] if r["trade"] == "T1_FOMC_DRIFT")
    assert t1["exit_px"] == 512.0   # MOO exit = decision-day OPEN
    assert t1["ret_pct"] == pytest.approx((512.0 / 505.0 - 1) * 100)


def test_realized_history_short_sign_and_open_mark():
    px = {"IWM": _px_frame(["2026-09-18", "2026-09-30"],
                           [200.0, 190.0], [201.0, 191.0])}
    records = [
        {"kind": "entry", "trade": "T3_SEP_POSTQUAD_SHORT", "ticker": "IWM",
         "action": "SELL_SHORT", "qty": 50, "order_type": "MOC",
         "date": "2026-09-18"},
    ]
    hist = es.realized_history(records, px)
    assert hist["closed"] == []
    o = hist["open"][0]
    # short marked to the last close: price fell 201 -> 191 = +$500
    assert o["entry_px"] == 201.0 and o["mark_px"] == 191.0
    assert o["pnl"] == pytest.approx(500.0)
    records.append(
        {"kind": "exit", "trade": "T3_SEP_POSTQUAD_SHORT", "ticker": "IWM",
         "action": "BUY_TO_COVER", "qty": 50, "order_type": "MOC",
         "date": "2026-09-30"})
    hist2 = es.realized_history(records, px)
    c = hist2["closed"][0]
    assert c["pnl"] == pytest.approx(500.0)
    assert c["ret_pct"] == pytest.approx(-(191.0 / 201.0 - 1) * 100)


def test_realized_history_missing_bar_stays_ungraded():
    px = {"SVXY": _px_frame(["2026-08-21"], [60.0], [61.0])}
    records = [
        {"kind": "entry", "trade": "V4_POSTOPEX_VOL", "ticker": "SVXY",
         "action": "BUY", "qty": 100, "order_type": "MOC",
         "date": "2026-08-21"},
        {"kind": "exit", "trade": "V4_POSTOPEX_VOL", "ticker": "SVXY",
         "action": "SELL", "qty": 100, "order_type": "MOC",
         "date": "2026-08-26"},   # no bar for the exit date
    ]
    hist = es.realized_history(records, px)
    assert len(hist["closed"]) == 1
    assert "ret_pct" not in hist["closed"][0]


def test_state_records_exit_schedule():
    _, _, state = run("2026-09-18", z10=0.5)
    pos = state["positions"]["T3_SEP_POSTQUAD_SHORT"]
    assert pos["exit_on"] == "2026-09-30"
    assert pos["exit_order_type"] == "MOC"
    _, _, state2 = run("2026-09-10", rank21=30)
    pos2 = state2["positions"]["T2_FOMC_MIDTERM_SHORT"]
    assert pos2["exit_on"] == "2026-09-16"
    assert pos2["exit_order_type"] == "MOO"
