"""Guard for the Risk Agent paper ledger and grader. Synthetic bars/chains, no network."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import grade_risk_agent as G  # noqa: E402
import risk_agent_ledger as L  # noqa: E402

DAYS = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2026-03-02", periods=14)]
CAP = 200_000.0


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import cache_io
    monkeypatch.setattr(cache_io, "is_configured", lambda: False)


def bar(o, h=None, l=None, c=None):
    c = o if c is None else c
    return (o, max(o, c) if h is None else h, min(o, c) if l is None else l, c)


def prices(tmp_path, series: dict) -> Path:
    """series: ticker -> {day_index: (o,h,l,c)}; missing days repeat the last bar flat."""
    rows = []
    for t, over in series.items():
        last = over[min(over)]
        for i, d in enumerate(DAYS):
            last = over.get(i, bar(last[3]))
            rows.append((t, pd.Timestamp(d), *last, 1e6))
    p = tmp_path / "prices.parquet"
    pd.DataFrame(rows, columns=["ticker", "date", "Open", "High", "Low", "Close", "Volume"]).to_parquet(p)
    return p


def etf_order(pid="RA-2026-03-02-1", sym="XYZ", side="long", qty=100, entry=None,
              exit_=None, asof=DAYS[0], risk_bps=20.0, kind="etf", series=None, mult=1.0):
    o = {"type": "open", "position_id": pid, "kind": kind, "symbol": sym,
         "series": series or sym, "side": side, "qty": qty, "multiplier": mult,
         "ref_price": 100.0, "stop_distance": 2.0, "risk_bps": risk_bps,
         "notional": qty * 100.0 * mult, "entry": entry or {"type": "MOO"},
         "exit": exit_ or {"time_td": 3, "stop": None, "target": None}}
    return L.orders_to_records([o], asof, "d1")


def seed(tmp_path, recs):
    j = tmp_path / "j.jsonl"
    L.append(recs, j)
    return j


def run(tmp_path, journal, price_series, asof=DAYS[-1], chains=None):
    p = prices(tmp_path, price_series)
    args = ["--journal", str(journal), "--prices", str(p), "--asof", asof, "--no-push",
            "--chains", str(chains or tmp_path / "none.parquet")]
    rc = G.main(args, book_out=tmp_path / "book.json", scoreboard_out=tmp_path / "sb.json")
    assert rc == 0
    return L.replay(L.load(journal)), json.loads((tmp_path / "sb.json").read_text())


def kinds(journal):
    return [r["kind"] for r in L.load(journal)]


def identity(book):
    open_costs = sum(p["entry_cost"] for p in book["positions"].values())
    unreal = sum(p["unrealized_pnl"] for p in book["positions"].values())
    assert book["nav"] == pytest.approx(book["capital"] + book["realized_pnl"] + unreal - open_costs)


def flat(px, n=1):
    return {i: bar(px) for i in range(n)}


def test_empty_journal_scoreboard(tmp_path):
    j = tmp_path / "j.jsonl"
    book, sb = run(tmp_path, j, {"SPY": flat(500.0)})
    assert book["nav"] == CAP and sb["nav"] == CAP
    assert sb["total_return_pct"] is None and sb["sharpe"] is None
    assert sb["forecasts"]["5"]["n"] == 0 and sb["forecasts"]["5"]["brier"] is None
    assert not j.exists() or L.load(j) == []


def test_moo_fill_time_exit_and_idempotent(tmp_path):
    j = seed(tmp_path, etf_order())
    ser = {"SPY": flat(500.0), "XYZ": {0: bar(100), 1: bar(101, c=102), 4: bar(103, c=104)}}
    book, _ = run(tmp_path, j, ser)
    fill = next(r for r in L.load(j) if r["kind"] == "fill")
    assert fill["date"] == DAYS[1] and fill["price"] == 101
    ex = next(r for r in L.load(j) if r["kind"] == "exit")
    assert ex["exit_kind"] == "time" and ex["date"] == DAYS[4] and ex["price"] == 104
    # (104-101)*100 - 1bp entry (1.01) - 1bp exit (1.04)
    assert book["realized_pnl"] == pytest.approx(300 - 1.01 - 1.04)
    assert not book["positions"] and not book["pending"]
    identity(book)
    before = (tmp_path / "j.jsonl").read_bytes()
    run(tmp_path, j, ser)
    assert (tmp_path / "j.jsonl").read_bytes() == before        # append-only, nothing new


def test_marks_one_per_session_and_open_position_identity(tmp_path):
    j = seed(tmp_path, etf_order(exit_={"time_td": 50, "stop": None, "target": None}))
    book, _ = run(tmp_path, j, {"SPY": flat(500.0), "XYZ": {0: bar(100), 3: bar(110)}})
    recs = L.load(j)
    mark_days = [r["date"] for r in recs if r["kind"] == "mark"]
    assert mark_days == DAYS[1:] and len(set(mark_days)) == len(mark_days)
    pos = book["positions"]["RA-2026-03-02-1"]
    assert pos["days_held"] == len(DAYS) - 2 and pos["mark"] == 110
    identity(book)


def test_limit_fill_and_expire(tmp_path):
    lim = {"type": "LIMIT", "limit": 98.0, "fill_window_td": 2}
    recs = etf_order(pid="RA-2026-03-02-1", entry=lim, exit_={"time_td": 50}) + \
        etf_order(pid="RA-2026-03-02-2", sym="ABC", entry=lim, exit_={"time_td": 50})
    j = seed(tmp_path, recs)
    ser = {"SPY": flat(500.0),
           "XYZ": {0: bar(100), 2: bar(99, l=97.0, c=99)},          # touches on session 2 -> fills at 98
           "ABC": {0: bar(100)}}                                     # never touched -> expires
    book, _ = run(tmp_path, j, ser)
    fills = [r for r in L.load(j) if r["kind"] == "fill"]
    assert [(f["position_id"], f["price"], f["date"]) for f in fills] == [("RA-2026-03-02-1", 98.0, DAYS[2])]
    exp = [r for r in L.load(j) if r["kind"] == "expire"]
    assert len(exp) == 1 and exp[0]["position_id"] == "RA-2026-03-02-2" and exp[0]["date"] == DAYS[2]
    assert not book["pending"]


def test_limit_gap_below_buy_fills_at_open(tmp_path):
    j = seed(tmp_path, etf_order(entry={"type": "LIMIT", "limit": 98.0, "fill_window_td": 1},
                                 exit_={"time_td": 50}))
    run(tmp_path, j, {"SPY": flat(500.0), "XYZ": {0: bar(100), 1: bar(95, h=96, l=94, c=95)}})
    assert next(r for r in L.load(j) if r["kind"] == "fill")["price"] == 95


def test_stop_and_target_same_bar_books_stop(tmp_path):
    j = seed(tmp_path, etf_order(exit_={"time_td": 50, "stop": 95.0, "target": 110.0}))
    ser = {"SPY": flat(500.0),
           "XYZ": {0: bar(100), 1: bar(100), 2: bar(100, h=112, l=94, c=100)}}
    book, _ = run(tmp_path, j, ser)
    ex = next(r for r in L.load(j) if r["kind"] == "exit")
    assert ex["exit_kind"] == "stop" and ex["date"] == DAYS[2]
    assert ex["price"] == pytest.approx(95.0 * (1 - 3e-4))      # worse of stop/open, plus 3 bps
    assert book["closed"][0]["pnl"] < 0


def test_stop_not_armed_on_fill_day(tmp_path):
    j = seed(tmp_path, etf_order(exit_={"time_td": 50, "stop": 95.0}))
    run(tmp_path, j, {"SPY": flat(500.0), "XYZ": {0: bar(100), 1: bar(100, h=100, l=90, c=100)}})
    assert "exit" not in kinds(j)


def test_short_etf_pnl_sign_and_cash(tmp_path):
    j = seed(tmp_path, etf_order(side="short", exit_={"time_td": 2}))
    ser = {"SPY": flat(500.0), "XYZ": {0: bar(100), 1: bar(100), 3: bar(90)}}
    book, _ = run(tmp_path, j, ser)
    assert book["closed"][0]["pnl"] == pytest.approx(1000 - 1.0 - 0.9)   # price fell: short wins
    identity(book)
    # mid-trade: short credit sits in cash, liability in market value
    j2 = tmp_path / "j2.jsonl"
    L.append(etf_order(side="short", exit_={"time_td": 50}), j2)
    mid = L.replay(L.load(j2))
    assert mid["nav"] == CAP
    G.main(["--journal", str(j2), "--prices", str(prices(tmp_path, ser)), "--asof", DAYS[1],
            "--no-push", "--chains", str(tmp_path / "none.parquet")],
           book_out=tmp_path / "b2.json", scoreboard_out=tmp_path / "s2.json")
    b = L.replay(L.load(j2))
    assert b["cash"] == pytest.approx(CAP + 100 * 100 - 1.0)
    assert b["nav"] == pytest.approx(CAP - 1.0)


def test_future_multiplier_mes(tmp_path):
    o = {"type": "open", "position_id": "RA-2026-03-02-3", "kind": "future", "symbol": "MES",
         "series": "ES=F", "side": "long", "qty": 2, "multiplier": 5.0, "ref_price": 5000.0,
         "stop_distance": 50.0, "risk_bps": 10.0, "notional": 50000.0,
         "entry": {"type": "MOO"}, "exit": {"time_td": 2}, "contract_month": "2026-06"}
    j = seed(tmp_path, L.orders_to_records([o], DAYS[0], "d1"))
    ser = {"SPY": flat(500.0), "ES=F": {0: bar(5000), 1: bar(5000), 3: bar(5100)}}
    book, _ = run(tmp_path, j, ser)
    # 100 pts x $5 x 2 contracts = 1000, less $2.50 x 2 per side
    assert book["closed"][0]["pnl"] == pytest.approx(1000 - 5.0 - 5.0)
    assert book["cash"] == pytest.approx(CAP + 1000 - 10.0)
    identity(book)


def test_close_order_fills_next_open(tmp_path):
    recs = etf_order(exit_={"time_td": 50})
    close = L.orders_to_records([{"type": "close", "position_id": "RA-2026-03-02-1",
                                  "entry": {"type": "MOO"}}], DAYS[2], "d2")
    j = seed(tmp_path, recs)
    ser = {"SPY": flat(500.0), "XYZ": {0: bar(100), 3: bar(120, c=121)}}
    # first run through day 2 only, then the close arrives, then a second run
    run(tmp_path, j, ser, asof=DAYS[2])
    L.append(close, j)
    book, _ = run(tmp_path, j, ser)
    ex = next(r for r in L.load(j) if r["kind"] == "exit")
    assert ex["exit_kind"] == "close" and ex["date"] == DAYS[3] and ex["price"] == 120
    assert not book["positions"] and not book["pending"]


def chain_frame(rows):
    return pd.DataFrame(rows, columns=["date", "ticker", "spot", "pulled_at", "expiry", "dte", "strike",
                                       "right", "con_id", "bid", "ask", "mid"])


def test_option_spread_fill_and_expiry_settlement(tmp_path):
    expiry = DAYS[5]
    legs = [{"right": "C", "strike": 100.0, "expiry": expiry, "qty": 1, "con_id": 1},
            {"right": "C", "strike": 110.0, "expiry": expiry, "qty": -1, "con_id": 2}]
    o = {"type": "open", "position_id": "RA-2026-03-02-4", "kind": "option", "underlying": "SPY",
         "legs": legs, "structure_qty": 2, "risk_bps": 50.0, "notional": 1000.0,
         "entry": {"type": "CHAIN"}, "exit": {"time_td": 60}}
    j = seed(tmp_path, L.orders_to_records([o], DAYS[0], "d1"))
    exp = expiry.replace("-", "")
    rows = []
    for d in DAYS[:6]:
        rows += [[d, "SPY", 105.0, 1.0, exp, 3, 100.0, "C", 1, 4.8, 5.0, 4.9],
                 [d, "SPY", 105.0, 1.0, exp, 3, 110.0, "C", 2, 1.0, 1.2, 1.1]]
    cp = tmp_path / "chains.parquet"
    chain_frame(rows).to_parquet(cp)
    ser = {"SPY": {0: bar(105), 5: bar(105, c=108)}}                  # expiry close 108
    book, _ = run(tmp_path, j, ser, chains=cp)
    fill = next(r for r in L.load(j) if r["kind"] == "fill")
    assert fill["date"] == DAYS[1] and fill["leg_prices"] == [5.0, 1.0]   # long at ask, short at bid
    assert fill["costs"] == pytest.approx(0.65 * 4)
    assert fill["quote_sources"] == ["exact", "exact"]
    ex = next(r for r in L.load(j) if r["kind"] == "exit")
    assert ex["exit_kind"] == "expiry" and ex["leg_prices"] == [8.0, 0.0]  # intrinsic on raw close
    # debit (5-1)*100 = 400/unit x2; settle (8-0)*100 = 800/unit x2
    assert book["realized_pnl"] == pytest.approx((800 - 400) * 2 - 0.65 * 4)
    identity(book)


def test_option_missing_quote_expires_after_three_sessions(tmp_path):
    legs = [{"right": "C", "strike": 100.0, "expiry": DAYS[9], "qty": 1}]
    o = {"type": "open", "position_id": "RA-2026-03-02-5", "kind": "option", "underlying": "SPY",
         "legs": legs, "structure_qty": 1, "risk_bps": 50.0, "notional": 500.0,
         "entry": {"type": "CHAIN"}, "exit": {"time_td": 60}}
    j = seed(tmp_path, L.orders_to_records([o], DAYS[0], "d1"))
    book, _ = run(tmp_path, j, {"SPY": flat(105.0)})
    exp = [r for r in L.load(j) if r["kind"] == "expire"]
    assert len(exp) == 1 and exp[0]["date"] == DAYS[3] and exp[0]["reason"] == "no_quote"
    assert not book["pending"]


def test_orders_to_records_and_validator_positions(tmp_path):
    recs = etf_order()
    assert recs[0]["kind"] == "order" and recs[0]["status"] == "pending"
    assert recs[0]["order_id"] == "RA-2026-03-02-1-open" and recs[0]["instrument_kind"] == "etf"
    j = seed(tmp_path, recs)
    book, _ = run(tmp_path, j, {"SPY": flat(500.0), "XYZ": {0: bar(100)}}, asof=DAYS[2])
    vp = L.validator_positions(book)["RA-2026-03-02-1"]
    assert vp["kind"] == "etf" and vp["symbol"] == "XYZ" and vp["qty"] == 100
    assert vp["risk_bps"] == 20.0 and vp["notional"] == pytest.approx(10000.0)


def test_brier_of_matured_forecast(tmp_path):
    dec = {"kind": "decision", "asof": DAYS[0], "decision_id": "d1", "model": "m1",
           "forecasts": [{"horizon_td": 5, "p_up": 0.8, "q10_pct": -3.0, "q90_pct": 3.0},
                         {"horizon_td": 21, "p_up": 0.6, "q10_pct": -5.0, "q90_pct": 5.0}]}
    j = seed(tmp_path, [dec])
    ser = {"SPY": {0: bar(500), 5: bar(510)}}                         # +2% at 5 TD; 21 TD not matured
    _, sb = run(tmp_path, j, ser)
    f5 = sb["forecasts"]["5"]
    assert f5["n"] == 1 and f5["brier"] == pytest.approx(0.04)
    assert f5["inside_q10_q90"] == 1.0
    assert sb["forecasts"]["21"]["n"] == 0 and sb["forecasts"]["21"]["brier"] is None
    assert sb["by_model"]["m1"]["forecasts"]["5"]["brier"] == pytest.approx(0.04)
    assert sb["n_marks"] == len(DAYS) - 1 and sb["nav"] == CAP
    assert sb["spy_return_pct"] == pytest.approx(2.0)


def _bs_chain(day_rows_strikes, expiry, spot=105.0, iv=0.2, days=(1,)):
    """Calls quoted at the given strikes on each DAYS[i], priced by BS with a 2% half-spread."""
    exp = expiry.replace("-", "")
    rows = []
    for i in days:
        t = (pd.Timestamp(expiry) - pd.Timestamp(DAYS[i])).days / 365
        for k in day_rows_strikes:
            m = G.bs_price(spot, k, t, 0.04, iv, "C")
            rows.append([DAYS[i], "SPY", spot, 1.0, exp, 10, k, "C", int(k), m * 0.98, m * 1.02, m, iv])
    return pd.DataFrame(rows, columns=["date", "ticker", "spot", "pulled_at", "expiry", "dte",
                                       "strike", "right", "con_id", "bid", "ask", "mid", "iv"])


def _single_leg_order(strike, expiry):
    legs = [{"right": "C", "strike": strike, "expiry": expiry, "qty": 1}]
    o = {"type": "open", "position_id": "RA-2026-03-02-6", "kind": "option", "underlying": "SPY",
         "legs": legs, "structure_qty": 1, "risk_bps": 50.0, "notional": 500.0,
         "entry": {"type": "CHAIN"}, "exit": {"time_td": 60}}
    return L.orders_to_records([o], DAYS[0], "d1")


def test_option_strike_between_quotes_is_interpolated(tmp_path):
    expiry = DAYS[12]
    cp = tmp_path / "c.parquet"
    _bs_chain([100.0, 110.0], expiry, days=range(1, 14)).to_parquet(cp)
    j = seed(tmp_path, _single_leg_order(105.0, expiry))
    book, _ = run(tmp_path, j, {"SPY": flat(105.0)}, asof=DAYS[3], chains=cp)
    fill = next(r for r in L.load(j) if r["kind"] == "fill")
    assert fill["date"] == DAYS[1] and fill["quote_sources"] == ["interpolated"]
    t = (pd.Timestamp(expiry) - pd.Timestamp(DAYS[1])).days / 365
    ask100 = G.bs_price(105.0, 100.0, t, 0.04, 0.2, "C") * 1.02
    ask110 = G.bs_price(105.0, 110.0, t, 0.04, 0.2, "C") * 1.02
    assert ask110 < fill["leg_prices"][0] < ask100
    # flat 20% vol: the interpolated ask is BS(105) plus the 2% half-spread
    assert fill["leg_prices"][0] == pytest.approx(G.bs_price(105.0, 105.0, t, 0.04, 0.2, "C") * 1.02, rel=1e-6)
    marks = [r for r in L.load(j) if r["kind"] == "mark" and r["date"] == DAYS[3]]
    m = marks[0]["positions"]["RA-2026-03-02-6"]
    assert m["quote_sources"] == ["interpolated"] and m["stale_mark"] is False
    identity(book)


def test_option_strike_outside_quoted_range_expires(tmp_path):
    expiry = DAYS[12]
    cp = tmp_path / "c.parquet"
    _bs_chain([100.0, 110.0], expiry, days=range(1, 14)).to_parquet(cp)
    j = seed(tmp_path, _single_leg_order(120.0, expiry))
    book, _ = run(tmp_path, j, {"SPY": flat(105.0)}, chains=cp)
    exp = [r for r in L.load(j) if r["kind"] == "expire"]
    assert len(exp) == 1 and exp[0]["date"] == DAYS[3] and exp[0]["reason"] == "no_quote"
    assert "fill" not in kinds(j) and not book["pending"]


def _surf_rows(spot=105.0, iv=0.2):
    rows = []
    for e in ("2026-04-20", "2026-05-20"):
        for k in (95.0, 100.0, 105.0, 110.0, 115.0):
            t = (pd.Timestamp(e) - pd.Timestamp(DAYS[3])).days / 365
            m = G.bs_price(spot, k, t, 0.04, iv, "C")
            rows.append({"expiry": e, "strike": k, "right": "C", "bid": m * 0.98,
                         "ask": m * 1.02, "mid": m, "iv": iv, "spot": spot})
    return rows


def test_surface_prices_rolled_off_expiry_above_intrinsic():
    snap = G.Snapshot(_surf_rows(), DAYS[3], 0.04)
    gone = "2026-03-27"          # earlier than every quoted expiry: flat vol of the shortest
    legs = [{"right": "C", "strike": 108.0, "expiry": gone, "qty": 1},
            {"right": "C", "strike": 112.0, "expiry": gone, "qty": -1}]
    q = [snap.quote(l, surface=True) for l in legs]
    assert all(x and x["source"] == "surface" for x in q)
    spread = q[0]["mid"] - q[1]["mid"]
    assert spread > 0.0                                 # intrinsic of an OTM spread is 0
    t = (pd.Timestamp(gone) - pd.Timestamp(DAYS[3])).days / 365
    assert q[0]["mid"] == pytest.approx(G.bs_price(105.0, 108.0, t, 0.04, 0.2, "C"), rel=1e-6)
    # in between two quoted expiries: variance interpolation of a flat 20% surface stays 20%
    mid = snap.quote({"right": "C", "strike": 108.0, "expiry": "2026-05-05"}, surface=True)
    t2 = (pd.Timestamp("2026-05-05") - pd.Timestamp(DAYS[3])).days / 365
    assert mid["mid"] == pytest.approx(G.bs_price(105.0, 108.0, t2, 0.04, 0.2, "C"), rel=1e-6)


def test_fills_refuse_surface_pricing():
    snap = G.Snapshot(_surf_rows(), DAYS[3], 0.04)
    legs = [{"right": "C", "strike": 108.0, "expiry": "2026-03-27", "qty": 1}]
    assert G._leg_entry_prices(legs, snap, closing=False) is None
    assert G._leg_entry_prices(legs, snap, closing=True) is not None
    assert G._leg_exit_mid(legs, snap)[1] == ["surface"]
    assert snap.quote(legs[0]) is None
