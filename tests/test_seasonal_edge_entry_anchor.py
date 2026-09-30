"""Guards the 2026-09-30 seasonal-board fixes (docs/claude_ref/private_site.md):

1. Entry anchoring: a ticket that enters T+k measures every realized stat
   (cycle/all-years k/n, ATR magnitudes, expected move) from ITS entry close,
   entry_offset_days + 1 sessions past the as-of analog, forward h sessions.
   A seasonal move that is over by the entry must not pass the gates.
2. Ticket shape: time-exit primary, 3.0 ATR catastrophe stop, no target, and
   an expected-move gate (>= 1.0 ATR) in place of R/R >= 2.
"""
import numpy as np
import pandas as pd
import pytest

from scripts import seasonal_edge as se
from scripts.seasonal_ticket_sim import parse_ticket, simulate_ticket

ASOF = pd.Timestamp("2026-09-29")
H = 10
OFFSET = 4  # enter T+5


def _series(move_start: int, move_len: int = 5, step: float = -0.01) -> pd.DataFrame:
    """Flat business-day series except that in every prior year the sessions
    p+move_start .. p+move_start+move_len-1 (p = the as-of trading-day-of-year
    analog) each move `step`."""
    idx = pd.bdate_range("2000-01-03", ASOF)
    doy = se._trading_doy(idx).values
    target = int(doy[-1])
    rets = np.zeros(len(idx))
    for y in range(2000, ASOF.year):
        pos = np.flatnonzero((idx.year == y) & (doy == target))
        if pos.size:
            p = int(pos[0])
            rets[p + move_start:p + move_start + move_len] = step
    close = 100.0 * np.cumprod(1.0 + rets)
    return pd.DataFrame({"Open": close, "High": close * 1.005, "Low": close * 0.995,
                         "Close": close, "Volume": 1_000_000}, index=idx)


def _peak_path(offset: int):
    path = np.zeros(H)
    path[offset] = 1.0  # argmax -> short entry offset
    return path


def _patch_scan(monkeypatch, px):
    ranks = pd.DataFrame({"Date": [ASOF], "atr_sznl_5d": [50.0],
                          "atr_sznl_10d": [5.0], "atr_sznl_21d": [50.0]}, index=["TEST"])
    monkeypatch.setattr(se, "seasonal_cross_section", lambda **_k: ranks)
    monkeypatch.setattr(se, "load_prices", lambda _names: {"TEST": px})
    monkeypatch.setattr(se, "recent_dollar_volume", lambda _px: 1e9)
    monkeypatch.setattr(se, "trailing_return_pctile", lambda _px, _w, _a=None: 95.0)
    monkeypatch.setattr(se, "expected_seasonal_path", lambda *_a, **_k: _peak_path(OFFSET))


def test_move_before_the_entry_fails_the_gate(monkeypatch):
    # the drop happens on sessions p+1..p+5 = through the T+5 entry close
    px = _series(move_start=1)
    # the OLD lag-0 measurement sees the drop and confirms the short 26/26
    old = se.seasonal_window_blended(px, ASOF, H, entry_lag=0)
    assert se._confirms_blended(old, "short")
    assert old["all"]["n_down"] == old["all"]["n"] >= 20
    # measured from the T+5 entry close the window is flat
    new = se.seasonal_window_blended(px, ASOF, H, entry_lag=OFFSET + 1)
    assert new["all"]["n_down"] == 0
    assert not se._confirms_blended(new, "short")

    _patch_scan(monkeypatch, px)
    assert se.scan_seasonal_tickets(["TEST"], ASOF, "detect_seasonal", horizons=(H,)) == []


def test_move_after_the_entry_passes_and_stats_are_entry_anchored(monkeypatch):
    # the drop happens on sessions p+6..p+10, after the T+5 entry close
    px = _series(move_start=OFFSET + 2)
    _patch_scan(monkeypatch, px)
    cands = se.scan_seasonal_tickets(["TEST"], ASOF, "detect_seasonal", horizons=(H,))
    assert len(cands) == 1
    c = cands[0]
    assert c["entry_offset_days"] == OFFSET
    ev = c["evidence"]
    assert "from T+5 close" in ev["all-years"]
    n_prior = ASOF.year - 2000
    assert ev["all-years"].startswith(f"{n_prior}/{n_prior} lower")
    # expected move equals the entry-anchored blend, not the lag-0 one
    blend = se.seasonal_window_blended(px, ASOF, H, entry_lag=OFFSET + 1)
    assert c["expected_move_atr"] == pytest.approx(blend["ea"], abs=0.01)


def test_ticket_is_time_exit_with_catastrophe_stop(monkeypatch):
    px = _series(move_start=OFFSET + 2)
    _patch_scan(monkeypatch, px)
    c = se.scan_seasonal_tickets(["TEST"], ASOF, "detect_seasonal", horizons=(H,))[0]
    line = c["evidence"]["TICKET"]
    assert line.startswith("SELL ~")
    assert "cat-stop" in line and "(3.0 ATR)" in line
    assert f"time-exit {H}td" in line and f"ATR / {H}td" in line
    assert "target" not in line and "R/R" not in line
    assert "blended target" not in c["evidence"]
    assert "expected move" in c["evidence"]

    tk = parse_ticket(c)
    assert tk["target"] is None and tk["rr"] is None and tk["hold_from_entry"]
    atr = float(se.atr_wilder(px).iloc[-1])
    assert tk["stop"] - tk["entry"] == pytest.approx(3.0 * atr, abs=0.01)
    assert tk["expected_move_atr"] == pytest.approx(c["expected_move_atr"], abs=0.01)


def test_expected_move_gate_replaces_rr():
    px = _series(move_start=1)
    atr = float(se.atr_wilder(px).iloc[-1])
    entry = float(px["Close"].iloc[-1])
    weak = se.build_trade_ticket(px, ASOF, "long", 21, 0.99)
    strong = se.build_trade_ticket(px, ASOF, "long", 21, 1.0)
    assert not weak["is_ticket"] and strong["is_ticket"]
    for h in (5, 10, 21):
        tk = se.build_trade_ticket(px, ASOF, "long", h, 2.5)
        assert tk["target"] is None and tk["rr"] is None
        assert tk["stop_atr"] == 3.0
        assert entry - tk["stop"] == pytest.approx(3.0 * atr, abs=1e-3)
        assert tk["time_stop_days"] == h
    short = se.build_trade_ticket(px, ASOF, "short", 10, -1.5)
    assert short["stop"] - entry == pytest.approx(3.0 * atr, abs=1e-3)


def test_time_exit_ticket_holds_n_sessions_from_entry():
    # asof bar + 8 forward bars; delayed entry at forward bar 2's open, hold 3
    idx = pd.bdate_range("2020-01-02", periods=9)
    px = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0}, index=idx)
    px.iloc[6, px.columns.get_loc("Close")] = 104.0  # close of fwd bar 5 = entry bar + 3
    tk = {"direction": "long", "entry": 100.0, "stop": 94.0, "target": None,
          "time_stop_days": 3}
    out = simulate_ticket(tk, px, idx[0], entry_mode="delayed", entry_window=2, reanchor=True)
    assert out["exit_type"] == "Time"
    assert out["entry_date"] == idx[3] and out["exit_date"] == idx[6]
    assert out["R"] == pytest.approx(4.0 / 6.0, abs=1e-3)
    # not matured until forward bar entry+N exists
    assert simulate_ticket(tk, px.iloc[:6], idx[0], entry_mode="delayed", entry_window=2) is None


def test_order_staging_parses_time_exit_ticket():
    # staging is OFF today; when re-enabled it must not silently drop the new line
    from seasonal_order_staging import _row, parse_seasonal_ticket
    cand = {"ticker": "TRV", "channel": "Equity Seasonal Tickets", "direction": "long",
            "horizon": "21d", "conviction": "A", "entry_offset_days": 1,
            "evidence": {"TICKET": "BUY ~363.21 | cat-stop 343.35 (3.0 ATR) | "
                                   "expected +2.79 ATR / 21td | time-exit 21td"}}
    tk = parse_seasonal_ticket(cand)
    assert tk["target"] is None and tk["stop_atr"] == 3.0 and tk["tgt_atr"] == 0.0
    row = _row(cand, tk, 20.0, 750_000.0, "2026-09-29")
    assert row["Use_Target"] is False and row["Stop_ATR_Mult"] == 3.0
    # 21 sessions held from the T+2 entry -> asof + 23 business days
    assert row["Time_Exit_Date"] == (pd.Timestamp("2026-09-29")
                                     + pd.tseries.offsets.BDay(23)).strftime("%Y-%m-%d")


def test_legacy_target_ticket_keeps_asof_window():
    idx = pd.bdate_range("2020-01-02", periods=9)
    px = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0}, index=idx)
    tk = {"direction": "long", "entry": 100.0, "stop": 98.0, "target": 110.0,
          "time_stop_days": 5}
    out = simulate_ticket(tk, px, idx[0], entry_mode="delayed", entry_window=2)
    assert out["exit_type"] == "Time" and out["exit_date"] == idx[5]  # asof + 5
