"""Guards the Seasonal Signals any-window price-stretch rule."""

import pandas as pd

from scripts import seasonal_edge as se


def test_long_extension_can_qualify_on_a_different_horizon(monkeypatch):
    readings = {5: 42.0, 10: 28.0, 21: 11.0}
    monkeypatch.setattr(
        se, "trailing_return_pctile",
        lambda _price, window, _asof=None: readings[window],
    )

    assert se.qualifying_extension(None, "long") == (21, 11.0)


def test_short_extension_uses_the_most_extreme_passing_window(monkeypatch):
    readings = {5: 86.0, 10: 94.0, 21: 89.0}
    monkeypatch.setattr(
        se, "trailing_return_pctile",
        lambda _price, window, _asof=None: readings[window],
    )

    assert se.qualifying_extension(None, "short") == (10, 94.0)


def test_extension_fails_when_no_window_clears_the_gate(monkeypatch):
    readings = {5: 20.0, 10: 50.0, 21: 80.0}
    monkeypatch.setattr(
        se, "trailing_return_pctile",
        lambda _price, window, _asof=None: readings[window],
    )

    assert se.qualifying_extension(None, "long") is None
    assert se.qualifying_extension(None, "short") is None


def test_strong_atr_move_threshold_remains_ticket_horizon_specific():
    # 0.80 ATR is STRONG for a 5d ticket after sqrt-time scaling, but only OK
    # for a 21d ticket. The independent extension window must not change this.
    assert se._leg_grade(0.67, 0.80, 5) == "STRONG"
    assert se._leg_grade(0.67, 0.80, 21) == "OK"


def test_candidate_labels_the_stretch_window_that_qualified(monkeypatch):
    monkeypatch.setattr(se, "expected_seasonal_path", lambda *_args, **_kwargs: None)
    blend = {
        "all": {"n": 12, "n_up": 9, "n_down": 3},
        "cyc": {"n": 3, "n_up": 2, "n_down": 1},
        "ea": 1.0, "ea_all": 1.0, "ea_cyc": 1.0,
        "cyc_ok": True, "disagree": False,
    }
    ticket = {
        "entry": 100.0, "stop": 99.5, "target": 101.0,
        "stop_atr": 0.5, "rr": 2.0,
    }

    candidate = se._seasonal_candidate(
        "detect_seasonal", "TEST", None, "2026-08-13", 5, "long",
        blend, ticket, 90.0, "tactical", ext_pct=11.0, ext_window=21,
    )

    assert candidate["evidence"]["extension"].startswith("21d return at 11th %ile")


def test_five_day_ticket_passes_on_twenty_one_day_stretch(monkeypatch):
    asof = pd.Timestamp("2026-08-13")
    index = pd.bdate_range(end=asof, periods=320)
    prices = pd.DataFrame({
        "Open": 100.0, "High": 101.0, "Low": 99.0,
        "Close": 100.0, "Volume": 1_000_000,
    }, index=index)
    ranks = pd.DataFrame({
        "Date": [asof], "atr_sznl_5d": [90.0],
        "atr_sznl_10d": [50.0], "atr_sznl_21d": [50.0],
    }, index=["TEST"])
    blend = {
        "mean": 0.01, "pct_down": 0.25,
        "all": {"n": 12, "n_up": 9, "n_down": 3},
        "cyc": {"n": 4, "n_up": 3, "n_down": 1},
        "ea": 1.2, "ea_all": 1.1, "ea_cyc": 1.3,
        "cyc_ok": True, "disagree": False,
    }
    monkeypatch.setattr(se, "seasonal_cross_section", lambda **_kwargs: ranks)
    monkeypatch.setattr(se, "load_prices", lambda _names: {"TEST": prices})
    monkeypatch.setattr(se, "recent_dollar_volume", lambda _prices: 100_000_000.0)
    monkeypatch.setattr(se, "seasonal_window_blended", lambda *_args, **_kwargs: blend)
    monkeypatch.setattr(
        se, "trailing_return_pctile",
        lambda _price, window, _asof=None: {5: 50.0, 10: 40.0, 21: 10.0}[window],
    )
    monkeypatch.setattr(se, "expected_seasonal_path", lambda *_args, **_kwargs: None)

    candidates = se.scan_seasonal_tickets(["TEST"], asof, "detect_seasonal")

    assert len(candidates) == 1
    assert candidates[0]["horizon"] == "5d"
    assert candidates[0]["evidence"]["extension"].startswith("21d return at 10th %ile")
