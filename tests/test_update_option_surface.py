import datetime as dt
import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("update_option_surface", os.path.join(ROOT, "scripts", "update_option_surface.py"))
uos = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(uos)

TODAY = dt.date(2026, 10, 7)  # Wednesday


def _exp(days):
    return (TODAY + dt.timedelta(days=days)).strftime("%Y%m%d")


def _dtes(expiries):
    return [uos._dte(e, TODAY) for e in expiries]


def test_chain_expiries_cover_front_weeklies_30_60_and_drop_90():
    listed = [_exp(d) for d in (0, 1, 2, 7, 9, 14, 21, 28, 35, 63, 91, 120)]
    got = uos._chain_expiries(listed, TODAY)
    assert _dtes(got) == [1, 2, 7, 14, 28, 63]  # no 0DTE, no 90
    assert len(set(got)) == len(got)


def test_chain_expiries_dedupe_on_thin_products():
    listed = [_exp(d) for d in (3, 24, 52, 94)]  # monthlies only
    got = uos._chain_expiries(listed, TODAY)
    assert len(set(got)) == len(got)
    # 7/14 targets must not re-claim a far monthly; 90+ is not recorded
    assert _dtes(got) == [3, 24, 52]


def test_chain_expiries_empty_and_all_expired():
    assert uos._chain_expiries([], TODAY) == []
    assert uos._chain_expiries([_exp(0), _exp(-3)], TODAY) == []


def test_chain_band_short_dte_is_tight_and_keeps_atm_neighbours():
    strikes = [float(s) for s in range(400, 500)]  # $1 grid
    spot = 450.4
    band = uos._chain_band(strikes, spot, 1)
    assert min(band) >= spot * 0.97 and max(band) <= spot * 1.03
    assert len(band) <= uos._strike_cap(1)
    nearest3 = sorted(strikes, key=lambda s: abs(s - spot))[:3]
    assert all(s in band for s in nearest3)


def test_chain_band_widens_with_dte_but_caps_far_tenors():
    strikes = [float(s) for s in range(200, 800)]
    spot = 500.0
    near, far = uos._chain_band(strikes, spot, 7), uos._chain_band(strikes, spot, 60)
    assert max(far) - min(far) > max(near) - min(near)
    assert len(near) <= 20 and len(far) <= 10
    assert 500.0 in far
