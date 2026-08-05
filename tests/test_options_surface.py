import math

from options_surface import (
    basket_vol,
    constant_maturity_iv,
    forward_vol,
    implied_correlation,
    percentile_rank,
    summarize_surface,
)


def test_constant_maturity_uses_total_variance_and_never_extrapolates():
    points = [{"dte": 20, "atm_iv": 0.20}, {"dte": 40, "atm_iv": 0.24}]
    expected = math.sqrt(((0.20**2 * 20) + 0.5 * ((0.24**2 * 40) - (0.20**2 * 20))) / 30)
    assert math.isclose(constant_maturity_iv(points, 30), expected)
    assert constant_maturity_iv(points, 10) is None
    assert forward_vol(0.20, 20, 0.24, 40) > 0.24


def test_implied_correlation_inverts_the_basket_variance_equation():
    component_vols = [0.22, 0.27, 0.31, 0.18]
    target_rho = 0.43
    index_vol = basket_vol(component_vols, target_rho)
    assert math.isclose(implied_correlation(index_vol, component_vols), target_rho, abs_tol=1e-12)


def test_surface_summary_captures_skew_and_unsigned_positioning_without_dealer_claim():
    term = [
        {"dte": 20, "atm_iv": 0.20},
        {"dte": 40, "atm_iv": 0.24},
        {"dte": 90, "atm_iv": 0.26},
    ]
    chain = [
        {"expiry": "20260918", "dte": 30, "strike": 95, "right": "P", "delta": -0.25,
         "iv": 0.25, "gamma": 0.02, "oi": 200},
        {"expiry": "20260918", "dte": 30, "strike": 105, "right": "C", "delta": 0.25,
         "iv": 0.22, "gamma": 0.02, "oi": 100},
    ]
    result = summarize_surface("spy", "2026-08-05", 100, term, chain)
    assert result["ticker"] == "SPY"
    assert math.isclose(result["rr25"], 0.03)
    assert result["total_oi"] == 300
    assert result["put_call_oi"] == 2
    assert result["gamma_abs_1pct"] > 0
    assert result["call_minus_put_gamma_proxy"] < 0
    assert result["oi_coverage"] == 1


def test_history_percentiles_require_a_real_sample():
    assert percentile_rank(range(19), 10, min_obs=20) is None
    assert percentile_rank(range(20), 10, min_obs=20) == 50
