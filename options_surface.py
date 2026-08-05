"""Shared ETF/index option-surface math and data contract.

The live collector, site builder, and tests all use this module so the
Moontower-style rankings do not quietly drift from the data being recorded.
All volatility inputs are decimals (0.20 == 20%). Constant-maturity values are
interpolated in total-variance space and are never extrapolated.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Iterable, Mapping, Sequence


OPTIONS_ETF_GROUPS = {
    "Index": ["SPY", "QQQ", "IWM", "DIA"],
    "US sectors": ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"],
    "Industry": ["SMH", "XBI", "IBB", "IHI", "ITA", "ITB", "KRE", "OIH", "XHB", "XME", "XOP", "XRT"],
    "Macro": ["TLT", "HYG", "LQD", "UUP", "GLD", "SLV", "USO", "UNG", "EEM", "EFA", "EWJ"],
    "Real estate": ["IYR", "VNQ"],
    "Alternative": ["IBIT"],
}
OPTIONS_MACRO_ETFS = tuple(dict.fromkeys(
    ticker for tickers in OPTIONS_ETF_GROUPS.values() for ticker in tickers
))

SURFACE_HISTORY_R2_KEY = "options/surface_history.parquet"
POSITIONING_HISTORY_R2_KEY = "options/positioning_history.parquet"

CMIV_TENORS = (10, 20, 30, 60, 90, 180, 365)
CHAIN_TENORS = (30, 60, 90)


def _finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def constant_maturity_iv(points: Iterable[Mapping], target_dte: int):
    """Variance-interpolate a constant-maturity IV; return None outside range."""
    clean = sorted(
        (int(p["dte"]), float(p["atm_iv"]))
        for p in points
        if _finite(p.get("dte")) and _finite(p.get("atm_iv")) and float(p["dte"]) > 0 and float(p["atm_iv"]) > 0
    )
    if not clean:
        return None
    exact = next((iv for dte, iv in clean if dte == target_dte), None)
    if exact is not None:
        return exact
    lo = next(((dte, iv) for dte, iv in reversed(clean) if dte < target_dte), None)
    hi = next(((dte, iv) for dte, iv in clean if dte > target_dte), None)
    if lo is None or hi is None:
        return None
    lo_dte, lo_iv = lo
    hi_dte, hi_iv = hi
    weight = (target_dte - lo_dte) / (hi_dte - lo_dte)
    total_variance = lo_iv * lo_iv * lo_dte + weight * (
        hi_iv * hi_iv * hi_dte - lo_iv * lo_iv * lo_dte
    )
    return math.sqrt(total_variance / target_dte) if total_variance > 0 else None


def forward_vol(iv_near, dte_near, iv_far, dte_far):
    iv_near, iv_far = _finite(iv_near), _finite(iv_far)
    dte_near, dte_far = _finite(dte_near), _finite(dte_far)
    if not (iv_near and iv_far and dte_near and dte_far and dte_far > dte_near):
        return None
    variance = (iv_far * iv_far * dte_far - iv_near * iv_near * dte_near) / (dte_far - dte_near)
    return math.sqrt(variance) if variance > 0 else None


def percentile_rank(values: Iterable, current, min_obs: int = 20):
    current = _finite(current)
    clean = sorted(v for raw in values if (v := _finite(raw)) is not None)
    if current is None or len(clean) < min_obs:
        return None
    return 100.0 * sum(v < current for v in clean) / len(clean)


def nearest_delta(rows: Sequence[Mapping], right: str, target: float):
    candidates = []
    for row in rows:
        delta = _finite(row.get("delta"))
        iv = _finite(row.get("iv"))
        if str(row.get("right", "")).upper() == right and delta is not None and iv and iv > 0:
            candidates.append((abs(abs(delta) - target), row))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def implied_correlation(index_vol, component_vols: Sequence, weights: Sequence | None = None):
    """Return the constant-correlation solution in variance space.

    This is only as representative as the supplied basket. Passing equal
    sector-ETF weights therefore creates a sector proxy, not constituent-level
    SPX implied correlation.
    """
    index_vol = _finite(index_vol)
    vols = [_finite(v) for v in component_vols]
    if not index_vol or len(vols) < 2 or any(v is None or v <= 0 for v in vols):
        return None
    if weights is None:
        w = [1.0 / len(vols)] * len(vols)
    else:
        w = [_finite(x) for x in weights]
        if len(w) != len(vols) or any(x is None or x < 0 for x in w) or not sum(w):
            return None
        total = sum(w)
        w = [x / total for x in w]
    own = sum((wi * vi) ** 2 for wi, vi in zip(w, vols))
    cross = 2.0 * sum(
        w[i] * w[j] * vols[i] * vols[j]
        for i in range(len(vols)) for j in range(i + 1, len(vols))
    )
    if cross <= 0:
        return None
    return (index_vol * index_vol - own) / cross


def basket_vol(component_vols: Sequence, correlation: float, weights: Sequence | None = None):
    vols = [_finite(v) for v in component_vols]
    rho = _finite(correlation)
    if rho is None or len(vols) < 2 or any(v is None or v <= 0 for v in vols):
        return None
    w = [1.0 / len(vols)] * len(vols) if weights is None else list(weights)
    total = sum(w)
    if not total:
        return None
    w = [float(x) / total for x in w]
    variance = sum((wi * vi) ** 2 for wi, vi in zip(w, vols))
    variance += 2.0 * rho * sum(
        w[i] * w[j] * vols[i] * vols[j]
        for i in range(len(vols)) for j in range(i + 1, len(vols))
    )
    return math.sqrt(variance) if variance > 0 else None


def summarize_surface(ticker: str, date: str, spot, term_rows: Sequence[Mapping], chain_rows: Sequence[Mapping],
                      market_data_type=None, pulled_at=None):
    """Collapse one nightly ticker snapshot into a history-friendly row."""
    spot = _finite(spot)
    out = {
        "date": str(date), "ticker": str(ticker).upper(), "spot": spot,
        "market_data_type": market_data_type, "pulled_at": pulled_at,
        "term_quote_count": sum(1 for r in term_rows if _finite(r.get("atm_iv"))),
    }
    for tenor in CMIV_TENORS:
        out[f"cmiv{tenor}"] = constant_maturity_iv(term_rows, tenor)
    iv30, iv60, iv90 = out.get("cmiv30"), out.get("cmiv60"), out.get("cmiv90")
    out["term_30_90"] = iv30 / iv90 - 1.0 if iv30 and iv90 else None
    out["term_30_60"] = iv30 / iv60 - 1.0 if iv30 and iv60 else None
    out["fwd30_90"] = forward_vol(iv30, 30, iv90, 90)

    by_expiry = defaultdict(list)
    for row in chain_rows:
        by_expiry[str(row.get("expiry") or "")].append(row)
    expiries = []
    for expiry, rows in by_expiry.items():
        dtes = [_finite(r.get("dte")) for r in rows]
        dtes = [d for d in dtes if d is not None]
        if dtes:
            expiries.append((abs(min(dtes) - 30), min(dtes), expiry, rows))
    skew_rows = min(expiries, key=lambda x: x[0])[3] if expiries else []
    p25, c25 = nearest_delta(skew_rows, "P", 0.25), nearest_delta(skew_rows, "C", 0.25)
    p10, c10 = nearest_delta(skew_rows, "P", 0.10), nearest_delta(skew_rows, "C", 0.10)
    p25iv = _finite(p25.get("iv")) if p25 else None
    c25iv = _finite(c25.get("iv")) if c25 else None
    p10iv = _finite(p10.get("iv")) if p10 else None
    c10iv = _finite(c10.get("iv")) if c10 else None
    out.update({
        "rr25": p25iv - c25iv if p25iv and c25iv else None,
        "rr10": p10iv - c10iv if p10iv and c10iv else None,
        "put25_norm": p25iv / iv30 - 1.0 if p25iv and iv30 else None,
        "call25_norm": c25iv / iv30 - 1.0 if c25iv and iv30 else None,
        "skew_expiry": min(expiries, key=lambda x: x[0])[2] if expiries else None,
    })

    call_oi = put_oi = gamma_abs = gamma_proxy = 0.0
    oi_seen = gamma_seen = 0
    strike_oi, strike_gamma = defaultdict(float), defaultdict(float)
    for row in chain_rows:
        oi, gamma, strike = _finite(row.get("oi")), _finite(row.get("gamma")), _finite(row.get("strike"))
        if oi is not None and oi >= 0:
            oi_seen += 1
            if str(row.get("right", "")).upper() == "C":
                call_oi += oi
            else:
                put_oi += oi
            if strike is not None:
                strike_oi[strike] += oi
        if oi is not None and oi >= 0 and gamma is not None and spot:
            gamma_seen += 1
            dollar_gamma_1pct = abs(gamma) * spot * spot * 0.01 * 100.0 * oi
            gamma_abs += dollar_gamma_1pct
            gamma_proxy += dollar_gamma_1pct * (1.0 if str(row.get("right", "")).upper() == "C" else -1.0)
            if strike is not None:
                strike_gamma[strike] += dollar_gamma_1pct
    total_oi = call_oi + put_oi if oi_seen else None
    out.update({
        "call_oi": call_oi if oi_seen else None,
        "put_oi": put_oi if oi_seen else None,
        "total_oi": total_oi,
        "put_call_oi": put_oi / call_oi if oi_seen and call_oi > 0 else None,
        "gamma_abs_1pct": gamma_abs if gamma_seen else None,
        "call_minus_put_gamma_proxy": gamma_proxy if gamma_seen else None,
        "max_oi_strike": max(strike_oi, key=strike_oi.get) if strike_oi else None,
        "max_gamma_strike": max(strike_gamma, key=strike_gamma.get) if strike_gamma else None,
        "top_gamma_strikes": json.dumps(sorted(strike_gamma.items(), key=lambda x: x[1], reverse=True)[:5]),
        "chain_expiry_count": len(by_expiry),
        "chain_contract_count": len(chain_rows),
        "oi_coverage": oi_seen / len(chain_rows) if chain_rows else 0.0,
        "gamma_coverage": gamma_seen / len(chain_rows) if chain_rows else 0.0,
    })
    return out
