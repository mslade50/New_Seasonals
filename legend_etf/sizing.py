"""Account-local, overlap-aware sizing for the ETF implementation."""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

from .config import RULES, RiskProfile


@dataclass(frozen=True)
class SizeRequest:
    root: str
    etf: str
    direction: int
    atr14: float
    reference_price: float


@dataclass(frozen=True)
class SizeResult:
    root: str
    etf: str
    direction: int
    shares: int
    requested_bps: float
    stress_distance: float
    stress_risk_usd: float
    reference_notional_usd: float
    scale: float
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def risk_profile_from_env(
    label: str,
    baseline: RiskProfile,
    *,
    environment: Mapping[str, str] | None = None,
) -> RiskProfile:
    """Allow deployment-time reductions, never increases, to reviewed limits."""

    prefix = f"LEGEND_ETF_{label.upper()}"
    values = os.environ if environment is None else environment

    def number(name: str, default: float) -> float:
        value = float(values.get(f"{prefix}_{name}", str(default)))
        if not math.isfinite(value) or value < 0 or value > default:
            raise ValueError(f"{prefix}_{name} must be between 0 and {default}")
        return value

    max_shares_value = number(
        "MAX_SHARES_PER_ROOT", float(baseline.max_shares_per_root)
    )
    if max_shares_value != int(max_shares_value):
        raise ValueError(f"{prefix}_MAX_SHARES_PER_ROOT must be a whole number")
    max_shares = int(max_shares_value)
    return RiskProfile(
        long_bps=number("LONG_BPS", baseline.long_bps),
        short_bps=number("SHORT_BPS", baseline.short_bps),
        cluster_bps=number("CLUSTER_BPS", baseline.cluster_bps),
        max_shares_per_root=max_shares,
        max_notional_pct=number("MAX_NOTIONAL_PCT", baseline.max_notional_pct),
    )


def _validate_request(request: SizeRequest) -> None:
    if request.direction not in (-1, 1):
        raise ValueError(f"{request.root}: direction must be -1 or 1")
    if not math.isfinite(request.atr14) or request.atr14 <= 0:
        raise ValueError(f"{request.root}: ATR14 must be positive and finite")
    if not math.isfinite(request.reference_price) or request.reference_price <= 0:
        raise ValueError(f"{request.root}: reference price must be positive and finite")


def size_batch(
    requests: Iterable[SizeRequest],
    *,
    nlv: float,
    profile: RiskProfile,
    available_long_bps: float | None = None,
    available_short_bps: float | None = None,
    available_gross_bps: float | None = None,
) -> list[SizeResult]:
    """Size each economic root, then enforce one gross cluster cap.

    Integer rounding is performed only after pro-rata scaling.  Shares are never
    forced to one, and unused rounding capacity is not recycled into another
    root.  Thus adding a correlated signal cannot increase a prior root's size.
    """

    if not math.isfinite(nlv) or nlv <= 0:
        raise ValueError("NLV must be positive and finite")
    for label, value in {
        "available_long_bps": available_long_bps,
        "available_short_bps": available_short_bps,
        "available_gross_bps": available_gross_bps,
    }.items():
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f"{label} must be finite and non-negative")
    items = list(requests)
    roots = [item.root for item in items]
    etfs = [item.etf for item in items]
    if len(set(roots)) != len(roots) or len(set(etfs)) != len(etfs):
        raise ValueError("batch must contain at most one request per economic root/ETF")
    for item in items:
        _validate_request(item)

    provisional: list[dict[str, float | int | str]] = []
    for item in items:
        bps = profile.long_bps if item.direction > 0 else profile.short_bps
        stress = RULES.atr_stress_multiple * item.atr14
        risk_budget = nlv * bps / 10_000.0
        raw_shares = math.floor(risk_budget / stress)
        notional_limit = nlv * profile.max_notional_pct
        notional_shares = math.floor(notional_limit / item.reference_price)
        shares = max(
            0,
            min(raw_shares, profile.max_shares_per_root, notional_shares),
        )
        provisional.append(
            {
                "root": item.root,
                "etf": item.etf,
                "direction": item.direction,
                "bps": bps,
                "stress": stress,
                "price": item.reference_price,
                "shares": shares,
            }
        )

    side_totals = {
        direction: sum(
            float(row["stress"]) * int(row["shares"])
            for row in provisional
            if int(row["direction"]) == direction
        )
        for direction in (-1, 1)
    }
    side_bps = {1: available_long_bps, -1: available_short_bps}
    side_scales: dict[int, float] = {}
    for direction in (-1, 1):
        bps = side_bps[direction]
        budget = math.inf if bps is None else nlv * bps / 10_000.0
        total = side_totals[direction]
        side_scales[direction] = (
            min(1.0, budget / total) if total > 0 else 1.0
        )
    after_side = sum(
        float(row["stress"])
        * int(row["shares"])
        * side_scales[int(row["direction"])]
        for row in provisional
    )
    gross_bps = profile.cluster_bps
    if available_gross_bps is not None:
        gross_bps = min(gross_bps, available_gross_bps)
    cluster_budget = nlv * gross_bps / 10_000.0
    gross_scale = (
        min(1.0, cluster_budget / after_side) if after_side > 0 else 1.0
    )

    results: list[SizeResult] = []
    for row in provisional:
        scale = side_scales[int(row["direction"])] * gross_scale
        shares = math.floor(int(row["shares"]) * scale + 1e-12)
        stress_risk = shares * float(row["stress"])
        reason = "sized" if shares > 0 else "below_one_share_or_capped"
        results.append(
            SizeResult(
                root=str(row["root"]),
                etf=str(row["etf"]),
                direction=int(row["direction"]),
                shares=shares,
                requested_bps=float(row["bps"]),
                stress_distance=float(row["stress"]),
                stress_risk_usd=stress_risk,
                reference_notional_usd=shares * float(row["price"]),
                scale=scale,
                reason=reason,
            )
        )
    if sum(result.stress_risk_usd for result in results) > cluster_budget + 1e-8:
        raise AssertionError("integer sizing exceeded the cluster risk cap")
    return results
