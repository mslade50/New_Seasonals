"""User-parameterized capital-reuse feasibility for research trades.

This is arithmetic, not a regulatory or broker-account model.  In particular,
it never infers PDT status, margin eligibility, settlement treatment, or reuse
permission from account size.  The caller must supply every applicable limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
import pandas as pd


class AmbiguousCapitalTieError(ValueError):
    """Raised when row order would decide an oversubscribed simultaneous batch."""


@dataclass(frozen=True)
class CapitalReuseConfig:
    """Explicit research constraints applied independently to each session.

    ``starting_settled_cash`` and ``intraday_buying_power`` are independent
    ceilings when both are supplied.  ``same_day_reuse_allowed=False`` means
    accepted notional consumes each supplied ceiling for the rest of that
    session even after a trade exits.  It defaults false intentionally.
    """

    default_notional_per_trade: float | None = None
    starting_settled_cash: float | None = None
    intraday_buying_power: float | None = None
    same_day_reuse_allowed: bool = False
    max_concurrent_notional: float | None = None

    def __post_init__(self) -> None:
        amounts = {
            "default_notional_per_trade": self.default_notional_per_trade,
            "starting_settled_cash": self.starting_settled_cash,
            "intraday_buying_power": self.intraday_buying_power,
            "max_concurrent_notional": self.max_concurrent_notional,
        }
        for name, value in amounts.items():
            if value is not None and (not isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive when supplied")
        if (
            self.starting_settled_cash is None
            and self.intraday_buying_power is None
            and self.max_concurrent_notional is None
        ):
            raise ValueError("at least one capital ceiling must be supplied")


@dataclass
class CapitalFeasibilityResult:
    audit: pd.DataFrame
    feasible_trades: pd.DataFrame
    rejected_trades: pd.DataFrame


def _available(limit: float | None, used: float) -> float | None:
    return None if limit is None else max(float(limit) - used, 0.0)


def apply_capital_feasibility(
    trades: pd.DataFrame,
    config: CapitalReuseConfig,
    *,
    notional_column: str = "requested_notional",
    priority_column: str | None = None,
    priority_ascending: bool = False,
) -> CapitalFeasibilityResult:
    """Gate fixed-clock trades against explicit user-supplied capital limits.

    Entries are processed chronologically.  An oversubscribed simultaneous
    batch requires an explicit numeric priority; otherwise row order would
    allocate capital and the function raises ``AmbiguousCapitalTieError``. The
    function requires intraday trades and resets supplied daily ceilings at
    each new session; it does not model a multi-day settlement calendar.
    """

    if trades.empty:
        empty = trades.copy()
        return CapitalFeasibilityResult(empty, empty.copy(), empty.copy())
    required = {"ticker", "entry_ts", "exit_ts"}
    missing = sorted(required.difference(trades.columns))
    if missing:
        raise ValueError(f"trades missing capital-clock columns: {missing}")
    if priority_column is not None and priority_column not in trades.columns:
        raise ValueError(f"priority column is missing: {priority_column}")

    work = trades.copy().reset_index(drop=True)
    work["_original_order"] = range(len(work))
    work["entry_ts"] = pd.to_datetime(work["entry_ts"], errors="coerce")
    work["exit_ts"] = pd.to_datetime(work["exit_ts"], errors="coerce")
    if work[["entry_ts", "exit_ts"]].isna().any().any():
        raise ValueError("trades contain invalid capital-clock timestamps")
    if work["exit_ts"].le(work["entry_ts"]).any():
        raise ValueError("capital feasibility requires exits after entries")
    if not work["entry_ts"].dt.normalize().eq(work["exit_ts"].dt.normalize()).all():
        raise ValueError("capital feasibility v0 supports intraday trades only")

    if notional_column in work.columns:
        work["requested_notional"] = pd.to_numeric(
            work[notional_column], errors="coerce"
        )
    elif config.default_notional_per_trade is not None:
        work["requested_notional"] = float(config.default_notional_per_trade)
    else:
        raise ValueError(
            f"trades lack {notional_column!r} and no default_notional_per_trade was supplied"
        )
    if (
        work["requested_notional"].isna().any()
        or (~np.isfinite(work["requested_notional"])).any()
        or work["requested_notional"].le(0).any()
    ):
        raise ValueError("requested notionals must be finite positive numbers")
    if priority_column is not None:
        work[priority_column] = pd.to_numeric(work[priority_column], errors="coerce")
        if (
            work[priority_column].isna().any()
            or (~np.isfinite(work[priority_column])).any()
        ):
            raise ValueError("capital priority values must be finite numbers")

    sort_columns = ["entry_ts"]
    ascending = [True]
    if priority_column is not None:
        sort_columns.append(priority_column)
        ascending.append(priority_ascending)
        sort_columns.append("ticker")
        ascending.append(True)
    sort_columns.append("_original_order")
    ascending.append(True)
    work = work.sort_values(
        sort_columns, ascending=ascending, kind="stable"
    ).reset_index(drop=True)

    audit_rows: list[dict] = []
    sequence = 0
    for trade_day, day_trades in work.groupby(
        work["entry_ts"].dt.normalize(), sort=True
    ):
        used_settled_cash = 0.0
        used_buying_power = 0.0
        open_positions: list[tuple[pd.Timestamp, float]] = []
        open_notional = 0.0

        for entry_ts, entry_batch in day_trades.groupby("entry_ts", sort=True):
            entry_ts = pd.Timestamp(entry_ts)
            still_open: list[tuple[pd.Timestamp, float]] = []
            for exit_ts, open_amount in open_positions:
                if exit_ts <= entry_ts:
                    open_notional -= open_amount
                    if config.same_day_reuse_allowed:
                        used_settled_cash -= open_amount
                        used_buying_power -= open_amount
                else:
                    still_open.append((exit_ts, open_amount))
            open_positions = still_open
            used_settled_cash = max(used_settled_cash, 0.0)
            used_buying_power = max(used_buying_power, 0.0)
            open_notional = max(open_notional, 0.0)

            cash_available = _available(config.starting_settled_cash, used_settled_cash)
            buying_power_available = _available(
                config.intraday_buying_power, used_buying_power
            )
            concurrent_available = _available(
                config.max_concurrent_notional, open_notional
            )
            if priority_column is None and len(entry_batch) > 1:
                batch_notional = float(entry_batch["requested_notional"].sum())
                capacities = (
                    cash_available,
                    buying_power_available,
                    concurrent_available,
                )
                if any(
                    capacity is not None and batch_notional > capacity + 1e-9
                    for capacity in capacities
                ):
                    raise AmbiguousCapitalTieError(
                        f"{trade_day.date()} {entry_ts.time()} simultaneous batch "
                        "is oversubscribed; supply a pre-registered priority column"
                    )

            for record in entry_batch.to_dict("records"):
                requested = float(record["requested_notional"])
                cash_available = _available(
                    config.starting_settled_cash, used_settled_cash
                )
                buying_power_available = _available(
                    config.intraday_buying_power, used_buying_power
                )
                concurrent_available = _available(
                    config.max_concurrent_notional, open_notional
                )
                reasons: list[str] = []
                if cash_available is not None and requested > cash_available + 1e-9:
                    reasons.append("settled_cash")
                if (
                    buying_power_available is not None
                    and requested > buying_power_available + 1e-9
                ):
                    reasons.append("intraday_buying_power")
                if (
                    concurrent_available is not None
                    and requested > concurrent_available + 1e-9
                ):
                    reasons.append("max_concurrent_notional")

                accepted = not reasons
                result = dict(record)
                result.update(
                    {
                        "capital_sequence": sequence,
                        "capital_trade_date": trade_day,
                        "capital_feasible": accepted,
                        "capital_rejection_reason": "|".join(reasons),
                        "same_day_reuse_assumed": config.same_day_reuse_allowed,
                        "open_notional_before": open_notional,
                        "settled_cash_available_before": cash_available,
                        "intraday_buying_power_available_before": buying_power_available,
                        "concurrent_notional_available_before": concurrent_available,
                    }
                )
                sequence += 1
                if accepted:
                    used_settled_cash += requested
                    used_buying_power += requested
                    open_notional += requested
                    open_positions.append((pd.Timestamp(record["exit_ts"]), requested))
                audit_rows.append(result)

    audit = pd.DataFrame(audit_rows).drop(columns=["_original_order"], errors="ignore")
    feasible = audit.loc[audit["capital_feasible"]].reset_index(drop=True)
    rejected = audit.loc[~audit["capital_feasible"]].reset_index(drop=True)
    return CapitalFeasibilityResult(audit.reset_index(drop=True), feasible, rejected)
