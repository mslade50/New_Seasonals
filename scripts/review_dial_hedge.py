"""Read-only sensitivity audit of the September 2 dial hedge research.

Uses frozen research book/dial inputs and an explicitly supplied price cache.
No trading imports, canonical writes, parameter search, or provider requests.
SPY is a research proxy; these results do not model executable MES contracts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

NAV = 750_000.0


def hysteresis(values: pd.Series) -> pd.Series:
    state = False
    result = []
    for value in values:
        if pd.notna(value):
            if value >= 50:
                state = True
            elif value < 45:
                state = False
        result.append(state)
    return pd.Series(result, index=values.index)


def next_open_pnl(
    exposure: pd.Series, opening: pd.Series, closing: pd.Series, cost_bps: float = 2.0
) -> tuple[pd.Series, pd.Series]:
    """Exposure known before today's open; old shares carry overnight gaps.

    Positive exposure is long. The final close liquidates the last position so
    terminal trading costs are included. Fractional shares are diagnostic only.
    """
    shares = exposure * NAV / opening
    previous = shares.shift(1).fillna(0)
    overnight = previous * (opening - closing.shift(1)).fillna(0)
    intraday = shares * (closing - opening)
    costs = (shares - previous).abs() * opening * cost_bps / 1e4
    if len(costs):
        costs.iloc[-1] += abs(shares.iloc[-1]) * closing.iloc[-1] * cost_bps / 1e4
    return (overnight + intraday - costs) / NAV, costs / NAV


def statistics(
    book: pd.Series, hedge: pd.Series, costs: pd.Series, armed: pd.Series
) -> dict:
    def sharpe(values):
        return (
            float(values.mean() / values.std() * np.sqrt(252)) if values.std() else None
        )

    def drawdown(values):
        equity = values.cumsum()
        high = equity.cummax().clip(lower=0)
        return float((equity - high).min() * 100)

    # Preserve the original study's 21-session episode clustering.
    indexes = np.flatnonzero(armed.to_numpy())
    groups = []
    if len(indexes):
        first = previous = indexes[0]
        for i in indexes[1:]:
            if i - previous > 21:
                groups.append((first, previous))
                first = i
            previous = i
        groups.append((first, previous))
    # Include the release-day overnight P&L and close/resize fee in the episode.
    totals = np.array(
        [hedge.iloc[a : min(b + 2, len(hedge))].sum() * NAV for a, b in groups]
    )
    t = (
        float(totals.mean() / (totals.std(ddof=1) / np.sqrt(len(totals))))
        if len(totals) > 2 and totals.std(ddof=1)
        else None
    )
    return {
        "hedge_pnl_usd": float(hedge.sum() * NAV),
        "cost_usd": float(costs.sum() * NAV),
        "sharpe_unhedged": sharpe(book),
        "sharpe_hedged": sharpe(book + hedge),
        "max_drawdown_unhedged_pct": drawdown(book),
        "max_drawdown_hedged_pct": drawdown(book + hedge),
        "worst_21d_unhedged_pct": float(book.rolling(21).sum().min() * 100),
        "worst_21d_hedged_pct": float((book + hedge).rolling(21).sum().min() * 100),
        "episodes": len(totals),
        "episode_t": t,
        "drop_best_episode_pnl_usd": float(totals.sum() - totals.max())
        if len(totals)
        else 0,
        "2021_hedge_pnl_usd": float(hedge.loc[hedge.index.year == 2021].sum() * NAV),
    }


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-dir", type=Path, required=True)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "book": args.research_dir / "strategy_daily_extended.parquet",
        "dial": args.research_dir / "pit_dial_extended.parquet",
        "prices": args.prices,
    }
    hashes = {k: digest(p) for k, p in paths.items()}
    daily = pd.read_parquet(paths["book"])
    daily.index = pd.to_datetime(daily.index)
    dials = pd.read_parquet(paths["dial"])
    dials.index = pd.to_datetime(dials.index)
    prices = (
        pq.read_table(
            paths["prices"],
            columns=["ticker", "date", "Open", "Close"],
            filters=[("ticker", "==", "SPY")],
        )
        .to_pandas()
        .set_index("date")
        .sort_index()
    )
    prices.index = pd.to_datetime(prices.index)
    if prices.index.has_duplicates:
        raise ValueError("Duplicate SPY dates")
    keep = daily.index.intersection(prices.index[prices.Close.notna()])
    keep = keep[keep <= "2026-09-01"]
    book = daily.loc[keep, "book"] / NAV
    px = prices.reindex(keep)
    returns = px.Close.pct_change(fill_method=None)
    beta = (
        (book.rolling(126).cov(returns) / returns.rolling(126).var())
        .shift(1)
        .clip(-1, 2)
    )
    index = keep[keep >= "2018-01-02"]
    if (
        px.loc[index, ["Open", "Close"]].isna().any().any()
        or (px.loc[index, ["Open", "Close"]] <= 0).any().any()
    ):
        raise ValueError("Missing or invalid execution-price input")
    results = {}
    exported = pd.DataFrame(index=index)
    for vintage, column in [
        ("pit", "pit"),
        ("current", "cur_recompute"),
        ("live_vintage", "live"),
    ]:
        lagged = dials[column].shift(1).reindex(index)
        armed = hysteresis(lagged)
        exposure = -armed.astype(float) * beta.reindex(index).fillna(0)
        trigger = (armed.astype(int).diff() > 0) | (armed & armed.shift(1).isna())
        entry_costs = (
            trigger.astype(float) * beta.reindex(index).abs().fillna(0) * 2 / 1e4
        )
        old = exposure * returns.reindex(index) - entry_costs
        results[vintage] = {
            "original_proxy": statistics(book.reindex(index), old, entry_costs, armed),
            "missing_dial_days": int(lagged.isna().sum()),
            "imputed_zero_beta_days": int(beta.reindex(index).isna().sum()),
        }
        for cost in [2.0, 5.0]:
            pnl, fees = next_open_pnl(
                exposure, px.loc[index, "Open"], px.loc[index, "Close"], cost
            )
            results[vintage][f"next_open_all_turnover_{cost:g}bps"] = statistics(
                book.reindex(index), pnl, fees, armed
            )
        exported[f"{vintage}_dial_lag"] = lagged
        exported[f"{vintage}_exposure"] = exposure
        exported[f"{vintage}_original_pnl_usd"] = old * NAV
        pnl, _ = next_open_pnl(exposure, px.loc[index, "Open"], px.loc[index, "Close"])
        exported[f"{vintage}_next_open_pnl_usd"] = pnl * NAV
    if hashes != {k: digest(p) for k, p in paths.items()}:
        raise RuntimeError("Inputs changed during the review")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "asof": "2026-09-01",
        "reviewed_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "nav_basis": NAV,
        "inputs": {k: {"path": str(p), "sha256": hashes[k]} for k, p in paths.items()},
        "assumptions": [
            "SPY fractional-share proxy, not MES execution",
            "2/5 bps per side are sensitivity assumptions",
            "Original rolling return-beta, not the proposed holdings-beta policy",
            "Saved book includes its original strategy/model vintage; no live portfolio claimed",
        ],
        "results": results,
    }
    (args.output_dir / "hedge_audit.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    exported.to_csv(args.output_dir / "hedge_daily_audit.csv")
    print(json.dumps(results, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
