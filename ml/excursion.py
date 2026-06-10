"""
ml/excursion.py — Maximum Adverse Excursion (MAE) labels for the risk model.

For each resolved ledger trade, walk the daily bars from Entry Date through
Exit Date and measure the worst move against the position, normalized by the
same per-share risk used for y_R (Risk_flat_750k / Shares_flat):

    long:  mae_R = max(0, (Entry - min(Low))  / risk_per_share)
    short: mae_R = max(0, (max(High) - Entry) / risk_per_share)

Label: y_mae1 = mae_R >= 1.0 ("went a full risk-unit offside at some point").

Caveat (disclosed): the entry day's full Low/High is included even though part
of that range may predate the intraday fill — this slightly OVERSTATES
excursion, which is the conservative direction for a risk model. Trades with
unusable risk normalization are dropped (label NaN).
"""

import numpy as np
import pandas as pd

from ml import config


def trade_mae_r(bars: pd.DataFrame, entry_date: pd.Timestamp,
                exit_date: pd.Timestamp, entry_price: float,
                risk_per_share: float, direction: str) -> float:
    """MAE in R units for one trade; NaN when not computable."""
    if (pd.isna(entry_date) or pd.isna(exit_date) or pd.isna(entry_price)
            or not np.isfinite(risk_per_share) or risk_per_share <= 0):
        return float("nan")
    window = bars.loc[entry_date:exit_date]
    if window.empty:
        return float("nan")
    if str(direction).strip().lower() == "short":
        adverse = float(window["High"].max()) - float(entry_price)
    else:
        adverse = float(entry_price) - float(window["Low"].min())
    return max(0.0, adverse) / float(risk_per_share)


def build_excursion_labels(ledger: pd.DataFrame, price_map: dict,
                           verbose: bool = True) -> pd.DataFrame:
    """Frame indexed like `ledger` with mae_R and y_mae1 columns."""
    risk_per_share = ledger["Risk_flat_750k"] / ledger["Shares_flat"].replace(0, np.nan)

    out = pd.DataFrame(index=ledger.index, columns=["mae_R", "y_mae1"], dtype=float)
    for idx, row in ledger.iterrows():
        bars = price_map.get(row["Ticker"])
        if bars is None:
            continue
        out.at[idx, "mae_R"] = trade_mae_r(
            bars, row["Entry Date"], row["Exit Date"], row["Entry Price"],
            risk_per_share.at[idx], row["Direction"])
    out["y_mae1"] = (out["mae_R"] >= 1.0).astype(float).where(out["mae_R"].notna())
    if verbose:
        ok = out["mae_R"].notna()
        print(f"[excursion] labels for {int(ok.sum())}/{len(out)} trades; "
              f"P(MAE>=1R) base rate = {out.loc[ok, 'y_mae1'].mean():.3f}; "
              f"median MAE = {out.loc[ok, 'mae_R'].median():.2f}R")
    return out
