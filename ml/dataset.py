"""
ml/dataset.py — Assemble the training table: trade ledger + point-in-time
features + labels. Output: data/ml/dataset.parquet.

CLI:  python -m ml.dataset
"""

import numpy as np
import pandas as pd

from ml import config, features, market_features

# Strategy/Direction/Tier live in the feature matrix (categorical features),
# so the meta side carries only what the features don't.
META_COLS = ["trade_id", "Ticker", "Signal Date", "Entry Date", "Exit Date",
             "Exit Type"]


def load_ledger(path: str = None) -> pd.DataFrame:
    df = pd.read_parquet(path or config.LEDGER_PATH)
    for c in ("Signal Date", "Entry Date", "Exit Date"):
        df[c] = pd.to_datetime(df[c]).dt.normalize()
    return df


def build_labels(ledger: pd.DataFrame) -> pd.DataFrame:
    """y_R = PnL_flat / Risk_flat (risk-normalized R), falling back to the
    ledger's R_Multiple where Risk is 0/NaN. y_win = y_R > 0."""
    risk = ledger["Risk_flat_750k"].replace(0, np.nan)
    y_r = ledger["PnL_flat_750k"] / risk
    y_r = y_r.fillna(ledger["R_Multiple"])
    out = pd.DataFrame(index=ledger.index)
    out["y_R"] = y_r
    out["y_R_winsor"] = y_r.clip(config.R_WINSOR_LO, config.R_WINSOR_HI)
    out["y_win"] = (y_r > 0).astype(int)
    return out


def build_dataset(ledger_path: str = None, save: bool = True,
                  verbose: bool = True) -> pd.DataFrame:
    ledger = load_ledger(ledger_path)

    # Drop unresolved trades (no exit yet) — labels undefined.
    open_mask = ledger["Exit Date"].isna() | ledger["R_Multiple"].isna()
    if open_mask.any() and verbose:
        print(f"[dataset] dropping {int(open_mask.sum())} open/unlabeled trades")
    ledger = ledger[~open_mask].reset_index(drop=True)

    if verbose:
        print(f"[dataset] {len(ledger)} trades "
              f"({ledger['Signal Date'].min().date()} -> {ledger['Signal Date'].max().date()})")
        print("[dataset] loading price/seasonal/market data ...")

    tickers = ledger["Ticker"].unique()
    price_map = features.load_price_map(tickers)
    sznl_map = features.load_sznl_map()
    atr_map = features.load_atr_sznl_map(tickers=tickers)
    mkt = market_features.get_market_frame()

    missing_px = [t for t in tickers if t not in price_map]
    if missing_px and verbose:
        n_aff = int(ledger["Ticker"].isin(missing_px).sum())
        print(f"[dataset] WARNING: {len(missing_px)} tickers missing from master_prices "
              f"({n_aff} trades will carry NaN ticker features): {missing_px[:10]}")

    if verbose:
        print("[dataset] computing point-in-time features ...")
    X = features.assemble_features(ledger, price_map, sznl_map, atr_map, mkt)
    y = build_labels(ledger)

    ds = pd.concat([ledger[META_COLS], X, y], axis=1)
    if save:
        config.ensure_dirs()
        ds.to_parquet(config.DATASET_PATH, index=False)
        if verbose:
            print(f"[dataset] saved {len(ds)} rows x {ds.shape[1]} cols -> {config.DATASET_PATH}")
    return ds


if __name__ == "__main__":
    build_dataset()
