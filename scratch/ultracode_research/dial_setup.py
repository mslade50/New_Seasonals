"""Shared setup for dial-design study: join non-OVS trades to each dial's
smoothed fragility score as-of signal date (live convention: 10d MA,
ffill limit 5)."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DIALS = ["5d", "21d", "63d"]


def load_frag() -> pd.DataFrame:
    frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
    frag.index = pd.to_datetime(frag.index).normalize()
    return frag


def smooth(frag: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    out = {}
    for d in DIALS:
        s = frag[d].dropna().rolling(window, min_periods=1).mean()
        out[d] = s
    return pd.DataFrame(out)


def load_trades_joined(ma_window: int = 10) -> pd.DataFrame:
    trades = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
    trades["Signal Date"] = pd.to_datetime(trades["Signal Date"]).dt.normalize()
    frag = load_frag()
    sm = smooth(frag, ma_window)
    start = sm.index.min() + pd.Timedelta(days=30)  # MA warm-up
    t = trades[trades["Signal Date"] >= start].copy().sort_values("Signal Date")
    for d in DIALS:
        s = sm[d].dropna().rename(f"frag_{d}").reset_index()
        s.columns = ["Date", f"frag_{d}"]
        t[f"frag_{d}"] = pd.merge_asof(
            t[["Signal Date"]], s, left_on="Signal Date", right_on="Date",
            tolerance=pd.Timedelta(days=7),
        )[f"frag_{d}"].values
    t = t.dropna(subset=[f"frag_{d}" for d in DIALS] + ["R_Multiple"])
    t["is_ovs"] = t["Strategy"].str.contains("Overbot Vol|OVS", case=False, na=False)
    t["ym"] = t["Signal Date"].dt.to_period("M")
    t["yr"] = t["Signal Date"].dt.year
    return t


def clustered_t(hi: pd.DataFrame, lo: pd.DataFrame, col: str = "R_Multiple"):
    """Monthly-clustered Welch t between two trade sets. Returns (t, p, nmo_hi, nmo_lo)."""
    from scipy import stats
    hm = hi.groupby("ym")[col].mean()
    lm = lo.groupby("ym")[col].mean()
    if len(hm) < 3 or len(lm) < 3:
        return np.nan, np.nan, len(hm), len(lm)
    tt = stats.ttest_ind(hm, lm, equal_var=False)
    return tt.statistic, tt.pvalue, len(hm), len(lm)
