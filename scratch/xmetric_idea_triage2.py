"""Round 2: refine the promising combos from xmetric_idea_triage.py.

Focus: mom_10dec and dvol_confirm. Adds a 5d cross-sectional return rank
(the page's existing XSec filter, so combos stay UI-expressible), pullback
vs breakout entries, vol-regime splits, and a 126d horizon.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pages"))

from backtester import build_xsec_metric_matrices, build_xsec_rank_matrices  # noqa: E402
from strategy_config import CSV_UNIVERSE  # noqa: E402

HORIZONS = [21, 63, 126]
DEDUP_TD = 21
START = "2005-01-01"

print("Loading master_prices...")
raw = pd.read_parquet(ROOT / "data" / "master_prices.parquet")
raw["date"] = pd.to_datetime(raw["date"])
raw = raw[raw["ticker"].isin(set(CSV_UNIVERSE))]

data_dict = {}
for tkr, g in raw.groupby("ticker"):
    df = g.set_index("date").sort_index()[["Open", "High", "Low", "Close", "Volume"]]
    df = df[~df.index.duplicated(keep="last")]
    if len(df) >= 300:
        data_dict[tkr] = df
print(f"data_dict: {len(data_dict)} tickers")

specs = [
    {"metric": "mom_12_1"},
    {"metric": "adr20"},
    {"metric": "dvol_roc"},
    {"metric": "rvol_roc"},
]
mats = build_xsec_metric_matrices(data_dict, specs)
# existing-page XSec 5d return rank (double-rank scheme) for pullback/strength timing
mats["xsec5"] = build_xsec_rank_matrices(data_dict, [5])[5]
print("matrices built")

close = pd.DataFrame({t: d["Close"] for t, d in data_dict.items()})
dates = close.index
mats = {k: m.reindex(dates) for k, m in mats.items()}
fwd = {k: close.shift(-k) / close - 1.0 for k in HORIZONS}
bo21 = close >= close.rolling(21).max() - 1e-12
valid = mats["mom_12_1"].notna() & (dates.to_series() >= pd.Timestamp(START)).values[:, None]

COMBOS = {
    # anchors from round 1
    "mom>90": [("mom_12_1", 90, 100)],
    "mom>70 dvroc>90": [("mom_12_1", 70, 100), ("dvol_roc", 90, 100)],
    # tighter / looser volume-confirmation
    "mom>80 dvroc>90": [("mom_12_1", 80, 100), ("dvol_roc", 90, 100)],
    "mom>90 dvroc>90": [("mom_12_1", 90, 100), ("dvol_roc", 90, 100)],
    "mom>70 dvroc>95": [("mom_12_1", 70, 100), ("dvol_roc", 95, 100)],
    "mom>70 dvroc[80,95]": [("mom_12_1", 70, 100), ("dvol_roc", 80, 95)],
    # entry timing inside top-decile momentum
    "mom>90 pullback x5<20": [("mom_12_1", 90, 100), ("xsec5", 0, 20)],
    "mom>90 strength x5>80": [("mom_12_1", 90, 100), ("xsec5", 80, 100)],
    "mom>70 dvroc>90 pullback x5<30": [("mom_12_1", 70, 100), ("dvol_roc", 90, 100), ("xsec5", 0, 30)],
    "mom>70 dvroc>90 strength x5>70": [("mom_12_1", 70, 100), ("dvol_roc", 90, 100), ("xsec5", 70, 100)],
    # vol-state modulation of top-decile momentum
    "mom>90 rvroc<50": [("mom_12_1", 90, 100), ("rvol_roc", 0, 50)],
    "mom>90 rvroc>50": [("mom_12_1", 90, 100), ("rvol_roc", 50, 100)],
    # tradeable-range overlay
    "mom>70 dvroc>90 adr>50": [("mom_12_1", 70, 100), ("dvol_roc", 90, 100), ("adr20", 50, 100)],
}
BO = {
    "mom>90 +BO21": COMBOS["mom>90"],
    "mom>70 dvroc>90 +BO21": COMBOS["mom>70 dvroc>90"],
}


def band_mask(spec):
    m = pd.DataFrame(True, index=dates, columns=close.columns)
    for key, lo, hi in spec:
        r = mats[key]
        m &= r.notna() & (r >= lo) & (r <= hi)
    return m & valid


def dedup(sig):
    prior = sig.shift(1).rolling(DEDUP_TD, min_periods=1).sum() > 0
    return sig & ~prior


def stats_for(sig, label, rows):
    sig = dedup(sig)
    n_per_yr = sig.sum().sum() / ((dates[-1] - pd.Timestamp(START)).days / 365.25)
    for k in HORIZONS:
        f = fwd[k]
        base_date_mean = f[valid].mean(axis=1)
        exc = f.sub(base_date_mean, axis=0)
        sel = exc[sig]
        vals = sel.stack().dropna()
        if len(vals) < 30:
            rows.append({"combo": label, "k": k, "N": len(vals)})
            continue
        by_date = sel.mean(axis=1).dropna()
        t_clust = by_date.mean() / (by_date.std(ddof=1) / np.sqrt(len(by_date)))
        sub = by_date.iloc[::k]
        t_sub = sub.mean() / (sub.std(ddof=1) / np.sqrt(len(sub))) if len(sub) > 10 else np.nan
        raw_vals = f[sig].stack().dropna()
        pre16 = vals[vals.index.get_level_values(0) < pd.Timestamp("2016-01-01")]
        post16 = vals[vals.index.get_level_values(0) >= pd.Timestamp("2016-01-01")]
        rows.append({
            "combo": label, "k": k, "N": len(vals), "sig/yr": n_per_yr,
            "raw_mean%": 100 * raw_vals.mean(), "exc_mean%": 100 * vals.mean(),
            "hit_exc%": 100 * (vals > 0).mean(),
            "t_clust": t_clust, "t_nonovl": t_sub,
            "exc_pre16%": 100 * pre16.mean() if len(pre16) else np.nan,
            "exc_post16%": 100 * post16.mean() if len(post16) else np.nan,
        })


rows = []
for label, spec in COMBOS.items():
    stats_for(band_mask(spec), label, rows)
for label, spec in BO.items():
    stats_for(band_mask(spec) & bo21, label, rows)

out = pd.DataFrame(rows)
pd.set_option("display.width", 250)
pd.set_option("display.float_format", lambda x: f"{x:8.2f}")
print(out.to_string(index=False))
out.to_csv(ROOT / "scratch" / "xmetric_triage2_results.csv", index=False)
print("\nSaved scratch/xmetric_triage2_results.csv")
