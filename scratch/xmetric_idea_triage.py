"""Event-study triage of cross-sectional metric filter combos.

Imports build_xsec_metric_matrices from pages/backtester.py (the shipped
implementation) so ranks match exactly what the UI computes. Signals at close
D, forward returns Close[D] -> Close[D+k]. Excess = signal fwd ret minus the
same-date universe mean fwd ret (removes market beta / era drift).
Episode dedup: a ticker cannot re-signal within 21 trading days.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pages"))

from backtester import build_xsec_metric_matrices  # noqa: E402
from strategy_config import CSV_UNIVERSE  # noqa: E402

HORIZONS = [5, 21, 63]
DEDUP_TD = 21
START = "2005-01-01"

print("Loading master_prices...")
raw = pd.read_parquet(ROOT / "data" / "master_prices.parquet")
raw["date"] = pd.to_datetime(raw["date"])
universe = sorted(set(CSV_UNIVERSE))
raw = raw[raw["ticker"].isin(universe)]
print(f"rows={len(raw):,} tickers={raw['ticker'].nunique()} span={raw['date'].min().date()}..{raw['date'].max().date()}")

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
    {"metric": "sigma_mad", "window": 63},
    {"metric": "autocorr", "window": 63},
    {"metric": "dvol_roc"},
    {"metric": "rvol_roc"},
]
print("Building metric rank matrices (shipped builder)...")
mats = build_xsec_metric_matrices(data_dict, specs)
# second pass for a longer autocorr window, renamed locally
mats["autocorr126"] = build_xsec_metric_matrices(data_dict, [{"metric": "autocorr", "window": 126}])["autocorr"]
for k, m in mats.items():
    print(f"  {k}: {m.shape}")

close = pd.DataFrame({t: d["Close"] for t, d in data_dict.items()})
dates = close.index
mats = {k: m.reindex(dates) for k, m in mats.items()}

fwd = {k: close.shift(-k) / close - 1.0 for k in HORIZONS}

# breakout trigger: close at/above 21d rolling max (includes today)
bo21 = close >= close.rolling(21).max() - 1e-12

valid = mats["mom_12_1"].notna() & (dates.to_series() >= pd.Timestamp(START)).values[:, None]

COMBOS = {
    # user's consolidation-after-trend idea and variants
    "consol_user      mom[50,90] ac63>80 rvroc<20": [("mom_12_1", 50, 90), ("autocorr", 80, 100), ("rvol_roc", 0, 20)],
    "consol_loose     mom[50,90] ac63>70 rvroc<30": [("mom_12_1", 50, 90), ("autocorr", 70, 100), ("rvol_roc", 0, 30)],
    "consol_ac126     mom[50,90] ac126>80 rvroc<20": [("mom_12_1", 50, 90), ("autocorr126", 80, 100), ("rvol_roc", 0, 20)],
    "consol_no_ac     mom[50,90] rvroc<20": [("mom_12_1", 50, 90), ("rvol_roc", 0, 20)],
    "consol_sigma     mom[50,90] sigmad<20": [("mom_12_1", 50, 90), ("sigma_mad", 0, 20)],
    # coiled spring: stronger trend + compression
    "coil_rvroc       mom>80 rvroc<10": [("mom_12_1", 80, 100), ("rvol_roc", 0, 10)],
    "coil_sigma       mom>80 sigmad<20": [("mom_12_1", 80, 100), ("sigma_mad", 0, 20)],
    # tweet menu
    "mom_9dec         mom[80,90]": [("mom_12_1", 80, 90)],
    "mom_10dec        mom>90": [("mom_12_1", 90, 100)],
    "trendy_compress  ac63>90 sigmad<20": [("autocorr", 90, 100), ("sigma_mad", 0, 20)],
    "dvol_confirm     mom>70 dvroc>90": [("mom_12_1", 70, 100), ("dvol_roc", 90, 100)],
    "rvol_expand      mom>70 rvroc>90": [("mom_12_1", 70, 100), ("rvol_roc", 90, 100)],
}
# breakout-triggered versions of the consolidation setups
BO_COMBOS = {
    "consol_user+BO21": COMBOS["consol_user      mom[50,90] ac63>80 rvroc<20"],
    "consol_no_ac+BO21": COMBOS["consol_no_ac     mom[50,90] rvroc<20"],
    "consol_sigma+BO21": COMBOS["consol_sigma     mom[50,90] sigmad<20"],
    "coil_sigma+BO21": COMBOS["coil_sigma       mom>80 sigmad<20"],
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
    for k in HORIZONS:
        f = fwd[k]
        base_date_mean = f[valid].mean(axis=1)  # universe mean fwd ret per date
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
            "combo": label, "k": k, "N": len(vals), "dates": len(by_date),
            "raw_mean%": 100 * raw_vals.mean(), "exc_mean%": 100 * vals.mean(),
            "exc_med%": 100 * vals.median(), "hit_exc%": 100 * (vals > 0).mean(),
            "t_clust": t_clust, "t_nonovl": t_sub,
            "exc_pre16%": 100 * pre16.mean() if len(pre16) else np.nan,
            "exc_post16%": 100 * post16.mean() if len(post16) else np.nan,
        })


rows = []
for label, spec in COMBOS.items():
    stats_for(band_mask(spec), label, rows)
for label, spec in BO_COMBOS.items():
    stats_for(band_mask(spec) & bo21, label, rows)

out = pd.DataFrame(rows)
pd.set_option("display.width", 250)
pd.set_option("display.float_format", lambda x: f"{x:8.2f}")
print(out.to_string(index=False))
out.to_csv(ROOT / "scratch" / "xmetric_triage_results.csv", index=False)
print("\nSaved scratch/xmetric_triage_results.csv")
