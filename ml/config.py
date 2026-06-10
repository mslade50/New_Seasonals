"""
ml/config.py — Single source of truth for the ML meta-labeling layer.

Everything here is pre-registered per docs/ml_meta_layer_plan.md (v4).
Changing hyperparameters or thresholds after looking at test-fold results
invalidates the evaluation — bump RUN_NOTE and rerun honestly if you do.
"""

import os

# -----------------------------------------------------------------------------
# Paths (repo root = parent of this file's directory)
# -----------------------------------------------------------------------------
ML_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(ML_PKG_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data")
ML_DATA_DIR = os.path.join(DATA_DIR, "ml")
MODELS_DIR = os.path.join(ML_DATA_DIR, "models")
SCORES_DIR = os.path.join(ML_DATA_DIR, "scores")
REPORTS_DIR = os.path.join(ML_DATA_DIR, "reports")

LEDGER_PATH = os.path.join(DATA_DIR, "backtest_trades_full.parquet")
MASTER_PRICES_PATH = os.path.join(DATA_DIR, "master_prices.parquet")
SZNL_CSV_PRIMARY = os.path.join(REPO_ROOT, "sznl_ranks.csv")
SZNL_CSV_BACKUP = os.path.join(REPO_ROOT, "seasonal_ranks.csv")
ATR_SZNL_PATH = os.path.join(REPO_ROOT, "atr_seasonal_ranks.parquet")
DATASET_PATH = os.path.join(ML_DATA_DIR, "dataset.parquet")

# -----------------------------------------------------------------------------
# Labels
# -----------------------------------------------------------------------------
R_WINSOR_LO = -3.0
R_WINSOR_HI = 8.0

# -----------------------------------------------------------------------------
# Features (pre-registered). NEVER include look-ahead columns from
# indicators.calculate_indicators (NextOpen, is_pivot_*, LastPivot*).
# -----------------------------------------------------------------------------
TICKER_FEATURES = [
    "rank_ret_2d", "rank_ret_5d", "rank_ret_10d", "rank_ret_21d",
    "rank_ret_126d", "rank_ret_252d",
    "rank_ret_atr_5d", "rank_ret_atr_21d",
    "ATR_Pct", "today_return_atr", "range_in_atr", "RangePct",
    "vol_ratio_10d_rank",
    "close_vs_sma50_atr", "close_vs_sma200_atr", "sma50_gt_sma200",
    "pct_off_52w_high",
    "Sznl",
]
ATR_SZNL_FEATURES = ["atr_sznl_5d", "atr_sznl_21d", "atr_sznl_63d", "atr_sznl_252d"]
MARKET_FEATURES = [
    "vix_close", "vix_5d_chg", "vix_term",
    "spx_vs_sma200_pct", "spx_ret_21d_rank", "spx_rv_21d",
    "breadth_pct_above_200d", "hyg_ief_z63",
]
META_NUMERIC_FEATURES = ["hold_days_target", "stop_atr", "tgt_atr", "dow", "month"]
CATEGORICAL_FEATURES = ["Strategy", "Direction", "Tier"]

ALL_FEATURES = (
    TICKER_FEATURES + ATR_SZNL_FEATURES + MARKET_FEATURES
    + META_NUMERIC_FEATURES + CATEGORICAL_FEATURES
)
# Sensitivity-run ablation: everything seasonal
SEASONAL_FEATURES = ["Sznl"] + ATR_SZNL_FEATURES

# Market context tickers (all verified present in master_prices.parquet)
SECTOR_ETFS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
MARKET_TICKERS = ["^GSPC", "^VIX", "^VIX3M", "HYG", "IEF"] + SECTOR_ETFS

# -----------------------------------------------------------------------------
# Model (pre-registered — deliberately small for N ~ 3.4k)
# -----------------------------------------------------------------------------
SEED = 7
HGB_PARAMS = dict(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=15,
    min_samples_leaf=40,
    l2_regularization=1.0,
    early_stopping=True,
    validation_fraction=0.15,
    random_state=SEED,
)
# Fraction of each training fold used to fit the model; the time-ordered
# remainder fits the isotonic calibrator and the decision thresholds.
CALIB_FRACTION = 0.25

# -----------------------------------------------------------------------------
# Walk-forward CV (pre-registered)
# -----------------------------------------------------------------------------
FIRST_TEST_YEAR = 2012
EMBARGO_TRADING_DAYS = 5

# -----------------------------------------------------------------------------
# Decision policy (pre-registered): calibrated p quantiles from the TRAIN-side
# calibration segment only. Never sizes above 1.0x.
# -----------------------------------------------------------------------------
SKIP_QUANTILE = 0.20   # p below train q20  -> SKIP (0.0x)
TRIM_QUANTILE = 0.50   # p below train q50  -> TRIM (0.5x)
TRIM_MULTIPLIER = 0.5

# Per-strategy deployment gates (conservative direction only: failing a gate
# means pass-through, i.e. no ML influence on that strategy)
MIN_STRATEGY_RETENTION = 0.40
MIN_STRATEGY_OOS_TRADES = 100

# -----------------------------------------------------------------------------
# Pre-registered ship/no-ship criteria (see plan section 7)
# -----------------------------------------------------------------------------
SHIP_MIN_UPLIFT_R = 0.05
SHIP_MIN_RETENTION = 0.60
SHIP_MIN_POSITIVE_YEARS_FRAC = 8 / 14
BOOTSTRAP_N = 10_000

# Monitoring. Daily batches share one value for market/calendar features
# (single scan date), which inflates PSI mechanically — exclude them from
# per-batch drift checks (ml.monitor covers them over trailing windows).
PSI_WARN = 0.20
PSI_MIN_BATCH = 50
DAILY_PSI_EXCLUDE = set(MARKET_FEATURES) | {"dow", "month"}

RUN_NOTE = ("run-2: market-feature ffill plumbing fix (rolling-window NaN "
            "holes); design unchanged from run-1")


def ensure_dirs():
    for d in (ML_DATA_DIR, MODELS_DIR, SCORES_DIR, REPORTS_DIR):
        os.makedirs(d, exist_ok=True)
