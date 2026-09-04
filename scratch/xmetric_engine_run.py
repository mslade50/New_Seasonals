"""Run the volume-confirmed momentum candidates through the real backtester
engine (pages/backtester.run_engine) headlessly, full CSV_UNIVERSE.

Configs: T+1 Open entry, 2 ATR stop, time exit, 2 bps slippage, default
liquidity floors (min $10 / 100k vol). Filters expressed exactly as the UI
would (use_xmetric_filter + xmetric_filters), matrices from the shipped
builder.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pages"))

import backtester as bt  # noqa: E402
from strategy_config import CSV_UNIVERSE  # noqa: E402

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

sznl_map = bt.load_seasonal_map()

MOM90_DV90 = [
    {"metric": "mom_12_1", "window": 63, "logic": ">", "thresh": 90.0, "thresh_max": 100.0, "consecutive": 1},
    {"metric": "dvol_roc", "window": 63, "logic": ">", "thresh": 90.0, "thresh_max": 100.0, "consecutive": 1},
]
MOM90_ONLY = [
    {"metric": "mom_12_1", "window": 63, "logic": ">", "thresh": 90.0, "thresh_max": 100.0, "consecutive": 1},
]

print("Building metric matrices...")
mats = bt.build_xsec_metric_matrices(data_dict, MOM90_DV90)


def base_params(hold_days, xmetric_filters):
    return {
        "use_max_daily_risk": False, "max_daily_risk_pct": 3.0,
        "backtest_start_date": START, "trade_direction": "Long", "max_one_pos": True,
        "allow_same_day_reentry": False, "max_daily_entries": 20, "max_total_positions": 99,
        "use_stop_loss": True, "use_take_profit": False, "time_exit_only": False,
        "use_partial_exits": False, "partial_target_fraction": 0.5,
        "use_eod_dd_exit": False, "eod_dd_atr": 0.25, "eod_dd_weekdays": [],
        "stop_atr": 2.0, "tgt_atr": 8.0, "holding_days": hold_days,
        "entry_type": "T+1 Open", "use_ma_entry_filter": False, "require_close_gt_open": False,
        "use_intraday": False,
        "use_trailing_stop": False, "trail_atr": 2.0, "trail_anchor": "Peak High",
        "breakout_mode": "None", "use_range_filter": False, "range_min": 0.0, "range_max": 100.0,
        "use_dow_filter": False, "allowed_days": [],
        "allowed_cycles": [0, 1, 2, 3], "excluded_years": [],
        "min_price": 10.0, "min_vol": 100000, "max_vol": 0,
        "min_age": 0.25, "max_age": 100.0, "min_atr_pct": 0.2, "max_atr_pct": 10.0,
        "trend_filter": "None", "universe_tickers": list(data_dict.keys()),
        "slippage_bps": 2, "entry_conf_bps": 0,
        "perf_filters": [], "perf_atr_filters": [], "perf_first_instance": False, "perf_lookback": 21,
        "use_atr_ret_filter": False, "atr_ret_min": 0.0, "atr_ret_max": 10.0,
        "use_range_atr_filter": False, "range_atr_logic": ">", "range_atr_min": 0.0, "range_atr_max": 10.0,
        "use_open_gap_atr_filter": False, "open_gap_atr_logic": ">", "open_gap_atr_min": 0.0, "open_gap_atr_max": 10.0,
        "price_action_filters": [], "ma_consec_filters": [],
        "use_sznl": False, "sznl_logic": "<", "sznl_thresh": 50.0, "sznl_first_instance": False, "sznl_lookback": 21,
        "use_market_sznl": False, "market_sznl_logic": "<", "market_sznl_thresh": 50.0,
        "use_52w": False, "52w_type": "New High", "use_ath": False, "ath_type": "New ATH",
        "52w_first_instance": False, "52w_lookback": 21, "52w_lag": 0, "52w_window": 252, "exclude_52w_high": False,
        "use_vix_filter": False, "vix_min": 0.0, "vix_max": 100.0,
        "use_recent_52w": False, "recent_52w_invert": False, "recent_52w_lookback": 21,
        "use_recent_52w_low": False, "recent_52w_low_invert": False, "recent_52w_low_lookback": 21,
        "vol_gt_prev": False, "use_vol": False, "vol_logic": ">", "vol_thresh": 1.0, "vol_thresh_max": 0.0,
        "use_vol_rank": False, "vol_rank_logic": "<", "vol_rank_thresh": 15.0,
        "use_ma_dist_filter": False, "dist_ma_type": "SMA50", "dist_logic": ">", "dist_min": 0.0, "dist_max": 10.0,
        "use_weekly_ma_pullback": False, "wma_type": "SMA", "wma_period": 10,
        "wma_min_ext_pct": 10.0, "wma_lookback_months": 6, "wma_touch_logic": "Touch",
        "use_volret_delta": False, "vrd_method": "Z-score diff", "vrd_rank_window": "Expanding",
        "vrd_vol_halflife": 20, "vrd_ret_horizon": 20, "vrd_delta_n": 5, "vrd_min_periods": 252,
        "vrd_pctile_min": 70.0, "vrd_pctile_max": 90.0,
        "use_tr_vcr_filter": False, "tr_vcr_metric": "VCR", "tr_vcr_window": 63, "tr_vcr_sample_freq": 5,
        "tr_vcr_min_periods": 252, "tr_vcr_rank_window": "Expanding", "tr_vcr_filter_mode": "Percentile rank",
        "tr_vcr_pctile_min": 70.0, "tr_vcr_pctile_max": 100.0, "tr_vcr_raw_min": 0.0, "tr_vcr_raw_max": 10.0,
        "tr_vcr_raw_logic": ">", "tr_vcr_regime_quadrants": [], "tr_vcr_min_consec": 1, "tr_vcr_consec_first": False,
        "use_gap_filter": False, "gap_lookback": 21, "gap_logic": ">", "gap_thresh": 3,
        "use_earnings_filter": False, "earnings_logic": "Not Between", "earnings_value": 0,
        "earnings_min": -10, "earnings_max": 10,
        "use_eps_surp_filter": False, "eps_surp_logic": ">", "eps_surp_min": 0.0, "eps_surp_max": 100.0,
        "use_rev_surp_filter": False, "rev_surp_logic": ">", "rev_surp_min": 0.0, "rev_surp_max": 100.0,
        "use_eps_yoy_filter": False, "eps_yoy_logic": ">", "eps_yoy_min": 0.0, "eps_yoy_max": 100.0,
        "use_rev_yoy_filter": False, "rev_yoy_logic": ">", "rev_yoy_min": 0.0, "rev_yoy_max": 100.0,
        "use_grades_filter": False, "grades_window_days": 21, "grades_logic": ">", "grades_thresh": 0,
        "use_acc_count_filter": False, "acc_count_window": 21, "acc_count_logic": ">", "acc_count_thresh": 3,
        "use_dist_count_filter": False, "dist_count_window": 21, "dist_count_logic": ">", "dist_count_thresh": 3,
        "use_acc_dist_v2": False,
        "use_t1_open_filter": False, "t1_open_filters": [],
        "use_recent_ath": False, "recent_ath_invert": False, "ath_lookback_days": 63,
        "use_ref_ticker_filter": False, "ref_ticker": "", "ref_filters": [],
        "use_xsec_filter": False, "xsec_filters": [],
        "use_xmetric_filter": True, "xmetric_filters": xmetric_filters,
        "atr_sznl_filters": [], "dial_filters": [],
    }


def summarize(name, trades_df, rejected_df, total_signals):
    if trades_df.empty:
        print(f"\n=== {name}: NO TRADES (signals={total_signals}) ===")
        return None
    t = trades_df.copy()
    t["ExitDate"] = pd.to_datetime(t["ExitDate"])
    t["EntryDate"] = pd.to_datetime(t["EntryDate"])
    r = t["R"].astype(float)
    wins = r[r > 0]
    losses = r[r <= 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else np.inf
    eq = r.groupby(t["ExitDate"].dt.to_period("D")).sum().cumsum()
    dd = (eq - eq.cummax()).min()
    yearly = r.groupby(t["ExitDate"].dt.year).sum()
    print(f"\n=== {name} ===")
    print(f"trades={len(t)}  rejected={len(rejected_df)}  signals={total_signals}")
    print(f"win%={100*(r>0).mean():.1f}  avgR={r.mean():+.3f}  medR={r.median():+.3f}  "
          f"totR={r.sum():+.1f}  PF={pf:.2f}  maxDD_R={dd:+.1f}")
    print(f"avg hold={((t['ExitDate']-t['EntryDate']).dt.days).mean():.1f}cd  "
          f"exits: {t['ExitReason'].value_counts().to_dict() if 'ExitReason' in t else 'n/a'}")
    neg_years = yearly[yearly < 0]
    print(f"yearly totR: {' '.join(f'{y}:{v:+.0f}' for y, v in yearly.items())}")
    print(f"negative years: {len(neg_years)}/{len(yearly)}")
    return t


CONFIGS = [
    ("mom90_dv90_h21", 21, MOM90_DV90),
    ("mom90_dv90_h63", 63, MOM90_DV90),
    ("mom90_only_h63", 63, MOM90_ONLY),
]

all_trades = {}
for name, hold, filts in CONFIGS:
    print(f"\n>>> running {name} ...")
    params = base_params(hold, filts)
    trades_df, rejected_df, total_signals = bt.run_engine(
        data_dict, params, sznl_map, xsec_metric_matrices=mats)
    t = summarize(name, trades_df, rejected_df, total_signals)
    if t is not None:
        all_trades[name] = t
        t.to_parquet(ROOT / "scratch" / f"xmetric_trades_{name}.parquet")

print("\nDone.")
