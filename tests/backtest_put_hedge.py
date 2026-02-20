"""
Backtest: Systematic Put Hedge Overlay Driven by Fragility Score
================================================================
Tests a rule-based put-buying hedge that scales with the risk dashboard's
63-day fragility score. Higher fragility = larger delta-dollar hedge.

Hedge structure: 50/50 barbell of 40-delta (core protection) and 5-delta
(tail/crash insurance) 3-month SPY puts, rebalanced daily.

Multiple activation thresholds are tested simultaneously (50/60/70/75).

Known limitations:
1. No skew: BS with flat vol underprices OTM puts (especially 5-delta).
   Real 5-delta puts cost more due to volatility skew. Results are optimistic
   on the 5-delta leg.
2. No bid-ask spread: Real execution would cost ~$0.02-0.10 per contract.
3. VIX3M as proxy: Matches 3-month tenor well, but doesn't capture
   strike-dependent IV.
4. D/A elevated tier: Signal fire history doesn't distinguish base vs
   elevated. Uses base edge (2.42) — conservative.

Usage:
    python tests/backtest_put_hedge.py              # run backtest (needs signal history)
    python tests/backtest_put_hedge.py --generate   # generate signal history first, then backtest

Requires: data/signal_horizon_stats.json
          data/signal_fire_history.parquet (auto-generated with --generate)
"""

import os
import sys
import json
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
sys.path.insert(0, PROJECT_DIR)

SIGNAL_FIRE_HISTORY_PATH = os.path.join(DATA_DIR, "signal_fire_history.parquet")
HORIZON_STATS_PATH = os.path.join(DATA_DIR, "signal_horizon_stats.json")
FRAGILITY_OUTPUT_PATH = os.path.join(DATA_DIR, "fragility_63d_history.parquet")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PORTFOLIO_VALUE = 100_000  # Notional portfolio for sizing
PUT_TENOR_DAYS = 63        # 3 months in trading days
TRADING_DAYS_PER_YEAR = 252
REBALANCE_THRESHOLD = 0.05  # Only rebalance if gap > 5% of target

SIGNAL_NAMES = [
    "Distribution Dominance",
    "VIX Range Compression",
    "Defensive Leadership",
    "Pre-FOMC Rally",
    "Low Absorption Ratio",
    "Seasonal Rank Divergence",
]

THRESHOLDS = [50, 60, 70, 75]


# ===========================================================================
# Black-Scholes functions
# ===========================================================================

def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """European put price via Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_put_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Put delta (negative, -1 to 0)."""
    if T <= 0 or sigma <= 0:
        return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1.0


def find_strike_for_delta(
    S: float, T: float, r: float, sigma: float, target_delta: float
) -> float:
    """
    Find strike K such that put delta = target_delta.

    For puts, delta = N(d1) - 1, so d1 = norm.ppf(1 + target_delta).
    Solve for K from d1 definition.
    """
    if T <= 0 or sigma <= 0:
        return round(S)
    d1_target = norm.ppf(1.0 + target_delta)  # e.g., target=-0.40 -> ppf(0.60)
    # d1 = [ln(S/K) + (r + 0.5*sigma^2)*T] / (sigma*sqrt(T))
    # => ln(S/K) = d1 * sigma * sqrt(T) - (r + 0.5*sigma^2)*T
    # => K = S * exp(-(d1*sigma*sqrt(T) - (r + 0.5*sigma^2)*T))
    ln_s_over_k = d1_target * sigma * np.sqrt(T) - (r + 0.5 * sigma**2) * T
    K = S * np.exp(-ln_s_over_k)
    return round(K)


# ===========================================================================
# Data loading
# ===========================================================================

def load_signal_fire_history() -> pd.DataFrame:
    """Load boolean signal fire history from parquet."""
    if not os.path.exists(SIGNAL_FIRE_HISTORY_PATH):
        raise FileNotFoundError(
            f"Signal fire history not found at {SIGNAL_FIRE_HISTORY_PATH}.\n"
            "Run the risk dashboard (pages/risk_dashboard_v2.py) once first to generate it."
        )
    df = pd.read_parquet(SIGNAL_FIRE_HISTORY_PATH)
    df.index = pd.to_datetime(df.index)
    return df.astype(bool)


def load_horizon_stats() -> dict:
    """Load backtested signal horizon stats from JSON."""
    if not os.path.exists(HORIZON_STATS_PATH):
        raise FileNotFoundError(
            f"Horizon stats not found at {HORIZON_STATS_PATH}."
        )
    with open(HORIZON_STATS_PATH, "r") as f:
        return json.load(f)


def download_market_data(start_date: str, end_date: str) -> dict:
    """Download SPY, ^VIX3M, ^IRX via yfinance."""
    import yfinance as yf

    result = {}

    # SPY OHLC
    spy_raw = yf.download("SPY", start=start_date, end=end_date, progress=False)
    if isinstance(spy_raw.columns, pd.MultiIndex):
        spy_raw.columns = spy_raw.columns.get_level_values(0)
    spy_raw.columns = [c.capitalize() for c in spy_raw.columns]
    result["spy"] = spy_raw

    # VIX3M (3-month IV proxy)
    vix3m_raw = yf.download("^VIX3M", start=start_date, end=end_date, progress=False)
    if isinstance(vix3m_raw.columns, pd.MultiIndex):
        vix3m_raw.columns = vix3m_raw.columns.get_level_values(0)
    vix3m_raw.columns = [c.capitalize() for c in vix3m_raw.columns]
    result["vix3m"] = vix3m_raw["Close"]

    # IRX (risk-free rate proxy — 3-month T-bill rate)
    irx_raw = yf.download("^IRX", start=start_date, end=end_date, progress=False)
    if isinstance(irx_raw.columns, pd.MultiIndex):
        irx_raw.columns = irx_raw.columns.get_level_values(0)
    irx_raw.columns = [c.capitalize() for c in irx_raw.columns]
    result["irx"] = irx_raw["Close"] if not irx_raw.empty else None

    return result


# ===========================================================================
# Fragility score reconstruction (vectorized)
# ===========================================================================

def signal_edge(stats: dict, signal_key: str, horizon: str = "63d") -> float:
    """Return the downside edge (positive = worse) for a signal at a horizon."""
    sig = stats.get("signals", {}).get(signal_key, {})
    dm = sig.get("horizons", {}).get(horizon, {}).get("diff_mean", 0)
    if dm is None:
        return 0.0
    return max(0.0, -dm)


def compute_fragility_history(
    signal_fires: pd.DataFrame,
    spy_close: pd.Series,
    horizon_stats: dict,
) -> pd.Series:
    """
    Reconstruct historical 63d fragility score for every date.

    Replicates compute_horizon_fragility() from risk_dashboard_v2.py
    but vectorized across the full time series.

    Returns pd.Series of fragility scores (0-100), indexed by date.
    """
    horizon = "63d"
    h_days = 63

    # --- Price context vectors ---
    ret_12m = spy_close / spy_close.shift(252) - 1
    sma_200 = spy_close.rolling(200).mean()
    extension_200d = spy_close / sma_200 - 1
    high_52w = spy_close.rolling(252).max()
    drawdown = spy_close / high_52w - 1

    # --- Vectorized regime multiplier ---
    m = pd.Series(1.0, index=spy_close.index)

    # 12-month return component
    m = m + np.where(ret_12m > 0.25, 0.25,
            np.where(ret_12m > 0.15, 0.10,
            np.where(ret_12m < -0.05, -0.15, 0.0)))

    # Extension component
    m = m + np.where(extension_200d > 0.10, 0.25,
            np.where(extension_200d > 0.05, 0.10,
            np.where(extension_200d < -0.02, -0.15, 0.0)))

    # Drawdown component
    m = m + np.where(drawdown > -0.02, 0.10,
            np.where(drawdown < -0.10, -0.20, 0.0))

    regime_mult = m.clip(0.6, 1.8)

    # --- Signal edges at 63d horizon ---
    edges = {}
    for name in SIGNAL_NAMES:
        edges[name] = signal_edge(horizon_stats, name, horizon)

    # --- SPY percent from high (positive = below high) ---
    spy_pct_from_high = (-drawdown).clip(lower=0.0)

    # --- Signal decay weights (vectorized) ---
    # Align signal fires to spy_close index
    fires = signal_fires.reindex(spy_close.index).fillna(False).astype(bool)

    active_weight = pd.Series(0.0, index=spy_close.index)
    fomc_weight_series = pd.Series(0.0, index=spy_close.index)

    for name in SIGNAL_NAMES:
        if name not in fires.columns:
            continue

        edge = edges[name]
        if edge == 0.0:
            continue

        sig_on = fires[name]

        # Compute days since last fire (vectorized):
        # Use cumsum trick: mark fire events, forward-fill, count distance
        fire_dates = sig_on.astype(int)
        group = fire_dates.cumsum()
        days_since = group.groupby(group).cumcount()
        ever_fired = group > 0
        days_since = days_since.where(ever_fired, other=np.nan)

        remaining_frac = ((h_days - days_since) / h_days).clip(0.0, 1.0)
        spy_factor = (1.0 - spy_pct_from_high / 0.10).clip(0.0, 1.0)

        weight = np.where(
            sig_on,
            1.0,
            np.where(
                ever_fired & (remaining_frac > 0),
                remaining_frac * spy_factor,
                0.0,
            ),
        )

        active_weight += edge * weight

        if name == "Pre-FOMC Rally":
            fomc_weight_series = pd.Series(weight, index=spy_close.index)

    # --- Dynamic max_weight: exclude FOMC edge on days it can't contribute ---
    base_max = sum(e for n, e in edges.items() if n != "Pre-FOMC Rally")
    fomc_edge = edges.get("Pre-FOMC Rally", 0.0)
    max_weight = base_max + np.where(fomc_weight_series > 0, fomc_edge, 0.0)
    max_weight = np.maximum(max_weight, 1e-9)

    # --- Final fragility score ---
    fragility = ((active_weight / max_weight) * 80 * regime_mult).clip(0.0, 100.0)
    fragility.name = "fragility_63d"

    return fragility


# ===========================================================================
# Put position tracking
# ===========================================================================

@dataclass
class PutPosition:
    entry_date: pd.Timestamp
    strike: float
    expiry_date: pd.Timestamp
    entry_price: float  # per-share price (× 100 for per-contract)
    delta_type: int      # 40 or 5 (target delta at entry)
    contracts: int
    remaining_contracts: int = 0

    def __post_init__(self):
        if self.remaining_contracts == 0:
            self.remaining_contracts = self.contracts

    def days_remaining(self, current_date: pd.Timestamp) -> int:
        """Trading days remaining — approximate using calendar days × 252/365."""
        cal_days = (self.expiry_date - current_date).days
        return max(0, int(cal_days * 252 / 365))

    def T_remaining(self, current_date: pd.Timestamp) -> float:
        """Time to expiry in years."""
        return max(0.0, self.days_remaining(current_date) / TRADING_DAYS_PER_YEAR)

    def current_value(self, S: float, r: float, sigma: float, current_date: pd.Timestamp) -> float:
        """Current per-share put value."""
        T = self.T_remaining(current_date)
        if T <= 0:
            return max(self.strike - S, 0.0)
        return bs_put_price(S, self.strike, T, r, sigma)

    def current_delta_dollar(self, S: float, r: float, sigma: float, current_date: pd.Timestamp) -> float:
        """Absolute delta-dollar exposure of this position."""
        T = self.T_remaining(current_date)
        d = abs(bs_put_delta(S, self.strike, T, r, sigma))
        return d * 100 * self.remaining_contracts * S


# ===========================================================================
# Simulation engine
# ===========================================================================

@dataclass
class ThresholdPath:
    threshold: int
    positions: list = field(default_factory=list)
    total_premium_paid: float = 0.0
    total_proceeds: float = 0.0
    daily_records: list = field(default_factory=list)
    premium_40d: float = 0.0
    premium_5d: float = 0.0
    proceeds_40d: float = 0.0
    proceeds_5d: float = 0.0


def run_simulation(
    fragility: pd.Series,
    spy_close: pd.Series,
    vix3m: pd.Series,
    irx: pd.Series | None,
) -> list[ThresholdPath]:
    """
    Run the daily rebalancing simulation across all thresholds.

    Returns list of ThresholdPath objects with full results.
    """
    # Align all series to common dates
    common_idx = fragility.dropna().index.intersection(spy_close.dropna().index)
    common_idx = common_idx.intersection(vix3m.dropna().index)
    if irx is not None:
        common_idx = common_idx.intersection(irx.dropna().index)
    common_idx = common_idx.sort_values()

    if len(common_idx) < 252:
        raise ValueError(f"Only {len(common_idx)} common dates — need at least 252.")

    # Skip warmup period (need 252d for price context)
    sim_start = common_idx[0]

    paths = [ThresholdPath(threshold=t) for t in THRESHOLDS]

    for date in common_idx:
        S = float(spy_close.loc[date])
        sigma = float(vix3m.loc[date]) / 100.0  # VIX3M quoted in %, convert
        r = float(irx.loc[date]) / 100.0 if irx is not None and date in irx.index else 0.04
        frag = float(fragility.loc[date])

        for path in paths:
            _step_one_day(path, date, S, r, sigma, frag)

    return paths


def _step_one_day(
    path: ThresholdPath,
    date: pd.Timestamp,
    S: float,
    r: float,
    sigma: float,
    fragility: float,
):
    """Process one day for a single threshold path."""
    # 1. Expire puts
    expired_pnl = 0.0
    still_open = []
    for pos in path.positions:
        if pos.T_remaining(date) <= 0:
            intrinsic = max(pos.strike - S, 0.0)
            proceeds = intrinsic * 100 * pos.remaining_contracts
            path.total_proceeds += proceeds
            if pos.delta_type == 40:
                path.proceeds_40d += proceeds
            else:
                path.proceeds_5d += proceeds
            expired_pnl += proceeds
        else:
            still_open.append(pos)
    path.positions = still_open

    # 2. Compute target delta-dollar
    if fragility >= path.threshold:
        target_dd = (fragility / 100.0) * PORTFOLIO_VALUE
    else:
        target_dd = 0.0

    # 3. Current delta-dollar from open positions
    current_dd = sum(
        pos.current_delta_dollar(S, r, sigma, date) for pos in path.positions
    )
    current_dd_40 = sum(
        pos.current_delta_dollar(S, r, sigma, date)
        for pos in path.positions if pos.delta_type == 40
    )
    current_dd_5 = sum(
        pos.current_delta_dollar(S, r, sigma, date)
        for pos in path.positions if pos.delta_type == 5
    )

    # 4. Rebalance if gap is significant
    gap = target_dd - current_dd
    premium_today = 0.0
    proceeds_today = 0.0

    if target_dd > 0 and abs(gap) > REBALANCE_THRESHOLD * target_dd:
        if gap > 0:
            # Need more protection — buy puts
            # Split gap 50/50 between 40-delta and 5-delta legs
            gap_per_leg = gap / 2.0
            premium_today += _buy_puts(path, date, S, r, sigma, gap_per_leg, target_delta=-0.40, delta_type=40)
            premium_today += _buy_puts(path, date, S, r, sigma, gap_per_leg, target_delta=-0.05, delta_type=5)
        else:
            # Need less protection — sell oldest puts first
            proceeds_today += _sell_puts(path, date, S, r, sigma, abs(gap))

    elif target_dd == 0 and current_dd > 0:
        # Liquidate everything
        proceeds_today += _liquidate_all(path, date, S, r, sigma)

    # 5. Current portfolio value of puts
    put_value = sum(
        pos.current_value(S, r, sigma, date) * 100 * pos.remaining_contracts
        for pos in path.positions
    )

    # 6. Record daily state
    path.daily_records.append({
        "date": date,
        "spy_close": S,
        "fragility": fragility,
        "target_dd": target_dd,
        "current_dd": current_dd + gap if abs(gap) > REBALANCE_THRESHOLD * max(target_dd, 1) else current_dd,
        "premium_paid": premium_today,
        "proceeds_received": proceeds_today + expired_pnl,
        "put_portfolio_value": put_value,
        "n_positions": len(path.positions),
    })


def _buy_puts(
    path: ThresholdPath,
    date: pd.Timestamp,
    S: float,
    r: float,
    sigma: float,
    target_dd_for_leg: float,
    target_delta: float,
    delta_type: int,
) -> float:
    """Buy puts to fill target_dd_for_leg. Returns premium paid."""
    T = PUT_TENOR_DAYS / TRADING_DAYS_PER_YEAR
    K = find_strike_for_delta(S, T, r, sigma, target_delta)
    put_price = bs_put_price(S, K, T, r, sigma)
    put_d = abs(bs_put_delta(S, K, T, r, sigma))

    if put_d <= 0 or put_price <= 0:
        return 0.0

    # Delta-dollar per contract = |delta| × 100 × S
    dd_per_contract = put_d * 100 * S
    contracts = max(1, int(target_dd_for_leg / dd_per_contract))

    premium = put_price * 100 * contracts
    path.total_premium_paid += premium
    if delta_type == 40:
        path.premium_40d += premium
    else:
        path.premium_5d += premium

    expiry = date + pd.Timedelta(days=int(PUT_TENOR_DAYS * 365 / 252))
    path.positions.append(PutPosition(
        entry_date=date,
        strike=K,
        expiry_date=expiry,
        entry_price=put_price,
        delta_type=delta_type,
        contracts=contracts,
    ))

    return premium


def _sell_puts(
    path: ThresholdPath,
    date: pd.Timestamp,
    S: float,
    r: float,
    sigma: float,
    dd_to_sell: float,
) -> float:
    """Sell oldest puts first to reduce delta-dollar by dd_to_sell. Returns proceeds."""
    # Sort by entry date (oldest first)
    path.positions.sort(key=lambda p: p.entry_date)
    proceeds = 0.0
    remaining_to_sell = dd_to_sell
    still_open = []

    for pos in path.positions:
        if remaining_to_sell <= 0:
            still_open.append(pos)
            continue

        pos_dd = pos.current_delta_dollar(S, r, sigma, date)
        pos_value_per_share = pos.current_value(S, r, sigma, date)

        if pos_dd <= remaining_to_sell:
            # Sell entire position
            p = pos_value_per_share * 100 * pos.remaining_contracts
            proceeds += p
            path.total_proceeds += p
            if pos.delta_type == 40:
                path.proceeds_40d += p
            else:
                path.proceeds_5d += p
            remaining_to_sell -= pos_dd
        else:
            # Sell partial — reduce contract count
            frac = remaining_to_sell / pos_dd
            sell_contracts = max(1, int(pos.remaining_contracts * frac))
            p = pos_value_per_share * 100 * sell_contracts
            proceeds += p
            path.total_proceeds += p
            if pos.delta_type == 40:
                path.proceeds_40d += p
            else:
                path.proceeds_5d += p
            pos.remaining_contracts -= sell_contracts
            if pos.remaining_contracts > 0:
                still_open.append(pos)
            remaining_to_sell = 0

    path.positions = still_open
    return proceeds


def _liquidate_all(
    path: ThresholdPath,
    date: pd.Timestamp,
    S: float,
    r: float,
    sigma: float,
) -> float:
    """Liquidate all positions at current mark. Returns proceeds."""
    proceeds = 0.0
    for pos in path.positions:
        val = pos.current_value(S, r, sigma, date) * 100 * pos.remaining_contracts
        proceeds += val
        path.total_proceeds += val
        if pos.delta_type == 40:
            path.proceeds_40d += val
        else:
            path.proceeds_5d += val
    path.positions = []
    return proceeds


# ===========================================================================
# Episode analysis
# ===========================================================================

def find_drawdown_episodes(spy_close: pd.Series, min_drawdown: float = 0.05) -> list[dict]:
    """
    Find SPY drawdown episodes exceeding min_drawdown.

    Returns list of dicts with: peak_date, trough_date, peak_price,
    trough_price, drawdown_pct.
    """
    rolling_max = spy_close.expanding().max()
    dd_series = spy_close / rolling_max - 1

    episodes = []
    in_episode = False
    peak_date = None
    trough_date = None
    trough_dd = 0.0

    for i, (date, dd) in enumerate(dd_series.items()):
        if dd < -min_drawdown and not in_episode:
            # Start of episode
            in_episode = True
            # Find peak date (most recent new high)
            mask = rolling_max.loc[:date] == rolling_max.loc[date]
            peak_date = mask[mask].index[0]
            trough_date = date
            trough_dd = dd
        elif in_episode and dd < trough_dd:
            trough_date = date
            trough_dd = dd
        elif in_episode and dd > -min_drawdown * 0.5:
            # Episode over (recovered to half the threshold)
            episodes.append({
                "peak_date": peak_date,
                "trough_date": trough_date,
                "peak_price": float(spy_close.loc[peak_date]),
                "trough_price": float(spy_close.loc[trough_date]),
                "drawdown_pct": trough_dd,
            })
            in_episode = False
            trough_dd = 0.0

    # Handle ongoing episode
    if in_episode:
        episodes.append({
            "peak_date": peak_date,
            "trough_date": trough_date,
            "peak_price": float(spy_close.loc[peak_date]),
            "trough_price": float(spy_close.loc[trough_date]),
            "drawdown_pct": trough_dd,
        })

    return episodes


def analyze_episodes(
    episodes: list[dict],
    paths: list[ThresholdPath],
) -> pd.DataFrame:
    """
    For each drawdown episode and threshold, compute put payoff and net cost/benefit.
    """
    rows = []
    for ep in episodes:
        peak = ep["peak_date"]
        trough = ep["trough_date"]
        dd_pct = ep["drawdown_pct"]

        for path in paths:
            df = pd.DataFrame(path.daily_records)
            if df.empty:
                continue
            df["date"] = pd.to_datetime(df["date"])
            ep_mask = (df["date"] >= peak) & (df["date"] <= trough)
            ep_data = df[ep_mask]

            if ep_data.empty:
                continue

            premium_during = ep_data["premium_paid"].sum()
            proceeds_during = ep_data["proceeds_received"].sum()
            put_value_start = ep_data["put_portfolio_value"].iloc[0] if len(ep_data) > 0 else 0
            put_value_end = ep_data["put_portfolio_value"].iloc[-1] if len(ep_data) > 0 else 0
            active_days = (ep_data["n_positions"] > 0).sum()

            rows.append({
                "peak_date": peak.strftime("%Y-%m-%d"),
                "trough_date": trough.strftime("%Y-%m-%d"),
                "drawdown_pct": f"{dd_pct:.1%}",
                "threshold": path.threshold,
                "active_days": active_days,
                "premium_paid": premium_during,
                "proceeds": proceeds_during,
                "put_value_change": put_value_end - put_value_start,
                "net_benefit": proceeds_during - premium_during + (put_value_end - put_value_start),
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ===========================================================================
# Output formatting
# ===========================================================================

def print_summary(paths: list[ThresholdPath], n_years: float):
    """Print the main summary table."""
    print("\n" + "=" * 80)
    print("HEDGE OVERLAY BACKTEST RESULTS")
    print(f"Portfolio: ${PORTFOLIO_VALUE:,.0f} | Put tenor: {PUT_TENOR_DAYS}d | Barbell: 40d/5d 50/50")
    print(f"Period: {n_years:.1f} years | Rebalance threshold: {REBALANCE_THRESHOLD:.0%}")
    print("=" * 80)

    header = f"{'Threshold':>10} {'Tot Premium':>14} {'Tot Proceeds':>14} {'Net P&L':>12} {'Ann Cost %':>11} {'40d Premium':>13} {'5d Premium':>13}"
    print(header)
    print("-" * len(header))

    for path in paths:
        net = path.total_proceeds - path.total_premium_paid
        ann_cost_pct = (path.total_premium_paid / n_years) / PORTFOLIO_VALUE * 100 if n_years > 0 else 0

        print(
            f"{path.threshold:>9}% "
            f"${path.total_premium_paid:>12,.0f} "
            f"${path.total_proceeds:>12,.0f} "
            f"${net:>10,.0f} "
            f"{ann_cost_pct:>9.2f}% "
            f"${path.premium_40d:>11,.0f} "
            f"${path.premium_5d:>11,.0f}"
        )

    # Proceeds breakdown
    print()
    print("Proceeds breakdown (40d / 5d):")
    for path in paths:
        print(
            f"  {path.threshold}%: "
            f"40d=${path.proceeds_40d:,.0f}  "
            f"5d=${path.proceeds_5d:,.0f}"
        )


def print_episode_analysis(episode_df: pd.DataFrame):
    """Print episode-level analysis."""
    if episode_df.empty:
        print("\nNo drawdown episodes found.")
        return

    print("\n" + "=" * 80)
    print("DRAWDOWN EPISODE ANALYSIS (SPY drawdowns > 5%)")
    print("=" * 80)

    for (peak, trough, dd), group in episode_df.groupby(
        ["peak_date", "trough_date", "drawdown_pct"]
    ):
        print(f"\n  {peak} -> {trough}  (SPY {dd})")
        print(f"  {'Threshold':>10} {'Active Days':>12} {'Premium':>12} {'Proceeds':>12} {'Put dVal':>12} {'Net':>12}")
        print(f"  {'-'*72}")
        for _, row in group.iterrows():
            print(
                f"  {row['threshold']:>9}% "
                f"{row['active_days']:>11} "
                f"${row['premium_paid']:>10,.0f} "
                f"${row['proceeds']:>10,.0f} "
                f"${row['put_value_change']:>10,.0f} "
                f"${row['net_benefit']:>10,.0f}"
            )


def print_monthly_premium(paths: list[ThresholdPath]):
    """Print monthly premium breakdown for the first threshold."""
    path = paths[0]  # Lowest threshold = most active
    df = pd.DataFrame(path.daily_records)
    if df.empty:
        return
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M")

    monthly = df.groupby("month").agg(
        premium=("premium_paid", "sum"),
        proceeds=("proceeds_received", "sum"),
        avg_fragility=("fragility", "mean"),
        max_fragility=("fragility", "max"),
        active_days=("n_positions", lambda x: (x > 0).sum()),
    )

    print(f"\n{'=' * 80}")
    print(f"MONTHLY DETAIL — Threshold {path.threshold}%")
    print(f"{'=' * 80}")
    print(f"{'Month':>10} {'Premium':>12} {'Proceeds':>12} {'Net':>10} {'Avg Frag':>9} {'Max Frag':>9} {'Active':>7}")
    print("-" * 75)

    for period, row in monthly.iterrows():
        net = row["proceeds"] - row["premium"]
        if row["premium"] > 0 or row["proceeds"] > 0:
            print(
                f"{str(period):>10} "
                f"${row['premium']:>10,.0f} "
                f"${row['proceeds']:>10,.0f} "
                f"${net:>8,.0f} "
                f"{row['avg_fragility']:>8.1f} "
                f"{row['max_fragility']:>8.1f} "
                f"{row['active_days']:>6.0f}d"
            )


# ===========================================================================
# Charts
# ===========================================================================

def chart_fragility_vs_spy(fragility: pd.Series, spy_close: pd.Series) -> str:
    """
    Create dual-axis time series: fragility score (area) + SPY (line).
    Saves to data/fragility_vs_spy.html and returns the path.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Align to common dates
    common = fragility.dropna().index.intersection(spy_close.dropna().index).sort_values()
    frag = fragility.reindex(common)
    spy = spy_close.reindex(common)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Fragility area chart with color-coded fill
    # Build color segments based on fragility level
    fig.add_trace(
        go.Scatter(
            x=common,
            y=frag.values,
            name="Fragility Score",
            fill="tozeroy",
            fillcolor="rgba(255, 140, 0, 0.15)",
            line=dict(color="rgba(255, 140, 0, 0.8)", width=1),
            hovertemplate="Fragility: %{y:.1f}<extra></extra>",
        ),
        secondary_y=False,
    )

    # Threshold bands (horizontal reference lines)
    for thresh, color, label in [
        (50, "rgba(255, 215, 0, 0.4)", "50"),
        (70, "rgba(255, 69, 0, 0.4)", "70"),
    ]:
        fig.add_hline(
            y=thresh,
            line_dash="dot",
            line_color=color,
            line_width=1,
            annotation_text=label,
            annotation_position="left",
            secondary_y=False,
        )

    # SPY price line
    fig.add_trace(
        go.Scatter(
            x=common,
            y=spy.values,
            name="SPY",
            line=dict(color="#4A90D9", width=1.5),
            hovertemplate="SPY: $%{y:.2f}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title="63-Day Fragility Score vs SPY Price",
        template="plotly_dark",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(l=60, r=60, t=60, b=40),
    )
    fig.update_yaxes(
        title_text="Fragility (0-100)",
        range=[0, 105],
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="SPY Price",
        secondary_y=True,
    )

    out_path = os.path.join(DATA_DIR, "fragility_vs_spy.html")
    fig.write_html(out_path)
    return out_path


# ===========================================================================
# Signal history generation (headless dashboard import)
# ===========================================================================

def _import_dashboard_headless():
    """
    Import signal computation functions from risk_dashboard_v2.py
    without triggering Streamlit or the main() entry point.
    """
    import types
    from unittest.mock import MagicMock

    dashboard_path = os.path.join(PROJECT_DIR, "pages", "risk_dashboard_v2.py")
    with open(dashboard_path, "r") as f:
        source = f.read()

    # Remove the bare main() call at module bottom to prevent execution
    lines = source.split("\n")
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == "main()":
            lines[i] = "# main()  # disabled for headless import"
            break
    modified_source = "\n".join(lines)

    # Mock streamlit — decorators must be no-ops
    mock_st = MagicMock()
    mock_st.cache_data = lambda **kw: lambda f: f
    mock_st.cache_resource = lambda f: f
    saved_st = sys.modules.get("streamlit")
    saved_plotly = sys.modules.get("plotly")
    saved_go = sys.modules.get("plotly.graph_objects")
    sys.modules["streamlit"] = mock_st
    sys.modules["plotly"] = MagicMock()
    sys.modules["plotly.graph_objects"] = MagicMock()

    mod = types.ModuleType("risk_dashboard_v2")
    mod.__file__ = dashboard_path
    exec(compile(modified_source, dashboard_path, "exec"), mod.__dict__)

    # Restore original modules
    if saved_st is not None:
        sys.modules["streamlit"] = saved_st
    else:
        sys.modules.pop("streamlit", None)
    if saved_plotly is not None:
        sys.modules["plotly"] = saved_plotly
    if saved_go is not None:
        sys.modules["plotly.graph_objects"] = saved_go

    return mod


def generate_signal_fire_history(lookback_years: int = 10):
    """
    Download data and compute all 6 signals headlessly,
    then save signal_fire_history.parquet.
    """
    import yfinance as yf

    print("Importing dashboard signal functions (headless)...")
    rd = _import_dashboard_headless()

    start_date = (
        pd.Timestamp.now() - pd.Timedelta(days=lookback_years * 365)
    ).strftime("%Y-%m-%d")

    # Step 1: Download signal tickers
    print(f"Downloading {len(rd.ALL_SIGNAL_TICKERS)} signal tickers from {start_date}...")
    all_data = {}
    raw = yf.download(rd.ALL_SIGNAL_TICKERS, start=start_date, auto_adjust=True, threads=True)
    if raw is None or raw.empty:
        raise RuntimeError("yfinance download returned no data.")

    for ticker in rd.ALL_SIGNAL_TICKERS:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                df = raw.xs(ticker, level="Ticker", axis=1)
            else:
                df = raw.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.capitalize() for c in df.columns]
            df = df.dropna(how="all")
            if not df.empty:
                all_data[ticker] = df
        except Exception as e:
            print(f"  Warning: Could not process {ticker}: {e}")

    print(f"  Got data for {len(all_data)} tickers")

    if "SPY" not in all_data:
        raise RuntimeError("Could not download SPY data.")

    spy_df = all_data["SPY"]
    spy_close = spy_df["Close"]

    # Build closes DataFrame
    close_frames = {t: d["Close"] for t, d in all_data.items() if "Close" in d.columns}
    closes = pd.DataFrame(close_frames)

    # Step 2: Download S&P 500 closes for Defensive Leadership signal
    sp500_closes = None
    try:
        from abs_return_dispersion import SP500_TICKERS
        if len(SP500_TICKERS) > 50:
            print(f"Downloading S&P 500 ({len(SP500_TICKERS)} tickers) in batches...")
            sp500_frames = {}
            batch_size = 50
            for i in range(0, len(SP500_TICKERS), batch_size):
                batch = SP500_TICKERS[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(SP500_TICKERS) + batch_size - 1) // batch_size
                print(f"  Batch {batch_num}/{total_batches}...")
                try:
                    batch_raw = yf.download(batch, start=start_date, auto_adjust=True, threads=True)
                    if batch_raw is not None and not batch_raw.empty:
                        for t in batch:
                            try:
                                if isinstance(batch_raw.columns, pd.MultiIndex):
                                    s = batch_raw.xs(t, level="Ticker", axis=1)["Close"]
                                else:
                                    s = batch_raw["Close"]
                                if isinstance(s, pd.DataFrame):
                                    s = s.iloc[:, 0]
                                sp500_frames[t] = s.dropna()
                            except Exception:
                                pass
                except Exception as e:
                    print(f"  Batch {batch_num} failed: {e}")
            if sp500_frames:
                sp500_closes = pd.DataFrame(sp500_frames)
                print(f"  Got S&P 500 data: {sp500_closes.shape[1]} tickers")
    except ImportError:
        print("  abs_return_dispersion.py not found — skipping S&P 500 download")
        print("  (Defensive Leadership signal will be empty)")

    # Step 3: Compute signals
    print("Computing signals...")

    # 1. Distribution Dominance
    print("  Distribution Dominance...")
    da = rd.compute_da_signal(spy_df)

    # 2. VIX Range Compression
    print("  VIX Range Compression...")
    vix_close = closes["^VIX"].dropna() if "^VIX" in closes.columns else pd.Series(dtype=float)
    vrc = rd.compute_vix_range_compression(vix_close)

    # 3. Defensive Leadership
    print("  Defensive Leadership...")
    dl = rd.compute_defensive_leadership(sp500_closes, spy_close)

    # 4. Pre-FOMC Rally
    print("  Pre-FOMC Rally...")
    fomc = rd.compute_fomc_signal(spy_close)

    # 5. Low Absorption Ratio
    print("  Low Absorption Ratio...")
    sector_cols = [c for c in rd.SECTOR_ETFS if c in closes.columns]
    sector_returns = closes[sector_cols].pct_change().dropna(how="all")
    ar = rd.compute_low_ar_signal(sector_returns, spy_close)

    # 6. Seasonal Rank Divergence
    print("  Seasonal Rank Divergence...")
    srd = rd.compute_seasonal_divergence_signal(spy_close)

    signals_ordered = {
        "Distribution Dominance": da,
        "VIX Range Compression": vrc,
        "Defensive Leadership": dl,
        "Pre-FOMC Rally": fomc,
        "Low Absorption Ratio": ar,
        "Seasonal Rank Divergence": srd,
    }

    # Step 4: Save signal fire history
    print("Saving signal fire history...")
    rd.save_signal_fire_history(signals_ordered, spy_close)

    if os.path.exists(SIGNAL_FIRE_HISTORY_PATH):
        df = pd.read_parquet(SIGNAL_FIRE_HISTORY_PATH)
        print(f"  Saved: {df.shape[0]} dates × {df.shape[1]} signals")
        print(f"  Columns: {list(df.columns)}")
        for col in df.columns:
            n_fires = df[col].sum()
            print(f"    {col}: {n_fires} fire days ({n_fires/len(df)*100:.1f}%)")
    else:
        raise RuntimeError("Failed to save signal fire history.")


# ===========================================================================
# Main
# ===========================================================================

def main():
    # Check for --generate flag
    if "--generate" in sys.argv:
        generate_signal_fire_history()
        print()

    print("Loading signal fire history...")
    signal_fires = load_signal_fire_history()
    print(f"  Signals: {list(signal_fires.columns)}")
    print(f"  Date range: {signal_fires.index.min().date()} to {signal_fires.index.max().date()}")
    print(f"  {len(signal_fires)} trading days")

    print("\nLoading horizon stats...")
    horizon_stats = load_horizon_stats()

    # Print signal edges at 63d
    print("\n  63d signal edges:")
    total_edge = 0.0
    for name in SIGNAL_NAMES:
        e = signal_edge(horizon_stats, name, "63d")
        total_edge += e
        print(f"    {name}: {e:.2f}")
    print(f"    MAX_WEIGHT: {total_edge:.2f}")

    print("\nDownloading market data (SPY, ^VIX3M, ^IRX)...")
    start = (signal_fires.index.min() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    end = signal_fires.index.max().strftime("%Y-%m-%d")
    market = download_market_data(start, end)

    spy_close = market["spy"]["Close"]
    spy_close.index = pd.to_datetime(spy_close.index)
    # Flatten index if needed
    if hasattr(spy_close.index, 'tz') and spy_close.index.tz is not None:
        spy_close.index = spy_close.index.tz_localize(None)

    vix3m = market["vix3m"]
    vix3m.index = pd.to_datetime(vix3m.index)
    if hasattr(vix3m.index, 'tz') and vix3m.index.tz is not None:
        vix3m.index = vix3m.index.tz_localize(None)

    irx = market["irx"]
    if irx is not None:
        irx.index = pd.to_datetime(irx.index)
        if hasattr(irx.index, 'tz') and irx.index.tz is not None:
            irx.index = irx.index.tz_localize(None)

    print(f"  SPY: {len(spy_close)} days ({spy_close.index.min().date()} to {spy_close.index.max().date()})")
    print(f"  VIX3M: {len(vix3m)} days")
    if irx is not None:
        print(f"  IRX: {len(irx)} days")
    else:
        print("  IRX: unavailable, using r=0.04 fallback")

    print("\nComputing historical fragility scores...")
    fragility = compute_fragility_history(signal_fires, spy_close, horizon_stats)
    fragility = fragility.dropna()
    print(f"  Fragility computed for {len(fragility)} dates")
    print(f"  Mean: {fragility.mean():.1f}  Median: {fragility.median():.1f}  Max: {fragility.max():.1f}")
    print(f"  Days >= 50: {(fragility >= 50).sum()}  >= 60: {(fragility >= 60).sum()}  >= 70: {(fragility >= 70).sum()}  >= 75: {(fragility >= 75).sum()}")

    # Save fragility history
    fragility.to_frame().to_parquet(FRAGILITY_OUTPUT_PATH)
    print(f"\n  Saved to {FRAGILITY_OUTPUT_PATH}")

    # Chart: fragility vs SPY
    chart_path = chart_fragility_vs_spy(fragility, spy_close)
    print(f"  Chart saved to {chart_path}")

    print("\nRunning put hedge simulation...")
    paths = run_simulation(fragility, spy_close, vix3m, irx)

    n_years = len(fragility) / TRADING_DAYS_PER_YEAR

    # Summary
    print_summary(paths, n_years)

    # Monthly detail for lowest threshold
    print_monthly_premium(paths)

    # Episode analysis
    print("\nAnalyzing drawdown episodes...")
    episodes = find_drawdown_episodes(spy_close.reindex(fragility.index).dropna())
    print(f"  Found {len(episodes)} episodes with >5% drawdown")
    episode_df = analyze_episodes(episodes, paths)
    print_episode_analysis(episode_df)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
