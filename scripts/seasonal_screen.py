"""seasonal_screen.py - in-repo generator for pages/seasonal_sigs.py.

Replaces the retired Dropbox/Sublime_Misc screener (dead since 2026-05-13 when
its host machine's Task Scheduler entry stopped running). Writes
seasonal_screener_results.csv at the repo root, which the Seasonal Signals
Streamlit page renders.

A ticker qualifies only when ALL THREE gates pass:

1. Seasonal strength - the blended seasonal rank (sznl_ranks.csv, forward-dated
   through year-end) reaches >= SEAS_HIGH (bull) or <= SEAS_LOW (bear) within
   the next LOOKAHEAD_ROWS trading days.
2. Win-rate quality - on at least one forward horizon (5d / 21d), the
   day-of-year-matched historical win rate clears WIN_MIN in BOTH cohorts:
   all prior years AND same presidential-cycle years (bear signals need the
   mirror image: pct positive <= 100 - WIN_MIN in both). Minimum samples
   MIN_N_ALL / MIN_N_CYC. This gate is the 2026-07-23 addition - the old
   screener only sanity-checked the median's sign, which let noisy names
   through.
3. Technical extension - ANY of the 5d/10d/21d trailing-return percentiles is
   < TECH_OVERSOLD (bull) or > TECH_OVERBOUGHT (bear). Relaxed from the old
   ALL-three requirement; the "Super" label survives for the all-three case.

Prices come cache-first from data/master_prices.parquet (+ overflow), with a
yfinance fallback for names the caches lack. All computations are relative
(percentile ranks, forward returns), so the adjusted basis is safe per the
dividend-adjustment rule in CLAUDE.md.

Output stamps `# as_of=YYYY-MM-DD` as the first line (comment) so the page can
show freshness even when zero names qualify. Run from anywhere:
    python scripts/seasonal_screen.py
"""

from __future__ import annotations

import os
import sys
import datetime as dt

import numpy as np
import pandas as pd

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts import seasonal_edge as se  # noqa: E402

RANKS_CSV = os.path.join(REPO_ROOT, "sznl_ranks.csv")
OUTPUT_CSV = os.path.join(REPO_ROOT, "seasonal_screener_results.csv")

# --- Gate thresholds (tune here) ---------------------------------------------
SEAS_HIGH = 90.0        # blended seasonal rank >= this within lookahead -> bull
SEAS_LOW = 10.0         # blended seasonal rank <= this within lookahead -> bear
LOOKAHEAD_ROWS = 3      # today + next 2 trading-day rank rows

WIN_MIN = 60.0          # required win% (both cohorts, >=1 horizon); bears mirror
FWD_HORIZONS = (5, 21)  # forward-return horizons the win gate may qualify on
MIN_N_ALL = 10          # min prior-year observations, all-years cohort
MIN_N_CYC = 3           # min prior-year observations, cycle cohort

TECH_OVERSOLD = 15.0    # any-window trailing-return percentile below -> bull ok
TECH_OVERBOUGHT = 85.0  # any-window trailing-return percentile above -> bear ok
TECH_WINDOWS = (5, 10, 21)

MIN_BARS = 252          # skip names with under a year of price history

LEGACY_COLUMNS = [
    "Ticker", "Date", "Type", "Seas_Score", "Tech_Streak_Days",
    "Curr_5d_Rank", "Curr_10d_Rank", "Curr_21d_Rank",
    "Seas_All_Avg_5d", "Seas_All_Med_5d", "Seas_All_Win_5d",
    "Seas_All_Avg_21d", "Seas_All_Med_21d", "Seas_All_Win_21d",
    "Seas_Cyc_Avg_5d", "Seas_Cyc_Med_5d", "Seas_Cyc_Win_5d",
    "Seas_Cyc_Avg_21d", "Seas_Cyc_Med_21d", "Seas_Cyc_Win_21d",
]
EXTRA_COLUMNS = [
    "Seas_All_N_5d", "Seas_All_N_21d", "Seas_Cyc_N_5d", "Seas_Cyc_N_21d",
    "Win_Horizons", "Trigger_Windows",
]
ALL_COLUMNS = LEGACY_COLUMNS + EXTRA_COLUMNS


def get_seasonal_candidates(today: pd.Timestamp) -> list[dict]:
    """Tickers whose blended rank hits the bull/bear zone in the lookahead."""
    ranks = pd.read_csv(RANKS_CSV, parse_dates=["Date"])
    ranks = ranks[ranks["Date"] >= today].sort_values(["ticker", "Date"])
    if ranks.empty:
        return []
    window = ranks.groupby("ticker").head(LOOKAHEAD_ROWS)
    agg = window.groupby("ticker")["seasonal_rank"].agg(["max", "min"])
    out = []
    for ticker, row in agg.iterrows():
        if row["max"] >= SEAS_HIGH:
            out.append({"ticker": str(ticker), "bias": "BULL", "seas_score": row["max"]})
        elif row["min"] <= SEAS_LOW:
            out.append({"ticker": str(ticker), "bias": "BEAR", "seas_score": row["min"]})
    return out


def fetch_prices(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Cache-first via seasonal_edge; yfinance fallback for missing names."""
    prices = se.load_prices(tickers)
    missing = [t for t in tickers if se._norm_ticker(t) not in prices]
    if missing:
        import yfinance as yf
        print(f"[INFO] {len(missing)} tickers not in caches; yfinance fallback: {missing[:10]}")
        for t in missing:
            try:
                df = yf.download(t, period="max", auto_adjust=True, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [str(c).capitalize() for c in df.columns]
                if not df.empty:
                    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
                    prices[se._norm_ticker(t)] = df
            except Exception as e:
                print(f"[WARN] yfinance failed for {t}: {e}")
    return prices


def tech_ranks_and_streak(px: pd.DataFrame, bias: str):
    """Current 5/10/21d return percentiles + trigger windows + streak.

    Percentile = full-history rank of the trailing return (same convention as
    the old screener). Trigger = ANY window in the zone; streak = consecutive
    sessions (ending today) the any-window trigger held.
    """
    close = px["Close"].dropna()
    rank_cols = {}
    for w in TECH_WINDOWS:
        ret = close.pct_change(w)
        rank_cols[w] = ret.rank(pct=True) * 100
    curr = {w: float(rank_cols[w].iloc[-1]) for w in TECH_WINDOWS}
    if any(np.isnan(v) for v in curr.values()):
        return curr, [], 0, False

    if bias == "BULL":
        in_zone = {w: rank_cols[w] < TECH_OVERSOLD for w in TECH_WINDOWS}
    else:
        in_zone = {w: rank_cols[w] > TECH_OVERBOUGHT for w in TECH_WINDOWS}

    trigger_windows = [w for w in TECH_WINDOWS if bool(in_zone[w].iloc[-1])]
    any_zone = pd.concat(in_zone.values(), axis=1).any(axis=1).values
    streak = 0
    for hit in any_zone[::-1]:
        if hit:
            streak += 1
        else:
            break
    all_zone_today = len(trigger_windows) == len(TECH_WINDOWS)
    return curr, trigger_windows, streak, all_zone_today


def window_stats(px: pd.DataFrame, asof: pd.Timestamp, cycle: int):
    """{horizon: {'all': stats|None, 'cyc': stats|None}} from seasonal_edge.

    Stats are simple returns; converted to % here. min_years=1 so we gate on n
    ourselves and can still report thin samples.
    """
    out = {}
    for h in FWD_HORIZONS:
        stats_all = se.seasonal_window_returns(px, asof, h, cycle_phase_filter=None, min_years=1)
        stats_cyc = se.seasonal_window_returns(px, asof, h, cycle_phase_filter=cycle, min_years=1)
        out[h] = {"all": stats_all, "cyc": stats_cyc}
    return out


def _metrics(stats: dict | None) -> dict:
    if not stats or stats.get("insufficient") or "mean" not in stats:
        n = int(stats["n"]) if stats and "n" in stats else 0
        return {"n": n, "avg": np.nan, "med": np.nan, "win": np.nan}
    n = stats["n"]
    win = 100.0 * stats["n_up"] / n if n else np.nan
    return {"n": n, "avg": 100.0 * stats["mean"], "med": 100.0 * stats["median"], "win": win}


def qualifying_horizons(stats: dict, bias: str) -> list[int]:
    """Horizons where win% clears the bar in BOTH cohorts with enough sample."""
    passed = []
    for h, cohorts in stats.items():
        a, c = _metrics(cohorts["all"]), _metrics(cohorts["cyc"])
        if a["n"] < MIN_N_ALL or c["n"] < MIN_N_CYC:
            continue
        if np.isnan(a["win"]) or np.isnan(c["win"]):
            continue
        if bias == "BULL":
            if a["win"] >= WIN_MIN and c["win"] >= WIN_MIN:
                passed.append(h)
        else:
            if a["win"] <= 100.0 - WIN_MIN and c["win"] <= 100.0 - WIN_MIN:
                passed.append(h)
    return passed


def run_screener() -> pd.DataFrame:
    today = pd.Timestamp.now().normalize()
    print(f"--- Seasonal screen {today.date()} ---")

    candidates = get_seasonal_candidates(today)
    print(f"Seasonal-zone candidates: {len(candidates)}")
    if not candidates:
        return pd.DataFrame(columns=ALL_COLUMNS)

    prices = fetch_prices([c["ticker"] for c in candidates])

    results = []
    for cand in candidates:
        ticker, bias = cand["ticker"], cand["bias"]
        px = prices.get(se._norm_ticker(ticker))
        if px is None or len(px["Close"].dropna()) < MIN_BARS:
            continue

        asof = px.index.max()
        curr, trigger_windows, streak, all_zone = tech_ranks_and_streak(px, bias)
        if not trigger_windows:
            continue

        stats = window_stats(px, asof, cycle=int(asof.year) % 4)
        win_horizons = qualifying_horizons(stats, bias)
        if not win_horizons:
            continue

        m = {h: {k: _metrics(v) for k, v in stats[h].items()} for h in FWD_HORIZONS}
        signal_type = f"Super {bias.title()}" if all_zone else bias.title()
        results.append({
            "Ticker": ticker,
            "Date": today.strftime("%Y-%m-%d"),
            "Type": signal_type,
            "Seas_Score": round(float(cand["seas_score"]), 1),
            "Tech_Streak_Days": streak,
            "Curr_5d_Rank": round(curr[5], 1),
            "Curr_10d_Rank": round(curr[10], 1),
            "Curr_21d_Rank": round(curr[21], 1),
            "Seas_All_Avg_5d": round(m[5]["all"]["avg"], 2),
            "Seas_All_Med_5d": round(m[5]["all"]["med"], 2),
            "Seas_All_Win_5d": round(m[5]["all"]["win"], 1),
            "Seas_All_Avg_21d": round(m[21]["all"]["avg"], 2),
            "Seas_All_Med_21d": round(m[21]["all"]["med"], 2),
            "Seas_All_Win_21d": round(m[21]["all"]["win"], 1),
            "Seas_Cyc_Avg_5d": round(m[5]["cyc"]["avg"], 2),
            "Seas_Cyc_Med_5d": round(m[5]["cyc"]["med"], 2),
            "Seas_Cyc_Win_5d": round(m[5]["cyc"]["win"], 1),
            "Seas_Cyc_Avg_21d": round(m[21]["cyc"]["avg"], 2),
            "Seas_Cyc_Med_21d": round(m[21]["cyc"]["med"], 2),
            "Seas_Cyc_Win_21d": round(m[21]["cyc"]["win"], 1),
            "Seas_All_N_5d": m[5]["all"]["n"],
            "Seas_All_N_21d": m[21]["all"]["n"],
            "Seas_Cyc_N_5d": m[5]["cyc"]["n"],
            "Seas_Cyc_N_21d": m[21]["cyc"]["n"],
            "Win_Horizons": ",".join(f"{h}d" for h in win_horizons),
            "Trigger_Windows": ",".join(f"{w}d" for w in trigger_windows),
        })

    df = pd.DataFrame(results, columns=ALL_COLUMNS)
    if not df.empty:
        df = df.sort_values(["Type", "Seas_Score", "Tech_Streak_Days"],
                            ascending=[True, False, False]).reset_index(drop=True)
    return df


def write_output(df: pd.DataFrame) -> None:
    # As-of = latest priced trading day, not wall clock, so the stamp advances
    # once per session and re-runs on the same day are commit no-ops.
    today = pd.Timestamp.now().normalize()
    spx = se.load_one_price("^GSPC")
    asof = min(today, spx.index.max()) if spx is not None and not spx.empty else today
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        f.write(f"# as_of={asof.date()}\n")
        df.to_csv(f, index=False)
    print(f"Wrote {len(df)} signal(s) -> {OUTPUT_CSV}")
    if not df.empty:
        print(df[["Ticker", "Type", "Seas_Score", "Win_Horizons", "Trigger_Windows",
                  "Curr_5d_Rank", "Curr_10d_Rank", "Curr_21d_Rank"]].to_string(index=False))


if __name__ == "__main__":
    write_output(run_screener())
