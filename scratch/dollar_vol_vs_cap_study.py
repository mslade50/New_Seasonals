"""Event study: forward returns after $ volume exceeds market cap.

Three tiers, per ticker-day:
  daily   — 1d dollar volume > market cap
  weekly  — trailing 5d summed dollar volume > market cap
  monthly — trailing 21d summed dollar volume > market cap

Dollar volume uses the adjusted-close basis of master_prices; market cap is
FMP raw. The ratio therefore understates turnover for heavy dividend payers
in old history (conservative: fewer triggers, never phantom ones).

Episodes are deduped per (ticker, tier): a trigger only starts a new episode
if the tier hasn't fired in the prior 63 trading days. Forward returns are
close-to-close from the trigger close, absolute and SPY-excess.
"""
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

HORIZONS = (1, 5, 10, 21, 63)
COOLDOWN_TD = 63
MIN_MKTCAP = 50e6
MIN_PRICE = 1.0
MAX_RATIO = 25.0          # daily turnover above this is a data error
TIERS = {"daily": 1, "weekly": 5, "monthly": 21}


def load_prices() -> pd.DataFrame:
    df = pd.read_parquet(os.path.join(_ROOT, "data", "master_prices.parquet"),
                         columns=["ticker", "date", "Close", "Volume"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def spy_forward(spy_close: pd.Series) -> dict[int, pd.Series]:
    return {h: spy_close.shift(-h) / spy_close - 1.0 for h in HORIZONS}


def episode_starts(trigger: pd.Series) -> pd.Series:
    """True on trigger days with no trigger in the prior COOLDOWN_TD sessions."""
    prior = trigger.shift(1).rolling(COOLDOWN_TD, min_periods=1).max()
    return trigger & (prior.fillna(0) == 0)


def main() -> None:
    prices = load_prices()
    caps = pd.read_parquet(os.path.join(_ROOT, "scratch", "mktcap_history.parquet"))
    caps["date"] = pd.to_datetime(caps["date"]).dt.normalize()

    spy = (prices[prices["ticker"] == "SPY"]
           .set_index("date")["Close"].sort_index())
    spy_fwd = spy_forward(spy)

    events: list[dict] = []
    tickers = sorted(set(caps["ticker"]) & set(prices["ticker"]))
    print(f"tickers with both prices and market cap: {len(tickers)}")

    caps_by_ticker = dict(tuple(caps.groupby("ticker")))
    prices_by_ticker = dict(tuple(prices.groupby("ticker")))

    for ticker in tickers:
        px = (prices_by_ticker[ticker]
              .drop_duplicates("date", keep="last")
              .set_index("date").sort_index())
        cap = (caps_by_ticker[ticker]
               .drop_duplicates("date", keep="last")
               .set_index("date")["marketCap"].sort_index())
        cap = pd.to_numeric(cap, errors="coerce")
        cap = cap.reindex(px.index).ffill(limit=5)

        close = px["Close"]
        dv = close * px["Volume"]
        ratio = {1: dv / cap,
                 5: dv.rolling(5).sum() / cap,
                 21: dv.rolling(21).sum() / cap}
        fwd = {h: close.shift(-h) / close - 1.0 for h in HORIZONS}

        valid = (cap > MIN_MKTCAP) & (close > MIN_PRICE) & (ratio[1] < MAX_RATIO)
        for tier, window in TIERS.items():
            trigger = (ratio[window] > 1.0) & valid
            if not trigger.any():
                continue
            for date in px.index[episode_starts(trigger)]:
                row = {
                    "ticker": ticker, "date": date, "tier": tier,
                    "mktcap": cap.loc[date],
                    "ratio_1d": ratio[1].loc[date],
                    "ratio_5d": ratio[5].loc[date],
                    "ratio_21d": ratio[21].loc[date],
                    "day_ret": close.loc[date] / close.shift(1).loc[date] - 1.0,
                }
                for h in HORIZONS:
                    r = fwd[h].loc[date]
                    s = spy_fwd[h].get(date, np.nan)
                    row[f"fwd_{h}d"] = r
                    row[f"xs_{h}d"] = r - s if pd.notna(r) and pd.notna(s) else np.nan
                events.append(row)

    ev = pd.DataFrame(events)
    out = os.path.join(_ROOT, "scratch", "dollar_vol_vs_cap_events.parquet")
    ev.to_parquet(out, index=False)
    print(f"saved {len(ev)} episodes -> {out}\n")

    # Universe baseline: mean forward returns over all valid ticker-days
    # would be slow to do exactly; SPY-excess columns already control the
    # market drift, so report absolute + excess side by side.
    pd.set_option("display.width", 160)
    for tier in TIERS:
        sub = ev[ev["tier"] == tier]
        if sub.empty:
            print(f"== {tier}: no episodes")
            continue
        print(f"== {tier}: {len(sub)} episodes, {sub['ticker'].nunique()} tickers, "
              f"{sub['date'].dt.year.min()}-{sub['date'].dt.year.max()}")
        stats = {}
        for h in HORIZONS:
            r = sub[f"fwd_{h}d"].dropna()
            x = sub[f"xs_{h}d"].dropna()
            stats[f"{h}d"] = {
                "N": len(r),
                "mean%": r.mean() * 100, "med%": r.median() * 100,
                "win%": (r > 0).mean() * 100,
                "xs_mean%": x.mean() * 100, "xs_med%": x.median() * 100,
                "xs_t": x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 2 else np.nan,
            }
        print(pd.DataFrame(stats).T.round(2))
        by_year = sub.groupby(sub["date"].dt.year).size()
        print("episodes by year:", by_year.to_dict())
        print()


if __name__ == "__main__":
    main()
