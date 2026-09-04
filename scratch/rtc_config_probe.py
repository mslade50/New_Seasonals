"""rtc_config_probe.py — joint signal-configuration history for the risk-page
trade block. Recomputes the 7 daily_risk_report signals from today's code
(streamlit stubbed), builds a per-day configuration frame, saves it to
scratch/rtc_config_history.parquet, and prints frequency tables.

Config semantics per signal, per day (tri-state):
  ON  — signal_history True today
  RCT — an activation (False->True episode start) occurred within the
        trailing 5 trading days, but the signal is not ON today
  off — neither
Context flags: SPY close >= 98% of trailing-252d max; SPY > 200d SMA.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)


class _NoOpStreamlit:
    def __getattr__(self, name):
        return self

    def __call__(self, *a, **k):
        return self

    def __bool__(self):
        return False

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cache_data(self, *a, **k):
        def deco(fn):
            return fn
        return deco

    cache_resource = cache_data


if "streamlit" not in sys.modules or not hasattr(sys.modules["streamlit"], "columns"):
    sys.modules["streamlit"] = _NoOpStreamlit()

ABBR = {
    "Distribution Dominance": "DA",
    "VIX Range Compression": "VRC",
    "Defensive Leadership": "DL",
    "Pre-FOMC Rally": "FOMC",
    "Low Absorption Ratio": "AR",
    "Seasonal Rank Divergence": "SRD",
    "Dispersion": "DISP",
}
OUT = os.path.join(_ROOT, "scratch", "rtc_config_history.parquet")


def main() -> None:
    from pages.risk_dashboard_v2 import load_cached_data
    from daily_risk_report import compute_all_signals

    spy_df, closes, sp500_closes = load_cached_data()
    if spy_df is None:
        raise RuntimeError("rd2 caches missing — run refresh_all_data first")
    print(f"cache: spy {spy_df.index[0].date()} -> {spy_df.index[-1].date()}, "
          f"{len(spy_df)} rows")

    # rd2_spy_ohlc can lag the closes cache (stale partial refresh). Splice
    # the missing tail from master_prices (same adjusted basis).
    closes_end = closes.index[-1]
    if spy_df.index[-1] < closes_end:
        mp = pd.read_parquet(os.path.join(_ROOT, "data", "master_prices.parquet"))
        spy_mp = mp[mp["ticker"] == "SPY"].copy()
        spy_mp["date"] = pd.to_datetime(spy_mp["date"])
        spy_mp = spy_mp.set_index("date").sort_index()
        tail = spy_mp.loc[spy_mp.index > spy_df.index[-1],
                          ["Open", "High", "Low", "Close", "Volume"]]
        if not tail.empty:
            spy_df = pd.concat([spy_df, tail[spy_df.columns.intersection(tail.columns)]])
            print(f"  spliced {len(tail)} SPY rows from master_prices "
                  f"-> now ends {spy_df.index[-1].date()}")

    computed = compute_all_signals(spy_df, closes, sp500_closes)
    signals_ordered = computed["signals_ordered"]
    spy_close = computed["spy_close"].dropna()

    hists: dict[str, pd.Series] = {}
    for name, sig in signals_ordered.items():
        h = sig.get("signal_history")
        if h is None or not hasattr(h, "empty") or h.empty:
            print(f"  !! {name}: NO signal_history")
            continue
        h = h.dropna().astype(bool)
        hists[ABBR[name]] = h
        print(f"  {ABBR[name]:5s} {name:26s} span {h.index[0].date()} -> "
              f"{h.index[-1].date()}  n_days_on={int(h.sum())}")

    # Common index: every signal defined (avoid missing-reads-as-off)
    start = max(h.index[0] for h in hists.values())
    idx = spy_close.index[spy_close.index >= start]
    idx = idx[idx <= min(h.index[-1] for h in hists.values())]
    print(f"\njoint index: {idx[0].date()} -> {idx[-1].date()} ({len(idx)} td)")

    df = pd.DataFrame(index=idx)
    for ab, h in hists.items():
        on = h.reindex(idx).fillna(False).astype(bool)
        starts = on & ~on.shift(1, fill_value=False)
        # activation within trailing 5 td (incl. today)
        act5 = starts.rolling(5, min_periods=1).max().astype(bool)
        df[f"on_{ab}"] = on
        df[f"rct5_{ab}"] = act5 & ~on  # recent activation but off now
        df[f"any_{ab}"] = on | act5

    on_cols = [c for c in df.columns if c.startswith("on_")]
    any_cols = [c for c in df.columns if c.startswith("any_")]
    df["n_on"] = df[on_cols].sum(axis=1)
    df["n_on_or_recent"] = df[any_cols].sum(axis=1)

    # SPY context
    spy = spy_close.reindex(idx)
    roll_max = spy_close.rolling(252, min_periods=100).max().reindex(idx)
    sma200 = spy_close.rolling(200, min_periods=100).mean().reindex(idx)
    df["near_52w_high"] = spy >= 0.98 * roll_max
    df["above_200d"] = spy > sma200

    def cfg(row) -> str:
        parts = []
        for ab in hists:
            if row[f"on_{ab}"]:
                parts.append(f"{ab}+")
            elif row[f"rct5_{ab}"]:
                parts.append(f"{ab}~")
        sig = ",".join(parts) if parts else "none"
        ctx = ("HI" if row["near_52w_high"] else "no-hi") + "/" + \
              ("200+" if row["above_200d"] else "200-")
        return f"[{sig}] {ctx}"

    df["config"] = df.apply(cfg, axis=1)
    df["spy_close"] = spy
    df.to_parquet(OUT)
    print(f"wrote {OUT} ({len(df)} rows)")

    n_cfg = df["config"].nunique()
    print(f"\ndistinct configurations: {n_cfg}")
    print(f"days with >=1 signal on:            {(df['n_on'] >= 1).sum():5d} "
          f"({(df['n_on'] >= 1).mean() * 100:.1f}%)")
    print(f"days with >=2 signals on:           {(df['n_on'] >= 2).sum():5d} "
          f"({(df['n_on'] >= 2).mean() * 100:.1f}%)")
    print(f"days with >=2 on-or-recent(5td):    {(df['n_on_or_recent'] >= 2).sum():5d} "
          f"({(df['n_on_or_recent'] >= 2).mean() * 100:.1f}%)")
    print(f"days with >=3 on-or-recent(5td):    {(df['n_on_or_recent'] >= 3).sum():5d} "
          f"({(df['n_on_or_recent'] >= 3).mean() * 100:.1f}%)")

    vc = df["config"].value_counts()
    print(f"\ntop 15 configurations by days (of {len(df)} td):")
    top = vc.head(15).rename("days").to_frame()
    top["pct"] = (top["days"] / len(df) * 100).round(1)
    pd.set_option("display.width", 160)
    print(top.to_string())

    # signal-only config (ignoring context) for a sense of the signal space
    sig_cfg = df["config"].str.extract(r"^\[(.*)\]")[0]
    print(f"\ndistinct signal-only configs (ignoring SPY context): {sig_cfg.nunique()}")
    print("\nconfigs among days with >=2 on-or-recent, top 10:")
    sub = df[df["n_on_or_recent"] >= 2]["config"].value_counts().head(10)
    print(sub.to_string())

    print(f"\ntoday ({idx[-1].date()}): {df['config'].iloc[-1]}  "
          f"n_on={int(df['n_on'].iloc[-1])} "
          f"n_on_or_recent={int(df['n_on_or_recent'].iloc[-1])}")


if __name__ == "__main__":
    main()
