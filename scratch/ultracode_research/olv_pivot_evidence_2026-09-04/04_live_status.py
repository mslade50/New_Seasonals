"""Live status of the OLV pivot policy on the cache vs the corrected series.

Sessions: 2026-09-02 close (what the 2026-09-03 AM scan read) and 2026-09-03
close (what the 2026-09-03 PM and 2026-09-04 AM scans read). For every name
in the OLV liquid universe: band on the cache (full history, production) and
on the yf fully adjusted series; whether the OLV mask fired that session
(from candidates_pivot_audit.csv); which non-default names flip.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import data_provider  # noqa: E402
import strategy_config as sc  # noqa: E402
from olv_pivot_entry import causal_close_pivot_context, resolve_olv_pivot_entry_from_row  # noqa: E402
from yf_pull import load as load_yf, universe  # noqa: E402

POLICY = next(s for s in sc.STRATEGY_BOOK if s["name"] == "Oversold Low Volume")["execution"]["pivot_entry_policy"]
SESSIONS = {"2026-09-03_scan_AM": pd.Timestamp("2026-09-02"),
            "2026-09-03_scan_PM_and_2026-09-04_scan_AM": pd.Timestamp("2026-09-03")}


def atr14(df: pd.DataFrame) -> pd.Series:
    prev = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev).abs(), (df["Low"] - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(14).mean()


def resolve(df: pd.DataFrame, date: pd.Timestamp) -> dict | None:
    if date not in df.index:
        return None
    ctx = df.copy()
    piv = causal_close_pivot_context(ctx["Close"], 40, 40)
    for c in piv.columns:
        ctx[c] = piv[c]
    ctx["ATR"] = atr14(ctx)
    row = ctx.loc[date]
    return resolve_olv_pivot_entry_from_row(row, row["ATR"], POLICY)


def main() -> None:
    tickers = universe()
    cands = pd.read_csv(HERE / "candidates_pivot_audit.csv", parse_dates=["signal_date"])
    fired = set(zip(cands["t_clean"], cands["signal_date"]))
    yf = load_yf()
    yf_frames = {t: g.set_index("date").sort_index() for t, g in yf.groupby("ticker")}
    cache = data_provider.get_history(tickers, start="2000-01-01")
    rows = []
    for label, d in SESSIONS.items():
        for t in tickers:
            cf = cache.get(t)
            yfr = yf_frames.get(t)
            c = resolve(cf, d) if cf is not None else None
            y = resolve(yfr, d) if yfr is not None else None
            rows.append({
                "session_label": label, "close_date": d, "ticker": t,
                "fired_olv": (t, d) in fired,
                "cache_rule": c["matched_rule"] if c else "no_bar", "cache_distance": c["distance_atr"] if c else np.nan,
                "cache_nearest": c["nearest_type"] if c else "", "cache_level": c["nearest_level"] if c else np.nan,
                "cache_age": c["nearest_source_age_bars"] if c else np.nan,
                "yf_rule": y["matched_rule"] if y else "no_bar", "yf_distance": y["distance_atr"] if y else np.nan,
                "yf_nearest": y["nearest_type"] if y else "", "yf_level": y["nearest_level"] if y else np.nan,
                "yf_age": y["nearest_source_age_bars"] if y else np.nan,
                "close_cache": float(cf.loc[d, "Close"]) if cf is not None and d in cf.index else np.nan,
                "close_yf": float(yfr.loc[d, "Close"]) if yfr is not None and d in yfr.index else np.nan,
            })
    df = pd.DataFrame(rows)
    df["nondefault_cache"] = df["cache_rule"] != "default"
    df["flip"] = df["nondefault_cache"] & (df["cache_rule"] != df["yf_rule"])
    df["nondefault_either"] = (df["cache_rule"] != "default") | (df["yf_rule"] != "default")
    df.to_csv(HERE / "live_status.csv", index=False)
    summary = {}
    for label, g in df.groupby("session_label"):
        nd = g[g["nondefault_cache"]]
        summary[label] = {
            "close_date": str(g["close_date"].iloc[0].date()),
            "n_universe": int(len(g)), "n_no_bar_cache": int((g["cache_rule"] == "no_bar").sum()),
            "n_no_bar_yf": int((g["yf_rule"] == "no_bar").sum()),
            "n_nondefault_cache": int(len(nd)),
            "n_flips_among_nondefault": int(nd["flip"].sum()),
            "n_nondefault_yf": int((g["yf_rule"] != "default").sum()),
            "n_rule_changes_any": int((g["cache_rule"] != g["yf_rule"]).sum()),
            "fired_olv_names": sorted(g.loc[g["fired_olv"], "ticker"]),
            "fired_nondefault": sorted(g.loc[g["fired_olv"] & g["nondefault_cache"], "ticker"]),
            "fired_flips": sorted(g.loc[g["fired_olv"] & g["flip"], "ticker"]),
            "nondefault_cache": nd.sort_values("ticker")[["ticker", "cache_rule", "cache_distance", "yf_rule", "yf_distance", "fired_olv"]].round(3).to_dict("records"),
            "flips": nd[nd["flip"]].sort_values("ticker")[["ticker", "cache_rule", "cache_distance", "cache_nearest", "yf_rule", "yf_distance", "yf_nearest", "close_cache", "close_yf"]].round(3).to_dict("records"),
            "newly_nondefault_on_yf": g[(g["cache_rule"] == "default") & (g["yf_rule"] != "default")].sort_values("ticker")[["ticker", "cache_rule", "cache_distance", "yf_rule", "yf_distance"]].round(3).to_dict("records"),
        }
    (HERE / "live_status.json").write_text(json.dumps(summary, indent=2, default=str))
    for label, s in summary.items():
        print(label, {k: v for k, v in s.items() if not isinstance(v, list)})
        print("  flips:", s["flips"])


if __name__ == "__main__":
    main()
