"""Band stability of the OLV pivot policy under basis correction.

For every LIQUID-tier OLV signal in the last 3 years (from
candidates_pivot_audit.csv, script 02), compute the 40/40 closing-pivot
context, ATR and policy band on:
  (a) cache_full  - master_prices.parquet full history (production replay),
  (b) cache_win   - the same cache truncated to the yf window (same-window
                    control: isolates basis from window truncation),
  (c) yf          - yfinance auto_adjust=True, 3 years (fully adjusted).
Band = matched_rule (default / above_high_2_3 / above_high_4_5 /
above_high_gt5). The overflow tier is out of scope: the permitted pull is
capped at the OLV liquid universe.
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
from yf_pull import load as load_yf  # noqa: E402

POLICY = next(s for s in sc.STRATEGY_BOOK if s["name"] == "Oversold Low Volume")["execution"]["pivot_entry_policy"]
WINDOW_START = pd.Timestamp("2023-09-04")   # 3 years before the run date
FULL_CONTEXT_BARS = 300                     # 252 max source age + 40 right bars + slack


def atr14(df: pd.DataFrame) -> pd.Series:
    # indicators.py line ~184: simple 14-bar mean of the true range.
    prev = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev).abs(), (df["Low"] - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(14).mean()


def context(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    piv = causal_close_pivot_context(out["Close"], 40, 40)
    for c in piv.columns:
        out[c] = piv[c]
    out["ATR"] = atr14(out)
    return out


def resolve_at(ctx: pd.DataFrame, date: pd.Timestamp) -> dict | None:
    if date not in ctx.index:
        return None
    row = ctx.loc[date]
    res = resolve_olv_pivot_entry_from_row(row, row["ATR"], POLICY)
    res["pos"] = int(ctx.index.get_loc(date))
    res["close"] = float(row["Close"])
    res["atr"] = float(row["ATR"]) if pd.notna(row["ATR"]) else np.nan
    return res


def band(res: dict | None) -> str:
    if res is None:
        return "no_bar"
    return res["matched_rule"]


def main() -> None:
    cands = pd.read_csv(HERE / "candidates_pivot_audit.csv", parse_dates=["signal_date"])
    sig = cands[(cands["tier"] == "Liquid") & (cands["signal_date"] >= WINDOW_START)].copy()
    print(f"{len(sig)} liquid OLV signals since {WINDOW_START.date()} ({sig['t_clean'].nunique()} tickers)")

    yf = load_yf()
    yf_frames = {t: g.set_index("date").sort_index()[["Open", "High", "Low", "Close", "Volume"]]
                 for t, g in yf.groupby("ticker")}
    yf_start = yf["date"].min()
    cache = data_provider.get_history(sorted(sig["t_clean"].unique()), start="2000-01-01")

    rows = []
    for r in sig.itertuples(index=False):
        t, d = r.t_clean, pd.Timestamp(r.signal_date)
        cf = cache.get(t)
        yfr = yf_frames.get(t)
        if cf is None or yfr is None:
            rows.append({"ticker": t, "signal_date": d, "status": "missing_series"})
            continue
        cf = cf.copy()
        cf.columns = [c.capitalize() for c in cf.columns]
        full = resolve_at(context(cf), d)
        win = resolve_at(context(cf[cf.index >= yf_start]), d)
        yres = resolve_at(context(yfr), d)
        # cache/yf close ratio and its steps between the pivot source date and the signal
        ratio = (cf["Close"] / yfr["Close"]).dropna()
        step = ratio / ratio.shift(1) - 1.0
        piv_date = pd.Timestamp(win["nearest_date"]) if win and win["nearest_date"] is not None and pd.notna(win["nearest_date"]) else None
        seg = step[(step.index > piv_date) & (step.index <= d)] if piv_date is not None else step.iloc[0:0]
        big = seg.abs().idxmax() if len(seg) and seg.abs().max() > 1e-4 else None
        rows.append({
            "ticker": t, "signal_date": d, "status": "ok",
            "engine_matched_rule": r.matched_rule, "engine_distance_atr": r.distance_atr,
            "cache_full_rule": band(full), "cache_full_distance": full["distance_atr"] if full else np.nan,
            "cache_full_nearest": full["nearest_type"] if full else "", "cache_full_level": full["nearest_level"] if full else np.nan,
            "cache_win_rule": band(win), "cache_win_distance": win["distance_atr"] if win else np.nan,
            "cache_win_nearest": win["nearest_type"] if win else "", "cache_win_level": win["nearest_level"] if win else np.nan,
            "cache_win_pivot_date": piv_date, "cache_win_atr": win["atr"] if win else np.nan,
            "yf_rule": band(yres), "yf_distance": yres["distance_atr"] if yres else np.nan,
            "yf_nearest": yres["nearest_type"] if yres else "", "yf_level": yres["nearest_level"] if yres else np.nan,
            "yf_pivot_date": yres["nearest_date"] if yres else None, "yf_atr": yres["atr"] if yres else np.nan,
            "yf_pos": yres["pos"] if yres else -1,
            "close_cache": full["close"] if full else np.nan, "close_yf": yres["close"] if yres else np.nan,
            "ratio_at_signal": float(ratio.get(d, np.nan)),
            "ratio_at_pivot": float(ratio.get(piv_date, np.nan)) if piv_date is not None else np.nan,
            "largest_ratio_step_date": big, "largest_ratio_step_pct": float(seg[big] * 100) if big is not None else 0.0,
            "implied_dividend_usd": float(-seg[big] * cf["Close"].get(big, np.nan)) if big is not None else 0.0,
            "n_ratio_steps_gt_5bps": int((seg.abs() > 5e-4).sum()),
        })
    out = pd.DataFrame(rows)
    ok = out[out["status"] == "ok"].copy()
    ok["full_context_both"] = ok["yf_pos"] >= FULL_CONTEXT_BARS
    ok["dist_shift_win_vs_yf"] = ok["yf_distance"] - ok["cache_win_distance"]
    ok["dist_shift_full_vs_yf"] = ok["yf_distance"] - ok["cache_full_distance"]
    ok["flip_win_vs_yf"] = ok["cache_win_rule"] != ok["yf_rule"]
    ok["flip_full_vs_yf"] = ok["cache_full_rule"] != ok["yf_rule"]
    ok["engine_reproduced"] = ok["engine_matched_rule"] == ok["cache_full_rule"]
    ok.to_csv(HERE / "basis_signals.csv", index=False)

    def flip_stats(df: pd.DataFrame, a: str, b: str) -> dict:
        affected = df[df[a] != "default"]
        flips = affected[affected[a] != affected[b]]
        union = df[(df[a] != "default") | (df[b] != "default")]
        return {
            "n_signals": int(len(df)), "n_affected_on_cache": int(len(affected)),
            "n_flips_among_affected": int(len(flips)),
            "flip_share_of_affected": float(len(flips) / len(affected)) if len(affected) else float("nan"),
            "n_affected_union": int(len(union)), "n_flips_union": int((union[a] != union[b]).sum()),
            "flip_share_union": float((union[a] != union[b]).mean()) if len(union) else float("nan"),
            "n_any_signal_rule_change": int((df[a] != df[b]).sum()),
            "crosstab": pd.crosstab(df[a], df[b]).to_dict(),
        }
    prim = ok[ok["full_context_both"]]
    summary = {
        "window_start": str(WINDOW_START.date()), "yf_date_min": str(yf_start.date()),
        "n_liquid_signals": int(len(sig)), "n_ok": int(len(ok)), "n_missing_series": int((out["status"] != "ok").sum()),
        "engine_vs_cache_full_reproduced": int(ok["engine_reproduced"].sum()),
        "n_full_context_both": int(len(prim)),
        "primary_cache_win_vs_yf_full_context": flip_stats(prim, "cache_win_rule", "yf_rule"),
        "cache_win_vs_yf_all": flip_stats(ok, "cache_win_rule", "yf_rule"),
        "cache_full_vs_yf_all": flip_stats(ok, "cache_full_rule", "yf_rule"),
        "cache_full_vs_cache_win_all": flip_stats(ok, "cache_full_rule", "cache_win_rule"),
        "abs_dist_shift_win_vs_yf_full_context": {
            "median": float(prim["dist_shift_win_vs_yf"].abs().median()) if len(prim) else None,
            "p90": float(prim["dist_shift_win_vs_yf"].abs().quantile(0.9)) if len(prim) else None,
            "max": float(prim["dist_shift_win_vs_yf"].abs().max()) if len(prim) else None,
        },
        "median_ratio_cache_over_yf_at_signal": float(ok["ratio_at_signal"].median()),
    }
    top = prim.reindex(prim["dist_shift_win_vs_yf"].abs().sort_values(ascending=False).index).head(10)
    cols = ["ticker", "signal_date", "cache_win_rule", "yf_rule", "cache_win_distance", "yf_distance",
            "dist_shift_win_vs_yf", "cache_win_nearest", "yf_nearest", "cache_win_level", "yf_level",
            "cache_win_pivot_date", "yf_pivot_date", "cache_win_atr", "yf_atr", "close_cache", "close_yf",
            "ratio_at_pivot", "ratio_at_signal", "largest_ratio_step_date", "largest_ratio_step_pct",
            "implied_dividend_usd", "n_ratio_steps_gt_5bps"]
    top[cols].to_csv(HERE / "basis_top10_shifts.csv", index=False)
    (HERE / "basis_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "cache_full_vs_cache_win_all"}, indent=1, default=str))
    print(top[cols[:8]].to_string())


if __name__ == "__main__":
    main()
