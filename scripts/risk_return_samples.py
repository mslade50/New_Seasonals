"""Exact research samples of the saved main dial, without changing the model.

The cloud risk producer supplies authoritative saved scores and adjusted prices.
Only descriptive outcomes and their denominators are computed here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


WINDOWS = (5, 10, 21)


def build_return_samples(main, spy_close, reduced):
    if main is None or reduced is None or spy_close is None:
        return None
    main = pd.Series(main).sort_index().dropna()
    spy = pd.Series(spy_close).sort_index().dropna()
    if main.empty or spy.empty:
        return None
    if main.index.has_duplicates or spy.index.has_duplicates:
        raise ValueError("Risk sample inputs contain duplicate dates")
    common = main.index.intersection(spy.index).sort_values()
    scores = main.reindex(common)
    price = spy.reindex(common)
    if not np.isfinite(scores).all() or not np.isfinite(price).all() or (price <= 0).any():
        raise ValueError("Risk sample inputs must contain finite scores and positive prices")
    current = float(main.iloc[-1])
    if abs(current - float(reduced["current_score"])) > 1e-9:
        raise ValueError("Risk sample current score differs from saved forward study")
    lo, hi = float(reduced["band_low"]), float(reduced["band_high"])
    positions = np.flatnonzero(((scores >= lo) & (scores <= hi)).to_numpy())
    kept, last = [], -11
    for pos in positions:
        if pos - last > 10:
            kept.append(int(pos))
            last = int(pos)
    # Fix the cohort before observing outcomes; keep the same anchors for all
    # horizons. Strictly separate the longest (21-session) windows, including
    # their anchor closes. The original 10-session sample remains compatible.
    nonoverlap, last = [], -22
    for pos in positions:
        if pos - last > max(WINDOWS):
            nonoverlap.append(int(pos))
            last = int(pos)
    dates = [date.strftime("%Y-%m-%d") for date in common]
    reduced_dates = [dates[pos] for pos in kept]
    expected = [pd.Timestamp(date).strftime("%Y-%m-%d") for date in reduced.get("episode_dates", [])]
    if reduced_dates != expected:
        raise ValueError("Risk samples do not reproduce the saved reduced anchors")

    def sample(anchor_positions):
        anchor_positions = [int(pos) for pos in anchor_positions]
        result = {"episode_dates": [dates[pos] for pos in anchor_positions],
                  "n_episodes": len(anchor_positions), "min_samples": 5,
                  "sample_counts": {}, "returns": {}, "outcomes": {}}
        for window in WINDOWS:
            fwd = (price.shift(-window) / price - 1).dropna()
            values, outcomes = [], []
            for pos in anchor_positions:
                end = pos + window
                if end >= len(price):
                    outcomes.append({"date": dates[pos], "status": "incomplete", "value": None,
                                     "available": len(price) - pos - 1})
                else:
                    value = float(price.iloc[end] / price.iloc[pos] - 1)
                    values.append(value)
                    outcomes.append({"date": dates[pos], "status": "complete", "value": value,
                                     "endDate": dates[end], "start": float(price.iloc[pos]),
                                     "finish": float(price.iloc[end])})
            result["outcomes"][str(window)] = outcomes
            result["sample_counts"][str(window)] = len(values)
            stats = None
            if len(values) >= 5:
                v = np.asarray(values)
                stats = {"n": len(values), "mean": float(v.mean()), "median": float(np.median(v)),
                         "pct_neg": float((v < 0).mean()), "worst": float(v.min()),
                         "best": float(v.max()), "uncond_mean": float(fwd.mean()),
                         "baseline_n": len(fwd),
                         **{name: float(np.quantile(v, q)) for name, q in
                            (("q10", .1), ("q25", .25), ("q75", .75), ("q90", .9))}}
            result["returns"][str(window)] = stats
        return result

    full, spaced = sample(positions), sample(kept)
    for window in WINDOWS:
        saved_n = reduced.get("sample_counts", {}).get(window,
                    reduced.get("sample_counts", {}).get(str(window)))
        if saved_n is not None and spaced["sample_counts"][str(window)] != saved_n:
            raise ValueError("Risk samples disagree on completed reduced counts")
        saved = reduced.get("returns", {}).get(window,
                    reduced.get("returns", {}).get(str(window)))
        if saved and spaced["returns"][str(window)]:
            if abs(float(saved["mean"]) - spaced["returns"][str(window)]["mean"]) > 1e-7:
                raise ValueError("Risk samples disagree on reduced outcomes")
    return {"version": 1, "asof": spy.index[-1].strftime("%Y-%m-%d"),
            "score_asof": main.index[-1].strftime("%Y-%m-%d"),
            "basis": "Saved main dial level; adjusted SPY close-to-close outcomes",
            "current_score": current, "band_low": lo, "band_high": hi, "min_gap": 10,
            "coverage_from": dates[0] if dates else None,
            "coverage_through": dates[-1] if dates else None,
            "all": full, "reduced": spaced, "nonoverlap": sample(nonoverlap),
            "nonoverlap_gap": max(WINDOWS)}


def build_downside_samples(samples, spy_df, vix_close=None, vix_high=None):
    """Low-touch downside for exactly the return cohorts, including non-breaches.

    Missing lows never shorten a forward window. Missing VIX never removes a
    SPY observation from the downside denominator. Threshold tests use raw ATR.
    """
    from scripts.build_atr_downside_stats import ATR_N, MULTS, wilder_atr

    if not samples or spy_df is None or spy_df.empty:
        return None
    if not {"High", "Low", "Close"}.issubset(spy_df.columns):
        return None
    spy = spy_df[["High", "Low", "Close"]].copy().sort_index()
    spy.index = pd.to_datetime(spy.index)
    if spy.index.tz is not None:
        spy.index = spy.index.tz_localize(None)
    if spy.index.has_duplicates:
        raise ValueError("Downside inputs contain duplicate dates")
    if spy.index[-1].strftime("%Y-%m-%d") != samples["asof"]:
        raise ValueError("Downside and return samples have different market dates")
    high, low, close = (spy[c].to_numpy(float) for c in ("High", "Low", "Close"))
    atr = wilder_atr(high, low, close)

    def align(values):
        series = pd.Series(dtype=float) if values is None else pd.Series(values, dtype=float).dropna().sort_index()
        series.index = pd.to_datetime(series.index)
        if series.index.tz is not None:
            series.index = series.index.tz_localize(None)
        if series.index.has_duplicates:
            raise ValueError("Downside VIX input contains duplicate dates")
        return series.reindex(spy.index)

    iv_close = align(vix_close).ffill(limit=1)
    using_high = vix_high is not None and len(vix_high) > 0
    iv_high = align(vix_high).combine_first(iv_close) if using_high else iv_close
    output = {k: samples[k] for k in ("version", "asof", "score_asof", "current_score", "band_low", "band_high")}
    output.update({"atr_period": ATR_N, "thresholds": MULTS,
                   "measure": "max(anchor close - future intraday low, 0) / anchor-date Wilder ATR(14)",
                   "iv_basis": "VIX intraday high" if using_high else "VIX daily close (high unavailable)"})
    for name in (name for name in ("all", "reduced", "nonoverlap") if name in samples):
        cohort = samples[name]
        result = {"episode_dates": list(cohort["episode_dates"]), "windows": {}}
        for window in WINDOWS:
            outcomes = []
            for recorded in cohort["outcomes"][str(window)]:
                date = recorded["date"]
                row = {"date": date, "anchor_date": date, "status": recorded["status"]}
                if recorded["status"] != "complete":
                    row["available"] = recorded.get("available")
                    outcomes.append(row)
                    continue
                pos = spy.index.get_indexer([pd.Timestamp(date)])[0]
                end = pos + window
                usable = (pos >= 0 and end < len(spy) and np.isfinite(atr[pos]) and atr[pos] > 0
                          and np.isfinite(close[pos]) and close[pos] > 0
                          and spy.index[end].strftime("%Y-%m-%d") == recorded["endDate"]
                          and np.isfinite(low[pos + 1:end + 1]).all()
                          and (low[pos + 1:end + 1] > 0).all())
                if not usable:
                    row["status"] = "unavailable"
                    outcomes.append(row)
                    continue
                low_pos = pos + 1 + int(np.argmin(low[pos + 1:end + 1]))
                amount = float(max(close[pos] - low[low_pos], 0) / atr[pos])
                iv_start = iv_close.iloc[pos]
                iv_window = iv_high.iloc[pos + 1:low_pos + 1].dropna()
                peak = float(iv_window.max()) if len(iv_window) else None
                delta = float(peak - iv_start) if peak is not None and np.isfinite(iv_start) else None
                row.update({"max_drawdown_atr": amount,
                            "max_drawdown_pct": float(min(low[low_pos] / close[pos] - 1, 0)),
                            "anchor_spy_close": float(close[pos]), "anchor_atr": float(atr[pos]),
                            "worst_low_date": spy.index[low_pos].strftime("%Y-%m-%d"),
                            "worst_spy_low": float(low[low_pos]), "sessions_to_low": int(low_pos - pos),
                            "breaches": {str(k): bool(amount >= k) for k in MULTS},
                            "iv_start_close": float(iv_start) if np.isfinite(iv_start) else None,
                            "iv_peak": peak, "iv_change_points": delta})
                outcomes.append(row)
            completed = [r for r in outcomes if r["status"] == "complete"]
            hits = {str(k): sum(r["breaches"][str(k)] for r in completed) for k in MULTS}
            result["windows"][str(window)] = {
                "n_selected": len(outcomes), "n_complete": len(completed),
                "n_incomplete": sum(r["status"] == "incomplete" for r in outcomes),
                "n_unavailable": sum(r["status"] == "unavailable" for r in outcomes),
                "hits": hits, "rates": {k: 100 * v / len(completed) if completed else None for k, v in hits.items()},
                "outcomes": outcomes}
        output[name] = result
    return output
