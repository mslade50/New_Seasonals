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
            "all": full, "reduced": spaced}
