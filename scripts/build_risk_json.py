"""
build_risk_json.py — serialize a condensed risk-dashboard summary for the
private site (risk.html). Reuses daily_risk_report's computation pipeline
(which itself wraps pages/risk_dashboard_v2).

Heavy: downloads ~10 years of yfinance data. Designed to run in the
deploy_site workflow right before build_site.py. ALWAYS exits 0 — on any
failure it just skips the write and the site ships without a risk page
payload (the page shows a "no data" note).

Output: data/site_risk.json
"""
import datetime
import json
import math
import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

OUT = os.path.join(_ROOT, "data", "site_risk.json")
MASTER_PRICES = os.path.join(_ROOT, "data", "master_prices.parquet")

# Adjusted OHLC shipped for the private risk page's bottom candlestick
# explorer.  Keeping this menu intentionally compact avoids turning the
# condensed risk payload into a full market-data endpoint.
PRICE_EXPLORER_TICKERS = [
    "SPY", "QQQ", "IWM", "DIA", "UVXY", "VXX",
    "TLT", "GLD", "SLV", "USO", "HYG", "LQD",
]


# Metric metadata for the private-site signal charts.  The series themselves
# stay owned by risk_dashboard_v2; this map only describes how to serialize
# and display the values that compute_all_signals() already returns.
SIGNAL_METRICS = {
    "Distribution Dominance": {
        "key": "da_ratio", "label": "D/A ratio", "unit": "ratio", "decimals": 2,
        "thresholds": [
            {"value": 3.75, "label": "Fire", "operator": ">"},
            {"value": 6.0, "label": "Elevated", "operator": ">"},
        ],
    },
    "VIX Range Compression": {
        "key": "compression_pctile", "label": "21d range percentile",
        "unit": "percentile", "decimals": 1,
        "thresholds": [{"value": 15.0, "label": "Fire", "operator": "<"}],
    },
    "Defensive Leadership": {
        "key": "spread", "label": "50d risk-on minus risk-off spread",
        "unit": "pp", "decimals": 1,
        "thresholds": [{"value": -10.0, "label": "Fire", "operator": "<"}],
    },
    "Pre-FOMC Rally": None,
    "Low Absorption Ratio": {
        "key": "ar_pctile", "label": "Absorption Ratio percentile",
        "unit": "percentile", "decimals": 1,
        "thresholds": [{"value": 10.0, "label": "Fire", "operator": "<"}],
    },
    "Seasonal Rank Divergence": {
        "key": "spread", "label": "Risk-off minus risk-on seasonal spread",
        "unit": "pp", "decimals": 1,
        "thresholds": [{"value": 10.0, "label": "Fire", "operator": ">"}],
    },
    "Dispersion": {
        "key": "composite_pctile", "label": "Composite dispersion percentile",
        "unit": "percentile", "decimals": 1,
        "thresholds": [{"value": 85.0, "label": "Fire", "operator": ">"}],
    },
    "Equity P/C Complacency": {
        "key": "pc_pctile", "label": "10d-MA equity P/C percentile (252d)",
        "unit": "percentile", "decimals": 1,
        "thresholds": [{"value": 10.0, "label": "Fire", "operator": "<"}],
    },
}


def _clean(v):
    import numpy as np
    import pandas as pd
    if v is None:
        return None
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return str(v)


def _aligned_values(series, dates, decimals):
    """Align a numeric Series to the payload's shared dates and JSON-clean it."""
    import pandas as pd

    if series is None or not hasattr(series, "reindex"):
        return [None] * len(dates)
    try:
        aligned = series.reindex(dates)
        return [
            None if pd.isna(v) else round(float(v), decimals)
            for v in aligned.to_numpy()
        ]
    except (KeyError, TypeError, ValueError):
        return [None] * len(dates)


def _latest_value(series, decimals):
    """Return the metric's own latest observation, independent of chart dates."""
    import pandas as pd

    if series is None or not hasattr(series, "dropna"):
        return None
    try:
        available = series.dropna()
        return None if available.empty else round(float(available.iloc[-1]), decimals)
    except (IndexError, TypeError, ValueError):
        return None


def _build_signal_detail(signals_ordered, dates, signal_periods_fn):
    """Serialize signal periods and metric values on one shared date index.

    Kept separate from ``main`` so the payload contract can be verified without
    invoking the heavy ten-year market-data refresh.
    """
    detail = {}
    for name, sig_raw in signals_ordered.items():
        sig = sig_raw or {}
        periods = []
        try:
            periods = [
                [_clean(start), _clean(end)]
                for start, end in signal_periods_fn(sig.get("signal_history"))
            ]
        except Exception:
            # One malformed optional history must not suppress the whole risk
            # payload; the builder's production contract is best-effort.
            periods = []

        config = SIGNAL_METRICS.get(name)
        metric = None
        current_value = None
        if config is not None:
            source_series = sig.get(config["key"])
            values = _aligned_values(source_series, dates, config["decimals"])
            current_value = _latest_value(source_series, config["decimals"])
            metric = {
                "key": config["key"],
                "label": config["label"],
                "unit": config["unit"],
                "decimals": config["decimals"],
                "thresholds": _clean(config["thresholds"]),
                "values": values,
            }

        detail[name] = {
            "periods": periods,
            "metric": metric,
            "current": {
                "value": current_value,
                "summary": _clean(sig.get("summary")),
            },
        }
    return detail


# The hedge-recommendation block (Layer 4C decision tree + protection-cost
# gauge) was deleted 2026-07-16: it rode a retired regime taxonomy, self-
# labeled placeholder hit rates, and the adversarial put-hedge backtest priced
# the advice negative-EV at every threshold. Institutional memory lives in
# scratch/ultracode_research/RISK_DIALS_2026-07-16.md section 4.

# First PIT row of rd2_fragility.parquet — history before this date is a
# recompute vintage (drifted up to ~7 pts); band episodes are only drawn on
# the PIT segment so near-threshold drift can't manufacture phantom episodes.
PIT_START = "2026-07-02"


def build_vol_kpi():
    """VIX / VIX3M term-structure KPI, read from master_prices.parquet."""
    import pandas as pd

    mp = os.path.join(_ROOT, "data", "master_prices.parquet")
    if not os.path.exists(mp):
        return None
    px = pd.read_parquet(mp, columns=["ticker", "date", "Close"],
                         filters=[("ticker", "in", ["^VIX3M", "^VIX"])])

    def last(tk):
        s = px[px["ticker"] == tk].dropna(subset=["Close"]).sort_values("date")
        return (float(s["Close"].iloc[-1]), s["date"].iloc[-1]) if len(s) else (None, None)

    vix, vix_dt = last("^VIX")
    vix3m, _ = last("^VIX3M")
    if vix is None or vix3m is None or not vix3m:
        return None
    return {
        "vix": round(vix, 2),
        "vix3m": round(vix3m, 2),
        "term_ratio": round(vix / vix3m, 3),
        "asof": pd.Timestamp(vix_dt).strftime("%Y-%m-%d"),
    }


def load_vix_high():
    """Daily VIX highs from the master-price cache, if available."""
    import pandas as pd

    if not os.path.exists(MASTER_PRICES):
        return None
    px = pd.read_parquet(
        MASTER_PRICES,
        columns=["ticker", "date", "High"],
        filters=[("ticker", "==", "^VIX")],
    )
    if px.empty:
        return None
    px["date"] = pd.to_datetime(px["date"])
    out = px.dropna(subset=["High"]).sort_values("date").set_index("date")["High"]
    out.index = out.index.tz_localize(None) if out.index.tz is not None else out.index
    return out[~out.index.duplicated(keep="last")].astype(float)


def build_similar_reading_drawdown_iv(
    spy_df,
    vix_close,
    similar_result,
    vix_high=None,
    horizons=None,
    thresholds=None,
):
    """ATR downside paths for the exact 63d similar-reading sample.

    Each anchor is one of ``compute_similar_reading_returns``' declustered
    ``episode_dates``.  For every forward window the drawdown is the existing
    risk-tab low-touch measure::

        max(Close_anchor - Low_{anchor+1..anchor+N}, 0) / WilderATR14_anchor

    VIX begins at the anchor-date close and peaks on an intraday-high basis
    from the next session through the date of that window's worst SPY low.
    This deliberately is not a market-wide peak-to-recovery correction sample.
    """
    import numpy as np
    import pandas as pd
    from scripts.build_atr_downside_stats import (
        ATR_N, HORIZONS, MULTS, wilder_atr,
    )

    if similar_result is None or not similar_result.get("episode_dates"):
        return None
    required = {"High", "Low", "Close"}
    if spy_df is None or spy_df.empty or not required.issubset(spy_df.columns):
        return None

    spy = spy_df.loc[:, ["High", "Low", "Close"]].dropna().copy().sort_index()
    spy.index = pd.to_datetime(spy.index)
    if spy.index.tz is not None:
        spy.index = spy.index.tz_localize(None)
    spy = spy[~spy.index.duplicated(keep="last")]

    vclose = pd.Series(vix_close).dropna().astype(float).sort_index()
    vclose.index = pd.to_datetime(vclose.index)
    if vclose.index.tz is not None:
        vclose.index = vclose.index.tz_localize(None)
    vclose = vclose[~vclose.index.duplicated(keep="last")]
    if vclose.empty:
        return None

    using_high = vix_high is not None and len(vix_high)
    vhigh = pd.Series(vix_high if using_high else vclose).dropna().astype(float).sort_index()
    vhigh.index = pd.to_datetime(vhigh.index)
    if vhigh.index.tz is not None:
        vhigh.index = vhigh.index.tz_localize(None)
    vhigh = vhigh[~vhigh.index.duplicated(keep="last")]

    # Match the rest of the risk tab exactly: adjusted SPY OHLC and Wilder-14
    # ATR fixed on the analog date. A one-session VIX fill only bridges a
    # calendar mismatch; it never bridges a real multi-day data gap.
    idx = spy.index
    H = spy["High"].to_numpy(float)
    L = spy["Low"].to_numpy(float)
    C = spy["Close"].to_numpy(float)
    atr = wilder_atr(H, L, C)
    aligned_close = vclose.reindex(idx).ffill(limit=1)
    aligned_high = vhigh.reindex(idx).combine_first(aligned_close)

    horizon_map = dict(horizons or HORIZONS)
    atr_thresholds = [float(x) for x in (thresholds or MULTS)]
    anchor_dates = pd.to_datetime(similar_result["episode_dates"])
    anchor_positions = idx.get_indexer(anchor_dates)
    rows_by_horizon = {lab: [] for lab in horizon_map}
    eligible_by_horizon = {lab: 0 for lab in horizon_map}

    for anchor_date, pos in zip(anchor_dates, anchor_positions):
        if pos < 0 or not np.isfinite(atr[pos]) or atr[pos] <= 0:
            continue
        anchor_close = float(C[pos])
        iv_start_raw = aligned_close.iloc[pos]
        iv_start = float(iv_start_raw) if pd.notna(iv_start_raw) else None

        for label, n_sessions in horizon_map.items():
            end = pos + int(n_sessions)
            if end >= len(spy):
                continue
            eligible_by_horizon[label] += 1
            future_lows = L[pos + 1:end + 1]
            low_offset = int(np.nanargmin(future_lows))
            low_pos = pos + 1 + low_offset
            worst_low = float(L[low_pos])
            dd_points = max(anchor_close - worst_low, 0.0)
            dd_atr = dd_points / float(atr[pos])
            dd_pct = worst_low / anchor_close - 1.0

            iv_window = aligned_high.iloc[pos + 1:low_pos + 1].dropna()
            iv_peak = float(iv_window.max()) if not iv_window.empty else None
            iv_peak_date = iv_window.idxmax() if not iv_window.empty else None
            delta = (iv_peak - iv_start
                     if iv_peak is not None and iv_start is not None else None)
            rows_by_horizon[label].append({
                "anchor_date": pd.Timestamp(anchor_date).strftime("%Y-%m-%d"),
                "anchor_spy_close": round(anchor_close, 2),
                "anchor_atr": round(float(atr[pos]), 3),
                "worst_low_date": idx[low_pos].strftime("%Y-%m-%d"),
                "worst_spy_low": round(worst_low, 2),
                "max_drawdown_atr": round(float(dd_atr), 3),
                "max_drawdown_pct": round(float(dd_pct), 5),
                "sessions_to_low": int(low_pos - pos),
                "iv_start_close": round(iv_start, 2) if iv_start is not None else None,
                "iv_peak": round(iv_peak, 2) if iv_peak is not None else None,
                "iv_peak_date": (iv_peak_date.strftime("%Y-%m-%d")
                                 if iv_peak_date is not None else None),
                "iv_change_points": round(delta, 2) if delta is not None else None,
                "iv_change_pct": (round(delta / iv_start, 5)
                                  if delta is not None and iv_start else None),
            })

    for rows in rows_by_horizon.values():
        rows.sort(key=lambda row: row["anchor_date"], reverse=True)
    counts = {
        label: {f"{t:g}": sum(row["max_drawdown_atr"] >= t for row in rows)
                for t in atr_thresholds}
        for label, rows in rows_by_horizon.items()
    }
    return {
        "sample_basis": "exact declustered anchors from the 63d similar-fragility table",
        "score_horizon": "63d",
        "current_score": round(float(similar_result["current_score"]), 1),
        "band_low": round(float(similar_result["band_low"]), 1),
        "band_high": round(float(similar_result["band_high"]), 1),
        "n_episodes": int(similar_result["n_episodes"]),
        "atr_period": int(ATR_N),
        "measure": ("max(anchor close - subsequent intraday low, 0) / "
                    "anchor-date Wilder ATR(14)"),
        "iv_basis": "VIX intraday high" if using_high else "VIX daily close (high unavailable)",
        "horizons": list(horizon_map.keys()),
        "eligible_by_horizon": eligible_by_horizon,
        "thresholds": atr_thresholds,
        "default_horizon": "63d" if "63d" in horizon_map else list(horizon_map)[-1],
        "default_threshold": 2.0,
        "counts": counts,
        "rows_by_horizon": rows_by_horizon,
    }


def _bands_from_book(strategy_book):
    """[{strategy, bands}] for every strategy carrying frag_risk_bands.
    Single source of truth is strategy_config; the site never hardcodes the
    band table (guarded by tests/test_frag_risk_bands.py)."""
    out = []
    for strat in strategy_book:
        bands = (strat.get("execution") or {}).get("frag_risk_bands")
        if bands:
            out.append({"strategy": strat.get("name"),
                        "bands": [[float(lo), float(hi), float(m)] for lo, hi, m in bands]})
    return out


def _band_mult(bands, score):
    for lo, hi, mult in bands:
        if lo <= score < hi:
            return float(mult)
    return 1.0


def _run_episodes(flags):
    """[[start, end], ...] date strings for contiguous True runs."""
    episodes, start, prev = [], None, None
    for date, on in flags.items():
        if on and start is None:
            start = date
        elif not on and start is not None:
            episodes.append([start.strftime("%Y-%m-%d"), prev.strftime("%Y-%m-%d")])
            start = None
        prev = date
    if start is not None:
        episodes.append([start.strftime("%Y-%m-%d"), prev.strftime("%Y-%m-%d")])
    return episodes


def build_sizing_state():
    """The number that sizes live orders: 10d MA of the 63d dial from the
    APPEND-ONLY PIT parquet (never the deploy-time recompute — vintages drift
    up to ~7 pts and near the threshold the recompute can contradict what
    daily_scan actually staged)."""
    import pandas as pd

    frag_path = os.path.join(_ROOT, "data", "rd2_fragility.parquet")
    if not os.path.exists(frag_path):
        print("risk: sizing_state skipped (no rd2_fragility.parquet)")
        return None
    frag = pd.read_parquet(frag_path)
    if "63d" not in frag.columns:
        print("risk: sizing_state skipped (no 63d column)")
        return None
    s63 = frag["63d"].dropna()
    if s63.empty:
        return None
    s63.index = pd.to_datetime(s63.index)
    s63 = s63.sort_index()
    ma = s63.rolling(10, min_periods=1).mean()
    score = float(ma.iloc[-1])

    from strategy_config import STRATEGY_BOOK
    banded = _bands_from_book(STRATEGY_BOOK)
    throttle_los = [lo for b in banded for lo, hi, m in b["bands"] if m < 1.0]
    threshold = min(throttle_los) if throttle_los else 50.0

    throttled = []
    for b in banded:
        mult = _band_mult(b["bands"], score)
        if mult != 1.0:
            throttled.append({"strategy": b["strategy"], "mult": mult})

    state = ma >= threshold
    current = bool(state.iloc[-1])
    days_in_state = 0
    for v in state.iloc[::-1]:
        if bool(v) != current:
            break
        days_in_state += 1

    # Full dial history (2016+). The frontend defaults its chart window to
    # trailing 1y and offers range presets; the hero spark slices the last
    # 252 sessions client-side. (Was ma.tail(252) until 2026-07-29.)
    # Pre-2026-07-02 rows are the frozen recompute vintage, not PIT.
    tail = ma
    pit_state = state[state.index >= pd.Timestamp(PIT_START)]

    expo = None
    expo_path = os.path.join(_ROOT, "data", "exposure_state.json")
    if os.path.exists(expo_path):
        try:
            with open(expo_path, encoding="utf-8") as f:
                e = json.load(f)
            expo = {"mult": e.get("mult"), "active_rule": e.get("active_rule"),
                    "reason": e.get("reason"), "asof": e.get("asof")}
        except Exception:
            expo = None

    sleeve = None
    sleeve_path = os.path.join(_ROOT, "data", "dial_sleeve_paper.json")
    if os.path.exists(sleeve_path):
        try:
            with open(sleeve_path, encoding="utf-8") as f:
                s = json.load(f)
            sleeve = {"position": s.get("position"), "since": s.get("since"),
                      "last_evaluated": s.get("last_evaluated"),
                      "n_transitions": len(s.get("transitions") or []),
                      "last_transition": (s.get("transitions") or [None])[-1],
                      "spec_version": s.get("spec_version")}
        except Exception:
            sleeve = None

    return {
        "asof": ma.index[-1].strftime("%Y-%m-%d"),
        "basis": "10d MA of 63d dial, append-only PIT parquet (sizes live orders)",
        "score": round(score, 1),
        "raw_63d": round(float(s63.iloc[-1]), 1),
        "threshold": float(threshold),
        "throttle_on": bool(current),
        "gap_to_threshold": round(threshold - score, 1),
        "days_in_state": int(days_in_state),
        "banded_strategies": banded,
        "throttled": throttled,
        "pit_start": PIT_START,
        "spark": {
            "dates": [d.strftime("%Y-%m-%d") for d in tail.index],
            "ma": [round(float(v), 1) for v in tail.values],
            # One bar per stored point-in-time 63d dial reading. Keep this
            # aligned to the MA dates so the frontend can share one x-axis.
            "daily": [round(float(v), 1)
                      for v in s63.reindex(tail.index).values],
        },
        "episodes": _run_episodes(pit_state),
        "exposure": expo,
        "sleeve": sleeve,
    }


def build_atr_downside(spy_df):
    """Downside-distribution tables for the risk tab.

    Two parts, same low-touch measure and Wilder-14 ATR throughout:
      - ``signals`` / ``baseline``: full-history per-signal tables, read from the
        committed ``data/atr_downside_stats.json`` (generator:
        scripts/build_atr_downside_stats.py). Precomputed because the live risk
        pipeline only fetches 10y, which is too thin for the rare signals.
      - ``dial``: computed live here (it depends on today's dial value) — days
        where the 10d MA of the 63d dial closed within +-3 of its current value.
        The ATR helpers are imported from the generator so this table is
        byte-for-byte comparable with the per-signal ones.
    """
    import numpy as np
    import pandas as pd

    stats_path = os.path.join(_ROOT, "data", "atr_downside_stats.json")
    if not os.path.exists(stats_path):
        print("risk: atr_downside skipped (no atr_downside_stats.json)")
        return None
    with open(stats_path, encoding="utf-8") as f:
        stats = json.load(f)

    out = {k: stats.get(k) for k in (
        "measure", "atr_period", "mults", "horizons",
        "baseline", "baseline_n", "data_from", "data_through", "signals")}

    # ---- dial-conditioned table (live) ----
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from build_atr_downside_stats import wilder_atr, worst_low_drop, HORIZONS, MULTS

    frag_path = os.path.join(_ROOT, "data", "rd2_fragility.parquet")
    if spy_df is None or spy_df.empty or not os.path.exists(frag_path):
        return out  # per-signal tables only
    frag = pd.read_parquet(frag_path)
    if "63d" not in frag.columns:
        return out
    frag.index = pd.to_datetime(frag.index)
    dial_ma = frag["63d"].dropna().sort_index().rolling(10, min_periods=1).mean()
    if dial_ma.empty:
        return out
    current = float(dial_ma.iloc[-1])
    band = 3.0
    lo, hi = current - band, current + band

    idx = spy_df.index
    L = spy_df["Low"].to_numpy(float)
    C = spy_df["Close"].to_numpy(float)
    atr = wilder_atr(spy_df["High"].to_numpy(float), L, C)
    atr_valid = np.isfinite(atr) & (atr > 0)
    n = len(idx)

    band_dates = dial_ma[(dial_ma >= lo) & (dial_ma <= hi)].index
    posi = idx.get_indexer(band_dates)
    band_mask = np.zeros(n, bool)
    band_mask[posi[posi >= 0]] = True

    table, n_by_h = {}, {}
    for lab, N in HORIZONS.items():
        worst, wv = worst_low_drop(L, C, N)
        v = wv & atr_valid & band_mask
        dd = worst[v] / atr[v]
        table[lab] = ({str(k): round(float(100 * np.mean(dd >= k)), 1) for k in MULTS}
                      if v.sum() else {str(k): None for k in MULTS})
        n_by_h[lab] = int(v.sum())

    used = idx[band_mask]
    out["dial"] = {
        "value": round(current, 1),
        "band": band,
        "lo": round(lo, 1),
        "hi": round(hi, 1),
        "table": table,
        "n_by_h": n_by_h,
        "band_from": used.min().strftime("%Y-%m-%d") if len(used) else None,
        "band_through": used.max().strftime("%Y-%m-%d") if len(used) else None,
    }
    return out


def build_nuggets(p):
    """Deterministic interpretation of the risk payload -> idea-page nuggets.

    Each nugget: {title, tone (good|warn|bad|info), lines: [str,...]}.
    Tones map to badges client-side. Every quantitative claim comes straight
    from the payload (fragility scores, conditional forward returns, price
    context) — no fabricated history.
    """
    out = []
    frag = p.get("fragility") or {}
    frag10 = p.get("fragility_10d") or {}
    ctx = p.get("price_ctx") or {}
    fwd = p.get("forward_returns") or {}
    sigs = p.get("signals") or []
    n_on = p.get("n_active", 0)

    def lvl(v):
        return "robust" if v < 33 else "neutral" if v < 66 else "fragile"

    # 1. fragility level + trend
    if frag.get("21d") is not None:
        f21 = frag["21d"]
        trend = ""
        if frag10.get("21d") is not None:
            d = f21 - frag10["21d"]
            trend = " and easing" if d < -1 else " and building" if d > 1 else ", flat"
        tone = "good" if f21 < 33 else "warn" if f21 < 66 else "bad"
        out.append({
            "title": f"Fragility: {lvl(f21)}{trend}",
            "tone": tone,
            "lines": [
                f"21d score {f21:.0f} / 100 ({lvl(f21)}){trend} vs its 10d average. "
                f"5d at {frag.get('5d', 0):.0f}, 63d at {frag.get('63d', 0):.0f}.",
            ],
        })

    # 2. conditional forward returns at the current readings
    fwd_lines, zs, z_by_h = [], [], {}
    for h in ["5d", "21d", "63d"]:
        r = fwd.get(h)
        if not r:
            continue
        w = h.replace("d", "")
        st = (r.get("returns") or {}).get(w)
        if not st:
            continue
        mz = st.get("mean_z") or 0.0
        zs.append(mz)
        z_by_h[h] = mz
        fwd_lines.append(
            f"{h} fragility {r['current_score']:.0f} ({r['n_episodes']} similar episodes): "
            f"SPY next {w}d averaged {st['mean']:+.2%} vs {st['uncond_mean']:+.2%} baseline "
            f"(mean Z {mz:+.2f}, {st['pct_neg']:.0%} negative).")
    if fwd_lines:
        avg_z = sum(zs) / len(zs)
        tone = ("bad" if avg_z <= -0.75 else "warn" if avg_z < -0.25 else
                "good" if avg_z >= 0.25 else "info")
        verdict = ("Net read: readings like today's have been a tailwind." if avg_z >= 0.25 else
                   "Net read: roughly baseline forward returns from here." if avg_z > -0.25 else
                   "Net read: readings like today's have dragged on forward returns — lean smaller.")
        if zs and (max(zs) - min(zs)) > 0.6:
            soft = min(z_by_h, key=z_by_h.get)
            firm = max(z_by_h, key=z_by_h.get)
            verdict += f" Horizons diverge: {firm} supportive, {soft} the soft spot."
        out.append({"title": "What similar readings led to", "tone": tone,
                    "lines": fwd_lines + [verdict]})

    # 3. signal roster
    on = [s for s in sigs if s.get("on")]
    decaying = [s for s in sigs if not s.get("on") and str(s.get("badge", "")).startswith("DECAYING")]
    lines = []
    if on:
        for s in on:
            det = s.get("detail")
            det = f" — {det}" if isinstance(det, str) and det else ""
            lines.append(f"ON: {s['name']}{det}")
    else:
        lines.append(f"No fragility signals active (0 of {len(sigs)}).")
    if decaying:
        lines.append("Recently cooled: " + ", ".join(
            f"{s['name']} ({s['badge'].split('(')[-1].rstrip(')')})" for s in decaying)
            + " — recent enough to re-ignite quickly.")
    out.append({
        "title": f"Signals: {n_on} of {len(sigs)} active",
        "tone": "good" if n_on == 0 else "warn" if n_on <= 2 else "bad",
        "lines": lines,
    })

    # 4. SPY price action context
    if ctx:
        bits = []
        if ctx.get("regime_label"):
            bits.append(str(ctx["regime_label"]))
        if ctx.get("extension_200d") is not None:
            bits.append(f"{ctx['extension_200d']:+.1%} vs the 200d")
        if ctx.get("drawdown") is not None:
            bits.append(f"{ctx['drawdown']:+.1%} off the 52w high")
        if ctx.get("ret_12m") is not None:
            bits.append(f"{ctx['ret_12m']:+.1%} over 12m")
        line2 = []
        if ctx.get("days_since_5pct") is not None:
            line2.append(f"{ctx['days_since_5pct']}d since a 5% pullback")
        if ctx.get("days_since_10pct") is not None:
            line2.append(f"{ctx['days_since_10pct']}d since a 10% drawdown")
        ext = ctx.get("extension_200d") or 0
        out.append({
            "title": f"SPY {p.get('spy_last', '')}: {ctx.get('regime_label', 'price context')}",
            "tone": "warn" if ext > 0.12 or (ctx.get("drawdown") or 0) < -0.05 else "info",
            "lines": ["; ".join(bits) + ".", "; ".join(line2) + "." if line2 else ""],
        })

    # 5. book posture from the regime multiplier
    rm = p.get("regime_mult")
    if rm is not None:
        rm = float(rm)
        posture = ("run full-to-augmented size" if rm >= 1.1 else
                   "full size" if rm >= 0.95 else
                   "trim core exposure" if rm >= 0.75 else "de-risk meaningfully")
        out.append({
            "title": f"Book posture: regime multiplier {rm:.2f}x",
            "tone": "good" if rm >= 0.95 else "warn" if rm >= 0.75 else "bad",
            "lines": [f"The fragility framework's core-exposure dial (0.6-1.8x) says {posture}."],
        })

    return out


# ---------------------------------------------------------------------------
# Trade Console (spec: scratch/ultracode_research/RISK_TRADE_CONSOLE_2026-07-16.md)
# Every rendered figure is formatted here from data/trade_console_stats.json —
# zero hand-typed numbers, zero client-side composition. Display-only.
# ---------------------------------------------------------------------------
TC_STATS_PATH = os.path.join(_ROOT, "data", "trade_console_stats.json")
TC_STATS_MAX_AGE_DAYS = 180
TC_INPUT_STALE_TD = 3

TC_HEADLINES = {
    "BEAR_2PLUS": "ELEVATED LEFT TAIL — MULTIPLE SIGNALS",
    "DL_TAIL": "ELEVATED LEFT TAIL",
    "AR_DRAWDOWN": "ELEVATED DRAWDOWN ODDS",
    "SRD_TAIL": "TAIL READ",
    "DA_SOFT": "SOFT PATCH",
    "VRC_CONTEXT": "CONTEXT ONLY",
    "NONE": "NO READABLE EDGE",
}
TC_FULL_NAMES = {
    "DA": "Distribution Dominance", "VRC": "VIX Range Compression",
    "DL": "Defensive Leadership", "AR": "Low Absorption Ratio",
    "SRD": "Seasonal Rank Divergence", "DISP": "Dispersion",
    "FOMC": "Pre-FOMC Rally",
}


def _tc_pct(v, dec=1, signed=True):
    if v is None:
        return "n/a"
    s = f"{v:+.{dec}f}%" if signed else f"{v:.{dec}f}%"
    return s


def build_price_explorer(spy_df, shared_dates, master_path=MASTER_PRICES):
    """Serialize adjusted OHLC for the private site's match explorer.

    All assets share the risk payload's SPY session calendar.  The exact
    similar-fragility match dates live in ``forward_returns``; this block only
    changes the price series drawn underneath those fixed vertical overlays.
    """
    import pandas as pd

    dates = pd.DatetimeIndex(pd.to_datetime(shared_dates))
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    if dates.empty:
        return None

    columns = ["Open", "High", "Low", "Close"]
    frames = {}
    if spy_df is not None and all(column in spy_df.columns for column in columns):
        spy = spy_df[columns].copy()
        spy.index = pd.to_datetime(spy.index)
        if spy.index.tz is not None:
            spy.index = spy.index.tz_localize(None)
        frames["SPY"] = spy[~spy.index.duplicated(keep="last")].sort_index()

    other_tickers = [ticker for ticker in PRICE_EXPLORER_TICKERS if ticker != "SPY"]
    if os.path.exists(master_path):
        cached = pd.read_parquet(
            master_path,
            columns=["ticker", "date", *columns],
            filters=[
                ("ticker", "in", other_tickers),
                ("date", ">=", dates.min()),
            ],
        )
        if not cached.empty:
            cached["date"] = pd.to_datetime(cached["date"])
            for ticker, group in cached.groupby("ticker", sort=False):
                frame = group.set_index("date")[columns]
                frames[str(ticker)] = frame[
                    ~frame.index.duplicated(keep="last")
                ].sort_index()

    series = {}
    for ticker in PRICE_EXPLORER_TICKERS:
        frame = frames.get(ticker)
        if frame is None or frame.empty:
            continue
        aligned_close = frame["Close"].reindex(dates)
        if aligned_close.notna().sum() < 20:
            continue
        series[ticker] = {
            column.lower(): _aligned_values(frame[column], dates, 4)
            for column in columns
        }

    if not series:
        return None
    return {"assets": list(series), "series": series}


def _tc_prob(v):
    return "n/a" if v is None else f"{v * 100:.0f}%"


def _tc_ci(ci, unit="%"):
    if not ci or len(ci) != 2:
        return ""
    if unit == "%":
        return f" [{ci[0]:+.1f}%, {ci[1]:+.1f}%]"
    return f" [{ci[0] * 100:+.0f}%, {ci[1] * 100:+.0f}%]"


def _tc_fired_line(fired, ctx_bits):
    parts = []
    for name, rec in fired:
        if rec == 0:
            parts.append(f"{name} is active")
        else:
            s = "session" if rec == 1 else "sessions"
            parts.append(f"{name} fired {rec} {s} ago (now off)")
    line = "; ".join(parts) if parts else "No bearish signals active or recent"
    if ctx_bits:
        line += ". " + "; ".join(ctx_bits) + "."
    else:
        line += "."
    return line


def _tc_dist_line(cls, base):
    yrs = cls.get("episode_years") or []
    span = f"{yrs[0]}–{str(yrs[-1])[2:]}" if yrs else "n/a"
    bits = [
        f"In {cls['n_episodes']} prior episodes ({span}): "
        f"median 3-month return {_tc_pct(cls.get('fwd63_median_pct'))} "
        f"(baseline {_tc_pct(base.get('fwd63_median_pct'))}), "
        f"mean {_tc_pct(cls.get('fwd63_mean_pct'))}"
        f"{_tc_ci(cls.get('fwd63_mean_ci_pct'))} vs {_tc_pct(base.get('fwd63_mean_pct'))}"
    ]
    if cls.get("fwd63_p10_pct") is not None:
        bits.append(f"10th percentile {_tc_pct(cls['fwd63_p10_pct'])} "
                    f"vs {_tc_pct(base.get('fwd63_p10_pct'))}")
    if cls.get("p_dd10_63td") is not None:
        bits.append(f"{_tc_prob(cls['p_dd10_63td'])} saw a >=10% drawdown "
                    f"vs {_tc_prob(base.get('p_dd10_63td'))} baseline")
    if cls.get("fwd63_drop_best_mean_pct") is not None:
        bits.append(f"mean excl. best episode {_tc_pct(cls['fwd63_drop_best_mean_pct'])}")
    t = cls.get("episode_t_vs_baseline")
    if t is not None:
        bits.append(f"episode-level t {t:+.1f} vs baseline")
    return "; ".join(bits) + "."


def _tc_structure_line(cls):
    ev = cls.get("structures") or {}
    if not ev.get("n_priced"):
        return None, False
    sp = ev.get("spread_ret_on_cost") or {}
    tl = ev.get("tail_ret_on_cost") or {}
    sp_ci = sp.get("ci5_95")
    tl_ci = tl.get("ci5_95")
    flip_spread = bool(sp_ci and sp_ci[0] is not None and sp_ci[0] > 0)
    flip_tail = bool(tl_ci and tl_ci[0] is not None and tl_ci[0] > 0)
    n = ev["n_priced"]
    if flip_spread or flip_tail:
        which = ("a 3m 30Δ/10Δ put spread" if flip_spread
                 else "3m ~10Δ tail puts")
        stat = sp if flip_spread else tl
        line = (f"STRUCTURE CHECK (MODEL-PRICED, held to expiry): historically "
                f"+EV for {which} in this class — mean "
                f"{stat['mean'] * 100:+.0f}% of cost{_tc_ci(stat.get('ci5_95'), 'raw')}, "
                f"{n} priced episodes. Model prices, no real chains.")
        return line, True
    tail_txt = (f"expired worthless {n - ev.get('tail_hits', 0)} of {n}"
                if ev.get("tail_hits", 0) == 0 else
                f"EV {tl.get('mean', 0) * 100:+.0f}% of premium"
                f"{_tc_ci(tl_ci, 'raw')}, paid off {ev.get('tail_hits')} of {n}")
    line = (f"STRUCTURE CHECK (MODEL-PRICED, held to expiry): 3m 30Δ/10Δ SPY put "
            f"spread ≈ {ev.get('spread_cost_pct_notional_mean', 0):.1f}% of notional, "
            f"EV {sp.get('mean', 0) * 100:+.0f}% of debit{_tc_ci(sp_ci, 'raw')}, "
            f"paid off {ev.get('spread_hits', 0)} of {n}. Tail puts: {tail_txt}. "
            f"Perfect exits would lift the spread to "
            f"{ev.get('spread_oracle_mean', 0) * 100:+.0f}% — "
            f"neither structure has cleared cost at model prices in this class "
            f"this decade.")
    return line, False


def _tc_action_line(class_id, cls, base, low_sample):
    if class_id == "NONE":
        return "Historical read: nothing to do."
    if class_id == "VRC_CONTEXT":
        return ("Historical read: near-term drag only; 3-month tails sit at "
                "or below baseline. No action.")
    if low_sample:
        return ("Historical read: sample too thin for a verdict — descriptive "
                "only.")
    p10c, p10b = cls.get("p_dd10_63td"), base.get("p_dd10_63td")
    if p10c is not None and p10b and p10c >= 1.5 * p10b:
        return ("Historical read: fat left tail, normal median — smaller size, "
                "not premium purchase.")
    if class_id == "SRD_TAIL":
        return "Historical read: tail-percentile context only. No action."
    return "Historical read: no trade."


def _render_trade_console(stats, row, fired, asof, spy_stale_td):
    """Pure renderer — testable without IO. Returns the trade_console dict."""
    from scripts.build_trade_console_stats import (
        CLASS_PRECEDENCE, MIN_READABLE, classify_row)

    if spy_stale_td > TC_INPUT_STALE_TD:
        return {"asof": asof, "state": "silent",
                "reason": (f"inputs stale ({spy_stale_td} trading days since "
                           f"last SPY close in the pipeline) — console withheld"),
                "class_set_version": stats.get("class_set_version")}

    degraded_reason = None
    try:
        import datetime as _dt
        built = _dt.datetime.strptime(stats["built_utc"][:10], "%Y-%m-%d")
        age_days = (_dt.datetime.utcnow() - built).days
        if age_days > TC_STATS_MAX_AGE_DAYS:
            degraded_reason = (f"evidence file is {age_days} days old "
                               f"(> {TC_STATS_MAX_AGE_DAYS}) — structure block "
                               f"withheld pending deliberate regeneration")
    except Exception:
        degraded_reason = "evidence file vintage unreadable"

    class_id = classify_row(row)
    base = stats.get("baseline") or {}
    cls = (stats.get("classes") or {}).get(class_id) or {}
    n_ep = cls.get("n_episodes", 0)

    if class_id != "NONE" and n_ep < MIN_READABLE:
        class_id, cls = "NONE", (stats.get("classes") or {}).get("NONE") or {}
        n_ep = cls.get("n_episodes", 0)

    low_sample = (MIN_READABLE <= n_ep < 20) or \
        ((cls.get("max_year_share") or 0) > 0.40)

    ctx_bits = []
    if bool(row.get("near_52w_high")):
        ctx_bits.append("SPY within 2% of a 52-week high")
    if bool(row.get("above_200d")):
        ctx_bits.append("above the 200-day")
    fired_line = _tc_fired_line(fired, ctx_bits)

    headline = TC_HEADLINES.get(class_id, "NO READABLE EDGE")
    if low_sample and class_id != "NONE":
        headline += " (LOW SAMPLE)"

    dist_line = _tc_dist_line(cls, base) if cls else None
    structure_line, flip = (None, False)
    if not degraded_reason and class_id not in ("VRC_CONTEXT",):
        structure_line, flip = _tc_structure_line(cls)
    action_line = _tc_action_line(class_id, cls, base, low_sample)

    extra = None
    if class_id == "NONE" and bool(row.get("any_DISP")):
        n_disp = stats.get("dispersion_episodes")
        extra = (f"Dispersion fired — only {n_disp} prior episodes, too few "
                 f"to condition on.")

    caveats = [
        f"N={n_ep} episodes; one decade of history containing three real "
        f"drawdowns (2018Q4, 2020, 2022).",
        "No class is statistically distinct after a multiple-comparisons "
        "correction — these are conditional distributions, not proven edges.",
        "Signal histories recomputed from today's code (lookahead PIT cannot "
        "cure).",
        "Option EV is BS + static skew + 5%/side haircut — no real option "
        "chains; MODEL-PRICED throughout.",
        "Display-only — nothing here feeds sizing or staging.",
        f"Stats vintage {stats.get('built_utc')}, class set "
        f"{stats.get('class_set_version', '?').split(' ')[0]}.",
    ]

    return {
        "asof": asof,
        "state": "degraded" if degraded_reason else "ok",
        **({"reason": degraded_reason} if degraded_reason else {}),
        "class_id": class_id,
        "class_set_version": stats.get("class_set_version"),
        "flip_gate_passed": bool(flip),
        "headline": headline,
        "fired_line": fired_line,
        "dist_line": dist_line,
        **({"structure_line": structure_line} if structure_line else {}),
        "action_line": action_line,
        **({"extra_line": extra} if extra else {}),
        "caveats": caveats,
        "n_ep": int(n_ep),
        "stats_built": stats.get("built_utc"),
        "stats_sha": stats.get("fingerprint"),
    }


def build_trade_console(computed):
    """Classify today against the frozen class set and render the console.
    Returns None (key omitted) only on hard failure; degraded/silent states
    are RENDERED with their reason."""
    import numpy as np
    import pandas as pd
    from scripts.build_trade_console_stats import (
        ABBR, build_config_frame, class_fingerprint)

    if not os.path.exists(TC_STATS_PATH):
        print("risk: trade_console skipped (no stats file)")
        return None
    with open(TC_STATS_PATH, encoding="utf-8") as f:
        stats = json.load(f)

    signals_ordered = computed["signals_ordered"]
    spy_close = computed["spy_close"].dropna()

    # The trade-console evidence file was built on the ABBR taxonomy (the 7
    # base signals). Signals added to the composite later (Equity P/C
    # Complacency, 2026-08-05) are filtered out here so the fingerprint gate
    # checks the taxonomy the evidence actually covers instead of degrading.
    signals_ordered = {k: v for k, v in signals_ordered.items() if k in ABBR}

    if class_fingerprint(signals_ordered.keys()) != stats.get("fingerprint"):
        return {"asof": spy_close.index[-1].strftime("%Y-%m-%d"),
                "state": "degraded",
                "reason": ("signal taxonomy changed since the evidence file "
                           "was built — regenerate "
                           "scripts/build_trade_console_stats.py"),
                "class_set_version": stats.get("class_set_version"),
                "stats_built": stats.get("built_utc"),
                "stats_sha": stats.get("fingerprint")}

    frame = build_config_frame(signals_ordered, spy_close)
    row = frame.iloc[-1]
    asof = frame.index[-1].strftime("%Y-%m-%d")
    spy_stale_td = int(np.busday_count(frame.index[-1].date(),
                                       pd.Timestamp.today().normalize().date()))

    fired = []
    for name, abbr in ABBR.items():
        if abbr == "FOMC":
            continue
        if bool(row[f"any_{abbr}"]):
            on_col = frame[f"on_{abbr}"]
            if bool(row[f"on_{abbr}"]):
                rec = 0
            else:
                trues = np.flatnonzero(on_col.to_numpy())
                rec = len(frame) - 1 - int(trues[-1]) if len(trues) else 0
            fired.append((name, rec))

    return _render_trade_console(stats, row, fired, asof, spy_stale_td)


def main():
    try:
        from daily_risk_report import (
            download_data,
            compute_all_signals,
            build_forward_returns_data,
            _status_badge,
        )
        from pages.risk_dashboard_v2 import _signal_periods

        print("risk: downloading data ...")
        spy_df, closes, sp500_closes = download_data()
        print("risk: computing signals ...")
        computed = compute_all_signals(spy_df, closes, sp500_closes)

        signals = []
        price_ctx = computed["price_ctx"] or {}
        for name, sig in computed["signals_ordered"].items():
            badge, color = _status_badge(sig or {}, price_ctx)
            signals.append({
                "name": name,
                "on": bool((sig or {}).get("on")),
                "elevated": bool((sig or {}).get("elevated")),
                "badge": badge,
                "color": color,
                "detail": _clean((sig or {}).get("detail")),
            })

        fwd_raw = {}
        fwd = {}
        if computed.get("frag_df") is not None and computed.get("h_scores"):
            fwd_raw = build_forward_returns_data(
                computed["frag_df"], computed["spy_close"], computed["h_scores"])
            fwd = _clean(fwd_raw)

        spy_close = computed["spy_close"].dropna()
        shared_dates = spy_close.index
        payload = {
            "built_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "asof": spy_close.index[-1].strftime("%Y-%m-%d"),
            "spy_last": round(float(spy_close.iloc[-1]), 2),
            "price_ctx": _clean(price_ctx),
            "regime_mult": _clean(computed.get("regime_mult")),
            "fragility": _clean(computed.get("h_scores")),
            "fragility_10d": _clean(computed.get("h_scores_10d")),
            "signals": signals,
            "n_active": sum(1 for s in signals if s["on"]),
            "forward_returns": fwd,
            "dates": [d.strftime("%Y-%m-%d") for d in shared_dates],
        }
        # Full history uses one shared date array.  The browser opens on the
        # latest year and can autorange to all history on double-click.
        payload["spy_series"] = {
            "close": [round(float(v), 2) for v in spy_close.values],
        }
        try:
            price_explorer = build_price_explorer(spy_df, shared_dates)
            if price_explorer:
                payload["price_explorer"] = price_explorer
                print(
                    "risk: price_explorer ok "
                    f"({len(price_explorer['assets'])} assets)"
                )
        except Exception:
            print("risk: price_explorer FAILED (continuing without it)")
            traceback.print_exc()
        frag_df = computed.get("frag_df")
        if frag_df is not None and not frag_df.empty:
            ft = frag_df.rolling(5, min_periods=1).mean()
            payload["fragility_series"] = {
                **{c: _aligned_values(ft[c], shared_dates, 1) for c in ft.columns},
            }
        payload["signal_detail"] = _build_signal_detail(
            computed["signals_ordered"], shared_dates, _signal_periods)

        # sizing_state + vol KPI are best-effort inside the best-effort
        # script: a failure here must not cost the rest of the risk payload
        try:
            sizing = build_sizing_state()
            if sizing:
                payload["sizing_state"] = sizing
                print(f"risk: sizing_state ok (score {sizing['score']}, "
                      f"throttle {'ON' if sizing['throttle_on'] else 'off'}, "
                      f"asof {sizing['asof']})")
        except Exception:
            print("risk: sizing_state FAILED (continuing without it)")
            traceback.print_exc()
        try:
            vol_kpi = build_vol_kpi()
            if vol_kpi:
                payload["vol_kpi"] = vol_kpi
        except Exception:
            print("risk: vol_kpi FAILED (continuing without it)")
            traceback.print_exc()
        try:
            vix_close = (closes["^VIX"].dropna()
                         if "^VIX" in closes.columns else None)
            similar_63d = (fwd_raw or {}).get("63d")
            dd_iv = build_similar_reading_drawdown_iv(
                spy_df, vix_close, similar_63d, load_vix_high()
            ) if vix_close is not None and similar_63d is not None else None
            if dd_iv:
                payload["drawdown_iv"] = dd_iv
                print(f"risk: drawdown_iv ok ({dd_iv['n_episodes']} exact 63d analogs, "
                      f"{len(dd_iv['rows_by_horizon'].get('63d', []))} complete 63d paths)")
        except Exception:
            print("risk: drawdown_iv FAILED (continuing without it)")
            traceback.print_exc()
        try:
            tc = build_trade_console(computed)
            if tc:
                payload["trade_console"] = tc
                print(f"risk: trade_console ok (class {tc.get('class_id')}, "
                      f"state {tc.get('state')}, n_ep {tc.get('n_ep')})")
        except Exception:
            print("risk: trade_console FAILED (continuing without it)")
            traceback.print_exc()
        try:
            atr_dn = build_atr_downside(spy_df)
            if atr_dn:
                payload["atr_downside"] = atr_dn
                dial = atr_dn.get("dial") or {}
                print(f"risk: atr_downside ok ({len(atr_dn.get('signals') or {})} signals"
                      + (f", dial {dial['value']} band [{dial['lo']},{dial['hi']}]"
                         if dial else ", no dial") + ")")
        except Exception:
            print("risk: atr_downside FAILED (continuing without it)")
            traceback.print_exc()

        payload["nuggets"] = build_nuggets(payload)

        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"), ensure_ascii=False)
        print(f"risk: wrote {OUT} ({os.path.getsize(OUT)/1024:.0f} KB)")
    except Exception:
        print("risk: FAILED (site will ship without risk payload)")
        traceback.print_exc()
    sys.exit(0)


if __name__ == "__main__":
    main()
