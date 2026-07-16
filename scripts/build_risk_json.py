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

    tail = ma.tail(252)
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
        },
        "episodes": _run_episodes(pit_state),
        "exposure": expo,
    }


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
            "lines": [f"The fragility framework's core-exposure dial (0.6-1.8x) says {posture}. "
                      f"This is the same multiplier the AM scan writes to exposure_state.json."],
        })

    return out


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
