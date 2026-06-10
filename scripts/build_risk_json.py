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
        }
        # 1y of SPY closes + fragility series for a small context chart
        tail = spy_close.tail(252)
        payload["spy_series"] = {
            "dates": [d.strftime("%Y-%m-%d") for d in tail.index],
            "close": [round(float(v), 2) for v in tail.values],
        }
        frag_df = computed.get("frag_df")
        if frag_df is not None and not frag_df.empty:
            ft = frag_df.rolling(5, min_periods=1).mean().tail(252)
            payload["fragility_series"] = {
                "dates": [d.strftime("%Y-%m-%d") for d in ft.index],
                **{c: [_clean(round(float(v), 1)) if v == v else None for v in ft[c].values]
                   for c in ft.columns},
            }

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
