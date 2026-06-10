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

        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"), ensure_ascii=False)
        print(f"risk: wrote {OUT} ({os.path.getsize(OUT)/1024:.0f} KB)")
    except Exception:
        print("risk: FAILED (site will ship without risk payload)")
        traceback.print_exc()
    sys.exit(0)


if __name__ == "__main__":
    main()
