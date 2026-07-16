"""Trade-console stats: configuration-class outcome + structure-EV tables.

The SINGLE writer of data/trade_console_stats.json, the frozen evidence file
behind the risk page's Trade Console (spec: scratch/ultracode_research/
RISK_TRADE_CONSOLE_2026-07-16.md). Productionizes scratch/rtc_config_probe.py
+ rtc_config_stats.py + rtc_structure_ev.py with the critiques applied:

- ONE episode rule everywhere: greedy non-overlapping entries spaced >= the
  63td horizon (the probes' 21td-cooldown stats overlapped windows ~3:1).
- Episode-level t vs an episode-deduped BASELINE (not day-level windows).
- Adds P(fwd63<=-5/-10%), medians, drop-best-episode, hit counts, oracle
  exit bounds, and the class fingerprint + built_utc + git sha.
- Class set v1 (frozen 2026-07-16, screened post-hoc, one-shot): BEAR_2PLUS,
  DL_TAIL, AR_DRAWDOWN, SRD_TAIL, DA_SOFT, VRC_CONTEXT, NONE. Pre-FOMC never
  counts bearish. Dispersion alone anchors nothing (8 episodes).

Pricing is deliberately adversarial and MODEL-ONLY (no historical chains):
BS-European, entry IV = VIX3M, r = ^IRX, linear skew +0.40 vol/OTM%
(0.30/0.50 sensitivity), 5% premium haircut per side, payoff = hold-to-expiry
intrinsic (understates early-vol-spike exits; oracle bound reported).

Regeneration is a DELIBERATE review action (freeze policy A2) — never wire
this into a nightly workflow.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

OUT_DEFAULT = os.path.join(_ROOT, "data", "trade_console_stats.json")

CLASS_SET_VERSION = "v1 (frozen 2026-07-16, screened post-hoc, one-shot)"
RECENT_TD = 5
TENOR_TD = 63
HAIRCUT = 0.05
SLOPES = {"base": 0.40, "lo": 0.30, "hi": 0.50}
N_BOOT = 4000
MIN_READABLE = 12

ABBR = {
    "Distribution Dominance": "DA",
    "VIX Range Compression": "VRC",
    "Defensive Leadership": "DL",
    "Pre-FOMC Rally": "FOMC",
    "Low Absorption Ratio": "AR",
    "Seasonal Rank Divergence": "SRD",
    "Dispersion": "DISP",
}
BEAR = ["DA", "VRC", "DL", "AR", "SRD", "DISP"]  # FOMC excluded: positive 21d edge

# Precedence, first match wins (spec section 3). The mask key names the
# class-defining condition evaluated on the config frame.
CLASS_PRECEDENCE = [
    "BEAR_2PLUS", "DL_TAIL", "AR_DRAWDOWN", "SRD_TAIL",
    "DA_SOFT", "VRC_CONTEXT", "NONE",
]


# ---------------------------------------------------------------------------
# Shared with scripts/build_risk_json.py — the classifier must be IDENTICAL
# between stats generation and the nightly payload build.
# ---------------------------------------------------------------------------
def tri_state_any(history: pd.Series, index: pd.Index) -> pd.Series:
    """on-or-recent-5td: True while the signal is ON, or within RECENT_TD
    sessions of an activation start while not ON."""
    h = history.reindex(index)
    h = h.astype(object).where(pd.notna(h), False).astype(bool)
    starts = h & ~h.shift(1, fill_value=False)
    recent = starts.rolling(RECENT_TD, min_periods=1).max().astype(bool)
    return h | recent


def build_config_frame(signals_ordered: dict, spy_close: pd.Series) -> pd.DataFrame:
    idx = spy_close.index
    out = pd.DataFrame(index=idx)
    for name, abbr in ABBR.items():
        hist = (signals_ordered.get(name) or {}).get("signal_history")
        if hist is None or not hasattr(hist, "empty") or hist.empty:
            raise RuntimeError(f"signal history missing for {name}")
        h = hist.reindex(idx)
        h = h.astype(object).where(pd.notna(h), False).astype(bool)
        out[f"on_{abbr}"] = h
        out[f"any_{abbr}"] = tri_state_any(hist, idx)
    out["n_bear_any"] = sum(out[f"any_{s}"].astype(int) for s in BEAR)
    out["near_52w_high"] = spy_close >= spy_close.rolling(252, min_periods=60).max() * 0.98
    out["above_200d"] = spy_close > spy_close.rolling(200).mean()
    out["spy_close"] = spy_close.astype(float)
    return out


def class_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Class-DEFINING masks (independent, for episode stats). Today's card
    classification applies CLASS_PRECEDENCE over these same conditions."""
    bear = frame["n_bear_any"]
    return {
        "BEAR_2PLUS": bear >= 2,
        "DL_TAIL": frame["any_DL"],
        "AR_DRAWDOWN": frame["any_AR"],
        "SRD_TAIL": frame["any_SRD"],
        "DA_SOFT": frame["any_DA"],
        "VRC_CONTEXT": frame["any_VRC"],
        "NONE": bear == 0,
    }


def classify_row(row: pd.Series) -> str:
    masks = {
        "BEAR_2PLUS": row["n_bear_any"] >= 2,
        "DL_TAIL": bool(row["any_DL"]),
        "AR_DRAWDOWN": bool(row["any_AR"]),
        "SRD_TAIL": bool(row["any_SRD"]),
        "DA_SOFT": bool(row["any_DA"]),
        "VRC_CONTEXT": bool(row["any_VRC"]),
        "NONE": True,
    }
    for name in CLASS_PRECEDENCE:
        if masks[name]:
            return name
    return "NONE"


def class_fingerprint(signal_names) -> str:
    """Guards against a retired/renamed signal taxonomy silently feeding the
    console (the old hedge block's failure). Catches name-set and class-set
    drift; threshold drift inside an unchanged signal name is NOT caught —
    that risk is carried by the vintage tripwire and deliberate review."""
    basis = "|".join(sorted(signal_names)) + "||" + CLASS_SET_VERSION + "||" + \
        ">".join(CLASS_PRECEDENCE)
    return hashlib.sha256(basis.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Pricing (conventions from scratch/ultracode_research/ca_overlays.py)
# ---------------------------------------------------------------------------
def bs_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def put_delta(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1.0


def skew_iv(iv_base_pct, moneyness, slope):
    return iv_base_pct / 100.0 + slope * max(0.0, 1.0 - moneyness)


def strike_for_delta(S, T, r, iv_base_pct, target, slope):
    def f(K):
        return put_delta(S, K, T, r, skew_iv(iv_base_pct, K / S, slope)) - target
    lo, hi = 0.45 * S, 1.10 * S
    if f(lo) * f(hi) > 0:
        return np.nan
    return brentq(f, lo, hi, xtol=1e-4)


def price_structures(S, iv_base, r, S_T, S_min, slope):
    """Per-entry economics: spread + tail, expiry payoff AND oracle bound
    (intrinsic at the worst close inside the window — a perfect-exit ceiling,
    not an attainable return)."""
    T = TENOR_TD / 252.0
    k30 = strike_for_delta(S, T, r, iv_base, -0.30, slope)
    k10 = strike_for_delta(S, T, r, iv_base, -0.10, slope)
    if not np.isfinite(k30) or not np.isfinite(k10) or k10 >= k30:
        return None
    p30 = bs_put(S, k30, T, r, skew_iv(iv_base, k30 / S, slope))
    p10 = bs_put(S, k10, T, r, skew_iv(iv_base, k10 / S, slope))
    debit = p30 * (1 + HAIRCUT) - p10 * (1 - HAIRCUT)
    tail_cost = p10 * (1 + HAIRCUT)
    if debit <= 0 or tail_cost <= 0:
        return None
    sp_pay = max(k30 - S_T, 0.0) - max(k10 - S_T, 0.0)
    tp_pay = max(k10 - S_T, 0.0)
    sp_oracle = min(max(k30 - S_min, 0.0), k30 - k10)
    tp_oracle = max(k10 - S_min, 0.0)
    return {
        "spread_ret_cost": (sp_pay - debit) / debit,
        "spread_cost_pct_notional": debit / S * 100,
        "spread_oracle_ret_cost": (sp_oracle - debit) / debit,
        "tail_ret_cost": (tp_pay - tail_cost) / tail_cost,
        "tail_cost_pct_notional": tail_cost / S * 100,
        "tail_oracle_ret_cost": (tp_oracle - tail_cost) / tail_cost,
        "spread_hit": sp_pay > debit,
        "tail_hit": tp_pay > tail_cost,
    }


# ---------------------------------------------------------------------------
# Episode machinery — ONE rule: greedy entries spaced >= TENOR_TD
# ---------------------------------------------------------------------------
def episodes(mask: np.ndarray, spacing: int = TENOR_TD) -> list[int]:
    locs, last = [], -10 ** 9
    for p in np.flatnonzero(mask):
        if p - last >= spacing:
            locs.append(int(p))
            last = p
    return locs


def boot_ci(vals, rng, n=N_BOOT):
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 3:
        return None
    means = rng.choice(vals, size=(n, len(vals)), replace=True).mean(axis=1)
    return [round(float(np.percentile(means, 5)), 4),
            round(float(np.percentile(means, 95)), 4)]


def welch_t(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return None
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    if va + vb == 0:
        return None
    return round(float((a.mean() - b.mean()) / np.sqrt(va + vb)), 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=OUT_DEFAULT)
    args = parser.parse_args()

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

    from daily_risk_report import download_data, compute_all_signals

    print("downloading data (production pipeline) ...")
    spy_df, closes, sp500_closes = download_data()
    computed = compute_all_signals(spy_df, closes, sp500_closes)
    signals_ordered = computed["signals_ordered"]
    spy_close = computed["spy_close"].dropna()

    frame = build_config_frame(signals_ordered, spy_close)
    n = len(frame)
    spy = frame["spy_close"].to_numpy()

    mp = pd.read_parquet(os.path.join(_ROOT, "data", "master_prices.parquet"),
                         filters=[("ticker", "in", ["^VIX", "^VIX3M", "^IRX"])])
    mp["date"] = pd.to_datetime(mp["date"])

    def aligned(tkr, default=None):
        s = (mp[mp["ticker"] == tkr].set_index("date")["Close"]
             .sort_index().reindex(frame.index).ffill())
        return s.fillna(default) if default is not None else s

    vix = aligned("^VIX")
    vix3m = aligned("^VIX3M")
    irx = aligned("^IRX", default=4.0) / 100.0

    fwd21 = np.full(n, np.nan)
    fwd63 = np.full(n, np.nan)
    worst63 = np.full(n, np.nan)
    vixmax63 = np.full(n, np.nan)
    vix_np = vix.to_numpy()
    for i in range(n):
        if i + 21 < n:
            fwd21[i] = spy[i + 21] / spy[i] - 1
        if i + TENOR_TD < n:
            fwd63[i] = spy[i + TENOR_TD] / spy[i] - 1
            window = spy[i + 1: i + TENOR_TD + 1]
            worst63[i] = window.min() / spy[i] - 1
            vixmax63[i] = np.nanmax(vix_np[i + 1: i + TENOR_TD + 1])

    valid63 = np.isfinite(fwd63)
    rng = np.random.default_rng(7)

    baseline_eps = episodes(valid63)
    base_f63 = fwd63[baseline_eps]

    def dist_block(eps: list[int]) -> dict:
        f63 = fwd63[eps]
        f21 = fwd21[eps]
        w63 = worst63[eps]
        vm = vixmax63[eps]
        f63v = f63[np.isfinite(f63)]
        years = sorted({frame.index[p].year for p in eps})
        year_counts = pd.Series([frame.index[p].year for p in eps]).value_counts()
        max_year_share = float(year_counts.max() / len(eps)) if eps else None
        drop_best = None
        if len(f63v) > 3:
            drop_best = round(float(np.sort(f63v)[:-1].mean()) * 100, 2)
        return {
            "n_episodes": len(eps),
            "episode_dates": [frame.index[p].strftime("%Y-%m-%d") for p in eps],
            "episode_years": years,
            "n_distinct_years": len(years),
            "max_year_share": round(max_year_share, 2) if max_year_share else None,
            "fwd21_mean_pct": round(float(np.nanmean(f21)) * 100, 2) if np.isfinite(f21).any() else None,
            "fwd63_mean_pct": round(float(f63v.mean()) * 100, 2) if len(f63v) else None,
            "fwd63_median_pct": round(float(np.median(f63v)) * 100, 2) if len(f63v) else None,
            "fwd63_p10_pct": round(float(np.percentile(f63v, 10)) * 100, 2) if len(f63v) >= 5 else None,
            "fwd63_p90_pct": round(float(np.percentile(f63v, 90)) * 100, 2) if len(f63v) >= 5 else None,
            "fwd63_mean_ci_pct": [round(x * 100, 2) for x in (boot_ci(f63v, rng) or [])] or None,
            "fwd63_drop_best_mean_pct": drop_best,
            "p_fwd63_le_m5": round(float(np.mean(f63v <= -0.05)), 3) if len(f63v) else None,
            "p_fwd63_le_m10": round(float(np.mean(f63v <= -0.10)), 3) if len(f63v) else None,
            "p_dd5_63td": round(float(np.mean(w63[np.isfinite(w63)] <= -0.05)), 3) if np.isfinite(w63).any() else None,
            "p_dd10_63td": round(float(np.mean(w63[np.isfinite(w63)] <= -0.10)), 3) if np.isfinite(w63).any() else None,
            "p_vix28_63td": round(float(np.mean(vm[np.isfinite(vm)] >= 28)), 3) if np.isfinite(vm).any() else None,
            "episode_t_vs_baseline": welch_t(f63v, base_f63),
        }

    def ev_block(eps: list[int]) -> dict:
        rows = {sl: [] for sl in SLOPES}
        for p in eps:
            if not valid63[p]:
                continue
            S = spy[p]
            S_T = spy[p + TENOR_TD]
            S_min = spy[p + 1: p + TENOR_TD + 1].min()
            ivb = vix3m.iloc[p]
            r = irx.iloc[p]
            if not np.isfinite(ivb):
                continue
            for sl, slope in SLOPES.items():
                econ = price_structures(S, ivb, r, S_T, S_min, slope)
                if econ is not None:
                    rows[sl].append(econ)
        base = rows["base"]
        if not base:
            return {"n_priced": 0}

        def stat(key):
            v = [r_[key] for r_ in base]
            return {"mean": round(float(np.mean(v)), 3),
                    "median": round(float(np.median(v)), 3),
                    "ci5_95": boot_ci(v, rng)}

        return {
            "n_priced": len(base),
            "spread_ret_on_cost": stat("spread_ret_cost"),
            "spread_cost_pct_notional_mean": round(float(np.mean(
                [r_["spread_cost_pct_notional"] for r_ in base])), 2),
            "spread_hits": int(sum(r_["spread_hit"] for r_ in base)),
            "spread_oracle_mean": round(float(np.mean(
                [r_["spread_oracle_ret_cost"] for r_ in base])), 3),
            "tail_ret_on_cost": stat("tail_ret_cost"),
            "tail_cost_pct_notional_mean": round(float(np.mean(
                [r_["tail_cost_pct_notional"] for r_ in base])), 2),
            "tail_hits": int(sum(r_["tail_hit"] for r_ in base)),
            "tail_oracle_mean": round(float(np.mean(
                [r_["tail_oracle_ret_cost"] for r_ in base])), 3),
            "skew_slope_sensitivity": {
                sl: {"spread_mean": round(float(np.mean(
                        [r_["spread_ret_cost"] for r_ in rows[sl]])), 3)
                     if rows[sl] else None,
                     "tail_mean": round(float(np.mean(
                        [r_["tail_ret_cost"] for r_ in rows[sl]])), 3)
                     if rows[sl] else None}
                for sl in ("lo", "hi")},
        }

    results = {}
    for name, mask in class_masks(frame).items():
        eps = episodes((mask.to_numpy() & valid63))
        block = dist_block(eps)
        block["days_in_class"] = int(mask.sum())
        block["readable"] = block["n_episodes"] >= MIN_READABLE
        block["structures"] = ev_block(eps)
        results[name] = block
        print(f"{name:14s} ep={block['n_episodes']:3d} "
              f"f63m={block['fwd63_mean_pct']} p(dd10)={block['p_dd10_63td']} "
              f"spreadEV={block['structures'].get('spread_ret_on_cost', {}).get('mean')}")

    baseline = dist_block(baseline_eps)
    baseline["days_in_class"] = int(valid63.sum())
    baseline["structures"] = ev_block(baseline_eps)

    disp_eps = episodes((frame["any_DISP"].to_numpy() & valid63))

    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, cwd=_ROOT,
                             timeout=10).stdout.strip()
    except Exception:
        sha = "unknown"

    payload = {
        "built_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "data_through": frame.index[-1].strftime("%Y-%m-%d"),
        "git_sha": sha,
        "class_set_version": CLASS_SET_VERSION,
        "class_precedence": CLASS_PRECEDENCE,
        "fingerprint": class_fingerprint(signals_ordered.keys()),
        "episode_rule": f"greedy non-overlapping entries spaced >= {TENOR_TD} td (unified)",
        "recent_td": RECENT_TD,
        "min_readable_episodes": MIN_READABLE,
        "pricing": ("BS-European, entry IV=VIX3M, r=^IRX, linear skew "
                    f"+{SLOPES['base']} vol/OTM% ({SLOPES['lo']}/{SLOPES['hi']} sens), "
                    f"{HAIRCUT:.0%}/side haircut, payoff=hold-to-expiry intrinsic; "
                    "oracle=intrinsic at worst close in window (ceiling, not attainable)"),
        "lookahead_caveat": ("signal fire histories recomputed from today's code; "
                             "definitions/thresholds carry hindsight PIT cannot cure"),
        "n_days": n,
        "span": [frame.index[0].strftime("%Y-%m-%d"),
                 frame.index[-1].strftime("%Y-%m-%d")],
        "baseline": baseline,
        "classes": results,
        "dispersion_episodes": len(disp_eps),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    print(f"\nwrote {args.out} (fingerprint {payload['fingerprint']}, "
          f"data through {payload['data_through']})")


if __name__ == "__main__":
    main()
