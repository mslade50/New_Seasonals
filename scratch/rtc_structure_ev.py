"""rtc_structure_ev.py — per-episode EV of 3m SPY put structures conditional on
pre-registered signal-configuration classes.

Structures priced (BS-European, VIX3M entry IV, linear skew +0.40 vol/OTM%,
5% premium haircut per side — conventions from scratch/ultracode_research/
ca_overlays.py, the adversarial model that killed the old hedge block):
  (a) 3m ~30/10-delta put spread (strikes solved for BS-delta under skewed IV)
  (b) 3m ~10-delta tail put
  (c) no trade -> conditional forward SPY distribution only

Payoff = hold-to-expiry INTRINSIC vs realized SPY 63td forward (understates
exit value of early vol spikes — conservative on the payoff side; the static
skew + VIX3M basis are the optimistic side).

Classes are PRE-REGISTERED (no config mining): count buckets of the 6
bearish-evidence signals (Pre-FOMC excluded — its 21d edge is POSITIVE),
plus DL-near-high (today's state) and multi-near-high. Episodes = greedy
non-overlapping entries spaced >= horizon td. Bootstrap CI over episodes.

Outputs: scratch/rtc_config_stats.json, scratch/rtc_structure_ev.json
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG_PATH = os.path.join(_ROOT, "scratch", "rtc_config_history.parquet")
MP_PATH = os.path.join(_ROOT, "data", "master_prices.parquet")
OUT_STATS = os.path.join(_ROOT, "scratch", "rtc_config_stats.json")
OUT_EV = os.path.join(_ROOT, "scratch", "rtc_structure_ev.json")

TENOR_TD = 63
HAIRCUT = 0.05
SLOPES = {"base": 0.40, "lo": 0.30, "hi": 0.50}
BEAR = ["DA", "VRC", "DL", "AR", "SRD", "DISP"]  # FOMC excluded (positive 21d)
N_BOOT = 4000
RNG = np.random.default_rng(7)


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
    """Solve K so BS put delta under the K-dependent skewed IV = target (<0)."""
    def f(K):
        return put_delta(S, K, T, r, skew_iv(iv_base_pct, K / S, slope)) - target
    lo, hi = 0.45 * S, 1.10 * S
    if f(lo) * f(hi) > 0:  # pathological vol; fall back to flat-vol solve
        return np.nan
    return brentq(f, lo, hi, xtol=1e-4)


def price_structures(S, iv_base, r, S_T, slope):
    """Returns dict of per-entry economics for spread + tail at one slope."""
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
    return {
        "spread_ret_cost": (sp_pay - debit) / debit,            # % of debit
        "spread_bps_notional": (sp_pay - debit) / S * 1e4,      # bps of spot
        "spread_cost_pct_notional": debit / S * 100,
        "tail_ret_cost": (tp_pay - tail_cost) / tail_cost,
        "tail_bps_notional": (tp_pay - tail_cost) / S * 1e4,
        "tail_cost_pct_notional": tail_cost / S * 100,
        "k30_m": k30 / S, "k10_m": k10 / S,
    }


def boot_ci(vals, n=N_BOOT):
    vals = np.asarray(vals, float)
    if len(vals) < 3:
        return [float("nan"), float("nan")]
    means = RNG.choice(vals, size=(n, len(vals)), replace=True).mean(axis=1)
    return [float(np.percentile(means, 5)), float(np.percentile(means, 95))]


def episodes(mask: pd.Series, spacing: int) -> list[int]:
    """Greedy non-overlapping entry positions (integer locs) spaced >= spacing."""
    locs, last = [], -10**9
    pos = np.flatnonzero(mask.to_numpy())
    for p in pos:
        if p - last >= spacing:
            locs.append(int(p))
            last = p
    return locs


def main() -> None:
    df = pd.read_parquet(CFG_PATH)
    mp = pd.read_parquet(MP_PATH)
    mp["date"] = pd.to_datetime(mp["date"])

    def series(tkr):
        s = mp.loc[mp["ticker"] == tkr, ["date", "Close"]].set_index("date")["Close"]
        return s.sort_index()

    spy = series("SPY")
    vix3m = series("^VIX3M").reindex(spy.index).ffill()
    irx = (series("^IRX") / 100.0).reindex(spy.index).ffill().fillna(0.04)

    idx = df.index
    spy_c = df["spy_close"]
    # forward closes on the full SPY calendar (so the last config days can
    # still look 63td ahead if master extends — here they coincide; drop tail)
    spy_pos = spy.index.get_indexer(idx)
    assert (spy_pos >= 0).all(), "config dates missing from master SPY"
    fwd63_pos = spy_pos + TENOR_TD
    fwd21_pos = spy_pos + 21
    valid63 = fwd63_pos < len(spy)
    valid21 = fwd21_pos < len(spy)
    fwd63 = np.full(len(idx), np.nan)
    fwd21 = np.full(len(idx), np.nan)
    fwd63[valid63] = spy.to_numpy()[fwd63_pos[valid63]] / spy_c.to_numpy()[valid63] - 1
    fwd21[valid21] = spy.to_numpy()[fwd21_pos[valid21]] / spy_c.to_numpy()[valid21] - 1
    # worst close within the 63td window (for "material correction" framing)
    spy_np = spy.to_numpy()
    worst63 = np.full(len(idx), np.nan)
    for i in range(len(idx)):
        if valid63[i]:
            lo = spy_np[spy_pos[i] + 1: fwd63_pos[i] + 1].min()
            worst63[i] = lo / spy_c.iloc[i] - 1

    n_bear_any = df[[f"any_{s}" for s in BEAR]].sum(axis=1)
    classes = {
        "baseline_all": pd.Series(True, index=idx),
        "quiet_0sig": n_bear_any == 0,
        "one_sig": n_bear_any == 1,
        "multi_2plus": n_bear_any >= 2,
        "multi_2plus_hi": (n_bear_any >= 2) & df["near_52w_high"],
        "dl_hi_200p": df["any_DL"] & df["near_52w_high"] & df["above_200d"],
    }
    today_row = df.iloc[-1]
    today_classes = [k for k, m in classes.items() if bool(m.iloc[-1])]

    cfg_stats, ev_out = {}, {}
    for name, mask in classes.items():
        m63 = mask & pd.Series(valid63, index=idx)
        m21 = mask & pd.Series(valid21, index=idx)
        ep63 = episodes(m63, TENOR_TD)
        ep21 = episodes(m21, 21)
        f63 = fwd63[ep63]
        f21 = fwd21[ep21]
        w63 = worst63[ep63]
        yrs63 = sorted({idx[p].year for p in ep63})
        cfg_stats[name] = {
            "n_days": int(mask.sum()),
            "n_episodes_63d": len(ep63), "n_episodes_21d": len(ep21),
            "episode_years_63d": yrs63,
            "fwd63_mean_pct": float(np.mean(f63) * 100) if len(f63) else None,
            "fwd63_median_pct": float(np.median(f63) * 100) if len(f63) else None,
            "fwd63_ci_mean_pct": [x * 100 for x in boot_ci(f63)] if len(f63) >= 3 else None,
            "p_fwd63_le_m5": float(np.mean(f63 <= -0.05)) if len(f63) else None,
            "p_fwd63_le_m10": float(np.mean(f63 <= -0.10)) if len(f63) else None,
            "p_worst63_le_m10": float(np.mean(w63 <= -0.10)) if len(w63) else None,
            "fwd21_mean_pct": float(np.mean(f21) * 100) if len(f21) else None,
            "p_fwd21_le_m5": float(np.mean(f21 <= -0.05)) if len(f21) else None,
        }

        # ---- structure EV at episode entries
        rows = {sl: [] for sl in SLOPES}
        for p in ep63:
            d = idx[p]
            S = spy_c.iloc[p]
            ivb = vix3m.loc[:d].iloc[-1]
            r = irx.loc[:d].iloc[-1]
            S_T = spy_np[spy_pos[p] + TENOR_TD]
            for sl, slope in SLOPES.items():
                econ = price_structures(S, ivb, r, S_T, slope)
                if econ is not None:
                    econ["date"] = str(d.date())
                    rows[sl].append(econ)

        base = rows["base"]
        def agg(key):
            v = [r_[key] for r_ in base]
            return {
                "mean": float(np.mean(v)) if v else None,
                "median": float(np.median(v)) if v else None,
                "ci5_95": boot_ci(v),
                "hit_rate": float(np.mean(np.asarray(v) > 0)) if v else None,
            }
        sens = {}
        for sl in ("lo", "hi"):
            sens[sl] = {
                "spread_ret_cost_mean": float(np.mean([r_["spread_ret_cost"] for r_ in rows[sl]])) if rows[sl] else None,
                "tail_ret_cost_mean": float(np.mean([r_["tail_ret_cost"] for r_ in rows[sl]])) if rows[sl] else None,
            }
        ev_out[name] = {
            "n_episodes": len(base),
            "episode_dates": [r_["date"] for r_ in base],
            "spread_ret_on_cost": agg("spread_ret_cost"),
            "spread_bps_notional": agg("spread_bps_notional"),
            "spread_cost_pct_notional_mean": float(np.mean([r_["spread_cost_pct_notional"] for r_ in base])) if base else None,
            "tail_ret_on_cost": agg("tail_ret_cost"),
            "tail_bps_notional": agg("tail_bps_notional"),
            "tail_cost_pct_notional_mean": float(np.mean([r_["tail_cost_pct_notional"] for r_ in base])) if base else None,
            "avg_k30_moneyness": float(np.mean([r_["k30_m"] for r_ in base])) if base else None,
            "avg_k10_moneyness": float(np.mean([r_["k10_m"] for r_ in base])) if base else None,
            "skew_slope_sensitivity": sens,
        }
        print(f"{name:16s} days={cfg_stats[name]['n_days']:5d} ep63={len(ep63):3d} "
              f"fwd63={cfg_stats[name]['fwd63_mean_pct']} "
              f"spreadEV={ev_out[name]['spread_ret_on_cost']['mean']} "
              f"tailEV={ev_out[name]['tail_ret_on_cost']['mean']}")

    meta = {
        "built": pd.Timestamp.utcnow().isoformat(),
        "span": [str(idx[0].date()), str(idx[-1].date())],
        "n_days_total": len(idx),
        "bearish_signals": BEAR,
        "fomc_excluded_reason": "Pre-FOMC Rally 21d episode_t is POSITIVE (+2.03)",
        "tenor_td": TENOR_TD, "haircut_per_side": HAIRCUT,
        "skew_slope": SLOPES, "pricing": "BS-European, entry IV=VIX3M, r=^IRX, "
        "linear skew, payoff=hold-to-expiry intrinsic (understates early-vol-spike exits)",
        "episode_rule": "greedy non-overlapping entries spaced >= horizon td",
        "lookahead_caveat": "signal fire histories recomputed from today's code; "
        "definitions/thresholds carry hindsight PIT cannot cure",
        "today": {"date": str(idx[-1].date()), "config": today_row["config"],
                  "classes": today_classes},
    }
    with open(OUT_STATS, "w") as f:
        json.dump({"meta": meta, "classes": cfg_stats}, f, indent=1)
    with open(OUT_EV, "w") as f:
        json.dump({"meta": meta, "classes": ev_out}, f, indent=1)
    print(f"\nwrote {OUT_STATS}\nwrote {OUT_EV}")
    print(f"today: {meta['today']}")


if __name__ == "__main__":
    main()
