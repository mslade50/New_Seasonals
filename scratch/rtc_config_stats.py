"""Configuration-class episode study for the risk-page trade block.

Pre-registered taxonomy (stated BEFORE looking at outcomes; motivated by the
prior-evidence ledger, not mined from this data):
  - count buckets over the 6 BEARISH signals (FOMC excluded: Pre-FOMC Rally
    is a positive-EV signal per signal_horizon_stats_candidate.json)
  - each single signal on-or-recent (fired within trailing 5td)
  - DL variants (the one signal with episode-level significance at 21/42/63d)
  - the one pair with >=25 co-occurrence days: VRC+AR near-high
Episode = first day ENTERING the class (False->True), 21td cooldown between
episode starts. Forward metrics on SPY closes (adjusted basis, same series).
VIX threshold uses ^VIX Close (matches the dashboard complacency counter).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COOLDOWN_TD = 21
H_SHORT = 21
H_LONG = 63

cfg = pd.read_parquet(ROOT / "scratch" / "rtc_config_history.parquet")
cfg = cfg.sort_index()

mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                     filters=[("ticker", "==", "^VIX")])
vix = (mp.assign(date=pd.to_datetime(mp["date"]))
         .set_index("date")["Close"].sort_index())
vix = vix.reindex(cfg.index).ffill()

spy = cfg["spy_close"].astype(float)
n = len(cfg)

BEAR = ["DA", "VRC", "DL", "AR", "SRD", "DISP"]
bear_any = sum(cfg[f"any_{s}"].astype(int) for s in BEAR)
hi = cfg["near_52w_high"].astype(bool)

classes: dict[str, pd.Series] = {
    "NONE_active": cfg["n_on_or_recent"] == 0,
    "BEAR_1": bear_any == 1,
    "BEAR_2plus": bear_any >= 2,
    "BEAR_2plus_HI": (bear_any >= 2) & hi,
    "DL_any": cfg["any_DL"],
    "DL_any_HI": cfg["any_DL"] & hi,
    "DL_plus_other": cfg["any_DL"] & (bear_any >= 2),
    "DA_any": cfg["any_DA"],
    "DA_any_HI": cfg["any_DA"] & hi,
    "AR_any": cfg["any_AR"],
    "SRD_any": cfg["any_SRD"],
    "VRC_any": cfg["any_VRC"],
    "DISP_any": cfg["any_DISP"],
    "FOMC_any": cfg["any_FOMC"],
    "VRC_AR_HI": cfg["any_VRC"] & cfg["any_AR"] & hi,
}


def episode_starts(mask: pd.Series) -> list[int]:
    m = mask.to_numpy(dtype=bool)
    entries = np.flatnonzero(m & ~np.roll(m, 1))
    if m[0]:
        entries = np.unique(np.concatenate([[0], entries]))
    out: list[int] = []
    last = -10**9
    for i in entries:
        if i - last >= COOLDOWN_TD:
            out.append(int(i))
            last = i
    return out


def fwd_metrics(i: int) -> dict:
    px0 = spy.iloc[i]
    r21 = spy.iloc[i + H_SHORT] / px0 - 1 if i + H_SHORT < n else np.nan
    if i + H_LONG < n:
        r63 = spy.iloc[i + H_LONG] / px0 - 1
        path = spy.iloc[i + 1: i + H_LONG + 1]
        mdd = path.min() / px0 - 1
        vmax = vix.iloc[i + 1: i + H_LONG + 1].max()
    else:
        r63 = mdd = vmax = np.nan
    return {"r21": r21, "r63": r63, "mdd63": mdd, "vixmax63": vmax}


def summarize(rows: pd.DataFrame, n_ep: int | None = None,
              years: list[int] | None = None) -> dict:
    r63 = rows["r63"].dropna()
    r21 = rows["r21"].dropna()
    mdd = rows["mdd63"].dropna()
    vmx = rows["vixmax63"].dropna()
    d = {
        "n_episodes": int(n_ep if n_ep is not None else len(rows)),
        "n_63td_complete": int(len(r63)),
        "fwd21_mean": round(float(r21.mean()) * 100, 2) if len(r21) else None,
        "fwd21_median": round(float(r21.median()) * 100, 2) if len(r21) else None,
        "fwd63_mean": round(float(r63.mean()) * 100, 2) if len(r63) else None,
        "fwd63_median": round(float(r63.median()) * 100, 2) if len(r63) else None,
        "fwd63_p10": round(float(r63.quantile(0.10)) * 100, 2) if len(r63) else None,
        "fwd63_p90": round(float(r63.quantile(0.90)) * 100, 2) if len(r63) else None,
        "p_dd5_63td": round(float((mdd <= -0.05).mean()), 3) if len(mdd) else None,
        "p_dd10_63td": round(float((mdd <= -0.10).mean()), 3) if len(mdd) else None,
        "p_vix28_63td": round(float((vmx >= 28).mean()), 3) if len(vmx) else None,
    }
    if years is not None:
        d["episode_years"] = sorted(set(years))
        d["n_distinct_years"] = len(set(years))
    return d


results: dict[str, dict] = {}
for name, mask in classes.items():
    idxs = episode_starts(mask.astype(bool))
    rows = pd.DataFrame([fwd_metrics(i) for i in idxs])
    years = [cfg.index[i].year for i in idxs]
    d = summarize(rows, n_ep=len(idxs), years=years)
    d["episode_dates"] = [cfg.index[i].strftime("%Y-%m-%d") for i in idxs]
    d["days_in_class"] = int(mask.sum())
    results[name] = d

# Unconditional day-level baseline (every day with the horizon available)
base_rows = pd.DataFrame([fwd_metrics(i) for i in range(n)])
baseline = summarize(base_rows, n_ep=n)
baseline["note"] = "day-level unconditional, all 2512 days (overlapping windows)"

out = {
    "built_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M"),
    "frame": "2016-07-18..2026-07-15 (rd2 10y lookback; signal histories recomputed with TODAY'S signal code - lookahead caveat applies)",
    "episode_rule": f"class False->True entry, {COOLDOWN_TD}td cooldown between starts",
    "min_readable_episodes": 12,
    "unreadable_classes": [k for k, v in results.items() if v["n_episodes"] < 12],
    "baseline": baseline,
    "classes": results,
}
# Preserve the parallel option-EV agent's output (different taxonomy/episode
# rule, written concurrently to this same path on 2026-07-16) under a
# namespaced key instead of clobbering it.
path = ROOT / "scratch" / "rtc_config_stats.json"
if path.exists():
    try:
        prior = json.loads(path.read_text())
        if "meta" in prior and "episode_rule" in prior.get("meta", {}):
            out["structure_ev_agent"] = prior
        elif "structure_ev_agent" in prior:
            out["structure_ev_agent"] = prior["structure_ev_agent"]
    except Exception:
        pass
path.write_text(json.dumps(out, indent=1))

hdr = (f"{'class':<16}{'N':>4}{'N63':>5}{'f21m':>7}{'f63m':>7}{'f63med':>8}"
       f"{'p10':>7}{'p90':>7}{'P(dd5)':>8}{'P(dd10)':>9}{'P(vix28)':>10}{'yrs':>5}")
print(hdr)
b = baseline
print(f"{'BASELINE':<16}{b['n_episodes']:>4}{b['n_63td_complete']:>5}"
      f"{b['fwd21_mean']:>7}{b['fwd63_mean']:>7}{b['fwd63_median']:>8}"
      f"{b['fwd63_p10']:>7}{b['fwd63_p90']:>7}{b['p_dd5_63td']:>8}"
      f"{b['p_dd10_63td']:>9}{b['p_vix28_63td']:>10}{'':>5}")
for name, d in results.items():
    flag = " <12EP" if d["n_episodes"] < 12 else ""
    print(f"{name:<16}{d['n_episodes']:>4}{d['n_63td_complete']:>5}"
          f"{str(d['fwd21_mean']):>7}{str(d['fwd63_mean']):>7}{str(d['fwd63_median']):>8}"
          f"{str(d['fwd63_p10']):>7}{str(d['fwd63_p90']):>7}{str(d['p_dd5_63td']):>8}"
          f"{str(d['p_dd10_63td']):>9}{str(d['p_vix28_63td']):>10}"
          f"{d['n_distinct_years']:>5}{flag}")
