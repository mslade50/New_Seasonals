"""dd_pit step 2: extend the point-in-time dial through the latest close.

Method = scratch/pit_reestimate.py / cross_strategy_regime_0_pit_dial.py,
unchanged: expanding-window diff_mean edges at each year-end (vintage Y-1
scores calendar year Y; 2026 is scored on weights fit through 2025-12-31),
production fragility_core.compute_fragility_timeseries, live basis = raw ->
rolling(5) -> rolling(10). Inputs are the fire histories re-extracted in
step 1 (pit_signals_extended.pkl), restricted to the 7 composite signals
(Equity P/C Complacency is a 5d-only contributor and is not in the 63d
column by construction).

Writes pit_dial_extended.parquet with columns:
  pit            vintage-lagged dial, 10d MA of the 63d 5d-smoothed score
  cur_recompute  current-weights dial (edges fit on the full sample through
                 the last close), same basis
  live           data/rd2_fragility.parquet 63d -> rolling(10)  (recompute
                 vintage before 2026-07-02, append-only PIT after)
  pit_raw63 / cur_raw63   the un-smoothed 63d composites
  pit_from_old   the study's original PIT series (cross_strategy_regime_pit_dial.parquet)
"""
from __future__ import annotations
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))


class _NoOp:
    def __getattr__(self, name): return self
    def __call__(self, *a, **k): return self
    def __bool__(self): return False
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def cache_data(self, *a, **k):
        def deco(fn): return fn
        return deco
    cache_resource = cache_data


sys.modules["streamlit"] = _NoOp()
from fragility_core import compute_fragility_timeseries  # noqa: E402

SEVEN = ["Distribution Dominance", "VIX Range Compression", "Defensive Leadership", "Pre-FOMC Rally",
         "Low Absorption Ratio", "Seasonal Rank Divergence", "Dispersion"]
raw = pickle.load(open(HERE / "pit_signals_extended.pkl", "rb"))
fires: pd.DataFrame = raw["fires"][SEVEN].copy()
spy: pd.Series = raw["spy_close"].copy()
spy.index = pd.to_datetime(spy.index); fires.index = pd.to_datetime(fires.index)
LAST = pd.Timestamp("2026-09-01")
spy = spy[spy.index <= LAST]; fires = fires[fires.index <= LAST]
HORIZONS = {"5d": 5, "21d": 21, "63d": 63}


def estimate_stats(end_date) -> dict:
    end = pd.Timestamp(end_date)
    out = {}
    for name in fires.columns:
        hor = {}
        for hl, h in HORIZONS.items():
            fwd = (spy.shift(-h) / spy - 1.0) * 100.0
            valid = fwd.index[fwd.index <= end - pd.Timedelta(days=int(h * 1.6))]
            fwd = fwd.reindex(valid).dropna()
            f = fires[name].reindex(fwd.index).fillna(False)
            dm = float(fwd[f].mean() - fwd.mean()) if f.sum() >= 10 else None
            hor[hl] = {"diff_mean": dm}
        out[name] = {"horizons": hor}
    return {"signals": out}


sig_dict = {name: {"signal_history": fires[name]} for name in fires.columns}
pit_frames = []; vintages = {}
for year in range(2018, 2027):
    vint = estimate_stats(f"{year - 1}-12-31")
    frame = compute_fragility_timeseries(sig_dict, spy, vint)
    pit_frames.append(frame[frame.index.year == year])
    vintages[year - 1] = {n: round(max(0.0, -(v["horizons"]["63d"]["diff_mean"] or 0)), 3) for n, v in vint["signals"].items()}
    print(f"vintage {year-1}: 63d edges {vintages[year-1]}")
pit_raw = pd.concat(pit_frames).sort_index()
cur_stats = estimate_stats(LAST)
cur_raw = compute_fragility_timeseries(sig_dict, spy, cur_stats)
print("current-weights (through 2026-09-01) 63d edges:",
      {n: round(max(0.0, -(v["horizons"]["63d"]["diff_mean"] or 0)), 3) for n, v in cur_stats["signals"].items()})
shipped = json.load(open(ROOT / "data/signal_horizon_stats.json"))["signals"]
print("shipped JSON 63d edges (frozen weights the live parquet uses):",
      {n: round(max(0.0, -(shipped.get(n, {}).get("horizons", {}).get("63d", {}).get("diff_mean") or 0)), 3) for n in SEVEN})


def live_basis(frame: pd.DataFrame) -> pd.Series:
    s = frame["63d"].rolling(5, min_periods=1).mean()
    return s.dropna().rolling(10, min_periods=1).mean()


pit = live_basis(pit_raw).rename("pit")
cur = live_basis(cur_raw).rename("cur_recompute")
frag = pd.read_parquet(ROOT / "data/rd2_fragility.parquet")
live = frag["63d"].rolling(10).mean().rename("live"); live.index = pd.to_datetime(live.index).normalize()
old = pd.read_parquet(ROOT / "scratch/ultracode_sizing_2026-09-02/cross_strategy_regime_pit_dial.parquet")["pit"].rename("pit_from_old")
out = pd.concat([pit, cur, live, pit_raw["63d"].rename("pit_raw63"), cur_raw["63d"].rename("cur_raw63"), old], axis=1)
out.index.name = "date"
out.to_parquet(HERE / "pit_dial_extended.parquet")
pd.Series({str(k): v for k, v in vintages.items()}).to_json(HERE / "pit_vintages_extended.json", indent=1)
print(f"\nwrote pit_dial_extended.parquet  PIT window {pit.index.min().date()} .. {pit.index.max().date()} N={len(pit)}")

# --- agreement with the study's original PIT over its window ---
b = out.dropna(subset=["pit", "pit_from_old"])
print(f"new PIT vs study PIT over {b.index.min().date()}..{b.index.max().date()}: corr {b.pit.corr(b.pit_from_old):.4f}, "
      f"max abs diff {(b.pit - b.pit_from_old).abs().max():.2f}, mean abs diff {(b.pit - b.pit_from_old).abs().mean():.2f}, "
      f">=50 agreement {((b.pit >= 50) == (b.pit_from_old >= 50)).mean() * 100:.1f}%")
b = out.dropna(subset=["pit", "live"])
print(f"PIT vs live parquet (all): corr {b.pit.corr(b.live):.3f}; >=50 agreement {((b.pit >= 50) == (b.live >= 50)).mean() * 100:.1f}%")
b2 = b[b.index >= "2026-07-02"]
print(f"PIT vs live parquet APPEND-ONLY rows (2026-07-02+): corr {b2.pit.corr(b2.live):.3f}; mean(PIT-live) {(b2.pit - b2.live).mean():+.1f}; >=50 agreement {((b2.pit >= 50) == (b2.live >= 50)).mean() * 100:.1f}%")
b3 = out.dropna(subset=["cur_recompute", "live"]); b3 = b3[b3.index >= "2026-07-02"]
print(f"cur_recompute vs live APPEND-ONLY rows: corr {b3.cur_recompute.corr(b3.live):.3f}; mean(cur-live) {(b3.cur_recompute - b3.live).mean():+.1f}")


# --- the Aug-2026 episode under the three vintages ---
def arm_dates(x: pd.Series, on=50, off=45):
    state = False; ev = []
    for d, v in x.dropna().items():
        if not state and v >= on:
            state = True; ev.append(("ARM", d))
        elif state and v < off:
            state = False; ev.append(("RELEASE", d))
    return ev


print("\n=== Aug-2026 episode: 10d-MA 63d dial, three vintages ===")
win = out.loc["2026-06-15":"2026-09-01", ["pit", "cur_recompute", "live"]]
pd.set_option("display.width", 200, "display.max_rows", 200)
print(win.round(1).to_string())
for c in ["pit", "cur_recompute", "live"]:
    ev = [(k, d.date().isoformat()) for k, d in arm_dates(out[c].loc["2026-01-01":]) ]
    print(f"{c:14s} 2026 arm/release events (50/45): {ev}; max in 2026 {out[c].loc['2026'].max():.1f} on {out[c].loc['2026'].idxmax().date()}; last {out[c].dropna().iloc[-1]:.1f}")
print("\n=== whole-history arm/release events (50/45), PIT extended ===")
print([(k, d.date().isoformat()) for k, d in arm_dates(out["pit"])])
print("=== same, live parquet ===")
print([(k, d.date().isoformat()) for k, d in arm_dates(out["live"])])
