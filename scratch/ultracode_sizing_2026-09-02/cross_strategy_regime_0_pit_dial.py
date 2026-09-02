"""cross_strategy_regime step 0: rebuild the point-in-time (vintage-lagged) dial
series from scratch/pit_signals.pkl using exactly the scratch/pit_reestimate.py
method (expanding-window diff_mean edges at each year-end; year Y scored with
vintage Y-1; production compute_fragility_timeseries; live basis = raw ->
rolling(5) -> rolling(10)). Writes a parquet with columns pit / cur_recompute /
live (the rd2_fragility parquet's 10d MA) so the hedge study can score on all
three. Nothing in the repo is touched.
"""
from __future__ import annotations
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
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

raw = pickle.load(open(ROOT / "scratch/pit_signals.pkl", "rb"))
fires: pd.DataFrame = raw["fires"]
spy: pd.Series = raw["spy_close"]
spy.index = pd.to_datetime(spy.index)
fires.index = pd.to_datetime(fires.index)
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
pit_frames = []
vintages = {}
for year in range(2018, 2027):
    vint = estimate_stats(f"{year - 1}-12-31")
    frame = compute_fragility_timeseries(sig_dict, spy, vint)
    pit_frames.append(frame[frame.index.year == year])
    vintages[year - 1] = {n: round(max(0.0, -(v["horizons"]["63d"]["diff_mean"] or 0)), 3)
                          for n, v in vint["signals"].items()}
    print(f"vintage {year-1}: 63d edges {vintages[year-1]}")
pit_raw = pd.concat(pit_frames).sort_index()
cur_raw = compute_fragility_timeseries(sig_dict, spy, estimate_stats(fires.index.max()))


def live_basis(frame: pd.DataFrame) -> pd.Series:
    s = frame["63d"].rolling(5, min_periods=1).mean()
    return s.dropna().rolling(10, min_periods=1).mean()


pit = live_basis(pit_raw).rename("pit")
cur = live_basis(cur_raw).rename("cur_recompute")
frag = pd.read_parquet(ROOT / "data/rd2_fragility.parquet")
live = frag["63d"].rolling(10).mean().rename("live")
live.index = pd.to_datetime(live.index).normalize()
out = pd.concat([pit, cur, live], axis=1)
out.index.name = "date"
both = out.dropna(subset=["pit", "live"])
print(f"\nPIT window {pit.index.min().date()} .. {pit.index.max().date()} N={len(pit)}")
print(f"PIT vs live corr {both.pit.corr(both.live):.3f}; >=50 agreement {((both.pit>=50)==(both.live>=50)).mean()*100:.1f}%; "
      f">=65 agreement {((both.pit>=65)==(both.live>=65)).mean()*100:.1f}%; days>=50 PIT {(both.pit>=50).mean()*100:.1f}% live {(both.live>=50).mean()*100:.1f}%; "
      f"days>=65 PIT {(both.pit>=65).mean()*100:.1f}% live {(both.live>=65).mean()*100:.1f}%")
b2 = out.dropna(subset=["cur_recompute", "live"])
print(f"recompute-from-pickle vs live parquet corr {b2.cur_recompute.corr(b2.live):.3f} (sanity: should be ~1 up to vintage drift)")
out.to_parquet(HERE / "cross_strategy_regime_pit_dial.parquet")
pd.Series({str(k): v for k, v in vintages.items()}).to_json(HERE / "cross_strategy_regime_pit_vintages.json", indent=1)
print("wrote cross_strategy_regime_pit_dial.parquet")
