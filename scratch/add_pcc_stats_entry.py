"""One-shot: add the Equity P/C Complacency entry (5d horizon ONLY) to
data/signal_horizon_stats.json without touching any frozen entry.

Numbers are the full-series day-level stats from scratch/putcall_dial_study.py
(PRIMARY pct252<10, 2007-11 -> 2026-07, run 2026-08-05): n=742 days,
78 episodes, 14.9% active; 5d diff -0.18 (sig 0.06 vs unc 0.24), hit 57.4,
p .009, episode mean 0.41, episode t 0.98. NO 21d/63d horizons by design —
_signal_edge weights the signal 0 there, leaving the sizing 63d column
untouched (freeze policy A2).
"""
from __future__ import annotations

import json
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "data" / "signal_horizon_stats.json"

with open(PATH, encoding="utf-8") as f:
    data = json.load(f)

before = {k: json.dumps(v, sort_keys=True) for k, v in data["signals"].items()}
assert "Equity P/C Complacency" not in data["signals"], "entry already present"

data["signals"]["Equity P/C Complacency"] = {
    "n_events": 742,
    "n_episodes": 78,
    "pct_active": 14.9,
    "params": ("10d-MA CBOE equity put/call, trailing-252d pctile < 10 "
               "(complacency tail; data/cboe_putcall.parquet 2006-11+)"),
    "note": ("added 2026-08-05, 5d horizon ONLY by design — 63d dial "
             "candidacy rejected (day-level edge is overlap inflation; "
             "episode t wrong-signed). Evidence: "
             "scratch/putcall_dial_study.py. Full-series day-level stats, "
             "not the 10y production window."),
    "horizons": {
        "5d": {
            "signal_mean": 0.06,
            "unconditional_mean": 0.24,
            "diff_mean": -0.18,
            "hit_rate": 57.4,
            "p_value": 0.009,
            "episode_mean": 0.41,
            "episode_t": 0.98,
        }
    },
}

for k, v in before.items():
    assert json.dumps(data["signals"][k], sort_keys=True) == v, f"frozen entry {k} changed!"

with open(PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=1)
print(f"added Equity P/C Complacency (5d only); {len(data['signals'])} signals now")
