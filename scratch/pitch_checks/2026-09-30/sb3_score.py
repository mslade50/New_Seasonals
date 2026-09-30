import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.seasonal_ticket_sim import simulate_ticket  # noqa: E402
from scripts.score_seasonal_ideas import _load_raw  # noqa: E402  (pure yfinance read, auto_adjust=False)

pd.set_option("display.width", 220)
HERE = Path(__file__).parent
CUTOFF = pd.Timestamp("2026-09-29")
d = pd.read_parquet(HERE / "sb3_ledger_r2.parquet")
d["asof"] = pd.to_datetime(d["asof"]).dt.normalize()
d = d[d["asof"] >= "2026-07-24"].reset_index(drop=True)

rows = []
for ticker, grp in d.groupby("ticker"):
    raw = _load_raw(ticker, (grp["asof"].min() - pd.Timedelta(days=10)).date(), "2026-09-30")
    if raw is not None:
        raw = raw[raw.index <= CUTOFF]
    for _, r in grp.iterrows():
        off = int(r["entry_offset_days"]) if pd.notna(r["entry_offset_days"]) else 0
        n = int(r["time_stop_days"])
        tk = {"ticker": ticker, "direction": r["direction"], "entry": float(r["entry"]),
              "stop": float(r["stop"]), "target": float(r["target"]), "time_stop_days": n}
        base = {"asof": r["asof"].date(), "ticker": ticker, "ch": "EQ" if r["channel"].startswith("Equity") else "MACRO",
                "conv": r["conviction"], "dir": r["direction"], "n": n, "off": off}
        out = simulate_ticket(tk, raw, r["asof"], entry_mode="delayed", entry_window=off, reanchor=True) if raw is not None else None
        if out is None or raw is None:
            rows.append({**base, "status": "open"})
            continue
        fwd = raw[raw.index > r["asof"]]
        dd = max(0, min(off, n - 1))
        e = float(fwd.iloc[dd]["Open"])
        x = float(fwd.iloc[n - 1]["Close"]) if len(fwd) >= n else np.nan
        sign = 1 if r["direction"] == "long" else -1
        rows.append({**base, "status": "done" if len(fwd) >= n else "early_exit_unmatured", "entry_date": out["entry_date"].date(), "entry_px": out["entry_price"],
                     "exit_date": pd.Timestamp(out["exit_date"]).date(), "exit_type": out["exit_type"],
                     "R": out["R"], "hold_ret_pct": sign * (x / e - 1) * 100,
                     "risk_pct": abs(r["entry"] - r["stop"]) / r["entry"] * 100})

res = pd.DataFrame(rows)
res.to_csv(HERE / "sb3_results.csv", index=False)
print(res.to_string())
done = res[res["status"] == "done"]


def agg(g: pd.DataFrame) -> pd.Series:
    return pd.Series({"N": len(g), "hit": (g["R"] > 0).mean(), "meanR": g["R"].mean(), "sumR": g["R"].sum(),
                      "worstR": g["R"].min(), "hold%": g["hold_ret_pct"].mean(),
                      "hold_hit": (g["hold_ret_pct"] > 0).mean(),
                      "types": g["exit_type"].value_counts().to_dict()})


print(agg(done))
for k in ["ch", "conv", "dir"]:
    print(done.groupby(k).apply(agg))
