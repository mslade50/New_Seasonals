"""sb1 TRV dev audit: per-year intraday MAE in ATR units from MOC T+2 entry (sanity check on the stop table)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sb1_engine as E

import numpy as np
import pandas as pd

f = load_prices(["TRV"])["TRV"]
idx = f.index
L, C = f["Low"].values, f["Close"].values
atr = wilder_atr(f["High"], f["Low"], f["Close"])
rows = []
for y, p in E.anchors(idx).items():
    e0, ex = p + 2, p + 23
    a = atr[p - 1]
    lows = L[e0 + 1:ex + 1]
    closes = C[e0 + 1:ex + 1]
    rows.append({"year": y, "atr%": round(100 * a / C[p - 1], 2), "mae_low_atr": round((lows.min() - C[e0]) / a, 2),
                 "mae_close_atr": round((closes.min() - C[e0]) / a, 2), "day_of_mae": int(np.argmin(lows)) + 1,
                 "final_atr": round((C[ex] - C[e0]) / a, 2)})
d = pd.DataFrame(rows).set_index("year")
print(d.to_string())
for k in (0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0):
    hit = d.mae_low_atr <= -k
    hitc = d.mae_close_atr <= -k
    print(f"stop {k} ATR: intraday touched {hit.sum()}/26 (of which final>0: {(hit & (d.final_atr>0)).sum()}), "
          f"close-basis touched {hitc.sum()}/26")
