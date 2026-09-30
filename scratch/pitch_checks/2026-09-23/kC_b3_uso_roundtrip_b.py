"""B3 kill robustness: is the wrong-signed short a definition artifact?
Neighbour grid r5 <= 2/3/5 x prior r21 >= 85/90/95 x lookback 5/10/15 on the
pre-specified SHORT, h=5, episode level, on USO; plus the matched control at
each rung (same flush without the thrust)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kC_b3_uso_roundtrip import state, eps  # noqa: E402

if __name__ == "__main__":
    for t in ("USO", "CL=F"):
        px, c, r5, r21, r63 = state(t)
        ret = vehicle_ret(px, [(t, -1.0)], 5)
        idx = px.index
        rows = []
        for r5m in (2, 3, 5):
            for r21m in (85, 90, 95):
                for lk in (5, 10, 15):
                    prior = r21.shift(1).rolling(lk).max()
                    e, v = eps(ret, (r5 <= r5m) & (prior >= r21m), 5, idx)
                    em, vm = eps(ret, (r5 <= r5m) & (prior < r21m), 5, idx)
                    w = int((v > 0).sum())
                    rows.append({"cell": f"r5<={r5m} r21>={r21m} lk{lk}", "n": len(v),
                                 "short_mean_pct": 100 * np.nanmean(v) if len(v) else np.nan,
                                 "record": f"{w}-{len(v)-w}",
                                 "matched_pct": 100 * np.nanmean(vm) if len(vm) else np.nan,
                                 "child_minus_matched": 100 * (np.nanmean(v) - np.nanmean(vm)) if len(v) and len(vm) else np.nan})
        df = pd.DataFrame(rows)
        show(df.to_dict("records"), f"{t} neighbour grid, SHORT h=5 (positive = short pays)")
        print(f"  {t}: cells with short mean > 0: {(df.short_mean_pct > 0).sum()} of {len(df)}; "
              f"child beats matched: {(df.child_minus_matched > 0).sum()} of {len(df)}")
