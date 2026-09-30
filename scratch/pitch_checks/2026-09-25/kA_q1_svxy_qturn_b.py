"""kA Q1 placebo ladder: SVXY residual (b=daily 2018-03+ beta) and outright,
quarter-ends 2018-03+, entry offset ke in -6..+2, exit kx in ke+1..+3. Rank of
the pre-specified (-3,+1) cell among all cells; same ladder on ordinary
month-ends for contrast. Round 1 already failed; this only records the placebo
rank the brief asked for (no cell here is a candidate)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd
import io
import contextlib

with contextlib.redirect_stdout(io.StringIO()):
    import kA_q1_svxy_qturn as q  # noqa: E402

rows = []
for ke in range(-6, 3):
    for kx in range(ke + 1, 4):
        r = q.wret("SVXY", q.post_q, ke, kx, beta=q.B)
        o = q.wret("SVXY", q.post_q, ke, kx)
        m = q.wret("SVXY", q.post_m, ke, kx, beta=q.B)
        rows.append({"ke": ke, "kx": kx, "n": len(r), "resid_mean": 100 * r.mean(),
                     "resid_hit": 100 * (r > 0).mean(), "outright_mean": 100 * o.mean(),
                     "ordME_resid_mean": 100 * m.mean(),
                     "resid_per_session": 100 * r.mean() / (kx - ke)})
L = pd.DataFrame(rows)
L["rank_resid"] = L["resid_mean"].rank(ascending=False).astype(int)
print(L.round(3).to_string(index=False))
t = L[(L.ke == -3) & (L.kx == 1)].iloc[0]
print(f"\npre-specified (-3,+1): resid {t.resid_mean:+.3f}% rank {int(t.rank_resid)} of {len(L)}; "
      f"cells with resid > 0: {int((L.resid_mean > 0).sum())} of {len(L)}")
ts = q.px["TS"] if "TS" in q.px else None
