"""C1x item 6: where does the >=2 ATR failed-thrust SHORT's edge live?
Earnings-day vs non-earnings, gap-led vs intraday-led, with the single best
episode removed, era split. Reuses k2_c1x_family's panels (its prints muted)."""
import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))
with contextlib.redirect_stdout(io.StringIO()):
    import k2_c1x_family as F  # noqa

import numpy as np
import pandas as pd
from pitch_lab import cluster_note, era_split, show, summarize

thrust = (F.R5P >= 80).fillna(False)
cells = {
    "r5>=80 >=2ATR EARNINGS (any lead)": F.mask(2.0, earn="only", lead="any") & thrust,
    "r5>=80 >=2ATR non-earn any lead": F.mask(2.0, earn="non", lead="any") & thrust,
    "r5>=80 >=2ATR non-earn GAP-led (GLW 09-14 form)": F.mask(2.0, earn="non", lead="gap") & thrust,
    "r5>=80 >=2ATR non-earn intraday": F.mask(2.0, earn="non", lead="intraday") & thrust,
    "r5<80 >=2ATR non-earn GAP-led": F.mask(2.0, earn="non", lead="gap") & ~thrust,
}
for h in (3, 5):
    rows = []
    for lbl, M in cells.items():
        keep, v, ctrl = F.episodes(M, h, "sector", F.NAMES)
        r = summarize(v, f"h={h} {lbl}")
        order = np.argsort(-v)
        r["drop_best1_pct"] = round(100 * np.delete(v, order[:1]).mean(), 3)
        r["drop_best2_pct"] = round(100 * np.delete(v, order[:2]).mean(), 3)
        r["ctrl_pct"] = round(100 * ctrl, 3)
        w = int((v > 0).sum())
        r["rec"] = f"{w}-{len(v)-w}"
        r["sign_p2"] = round(F.two_sided(w, len(v)), 4)
        rows.append(r)
    show(rows, f"thrust-short decomposition h={h} (short P&L, sector-beta hedge)")

keep, v, ctrl = F.episodes(cells["r5>=80 >=2ATR non-earn GAP-led (GLW 09-14 form)"], 5, "sector", F.NAMES)
show(era_split(keep, v), "GLW-form era split h=5")
print(cluster_note(keep, v))
A = F.PAIRS[(5, "sector")]
M = cells["r5>=80 >=2ATR non-earn GAP-led (GLW 09-14 form)"]
st = M.stack()
st = st[st]
top = sorted(((A.at[d, t], d, t) for d, t in st.index if pd.notna(A.at[d, t])), reverse=True)[:3]
print("top name-day shorts h=5:", [(round(100 * x, 2), str(d.date()), t) for x, d, t in top])
