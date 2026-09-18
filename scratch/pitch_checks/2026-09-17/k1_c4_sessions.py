"""c4 mechanism detail: EWJ per-session returns around Mar/Sep quarter-end
vs other month-ends, split by the Japanese settlement regime (T+3 until
July 2019, T+2 after), since the reinvestment futures buy lands on the last
cum-dividend JP session (QE-3 JP pre-2019, QE-2 JP after), which a US close
captures one US session later-labelled (US close k-1 -> k sees JP session k)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from k1_common import *  # noqa

idx = nyse_index()
raw = close_panel(["EWJ", "JPY=X"])
px = raw.reindex(idx.union(raw.index)).ffill().reindex(idx)
E = px["EWJ"].values
H = (px["EWJ"] * px["JPY=X"]).values  # yen-hedged proxy level
A = anchors(idx, -10)
A = A[A.me_pos + 5 < len(idx)].copy()
A["ms"] = A.month.isin([3, 9])
out = []
for k in range(-9, 6):
    rec = {"k": k}
    for tag, fr in [("MS<2019", A[A.ms & (A.year < 2019)]), ("MS2019+", A[A.ms & (A.year >= 2019)]),
                    ("other", A[~A.ms])]:
        r = np.array([E[m + k] / E[m + k - 1] - 1 for m in fr.me_pos])
        rec[f"{tag}_bp"] = 1e4 * np.nanmean(r)
        rec[f"{tag}_hit"] = 100 * np.nanmean(r > 0)
    rh = np.array([H[m + k] / H[m + k - 1] - 1 for m in A[A.ms].me_pos])
    rec["MS_hedged_bp"] = 1e4 * np.nanmean(rh)
    out.append(rec)
show(out, "EWJ per-session bp around month-end (k = US close k-1 -> k)")
print(f"EWJ all-days mean session bp {1e4*np.nanmean(E[1:]/E[:-1]-1):+.2f}")
ms = A[A.ms]
# predicted buy session: JP last cum day -> US label k=-3 (pre-2019, T+3) / k=-2 (2019+, T+2)
pre = ms[ms.year < 2019]
post = ms[ms.year >= 2019]
rp = [E[m - 3] / E[m - 4] - 1 for m in pre.me_pos]
rq = [E[m - 2] / E[m - 3] - 1 for m in post.me_pos]
show([stats_line(rp, pre.sig_date, "pre-2019 predicted buy session (k=-3)"),
      stats_line(rq, post.sig_date, "2019+ predicted buy session (k=-2)"),
      stats_line(rp + rq, list(pre.sig_date) + list(post.sig_date), "pooled predicted buy session")],
     "mechanism session")
