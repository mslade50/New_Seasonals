"""C8 round 2 follow-up: the month-end-inside-the-hold split, which contains the
LIVE configuration (entry 09-29, ME 09-30 = NFP-2, NFP 10-02).

For every NFP, me_off = position of the last month-end at or before the NFP
minus the NFP position (live: -2). Gate: USO 21d rank >= 75 at each form's
signal bar. Report the k=-3/-2/-1 entry forms split by me_off, the ungated
split, CL=F, and the episode list of the live-like subset.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kB_common import *  # noqa

px = nyse_panel(["USO", "CL=F"], ffill=("CL=F",))
px = px[px.index >= "2006-04-10"]
idx = px.index
U, C = px["USO"].values, px["CL=F"].values
R = pct_rank(px["USO"], 21).values
npos = np.array(anchor_positions(idx, load_events(["nfp"])["date"])[0])
me = month_end_positions(idx)

rows = []
for a in npos:
    if a + 3 >= len(idx) or a < 6:
        continue
    prior_me = me[me <= a]
    me_off = int(prior_me[-1] - a) if len(prior_me) else -99
    r = {"nfp": idx[a], "me_off": me_off}
    for k in (-3, -2, -1):
        s = a + k - 1
        r[f"g{k}"] = (not np.isnan(R[s])) and R[s] >= 75
        r[f"u{k}"] = span_ret(U, a + k, a)
        r[f"c{k}"] = span_ret(C, a + k, a)
    rows.append(r)
A = pd.DataFrame(rows)
live_like = A.me_off.isin([-1, -2])       # month-end strictly inside a k=-3 hold (me_off -3 = entry AT the ME close)
print("me_off distribution (all NFPs):", A.me_off.value_counts().sort_index().to_dict())
print("LIVE: NFP 2026-10-02, ME 2026-09-30 -> me_off = -2")

out = []
for k in (-3, -2, -1):
    g = A[f"g{k}"]
    for lbl, m in [("ME inside k=-3 hold (live-like, me_off -1,-2)", live_like),
                   ("me_off = -2 exactly (live)", A.me_off == -2),
                   ("ME before k=-3 entry", ~live_like)]:
        out.append(cell(A.loc[g & m, f"u{k}"], f"USO k={k} gated | {lbl}"))
        out.append(cell(A.loc[g & m, f"c{k}"], f"CL=F k={k} gated | {lbl}"))
show(out, "gated cell by month-end position")
show([cell(A.loc[live_like, "u-3"], "USO k=-3 UNGATED, ME inside"),
      cell(A.loc[~live_like, "u-3"], "USO k=-3 UNGATED, ME before entry")], "ungated split")
d, t = welch(A.loc[A["g-3"] & ~live_like, "u-3"], A.loc[A["g-3"] & live_like, "u-3"])
print(f"  gated k=-3: ME-before minus ME-inside {d:+.3f}pp (welch t {t:+.2f})")
sub = A[A["g-3"] & live_like]
print("\nlive-like gated k=-3 episodes (NFP date, me_off, USO %, CL=F %):")
for _, r in sub.iterrows():
    print(f"  {r.nfp.date()}  me_off {r.me_off}  USO {100*r['u-3']:+.2f}  CL=F {100*r['c-3']:+.2f}")
sub2 = A[A["g-1"] & live_like]
print("\nlive-like gated k=-1 (entry NFP-1, after the ME) episodes:")
print("  " + ", ".join(f"{r.nfp.date()}:{100*r['u-1']:+.2f}" for _, r in sub2.iterrows()))
