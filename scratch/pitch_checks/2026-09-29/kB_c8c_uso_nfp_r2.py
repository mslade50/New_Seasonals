"""C8 round 2 on the 21d form (USO 21d rank >= 75 at NFP-4, long NFP-3 -> NFP).

(a) where the return lands: per-session decomposition NFP-2, NFP-1, NFP day;
    NFP-day 1d return in the 21d thrust vs thrust non-NFP days.
(b) turn-of-month confound: the same gate anchored on the month-end (ME)
    ladder, and NFP-3 windows split by whether the month turn is inside.
(c) midterm split (REQUIRED in a midterm year), October prints, Q4 prints.
(d) entry/exit neighbours: entry k=-5..-1 to the NFP close; exit NFP-1, NFP,
    NFP+1, NFP+2 from the k=-3 entry.
(e) print specificity: the same 21d gate into CPI (k=-3 -> CPI close).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kB_common import *  # noqa

px = nyse_panel(["USO", "CL=F"], ffill=("CL=F",))
px = px[px.index >= "2006-04-10"]
idx = px.index
U = px["USO"].values
r21 = pct_rank(px["USO"], 21)
R = r21.values
ev = load_events(["nfp", "cpi"])
npos = np.array(anchor_positions(idx, ev.loc[ev.event == "nfp", "date"])[0])
cpos = np.array(anchor_positions(idx, ev.loc[ev.event == "cpi", "date"])[0])
me = month_end_positions(idx)
dret = pd.Series(U, index=idx).pct_change(fill_method=None).values


def gated(anchor_pos: np.ndarray, k_entry: int, thr: float = 75) -> np.ndarray:
    """Anchor positions whose signal bar (entry-1) has 21d rank >= thr."""
    out = []
    for a in anchor_pos:
        s = a + k_entry - 1
        if s >= 0 and a + 3 < len(idx) and not np.isnan(R[s]) and R[s] >= thr:
            out.append(a)
    return np.array(out)


G = gated(npos, -3)
base = np.array([span_ret(U, a - 3, a) for a in G])
print(f"cell: N={len(G)} mean {100*np.nanmean(base):+.3f}%")

# (a) per-session decomposition
dec = [cell(np.array([span_ret(U, a - 3 + i, a - 2 + i) for a in G]), f"session NFP{-2+i:+d}") for i in range(3)]
thr_mask = (R >= 75)
nfp_set = set(npos.tolist())
td = [p for p in range(1, len(idx)) if thr_mask[p - 1] and p not in nfp_set]   # thrust at prior close, non-NFP day
nd = [p for p in npos if p >= 1 and thr_mask[p - 1]]
dec += [cell(dret[nd], "NFP-day 1d, 21d thrust at prior close"),
        cell(dret[[p for p in npos if p >= 1]], "NFP-day 1d, all"),
        cell(dret[td], "non-NFP day 1d, 21d thrust at prior close")]
show(dec, "(a) where the return lands")

# (b) month-turn confound
ME_G = gated(me, -3)
tom_in = np.array([any((m > a - 3) and (m < a) for m in me) for a in G])   # a month-end close strictly inside (entry, exit)
show([cell([span_ret(U, a - 3, a) for a in ME_G], "ME anchor: 21d>=75, ME-3 -> ME"),
      cell([span_ret(U, a - 1, a + 2) for a in gated(me, -1)], "ME anchor: 21d>=75, ME-1 -> ME+2"),
      cell([span_ret(U, a, a + 3) for a in gated(me, 0)], "ME anchor: 21d>=75, ME -> ME+3"),
      cell(base[tom_in], "NFP cell, a month-end INSIDE the hold"),
      cell(base[~tom_in], "NFP cell, NO month-end inside the hold")], "(b) turn-of-month confound")
# month-turn windows that do NOT contain an NFP, gated
tom_no_nfp = []
for m in gated(me, 0):
    e, x = m - 1, m + 2
    if not any((n > e) and (n <= x) for n in npos):
        tom_no_nfp.append(span_ret(U, e, x))
show([cell(tom_no_nfp, "ME-1 -> ME+2, 21d>=75, no NFP inside")], "month turn without the print")

# (c) midterm / month splits
Gd = idx[G]
mid = (Gd.year % 4 == 2)
octo = Gd.month == 10
q4 = Gd.month >= 10
show([cell(base[mid], "midterm years"), cell(base[~mid], "non-midterm years"),
      cell(base[octo], "October prints"), cell(base[q4], "Q4 prints (Oct-Dec)"),
      cell(base[Gd.month.isin([9, 10])], "Sep+Oct prints")], "(c) midterm and month splits (REQUIRED)")
print("  midterm episodes:", ", ".join(f"{d.date()}:{100*v:+.2f}" for d, v in zip(Gd[mid], base[mid])))
print("  October episodes:", ", ".join(f"{d.date()}:{100*v:+.2f}" for d, v in zip(Gd[octo], base[octo])))

# (d) neighbours
nb = []
for k in (-5, -4, -3, -2, -1):
    g = gated(npos, k)
    nb.append(cell([span_ret(U, a + k, a) for a in g], f"entry k={k} -> NFP close (h={-k})"))
for xo in (-1, 0, 1, 2):
    nb.append(cell([span_ret(U, a - 3, a + xo) for a in G], f"entry k=-3 -> NFP{xo:+d}"))
show(nb, "(d) entry / exit neighbours (21d>=75 at each entry's signal bar)")

# (e) print specificity: CPI
CG = gated(cpos, -3)
show([cell([span_ret(U, a - 3, a) for a in CG], "21d>=75, CPI-3 -> CPI close"),
      cell([span_ret(U, a - 3, a) for a in cpos if a - 3 >= 0], "all CPI-3 -> CPI close")], "(e) same gate into CPI")
