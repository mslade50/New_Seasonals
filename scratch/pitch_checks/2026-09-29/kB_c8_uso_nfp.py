"""C8 round 1: long crude into payrolls after a 63-day crude thrust.

USO from the NFP-3 close to the NFP close (h=3), signal bar NFP-4 (lag=1 entry),
gated on USO 63d pct_rank >= 75 at the signal bar. Live: signal 09-28 (NFP-4),
entry 09-29 close, exit 10-02 NFP close.
Controls: battery (own drift, all days, local), the thrust state WITHOUT the
print (rank>=75 days whose h=3 hold has no NFP), the NFP run-in WITHOUT the
gate, and the placebo anchor ladder (same gate, entry offsets -10..+7 around
NFP). CL=F repeated with a roll-seam audit.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kB_common import *  # noqa

px = nyse_panel(["USO", "CL=F"], ffill=("CL=F",))
px = px[px.index >= "2006-04-10"]
cl = px["CL=F"].where(px["CL=F"] > 0)
px["CL=F"] = cl
idx = px.index
uso = px["USO"]
r63 = pct_rank(uso, 63)
r21 = pct_rank(uso, 21)
c63 = pct_rank(px["CL=F"], 63)
print(f"LIVE {idx[-1].date()}: USO 63d {100*(uso.iloc[-1]/uso.iloc[-64]-1):+.1f}%  63d rank {r63.iloc[-1]:.1f}  "
      f"21d rank {r21.iloc[-1]:.1f}  1d {100*(uso.iloc[-1]/uso.iloc[-2]-1):+.2f}%;  CL=F 63d rank {c63.iloc[-1]:.1f}")

nfp = load_events(["nfp"])["date"]
pos, kept = anchor_positions(idx, nfp)
nfp_pos = np.array(pos)
print(f"NFP anchors in USO span: {len(nfp_pos)} (last {idx[nfp_pos[-1]].date()})")


def sig_mask(offset_entry: int, gate: pd.Series | None, thr: float = 75) -> pd.Series:
    """Signal bar = entry - 1 (lag=1). offset_entry = entry position relative to NFP."""
    m = pd.Series(False, index=idx)
    sp = nfp_pos + offset_entry - 1
    sp = sp[(sp >= 0) & (sp < len(idx))]
    m.iloc[sp] = True
    if gate is not None:
        m &= (gate >= thr).reindex(idx, fill_value=False)
    return m


H = 3
mask = sig_mask(-3, r63, 75)
variants = {"no gate (all NFP-3 entries)": sig_mask(-3, None),
            "63d rank>=70": sig_mask(-3, r63, 70), "63d rank>=80": sig_mask(-3, r63, 80),
            "63d rank>=90": sig_mask(-3, r63, 90),
            "21d rank>=75": sig_mask(-3, r21, 75), "CL=F 63d rank>=75": sig_mask(-3, c63, 75)}
battery(px, mask, [("USO", 1.0)], H, "C8 long USO NFP-3 -> NFP, USO 63d rank>=75", cost_bps=3,
        variants=variants, min_gap=10, event_kinds=("nfp",))

# thrust state WITHOUT the print: every rank>=75 signal day whose h=3 hold has no NFP
ret = vehicle_ret(px, [("USO", 1.0)], H, 1)
valid = ret.notna()
thr_days = idx[((r63 >= 75).reindex(idx, fill_value=False) & valid).values]
nfp_in = event_in_window(thr_days, idx, H, 1, ("nfp",))
cond_days = idx[(mask & valid).values]
show([cell(ret.loc[cond_days].values, "COND gated NFP-3 entries (day-level)"),
      cell(ret.loc[thr_days[~nfp_in]].values, "thrust state, NO NFP in hold"),
      cell(ret.loc[thr_days[nfp_in]].values, "thrust state, NFP in hold (any offset)"),
      cell(ret.loc[idx[valid.values]].values, "USO all days h=3")],
     "C8 vs the thrust state without the print")
d, t = welch(ret.loc[cond_days].values, ret.loc[thr_days[~nfp_in]].values)
print(f"  gated NFP-3 minus thrust-no-print: {d:+.3f}pp (welch t {t:+.2f})")

# placebo anchor ladder: same gate, entry offsets -10..+7 relative to NFP, h=3
lad = []
for k in range(-10, 8):
    m = sig_mask(k, r63, 75)
    s = idx[(m & valid).values]
    r = cell(ret.loc[s].values, f"k={k:+d}")
    lad.append(r)
L = pd.DataFrame(lad)
L["rank"] = L["mean_pct"].rank(ascending=False).astype(int)
show(L.to_dict("records"), "placebo anchor ladder (gated, h=3); live rung k=-3")
print(f"  live rung k=-3 ranks {int(L.loc[L.label == 'k=-3', 'rank'].iloc[0])} of {len(L)}")

# era split of the gated cell
show(era_split(cond_days, ret.loc[cond_days].values), "era split gated cell (day-level = episodes, one per month)")
print("  concentration:", cluster_note(cond_days, ret.loc[cond_days].values))
print("  gated episode dates + returns:", ", ".join(f"{d.date()}:{100*ret.loc[d]:+.2f}" for d in cond_days))

# CL=F repeat + roll seams: flag |CL=F - USO| daily divergence > 3% inside any gated window
cret = vehicle_ret(px, [("CL=F", 1.0)], H, 1)
cmask = sig_mask(-3, c63, 75)
cdays = idx[(cmask & cret.notna()).values]
show([cell(cret.loc[cdays].values, "CL=F NFP-3 -> NFP, CL=F 63d rank>=75"),
      cell(cret.loc[cond_days].dropna().values, "CL=F on the USO-gated dates"),
      cell(cret.dropna().values, "CL=F all days h=3")], "CL=F repeat")
du = uso.pct_change(fill_method=None)
dc = px["CL=F"].pct_change(fill_method=None)
gap = (dc - du).abs()
p_ = pd.Series(range(len(idx)), index=idx)
seams = []
for d in cdays.union(cond_days):
    p = p_[d]
    w = gap.iloc[p + 2: p + 1 + H + 1]
    if (w > 0.03).any():
        seams.append((d.date(), [str(x.date()) for x in w[w > 0.03].index]))
print(f"  roll-seam audit: gated windows with a >3% CL=F-vs-USO daily gap: {seams if seams else 'none'}")
