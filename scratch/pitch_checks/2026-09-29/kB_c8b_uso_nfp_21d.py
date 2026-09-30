"""C8 round 2 (the 21d neighbour that round 1 surfaced): long USO NFP-3 -> NFP
gated on USO 21d pct_rank at the signal bar (NFP-4). The stated 63d>=75 gate
does not fire today (74.6); 21d is 76.6. This is a neighbour found by a gate
walk (63d/21d x 70/75/80/90 = 8 cells), so it carries that charge.

Tests: vs the 21d thrust state without the print; placebo anchor ladder;
threshold ladder; era split; ex-2026 / ex-top-2; one-year share; CL=F repeat;
the 21d gate alone at every NFP offset vs no gate; 21d AND 63d conjunction.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kB_common import *  # noqa

px = nyse_panel(["USO", "CL=F", "SPY"], ffill=("CL=F",))
px = px[px.index >= "2006-04-10"]
px["CL=F"] = px["CL=F"].where(px["CL=F"] > 0)
idx = px.index
uso = px["USO"]
r63, r21 = pct_rank(uso, 63), pct_rank(uso, 21)
c21 = pct_rank(px["CL=F"], 21)
print(f"LIVE {idx[-1].date()}: USO 21d rank {r21.iloc[-1]:.1f}, 63d rank {r63.iloc[-1]:.1f}, CL=F 21d rank {c21.iloc[-1]:.1f}; "
      f"USO 21d ret {100*(uso.iloc[-1]/uso.iloc[-22]-1):+.1f}%")
nfp = load_events(["nfp"])["date"]
pos, _ = anchor_positions(idx, nfp)
nfp_pos = np.array(pos)
H = 3
ret = vehicle_ret(px, [("USO", 1.0)], H, 1)
cret = vehicle_ret(px, [("CL=F", 1.0)], H, 1)
sret = vehicle_ret(px, [("SPY", 1.0)], H, 1)
valid = ret.notna()


def sig_days(k: int, gate: pd.Series | None, thr: float = 75) -> pd.DatetimeIndex:
    sp = nfp_pos + k - 1
    sp = sp[(sp >= 0) & (sp < len(idx))]
    d = idx[sp]
    if gate is not None:
        g = gate.reindex(d)
        d = d[(g >= thr).values]
    return d[valid.reindex(d).values]


cond = sig_days(-3, r21, 75)
v = ret.loc[cond].values
thr_days = idx[((r21 >= 75).reindex(idx, fill_value=False) & valid).values]
nin = event_in_window(thr_days, idx, H, 1, ("nfp",))
show([cell(v, "COND USO 21d>=75, NFP-3 -> NFP"),
      cell(ret.loc[thr_days[~nin]].values, "21d>=75 thrust, NO NFP in hold"),
      cell(ret.loc[thr_days[nin]].values, "21d>=75 thrust, NFP in hold (any offset)"),
      cell(ret.loc[sig_days(-3, None)].values, "all NFP-3 entries, no gate"),
      cell(ret[valid].values, "USO all days h=3")], "21d gate vs its own state without the print")
d, t = welch(v, ret.loc[thr_days[~nin]].values)
print(f"  COND minus 21d-thrust-no-print {d:+.3f}pp (welch t {t:+.2f})")
lr = ret.loc[local_control(idx[valid.values], cond)].values
print(f"  local +/-126td control {100*np.nanmean(lr):+.3f}%")

# thresholds
show([cell(ret.loc[sig_days(-3, r21, th)].values, f"21d>={th}") for th in (60, 65, 70, 75, 80, 85, 90)] +
     [cell(ret.loc[sig_days(-3, r63, th)].values, f"63d>={th}") for th in (60, 70, 75, 80, 90)],
     "threshold ladder (the walk this neighbour came from)")
both = cond[(r63.reindex(cond) >= 75).values]
only21 = cond[(r63.reindex(cond) < 75).values]
show([cell(ret.loc[both].values, "21d>=75 AND 63d>=75"), cell(ret.loc[only21].values, "21d>=75, 63d<75 (today's cell)")],
     "conjunction split")

# placebo anchor ladder with the 21d gate
lad = []
for k in range(-10, 8):
    lad.append(cell(ret.loc[sig_days(k, r21, 75)].values, f"k={k:+d}"))
L = pd.DataFrame(lad)
L["rank"] = L["mean_pct"].rank(ascending=False).astype(int)
show(L.to_dict("records"), "placebo anchor ladder, 21d>=75, h=3 (live rung k=-3)")
print(f"  live rung k=-3 ranks {int(L.loc[L.label == 'k=-3', 'rank'].iloc[0])} of {len(L)}")

# era, concentration, year shares
show(era_split(cond, v) + era_split(cond, v, "2013-01-01"), "era split (2018 cut; 2013 cut)")
print("  ", cluster_note(cond, v))
yr = pd.Series(v, index=cond).groupby(cond.year).sum().sort_values(ascending=False)
print(f"  one-year share: {yr.index[0]} = {100*yr.iloc[0]:+.2f}pp of {100*v.sum():+.2f}pp ({100*yr.iloc[0]/v.sum():.0f}%)")
ex26 = cond.year < 2026
show([cell(v[ex26], "ex-2026"), cell(np.sort(v)[:-2], "ex best two"),
      cell(v[(cond.year < 2020) | (cond.year > 2022)], "ex 2020-2022")], "robustness")
print("  episodes:", ", ".join(f"{d.date()}:{100*x:+.2f}" for d, x in zip(cond, v)))

# CL=F repeat and SPY on the same dates (is it risk-on beta?)
show([cell(cret.loc[cond].dropna().values, "CL=F on USO-21d dates"),
      cell(cret.loc[sig_days(-3, c21, 75)].dropna().values, "CL=F own 21d>=75 gate"),
      cell(sret.loc[cond].values, "SPY on the same dates"),
      cell((ret.loc[cond] - sret.loc[cond]).values, "USO minus SPY on the same dates")], "vehicle repeat")
