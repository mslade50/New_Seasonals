"""C9 round 1: long USO after a >= 4% one-day fall with USO's 63d rank >= 70.

Pre-specified: signal bar = USO close-to-close <= -4.0% with pct_rank(USO, 63) >= 70
(trailing-252 percentile of the 63-session return, measured ON the signal bar, as the
tape does: live 09-29 USO -4.44%, r63 72.2). Entry lag=1 MOC, long, h=1..5, h=3
pre-named. Gate attribution against the plain >= 4% day and against the 09-28 c6
definition (r21 >= 75, first <= -3% day in 10 sessions); Jaccard of the masks.
CL=F on the same dates (seam-flagged at CL expiries), XLE as the non-roll cross-check.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd


def own(t: str) -> pd.Series:
    s = close_panel([t])[t].dropna()
    return s[s > 0]


uso, cl, xle = own("USO"), own("CL=F"), own("XLE")
idx = uso.index
r1 = uso.pct_change()
r63 = pct_rank(uso, 63)
r21 = pct_rank(uso, 21)
r5 = pct_rank(uso, 5)

drop4 = r1 <= -0.04
gate = r63 >= 70
sig = drop4 & gate
# c6 definition (09-28): r21 >= 75, <= -3% day, no other <= -3% day in D-10..D-1
crack = r1 <= -0.03
prior = crack.shift(1).rolling(10).sum() > 0
c6 = (r21 >= 75) & crack & ~prior

print("live tail (USO):")
print(pd.DataFrame({"close": uso, "r1%": 100 * r1, "r63": r63, "r21": r21, "r5": r5,
                    "sig": sig}).tail(6).round(2).to_string())


def jac(a: pd.Series, b: pd.Series) -> str:
    a, b = a.fillna(False), b.fillna(False)
    i, u_ = int((a & b).sum()), int((a | b).sum())
    return f"|A|={int(a.sum())} |B|={int(b.sum())} both={i} Jaccard={i/max(u_,1):.3f}"


print("\nJaccard C9 vs c6 (r21>=75 first -3% crack):", jac(sig, c6))
print("Jaccard C9 vs (r21>=75 & -4% day):", jac(sig, drop4 & (r21 >= 75)))

px = pd.DataFrame({"USO": uso})
battery(px, sig, [("USO", 1.0)], 3, "C9 long USO after -4% day, r63 rank >= 70", cost_bps=3.0,
        variants={"plain -4% day, no gate": drop4,
                  "complement: -4% day, r63 < 70": drop4 & ~gate,
                  "r63 >= 60": drop4 & (r63 >= 60),
                  "r63 >= 80": drop4 & (r63 >= 80),
                  "r63 >= 90": drop4 & (r63 >= 90),
                  "-3.5% day, r63 >= 70": (r1 <= -0.035) & gate,
                  "-5% day, r63 >= 70": (r1 <= -0.05) & gate,
                  "r63(D-1) >= 70 (pre-drop trend)": drop4 & (r63.shift(1) >= 70),
                  "c6 def (r21>=75 first -3%)": c6,
                  "-4% & r21 >= 75": drop4 & (r21 >= 75)},
        event_kinds=("nfp",))


# ---------------------------------------------------------------- gate attribution across h, three vehicles
def eps(s: pd.Series, mask: pd.Series, h: int):
    ret = fwd_lag(s, h, 1)
    m = mask.reindex(s.index, fill_value=False).fillna(False)
    t = s.index[m.values & ret.notna().values]
    e = declusters(t, h, s.index)
    return e, ret.loc[e]


# CL expiry: last trade = 3 bd before the 25th of the month before delivery (approx.)
def cl_exp(y: int, m: int) -> pd.Timestamp:
    d25 = pd.Timestamp(y, m, 25)
    if d25.weekday() >= 5:
        d25 = d25 - pd.offsets.BDay(1)
    return d25 - pd.offsets.BDay(3)


CLX = pd.DatetimeIndex([cl_exp(y, m) for y in range(2000, 2027) for m in range(1, 13)])


def seam_free(e: pd.DatetimeIndex, s: pd.Series, h: int) -> np.ndarray:
    ok = []
    for d in e:
        p = s.index.get_loc(d)
        lo, hi = s.index[p + 1], s.index[min(p + 1 + h, len(s) - 1)]
        ok.append(not (((CLX + pd.Timedelta(days=1)) > lo) & ((CLX + pd.Timedelta(days=1)) <= hi + pd.Timedelta(days=1))).any())
    return np.array(ok, dtype=bool)


rows = []
for veh, s in (("USO", uso), ("CL=F", cl), ("XLE", xle)):
    for h in (1, 2, 3, 4, 5):
        drift = fwd_lag(s, h, 1).dropna()
        for lbl, m in (("C9 gated", sig), ("plain -4%", drop4), ("-4% & r63<70", drop4 & ~gate),
                       ("c6 def", c6)):
            e, v = eps(s, m, h)
            if veh == "CL=F" and lbl == "C9 gated":
                sf = seam_free(e, s, h)
                vs = v[sf]
                w = int((vs > 0).sum())
                rows.append({"veh": veh, "h": h, "cell": "C9 gated SEAM-FREE", "n": len(vs),
                             "mean_pct": 100 * vs.mean(), "rec": f"{w}-{len(vs)-w}",
                             "sign_p": sign_test(w, len(vs)), "drift_pct": 100 * drift.mean()})
            w = int((v > 0).sum())
            rows.append({"veh": veh, "h": h, "cell": lbl, "n": len(v), "mean_pct": 100 * v.mean(),
                         "rec": f"{w}-{len(v)-w}", "sign_p": sign_test(w, len(v)) if len(v) else np.nan,
                         "drift_pct": 100 * drift.mean()})
df = pd.DataFrame(rows)
for veh, grp in df.groupby("veh", sort=False):
    print(f"\n=== gate attribution, LONG {veh} from lag-1 close (episodes, signal on USO) ===")
    print(grp.round(3).to_string(index=False))

# ---------------------------------------------------------------- era / midterm on USO h=3
e, v = eps(uso, sig, 3)
print("\nC9 episodes USO h=3:")
print(", ".join(f"{d.date()}:{100*x:+.2f}" for d, x in v.items()))
for lbl, m in (("pre-2018", v.index < "2018-01-01"), ("2018+", v.index >= "2018-01-01"),
               ("midterm", v.index.year % 4 == 2), ("non-midterm", v.index.year % 4 != 2)):
    vv = v[m]
    w = int((vv > 0).sum())
    print(f"  {lbl:12s} {100*vv.mean():+.3f}% ({w}-{len(vv)-w})")

# ---------------------------------------------------------------- USO roll cost vs CL=F
a = pd.DataFrame({"u": uso.pct_change(), "c": cl.pct_change()}).dropna()
a = a[a.c.abs() < 0.25]
dd = a.u - a.c
print(f"\nUSO minus CL=F per session (all common days, |CL|<25%): {1e4*dd.mean():+.2f} bp; "
      f"2018+ {1e4*dd[dd.index>='2018'].mean():+.2f} bp; x3 sessions = {3e4*dd[dd.index>='2018'].mean():+.1f} bp")
