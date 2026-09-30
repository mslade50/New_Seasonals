"""C4 round 2: concentration, entry/exit neighbours, permutation vs the 25-year
first-Sunday-of-October window, municipal-election placebo, washout state,
runoff replication of the RUN-IN on every vehicle, and the pre-vote tape.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kB_common import *  # noqa

FIRST = ["2002-10-06", "2006-10-01", "2010-10-03", "2014-10-05", "2018-10-07", "2022-10-02"]
RUNOFF = ["2002-10-27", "2006-10-29", "2010-10-31", "2014-10-26", "2018-10-28", "2022-10-30"]
MUNI = {2004: "2004-10-03", 2008: "2008-10-05", 2012: "2012-10-07", 2016: "2016-10-02", 2024: "2024-10-06"}
ELEC_Y = [2002, 2006, 2010, 2014, 2018, 2022]
px = nyse_panel(["EWZ", "EEM"])
idx = px.index
E, M = px["EWZ"].values, px["EEM"].values
bv = close_panel(["^BVSP"])["^BVSP"].dropna().loc[:BAR]
bidx, B = bv.index, bv.values
r5 = pct_rank(px["EWZ"], 5)


def first_sunday_oct(y: int) -> pd.Timestamp:
    d = pd.Timestamp(f"{y}-10-01")
    return d + pd.Timedelta(days=(6 - d.weekday()) % 7)


def rets(sunday, k: int = -4, x: int = -1) -> dict:
    """entry k sessions before (k=-1 Friday), exit x (-1 Friday, +1 Monday, +2 Tuesday)."""
    f, m = pos_before(idx, sunday), pos_after(idx, sunday)
    e = f + k + 1
    xp = f + x + 1 if x < 0 else m + x - 1
    fb, mb = pos_before(bidx, sunday), pos_after(bidx, sunday)
    eb = fb + k + 1
    xb = fb + x + 1 if x < 0 else mb + x - 1
    ew, em = span_ret(E, e, xp), span_ret(M, e, xp)
    return {"EWZ": ew, "BVSP": span_ret(B, eb, xb), "PAIR": ew - em, "EEM": em,
            "sig_r5": r5.iloc[e - 1], "pre5": span_ret(E, e - 6, e - 1)}


# 1. concentration
for form, (k, x) in {"run-in k=-4->Fri": (-4, -1), "across k=-4->Mon": (-4, 1)}.items():
    W = pd.DataFrame([rets(v, k, x) for v in FIRST], index=pd.DatetimeIndex(FIRST))
    for c in ("EWZ", "BVSP", "PAIR"):
        v = W[c].dropna()
        print(f"{form} {c}: {cluster_note(v.index, v.values)}")

# 2. neighbours: entry x exit grid
g = []
for k in (-5, -4, -3):
    for x in (-1, 1, 2):
        W = pd.DataFrame([rets(v, k, x) for v in FIRST])
        P = pd.DataFrame([rets(str(first_sunday_oct(y).date()), k, x) for y in range(2001, 2026) if y not in ELEC_Y])
        for c in ("EWZ", "PAIR", "BVSP"):
            r = cell(W[c].values, f"k={k} exit {x:+d} {c}")
            r["placebo_mean"] = round(100 * P[c].mean(), 3)
            g.append(r)
show(g, "entry x exit neighbours (election years vs placebo-year mean)")

# 3. permutation: 6 of the 25 years at random; P(mean >= observed) and rank-sum
rng = np.random.default_rng(7)
yrs = list(range(2001, 2026))
ALL = {y: rets(str(first_sunday_oct(y).date())) for y in yrs}
ALLX = {y: rets(str(first_sunday_oct(y).date()), -4, 1) for y in yrs}
for lbl, D in (("run-in", ALL), ("across", ALLX)):
    for c in ("EWZ", "BVSP", "PAIR"):
        vals = pd.Series({y: D[y][c] for y in yrs}).dropna()
        obs = vals.loc[[y for y in ELEC_Y if y in vals.index]].mean()
        n_e = sum(1 for y in ELEC_Y if y in vals.index)
        draws = np.array([vals.sample(n_e, random_state=int(rng.integers(1e9))).mean() for _ in range(20000)])
        rk = vals.rank(ascending=False)
        print(f"  permutation {lbl} {c}: election mean {100*obs:+.2f}%  P(random {n_e} yrs >= obs) = {(draws >= obs).mean():.3f};"
              f" ranks {sorted(int(rk[y]) for y in ELEC_Y if y in rk.index)} of {len(vals)}")

# 4. municipal-election placebo (an election label with no national binary)
Mu = pd.DataFrame([rets(d) for d in MUNI.values()], index=list(MUNI))
MuX = pd.DataFrame([rets(d, -4, 1) for d in MUNI.values()], index=list(MUNI))
show([cell(Mu.EWZ, "municipal run-in EWZ"), cell(Mu.PAIR, "municipal run-in PAIR"), cell(Mu.BVSP, "municipal run-in BVSP"),
      cell(MuX.EWZ, "municipal across EWZ"), cell(MuX.PAIR, "municipal across PAIR")], "municipal-election placebo (2004-2024)")

# 5. runoff replication of the RUN-IN and ACROSS, all vehicles
Ro = pd.DataFrame([rets(v) for v in RUNOFF], index=RUNOFF)
RoX = pd.DataFrame([rets(v, -4, 1) for v in RUNOFF], index=RUNOFF)
show([cell(Ro[c], f"runoff run-in {c}") for c in ("EWZ", "BVSP", "PAIR")] +
     [cell(RoX[c], f"runoff across {c}") for c in ("EWZ", "BVSP", "PAIR")], "runoff replication")

# 6. pre-vote tape and the washout state (registry: the 5d washout long is dead)
W = pd.DataFrame([rets(v) for v in FIRST], index=FIRST)
print("\nelection years: EWZ 5d return into the signal bar and its 5d rank, then run-in:")
for d, r in W.iterrows():
    print(f"  {d}: pre5 {100*r.pre5:+.2f}%  r5 {r.sig_r5:.0f}  run-in EWZ {100*r.EWZ:+.2f}%  EEM {100*r.EEM:+.2f}%  pair {100*r.PAIR:+.2f}%")
print(f"  LIVE 2026: EWZ 5d {100*(E[-1]/E[-6]-1):+.2f}%  r5 {r5.iloc[-1]:.1f}")
ret3 = fwd_lag(px["EWZ"], 3)
wash = idx[(r5 <= 10).values & ret3.notna().values]
we = declusters(wash, 3, idx)
show([cell(ret3.loc[we].values, "EWZ 5d rank<=10, lag-1 h=3 (episodes)"),
      cell(ret3.dropna().values, "EWZ all days lag-1 h=3")], "washout state control (today's EWZ tape)")
