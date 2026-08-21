"""C11 round 2: the reference-class permutation the family caveat demands,
plus gate attribution, residual teardown and the yen.

Round 1: EWJ outright +1.564% over 42 episodes (excess +1.460, welch +2.69,
30-12 sign p 0.004), beta-neutral residual against EFA at 0.839 = +0.674% at
a 71.4% hit, sign p 0.004. Tape over-selection CLEAN (SPY above its 200d on
66.7% of episodes against a 71.6% base). Concentration NEGATIVE (top-2 are
losers). Thresholds monotone the right way.

The one thing left is the argument that killed KWEB: being the best of ten on
a correlated class is what the null does. Two nulls, 20,000 draws each:
  N1 random anchors (count and min-gap matched) per name -> max-of-K excess
  N2 common-mean class null: pool every peer's own-rule episode excesses,
     resample each name at its own N, take the max
Plus: does the EFA gate filter, is the residual era-stable, and is Japan just
the yen?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change  # noqa

import warnings
warnings.filterwarnings("ignore")

ASOF = pd.Timestamp("2026-08-20")
H = 5
PEERS = ["EWJ", "EWT", "EWW", "EWY", "EWZ", "EEM", "FXI", "INDA", "KWEB", "RSX"]

px = close_panel(PEERS + ["EFA", "SPY", "JPY=X"]).loc[:ASOF]
idx = px.index
r5 = {c: _valid_pct_change(px[c], 5) for c in px.columns}
rk5 = {c: pct_rank(px[c], 5) for c in px.columns}
gap = r5["EWJ"] - r5["EFA"]
mask = ((rk5["EWJ"] <= 5) & (gap <= -0.025)).fillna(False)
leg = fwd_lag(px["EWJ"], H, 1)
epi = declusters(idx[(mask & leg.notna()).values], 5, idx)
d = px[["EWJ", "EFA"]].pct_change().dropna()
B = float(np.polyfit(d["EFA"], d["EWJ"], 1)[0])
resid = vehicle_ret(px, [("EWJ", 1.0), ("EFA", -B)], H, 1)

print("=" * 100)
print("C11b-1  GATE ATTRIBUTION -- does 'EFA holds' filter, or is it just a washout?")
print("=" * 100)
base = leg.dropna()
for lbl, m in (("EWJ 5d rank<=5 ALONE (no EFA gate)", (rk5["EWJ"] <= 5).fillna(False)),
               ("gap<=-2.5pp ALONE (no washout gate)", (gap <= -0.025).fillna(False)),
               ("BOTH (the cell)", mask),
               ("EWJ rank<=5 AND gap > -2.5pp (complement)",
                ((rk5["EWJ"] <= 5) & (gap > -0.025)).fillna(False))):
    e = declusters(idx[(m & leg.notna()).values], 5, idx)
    v = leg.loc[e].values
    rr = resid.loc[e].values
    print(f"  {lbl:<42} N={len(e):<4} outright {100*v.mean():+7.3f}% (excess "
          f"{100*v.mean()-100*base.mean():+7.3f}%, hit {100*(v>0).mean():5.1f}%, "
          f"signp {sign_test(int((v>0).sum()), len(v)):.4f})  resid {100*rr.mean():+7.3f}%")
a = leg.loc[epi].values
e2 = declusters(idx[((rk5["EWJ"] <= 5).fillna(False) & leg.notna()).values], 5, idx)
b = leg.loc[e2].values
se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
print(f"  EFA-gate increment over washout-alone: {100*(a.mean()-b.mean()):+.3f}pp  welch t {(a.mean()-b.mean())/se:+.2f}")

print("\n" + "=" * 100)
print("C11b-2  RESIDUAL TEARDOWN (beta-neutral EWJ - %.2f*EFA)" % B)
print("=" * 100)
rv = resid.loc[epi].values
print(f"  {cluster_note(epi, rv)}")
show(era_split(epi, rv), "residual era split")
for g in (5, 10, 21, 42):
    e = declusters(idx[(mask & resid.notna()).values], g, idx)
    x = resid.loc[e].values
    print(f"  min_gap {g:>2}: N={len(e):<4} resid {100*x.mean():+7.3f}%  hit {100*(x>0).mean():5.1f}%  "
          f"signp {sign_test(int((x>0).sum()), len(x)):.4f}  boot P(<=0) {bootstrap_p_le0(x):.3f}")
yrs = pd.DatetimeIndex(epi).year
print("  drop-one-year floor on the residual:")
worst = None
for y in sorted(set(yrs)):
    m = yrs != y
    v2 = rv[m]
    if worst is None or v2.mean() < worst[1]:
        worst = (y, v2.mean())
print(f"     LOYO floor = {100*worst[1]:+.3f}% dropping {worst[0]}  (full {100*rv.mean():+.3f}%)")

print("\n" + "=" * 100)
print("C11b-3  NULL 1: random anchors, count- and gap-matched, max-of-K")
print("=" * 100)
rng = np.random.default_rng(42)
NB = 20000
# per-name: own-rule episodes, excess vs own drift
obs = {}
pool = {}
for c in PEERS:
    g = r5[c] - r5["EFA"]
    m = ((rk5[c] <= 5) & (g <= -0.025)).fillna(False)
    r = fwd_lag(px[c], H, 1)
    e = declusters(idx[(m & r.notna()).values], 5, idx)
    if len(e) < 5:
        continue
    bb = r.dropna()
    obs[c] = (len(e), 100 * r.loc[e].mean() - 100 * bb.mean())
    pool[c] = (bb.values, 100 * bb.mean())
print("  observed own-rule excess:", {k: round(v[1], 3) for k, v in obs.items()})
maxes = np.empty(NB)
names = list(obs)
draws = {c: pool[c][0] for c in names}
for i in range(NB):
    best = -1e9
    for c in names:
        n = obs[c][0]
        s = rng.choice(draws[c], size=n, replace=True)
        ex = 100 * s.mean() - pool[c][1]
        if ex > best:
            best = ex
    maxes[i] = best
ewj_ex = obs["EWJ"][1]
print(f"  N1 (random anchors, all-days resample): P(max-of-{len(names)} excess >= EWJ's "
      f"{ewj_ex:+.3f}) = {(maxes >= ewj_ex).mean():.4f}")
print(f"     null max-of-K: median {np.median(maxes):+.3f}  95th {np.percentile(maxes,95):+.3f}")
print(f"  EWJ ALONE against its own random-anchor null: "
      f"P = {(np.array([100*rng.choice(draws['EWJ'], size=obs['EWJ'][0], replace=True).mean() - pool['EWJ'][1] for _ in range(NB)]) >= ewj_ex).mean():.4f}")

print("\n" + "=" * 100)
print("C11b-4  NULL 2: common-mean class null -- if every name shared the class")
print("        effect, how often is the best of K >= EWJ's excess?")
print("=" * 100)
# pooled episode-level excesses across all peers
allex = []
for c in names:
    g = r5[c] - r5["EFA"]
    m = ((rk5[c] <= 5) & (g <= -0.025)).fillna(False)
    r = fwd_lag(px[c], H, 1)
    e = declusters(idx[(m & r.notna()).values], 5, idx)
    bb = r.dropna()
    allex.append(100 * r.loc[e].values - 100 * bb.mean())
pooled = np.concatenate(allex)
print(f"  pooled class episode-excess: N={len(pooled)}  mean {pooled.mean():+.3f}%  sd {pooled.std(ddof=1):.3f}%")
maxes2 = np.empty(NB)
ns = [obs[c][0] for c in names]
for i in range(NB):
    maxes2[i] = max(rng.choice(pooled, size=n, replace=True).mean() for n in ns)
print(f"  N2: P(max-of-{len(names)} >= EWJ's {ewj_ex:+.3f}) = {(maxes2 >= ewj_ex).mean():.4f}")
print(f"     null max-of-K: median {np.median(maxes2):+.3f}  95th {np.percentile(maxes2,95):+.3f}")
# same on the beta-neutral residual
allres, obsres = [], {}
for c in names:
    g = r5[c] - r5["EFA"]
    m = ((rk5[c] <= 5) & (g <= -0.025)).fillna(False)
    dd = px[[c, "EFA"]].pct_change().dropna()
    bb2 = float(np.polyfit(dd["EFA"], dd[c], 1)[0])
    rr = vehicle_ret(px, [(c, 1.0), ("EFA", -bb2)], H, 1)
    e = declusters(idx[(m & rr.notna()).values], 5, idx)
    allres.append(100 * rr.loc[e].values - 100 * rr.dropna().mean())
    obsres[c] = 100 * rr.loc[e].mean() - 100 * rr.dropna().mean()
pooledr = np.concatenate(allres)
mx3 = np.array([max(rng.choice(pooledr, size=n, replace=True).mean() for n in ns) for _ in range(NB)])
print(f"  N2 on the RESIDUAL: EWJ {obsres['EWJ']:+.3f}%, "
      f"P(max-of-{len(names)} >= it) = {(mx3 >= obsres['EWJ']).mean():.4f}   "
      f"peers: {{" + ", ".join(f'{k}: {v:+.2f}' for k, v in obsres.items()) + "}}")

print("\n" + "=" * 100)
print("C11b-5  IS JAPAN JUST THE YEN?  (JPY=X is USD per JPY quote convention check)")
print("=" * 100)
jp = px["JPY=X"].dropna()
print(f"  JPY=X history {jp.index[0].date()} .. {jp.index[-1].date()}, last {jp.iloc[-1]:.2f}")
j5 = _valid_pct_change(px["JPY=X"], 5)
print(f"  today JPY=X 5d move {100*j5.loc[ASOF]:+.2f}%")
sub = pd.concat([px["EWJ"].pct_change(), px["JPY=X"].pct_change()], axis=1).dropna()
sub.columns = ["ewj", "jpy"]
print(f"  daily corr EWJ vs JPY=X = {sub['ewj'].corr(sub['jpy']):+.3f}  (N={len(sub)})")
je = j5.loc[epi].dropna()
print(f"  on the {len(je)} episodes with FX data: median JPY=X 5d move {100*je.median():+.2f}%")
fwd_j = fwd_lag(px["JPY=X"], H, 1)
common = pd.DatetimeIndex([x for x in epi if not np.isnan(fwd_j.get(x, np.nan))])
print(f"  forward JPY=X over the hold: mean {100*fwd_j.loc[common].mean():+.3f}% "
      f"(all-days {100*fwd_j.dropna().mean():+.3f}%)")
fwd_e = leg.loc[common].values
fj = fwd_j.loc[common].values
print(f"  corr(EWJ fwd, JPY fwd) on episodes = {np.corrcoef(fwd_e, fj)[0,1]:+.3f}")
bj = float(np.polyfit(fj, fwd_e, 1)[0])
print(f"  EWJ fwd regressed on JPY fwd: beta {bj:+.3f}  alpha "
      f"{100*(fwd_e.mean() - bj*fj.mean()):+.3f}%")

print("\n" + "=" * 100)
print("C11b-6  MID-CLUSTER ENTRY + today's position in the run")
print("=" * 100)
pos = pd.Series(range(len(idx)), index=idx)
trig = idx[(mask & leg.notna()).values]
prior = np.array([int(mask.iloc[max(0, pos[dd] - 10):pos[dd]].sum()) for dd in trig])
dv = leg.loc[trig].values
for lo, hi in ((0, 1), (1, 3), (3, 11)):
    s = (prior >= lo) & (prior < hi)
    if s.sum() == 0:
        continue
    print(f"  prior-10td trigger count [{lo},{hi}): N={int(s.sum()):<4} mean {100*dv[s].mean():+7.3f}%  "
          f"hit {100*(dv[s]>0).mean():5.1f}% (day level)")
print(f"  TODAY prior-10td trigger count = {int(mask.iloc[pos[ASOF]-10:pos[ASOF]].sum())}")
print("  last 8 sessions EWJ5 rank / gap:")
for dd in idx[-8:]:
    print(f"     {dd.date()}  EWJ rk5 {rk5['EWJ'].loc[dd]:5.1f}  gap {100*gap.loc[dd]:+6.2f}pp  "
          f"{'TRIGGER' if mask.loc[dd] else ''}")
