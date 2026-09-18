import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

# Round 2 for C9: what carries QE-12->QE-2, placebo over h=10 windows by offset,
# drop-best, vol-matched legs, spread as a general (non-calendar) conditioner.
px = close_panel(["SPY", "TLT", "IEF"]).dropna()
idx = px.index
pos = pd.Series(range(len(idx)), index=idx)
g = pd.Series(range(len(idx)), index=idx).groupby([idx.year, idx.month]).max()
me_pos = [int(p) for p in g.values if idx[p] != idx[-1]]
is_q = np.array([idx[p].month in (3, 6, 9, 12) for p in me_pos])
qpos = [p for p, s in zip(me_pos, is_q) if s]
npos = [p for p, s in zip(me_pos, is_q) if not s]

L = [("TLT", 1.0), ("SPY", -1.0)]


def win(legs, a, b):
    if a < 0 or b >= len(idx):
        return np.nan
    return sum(w * (px[t].iat[b] / px[t].iat[a] - 1.0) for t, w in legs)


# A. placebo ladder: h=10 windows entering at QE-e, e = 30..-5 (exit QE-(e-10))
print("=== A. h=10 window by entry offset (QE months vs non-Q months), TLT-SPY ===")
rows = []
for e in range(30, -6, -1):
    vq = np.array([win(L, p - e, p - e + 10) for p in qpos])
    vn = np.array([win(L, p - e, p - e + 10) for p in npos])
    sq, sn = summarize(vq), summarize(vn)
    rows.append({"entry": f"QE-{e}", "exit": f"QE{-(e-10):+d}", "Q_mean": sq["mean_pct"],
                 "Q_t": sq["t"], "Q_hit": sq["hit"], "nonQ_mean": sn["mean_pct"],
                 "diff": sq["mean_pct"] - sn["mean_pct"]})
A = pd.DataFrame(rows)
print(A.round(3).to_string(index=False))
pitched = A.loc[A.entry == "QE-12"].iloc[0]
print(f"pitched QE-12 Q_mean {pitched.Q_mean:.3f}; rank among {len(A)} offsets by Q_mean: "
      f"{int((A.Q_mean > pitched.Q_mean).sum()) + 1}; by diff: {int((A['diff'] > pitched['diff']).sum()) + 1}")

# B. event tagging of sessions inside QE-12..QE-2
ev = load_events(["fomc_decision", "quad_witching"])
fomc = set(pd.DatetimeIndex(ev.loc[ev.event == "fomc_decision", "date"]))
qw = set(pd.DatetimeIndex(ev.loc[ev.event == "quad_witching", "date"]))
cnt_f, cnt_q = {}, {}
for p in qpos:
    for x in range(12, -1, -1):
        d = idx[p - x]
        if d in fomc:
            cnt_f[x] = cnt_f.get(x, 0) + 1
        if d in qw:
            cnt_q[x] = cnt_q.get(x, 0) + 1
print("\n=== B. offset of FOMC decision day / quad witching inside QE months ===")
print("FOMC:", dict(sorted(cnt_f.items(), reverse=True)))
print("QuadW:", dict(sorted(cnt_q.items(), reverse=True)))

# session returns labelled by event, inside the reachable window (sessions ending QE-11..QE-2)
recs = []
for p in qpos:
    for x in range(11, 1, -1):
        d = idx[p - x]
        r = win(L, p - x - 1, p - x)
        tag = "FOMC day" if d in fomc else ("FOMC+1" if idx[p - x - 1] in fomc else
               ("quadW" if d in qw else ("quadW+1" if idx[p - x - 1] in qw else "plain")))
        recs.append({"me": idx[p], "x": x, "tag": tag, "r": r})
S = pd.DataFrame(recs)
show([dict(summarize(S.loc[S.tag == t, "r"].values, t), sum_bp_per_q=10000 * S.loc[S.tag == t, "r"].sum() / len(qpos))
      for t in ["FOMC day", "FOMC+1", "quadW", "quadW+1", "plain"]],
     "B2. one-session TLT-SPY returns inside QE-12->QE-2 by event tag (sum_bp_per_q = contribution to window mean)")
plain_by_q = S[S.tag == "plain"].groupby("me")["r"].sum()
show([summarize(plain_by_q.values, "window ex FOMC/FOMC+1/quadW/quadW+1 sessions (additive)")], "B3")

# C. concentration / drop-best
v = np.array([win(L, p - 12, p - 2) for p in qpos])
d = idx[[p - 13 for p in qpos]]
o = np.argsort(-v)
print("\n=== C. concentration ===")
print("top5:", [(str(idx[qpos[i]].date()), round(100 * v[i], 2)) for i in o[:5]])
print("bottom5:", [(str(idx[qpos[i]].date()), round(100 * v[i], 2)) for i in o[-5:]])
for k in [0, 1, 2, 3]:
    keep = np.ones(len(v), bool)
    keep[o[:k]] = False
    s = summarize(v[keep])
    print(f"drop best {k}: mean {s['mean_pct']:+.3f}% t {s['t']:+.2f} n {s['n']}")
wins = int((v > 0).sum())
print(f"record {wins}-{len(v)-wins} sign p {sign_test(wins, len(v)):.4f}; median {100*np.median(v):+.3f}%")
# LOYO by year
yrs = pd.DatetimeIndex(idx[qpos]).year
loyo = [np.mean(v[yrs != y]) for y in np.unique(yrs)]
print(f"LOYO min {100*min(loyo):+.3f}% max {100*max(loyo):+.3f}%")

# D. vol-matched legs (TLT weight = SPYvol63/TLTvol63 at signal date, SPY -1)
rets = px.pct_change()
vol = rets.rolling(63).std()
vm, ief_vm = [], []
for p in qpos:
    s = p - 13
    if s < 63:
        continue
    wt = vol["SPY"].iat[s] / vol["TLT"].iat[s]
    wi = vol["SPY"].iat[s] / vol["IEF"].iat[s]
    vm.append(win([("TLT", wt), ("SPY", -1.0)], p - 12, p - 2))
    ief_vm.append(win([("IEF", wi), ("SPY", -1.0)], p - 12, p - 2))
allvm = []
for s in range(63, len(idx) - 11):
    wt = vol["SPY"].iat[s] / vol["TLT"].iat[s]
    allvm.append(win([("TLT", wt), ("SPY", -1.0)], s + 1, s + 11))
show([summarize(np.array(vm), "vol-matched TLT-SPY QE-12->QE-2"),
      summarize(np.array(ief_vm), "vol-matched IEF-SPY QE-12->QE-2"),
      summarize(np.array(allvm), "vol-matched TLT-SPY all 10d windows")], "D. vol-matched")
print(f"today vol-matched TLT weight: {vol['SPY'].iat[-1]/vol['TLT'].iat[-1]:.3f}; IEF {vol['SPY'].iat[-1]/vol['IEF'].iat[-1]:.3f}")

# E. spread as a GENERAL conditioner: all days with spread PIT pctile in today's band
spy63 = px["SPY"] / px["SPY"].shift(63) - 1
tlt63 = px["TLT"] / px["TLT"].shift(63) - 1
spread = spy63 - tlt63
exp_pct = spread.expanding(252).apply(lambda s: (s[:-1] < s[-1]).mean() * 100, raw=True)
r10 = vehicle_ret(px, L, 10, 1)
thr90 = spread.expanding(252).quantile(0.90)
print(f"\n=== E. today spread {100*spread.iat[-1]:+.2f}pp, PIT pct {exp_pct.iat[-1]:.1f}; "
      f"expanding 90th pctile level {100*thr90.iat[-1]:+.2f}pp (gap {100*(thr90.iat[-1]-spread.iat[-1]):.2f}pp) ===")
ok = r10.notna() & exp_pct.notna()
rows = []
for lbl, m in [("pct<33", exp_pct < 33.3), ("33-67", (exp_pct >= 33.3) & (exp_pct < 66.7)),
               ("67-90", (exp_pct >= 66.7) & (exp_pct < 90)), (">=90", exp_pct >= 90),
               ("70-80 (today band)", (exp_pct >= 70) & (exp_pct < 80))]:
    mm = m & ok
    rows.append(summarize(r10[mm].values, f"all days {lbl}"))
show(rows, "E. TLT-SPY h=10 lag1 on ALL days by spread PIT pctile (day-level, overlapping)")
# QE anchors in today's band 67-90
qd = idx[[p - 13 for p in qpos]]
qp = exp_pct.reindex(qd)
vq = pd.Series(v, index=qd)
for lbl, m in [("67-90", (qp >= 66.7) & (qp < 90)), (">=90", qp >= 90), ("<67", qp < 66.7)]:
    s = summarize(vq[m].values, lbl)
    w = int((vq[m] > 0).sum())
    print(f"QE {lbl}: n {s['n']} mean {s['mean_pct']:+.3f}% median {s['median_pct']:+.3f}% hit {s['hit']:.0f} "
          f"t {s['t']:+.2f} sign p {sign_test(w, s['n']):.3f}")
fr = filter_vs_reanchor(r10, pd.Series(idx.isin(qd), index=idx),
                        pd.Series(idx.isin(qd[(qp >= 90).values]), index=idx), idx, label="top-decile gate")
print("filter_vs_reanchor top-decile:", {k: (round(val, 4) if isinstance(val, float) else val) for k, val in fr.items() if k not in ("pairs", "deleted_dates", "shifts")})

# F. September QE by era (the only month with a visible cell)
sep = [(idx[p], win(L, p - 12, p - 2)) for p in qpos if idx[p].month == 9]
sp = pd.Series([r for _, r in sep], index=[d for d, _ in sep])
for lbl, m in [("Sep 2002-2016", sp.index.year <= 2016), ("Sep 2017-2025", sp.index.year >= 2017)]:
    s = summarize(sp[m].values, lbl)
    w = int((sp[m] > 0).sum())
    print(f"{lbl}: n {s['n']} mean {s['mean_pct']:+.3f}% record {w}-{s['n']-w}")
