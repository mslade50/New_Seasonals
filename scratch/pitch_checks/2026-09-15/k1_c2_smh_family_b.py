"""C2 round 2: the ex-SMH family replicated the r5<15 conditioner's sign
(+0.861pp mask form, +1.216pp split form). Round 2 asks whether that is a
family effect that survives concentration, definition neighbours, era/regime
and gate attribution, and whether TODAY's SMH instance sits inside the support.

(a) concentration: year shares, drop-2021, drop-2026, date clusters
(b) neighbours: r63 3/10, r5 10/20, r252 30%/50%, gap 5/21, h 5/15
(c) regime: SPY >/< 200d, SPY off-high, dial ma10 63d (2016+), midterm
(d) gate attribution: r5<15 filter vs re-anchor on the B parent; the 252d gate
    (r63<=5 & r5<15 with and without it); and the C7 shock-day split
(e) today's position in its own SMH cluster (mask form, first-in-10)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch_lab import *  # noqa
from k1_common import (ASOF, REF23, atr_series, cell_stats, date_clusters,  # noqa
                       dial_ma10, fwd, rec, vret)

px = load_prices(REF23 + ["SPY"])
spy = px["SPY"]["Close"].dropna()
spy = spy[spy.index <= ASOF]
spy200 = spy / spy.rolling(200).mean() - 1
spyoffhi = spy / spy.rolling(252).max() - 1
spy1d = spy.pct_change()
dial = dial_ma10()
EX = [t for t in REF23 if t != "SMH"]

S, A = {}, {}
for t in REF23:
    d = px[t][px[t].index <= ASOF]
    s = d["Close"].dropna()
    S[t] = s
    a = atr_series(d).shift(1)
    A[t] = ((s - s.shift(1)) / a).reindex(s.index)


def feats(t):
    s = S[t]
    return {"r63": pct_rank(s, 63), "r5": pct_rank(s, 5), "r252": vret(s, 252),
            "shock": (A[t] <= -1.5) & (spy1d.reindex(s.index) > -0.0075)}


F = {t: feats(t) for t in REF23}


def run(names, fn, h=10, gap=10):
    out = []
    for t in names:
        s = S[t]
        r = fwd(s, h, 1)
        cs = cell_stats(r, fn(F[t]), gap)
        for dd, x in zip(cs["_dates"], cs["_vals"]):
            out.append({"t": t, "d": dd, "raw": x, "exc": x - cs["_drift"],
                        "spy200": spy200.get(dd, np.nan), "offhi": spyoffhi.get(dd, np.nan),
                        "dial": dial.get(dd, np.nan), "shock": bool(F[t]["shock"].get(dd, False))})
    return pd.DataFrame(out)


def line(df, label):
    if df is None or len(df) == 0:
        return {"label": label, "n": 0}
    s = summarize(df.exc.values, label)
    s["raw_pct"] = 100 * df.raw.mean()
    cl, _ = date_clusters(df.d, df.exc.values, 14)
    s["clusters"] = len(cl)
    s["cl_exc_pct"] = 100 * cl.mean()
    s["cl_rec"] = rec(cl)
    return s


C = lambda f: (f["r63"] <= 5) & (f["r252"] >= 0.40) & (f["r5"] < 15)
B = lambda f: (f["r63"] <= 5) & (f["r252"] >= 0.40)
base = run(EX, C)

print("=" * 78)
print("(a) CONCENTRATION, ex-SMH C mask, h=10")
print("=" * 78)
yr = base.groupby(base.d.dt.year).agg(n=("exc", "size"), sum_exc=("exc", "sum"), mean=("exc", "mean"))
yr["share_pct"] = 100 * yr.sum_exc / base.exc.sum()
print((yr.assign(sum_exc=100 * yr.sum_exc, mean=100 * yr["mean"])).round(2).to_string())
print("  " + cluster_note(pd.DatetimeIndex(base.d), base.exc.values, k=2))
show([line(base, "ex-SMH C all"),
      line(base[base.d.dt.year != 2021], "drop 2021"),
      line(base[base.d.dt.year != 2026], "drop 2026 (live regime)"),
      line(base[~base.d.dt.year.isin([2021, 2026])], "drop 2021 and 2026")])

print("\n" + "=" * 78)
print("(b) DEFINITION NEIGHBOURS, ex-SMH pooled excess (h=10 gap 10 unless noted)")
print("=" * 78)
rows = []
for lbl, fn, h, gap in [
    ("W25 C: r63<=5 r252>=40 r5<15", C, 10, 10),
    ("r63<=3", lambda f: (f["r63"] <= 3) & (f["r252"] >= 0.40) & (f["r5"] < 15), 10, 10),
    ("r63<=10", lambda f: (f["r63"] <= 10) & (f["r252"] >= 0.40) & (f["r5"] < 15), 10, 10),
    ("r5<10", lambda f: (f["r63"] <= 5) & (f["r252"] >= 0.40) & (f["r5"] < 10), 10, 10),
    ("r5<20", lambda f: (f["r63"] <= 5) & (f["r252"] >= 0.40) & (f["r5"] < 20), 10, 10),
    ("r252>=30%", lambda f: (f["r63"] <= 5) & (f["r252"] >= 0.30) & (f["r5"] < 15), 10, 10),
    ("r252>=50%", lambda f: (f["r63"] <= 5) & (f["r252"] >= 0.50) & (f["r5"] < 15), 10, 10),
    ("gap 5", C, 10, 5), ("gap 21", C, 10, 21),
    ("h=5", C, 5, 10), ("h=15", C, 15, 15), ("h=3", C, 3, 10),
]:
    rows.append(line(run(EX, fn, h, gap), lbl))
show(rows)

print("\n" + "=" * 78)
print("(c) REGIME, ex-SMH C mask h=10  [today: SPY +6.7% over 200d, -2.19% off high, dial 85.2]")
print("=" * 78)
show([line(base[base.spy200 >= 0], "SPY >= 200d"),
      line(base[base.spy200 < 0], "SPY < 200d"),
      line(base[base.spy200 >= 0.05], "SPY >= +5% over 200d"),
      line(base[base.offhi >= -0.05], "SPY within 5% of 252 high"),
      line(base[base.dial.notna()], "dial available (2016+)"),
      line(base[base.dial >= 50], "dial >= 50"),
      line(base[base.dial < 50], "dial < 50"),
      line(base[base.d < "2018-01-01"], "pre-2018"),
      line(base[base.d >= "2018-01-01"], "2018+"),
      line(base[base.d.dt.year % 4 == 2], "midterm years"),
      line(base[base.d.dt.year % 4 != 2], "non-midterm")])
print(f"  dial on ex-SMH episodes: max {base.dial.max():.1f}, n>=80: {(base.dial>=80).sum()}")
print(f"  SPY over 200d on episodes: median {100*base.spy200.median():.1f}%  "
      f"share >= +5%: {100*(base.spy200>=0.05).mean():.0f}%")

print("\n" + "=" * 78)
print("(d) GATE ATTRIBUTION")
print("=" * 78)
# r5<15 conditioner: filter vs re-anchor on the B parent, pooled ex-SMH
agg = {"deleted": [], "kp": [], "kc": [], "pall": []}
for t in EX:
    r = fwd(S[t], 10, 1)
    valid = r.dropna().index
    bm = B(F[t]).reindex(r.index, fill_value=False).fillna(False)
    cm = C(F[t]).reindex(r.index, fill_value=False).fillna(False)
    be = declusters(r.index[bm.values].intersection(valid), 10, valid)
    ce = declusters(r.index[cm.values].intersection(valid), 10, valid)
    if len(be) == 0:
        continue
    pm = pd.Series(r.index.isin(be), index=r.index)
    cmk = pd.Series(r.index.isin(ce), index=r.index)
    fr = filter_vs_reanchor(r - r.loc[valid].mean(), pm, cmk, r.index, window_td=21)
    agg["deleted"] += [r.loc[d] - r.loc[valid].mean() for d in fr["deleted_dates"]]
    agg["kp"] += [r.loc[a] - r.loc[valid].mean() for a, _, _ in fr["pairs"]]
    agg["kc"] += [r.loc[b] - r.loc[valid].mean() for _, b, _ in fr["pairs"]]
    agg["pall"] += [r.loc[d] - r.loc[valid].mean() for d in be]
m = {k: 100 * np.nanmean(v) for k, v in agg.items()}
print(f"  r5<15 on B parent (ex-SMH pooled, excess): parent all {m['pall']:+.3f} (N={len(agg['pall'])}), "
      f"deleted {m['deleted']:+.3f} (N={len(agg['deleted'])}), kept@parent {m['kp']:+.3f}, "
      f"kept@child {m['kc']:+.3f} (N={len(agg['kp'])})")
print(f"  FILTERING {m['kp']-m['pall']:+.3f}pp   RE-ANCHORING {m['kc']-m['kp']:+.3f}pp")
# 252d gate
show([line(run(EX, lambda f: (f["r63"] <= 5) & (f["r5"] < 15)), "r63<=5 & r5<15, NO 252 gate"),
      line(run(EX, lambda f: (f["r63"] <= 5) & (f["r5"] < 15) & (f["r252"] < 0.40)), "  complement: 252 < 40%"),
      line(base, "  with 252 >= 40% (C)")], "252d gate attribution (ex-SMH)")
run_all = run(REF23, lambda f: (f["r63"] <= 5) & (f["r5"] < 15))
show([line(run_all[run_all.spy200 >= 0], "NO 252 gate, SPY>=200d (all 23)"),
      line(run_all[run_all.spy200 < 0], "NO 252 gate, SPY<200d (all 23)")])
# C7 shock-day split inside C, all 23 names
c23 = run(REF23, C)
show([line(c23[c23.shock], "C episodes whose date is a -1.5 ATR calm-SPY shock"),
      line(c23[~c23.shock], "C episodes, no shock that day")], "C7 shock split inside C (all 23)")
cs_any = run(REF23, lambda f: C(f) & f["shock"], 10, 10)
show([line(cs_any, "C AND shock-day as the mask (all 23)"),
      line(cs_any[cs_any.t != "SMH"], "  ex-SMH")])
print(cs_any[["t", "d", "raw", "exc", "spy200", "dial"]].round(4).to_string())

print("\n" + "=" * 78)
print("(e) TODAY inside SMH's own cluster")
print("=" * 78)
f = F["SMH"]
cm = C(f)
bm = B(f)
last = cm[cm].index[-12:]
print("  last 12 SMH C-mask days:", [str(x.date()) for x in last])
print("  last 5 SMH B-mask days:", [str(x.date()) for x in bm[bm].index[-5:]])
prev = cm[cm].index
p = list(S["SMH"].index).index(ASOF)
prior = [d for d in prev if d < ASOF]
gap_td = p - list(S["SMH"].index).index(prior[-1]) if prior else None
print(f"  today C={bool(cm.loc[ASOF])}; sessions since prior C day: {gap_td}")
b_run = 0
for d in reversed(S["SMH"].index[:p + 1]):
    if bm.get(d, False):
        b_run += 1
    else:
        break
print(f"  consecutive B days ending today: {b_run}; B days in last 63 sessions: "
      f"{int(bm.iloc[p-62:p+1].sum())}")
print(f"  SMH shock today {A['SMH'].loc[ASOF]:+.2f} ATR; SPY 1d {100*spy1d.loc[ASOF]:+.2f}%")
