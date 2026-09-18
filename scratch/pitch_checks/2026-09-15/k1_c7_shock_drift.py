"""C7 round 1: industry-shock drift. A W25 reference-class member (23 ETFs)
falls >= 1.5 Wilder ATR (move / prior-day ATR, the 00_live_arms convention)
on a session SPY falls less than 0.75%. Claim: SHORT the member against SPY
(beta-neutral residual, rolling 126d beta as of the signal day), h=1..5,
lag=1 (MOC 09-15).

Two-sided: the unconditioned parent is reported as the LONG residual
(reversal); the short claim is its negative. p-values are charged two-sided.
Declustering: per name, explicit min_gap (5 and 10) -- battery's default of
min_gap=h gives NO declustering at h<=2 (registry 2026-09-11). Cross-name
same-week episodes are collapsed into calendar clusters as a second count.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch_lab import *  # noqa
from k1_common import (ASOF, REF23, atr_series, date_clusters, dial_ma10, fwd, rec)  # noqa

px = load_prices(REF23 + ["SPY"])
spyd = px["SPY"][px["SPY"].index <= ASOF]
spy = spyd["Close"]
spy1 = spy.pct_change()
spy200 = spy / spy.rolling(200).mean() - 1
dial = dial_ma10()
HS = (1, 2, 3, 5)

recs = []   # one row per (name, shock day) with forward residuals
ctrl = {h: [] for h in HS}
live = []
for t in REF23:
    d = px[t][px[t].index <= ASOF]
    s = d["Close"]
    idx = s.index.intersection(spy.index)
    s = s.loc[idx]
    sp = spy.loc[idx]
    atr = atr_series(d).reindex(idx).shift(1)
    move_atr = (s - s.shift(1)) / atr
    r1 = s.pct_change()
    q1 = sp.pct_change()
    beta = (r1.rolling(126).cov(q1) / q1.rolling(126).var())
    shock = (move_atr <= -1.5) & (q1 > -0.0075)
    broad = (move_atr <= -1.5) & (q1 <= -0.0075)
    res = {h: fwd(s, h, 1) - beta * fwd(sp, h, 1) for h in HS}
    raw = {h: fwd(s, h, 1) for h in HS}
    res0 = fwd(s, 1, 0) - beta * fwd(sp, 1, 0)
    for h in HS:
        v = res[h].dropna()
        v = v[v.index >= "2001-01-01"]
        ctrl[h].append(v)
    live.append({"t": t, "move_atr": move_atr.iloc[-1], "spy1d": q1.iloc[-1], "beta126": beta.iloc[-1],
                 "shock_today": bool(shock.iloc[-1])})
    for kind, m in (("calm", shock), ("broad", broad)):
        days = idx[m.fillna(False).values & beta.notna().values]
        for dd in days:
            row = {"t": t, "d": dd, "kind": kind, "move_atr": move_atr.loc[dd],
                   "beta": beta.loc[dd], "spy200": spy200.get(dd, np.nan), "dial": dial.get(dd, np.nan),
                   "res0_1": res0.loc[dd]}
            for h in HS:
                row[f"res{h}"] = res[h].loc[dd]
                row[f"raw{h}"] = raw[h].loc[dd]
            recs.append(row)

R = pd.DataFrame(recs)
L = pd.DataFrame(live).set_index("t")
print("LIVE 2026-09-14 (shock = move <= -1.5 prior-day Wilder ATR AND SPY 1d > -0.75%):")
print(L[L.move_atr <= -1.0].round(3).to_string())
print(f"  SPY 1d {100*spy1.iloc[-1]:+.2f}%")


def decl(df, gap):
    """Per-name declustering on each name's own session index."""
    keep = []
    for t, g in df.groupby("t"):
        idx = px[t].index[px[t].index <= ASOF]
        dd = declusters(pd.DatetimeIndex(g.d), gap, idx)
        keep.append(g[g.d.isin(dd)])
    return pd.concat(keep) if keep else df.iloc[:0]


def two_sided(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    w = int((v > 0).sum())
    n = len(v)
    return min(1.0, 2 * min(sign_test(w, n), sign_test(n - w, n))) if n else np.nan


def block(df, gap, label):
    rows = []
    for h in HS:
        v = df[f"res{h}"].dropna()
        dd = df.loc[v.index, "d"]
        s = summarize(v.values, f"{label} h={h} LONG residual")
        cb = np.concatenate([c.values for c in ctrl[h]])
        s["ctrl_all_pct"] = 100 * np.nanmean(cb)
        cl, _ = date_clusters(dd, v.values, 7)
        s["clusters"] = len(cl)
        s["cl_mean_pct"] = 100 * cl.mean()
        s["cl_rec"] = rec(cl)
        s["cl_p2"] = round(two_sided(cl), 4)
        s["short_pct"] = -s["mean_pct"]
        rows.append(s)
    show(rows, f"{label} (gap {gap})")


for gap in (5, 10):
    calm = decl(R[R.kind == "calm"], gap)
    broad = decl(R[R.kind == "broad"], gap)
    block(calm, gap, "CALM-SPY shock (C7 parent)")
    block(broad, gap, "BROAD shock (SPY <= -0.75%) contrast")

calm = decl(R[R.kind == "calm"], 5)
print("\nlag=0 contrast h=1 (untradeable signal-close entry), calm shock, gap 5:",
      summarize(calm.res0_1.values, "")["mean_pct"].round(3))
print("\nraw member (unhedged) calm shock, gap 5:")
show([summarize(calm[f"raw{h}"].values, f"raw h={h}") for h in HS])

print("\n=== ERA / REGIME splits, calm shock gap 5, LONG residual (short = negative) ===")
rows = []
for lbl, sub in [("pre-2018", calm[calm.d < "2018-01-01"]), ("2018+", calm[calm.d >= "2018-01-01"]),
                 ("SPY>=200d", calm[calm.spy200 >= 0]), ("SPY<200d", calm[calm.spy200 < 0]),
                 ("dial>=50", calm[calm.dial >= 50]), ("dial<50", calm[calm.dial < 50]),
                 ("move <= -2 ATR", calm[calm.move_atr <= -2.0]),
                 ("-2 < move <= -1.5", calm[calm.move_atr > -2.0])]:
    for h in (1, 3, 5):
        s = summarize(sub[f"res{h}"].values, f"{lbl} h={h}")
        cl, _ = date_clusters(sub.d, sub[f"res{h}"].values, 7)
        s["clusters"], s["cl_mean_pct"], s["cl_rec"] = len(cl), 100 * np.nanmean(cl), rec(cl)
        rows.append(s)
show(rows)

print("\n=== per-name h=3 LONG residual, calm shock gap 5 ===")
g = calm.groupby("t")["res3"].agg(["size", "mean", "std"])
g["mean_pct"] = 100 * g["mean"]
g["t_stat"] = g["mean"] / (g["std"] / np.sqrt(g["size"]))
print(g[["size", "mean_pct", "t_stat"]].sort_values("mean_pct").round(3).T.to_string())
from k1_common import cochran  # noqa
q = cochran(g.index, 100 * g["mean"], 100 * g["std"] / np.sqrt(g["size"]))
print(f"  Cochran on h=3: FE {q['fe_pct']:+.3f}pp (se {q['fe_se_pct']:.3f})  Q p {q['p']:.3f}  I2 {q['I2']:.1f}%")
print("\n  SMH calm-shock history (gap 5):")
print(calm[calm.t == "SMH"][["d", "move_atr", "res1", "res3", "res5"]].round(4).tail(15).to_string())
print("\n  OIH calm-shock history (gap 5):")
print(calm[calm.t == "OIH"][["d", "move_atr", "res1", "res3", "res5"]].round(4).tail(15).to_string())
R.to_pickle(Path(__file__).with_name("k1_c7_shocks.pkl"))
