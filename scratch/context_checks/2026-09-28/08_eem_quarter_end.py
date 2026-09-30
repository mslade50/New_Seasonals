"""EEM's final two sessions: quarter-ends 62 of 93 up (+0.50%) vs other month-ends 89 of 187 (07). Is it EM-specific,
which quarters carry it, how concentrated, and does September belong?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

px = load_prices(["EEM", "SPY", "EFA", "IWM", "UUP"])
res = {}
for tk in ["EEM", "SPY", "EFA", "IWM"]:
    c = px[tk]["Close"].astype(float)
    c = c.loc[c.index.intersection(px["SPY"].index)]
    idx = c.index
    per = pd.Series(idx.to_period("M"), index=idx)
    months = sorted(set(per.values))
    rows = []
    for i, m in enumerate(months):
        if i == 0 or i + 1 >= len(months) or m == idx[-1].to_period("M"):
            continue
        d = idx[(per == m).values]
        if len(d) < 5:
            continue
        a, s2 = d[-3], d[-1]
        rows.append({"m": m, "mon": m.month, "q": m.month in (3, 6, 9, 12), "a": a, "last2": c[s2] / c[a] - 1})
    df = pd.DataFrame(rows)
    res[tk] = df
    q, o = df[df.q], df[~df.q]
    se = np.sqrt(q.last2.var(ddof=1) / len(q) + o.last2.var(ddof=1) / len(o))
    print(f"{tk} from {idx[0].date()}: quarter-end last2 {100 * q.last2.mean():.3f}% up {int((q.last2 > 0).sum())}/{len(q)}; "
          f"other {100 * o.last2.mean():.3f}% up {int((o.last2 > 0).sum())}/{len(o)}; Welch t {(q.last2.mean() - o.last2.mean()) / se:.2f}")
    by = df[df.q].groupby("mon")["last2"].agg(["count", "mean", lambda x: int((x > 0).sum())])
    by["mean"] *= 100
    print(by.round(3).to_string())

e = res["EEM"]
q = e[e.q]
print("\n" + cluster_note(pd.DatetimeIndex(q["a"]), q["last2"].values))
nsep = q[q.mon != 9]
print(f"EEM quarter-ends ex-September: {100 * nsep.last2.mean():.3f}% up {int((nsep.last2 > 0).sum())}/{len(nsep)}")
sep = q[q.mon == 9]
se = np.sqrt(sep.last2.var(ddof=1) / len(sep) + nsep.last2.var(ddof=1) / len(nsep))
print(f"Welch t Sept vs other quarter-ends: {(sep.last2.mean() - nsep.last2.mean()) / se:.2f}")
print("EEM Septembers:", [(r.m.year, round(100 * r.last2, 2)) for r in sep.itertuples()])
# EEM relative to SPY at quarter-ends
s = res["SPY"].set_index("m")["last2"]
rel = (q.set_index("m")["last2"] - s.reindex(q["m"])).dropna()
print(f"EEM minus SPY at quarter-ends: {100 * rel.mean():.3f}% EEM ahead {int((rel > 0).sum())}/{len(rel)}")
show(era_split(pd.DatetimeIndex(q["a"]), q["last2"].values), "EEM quarter-end last2 era")
