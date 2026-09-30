"""Follow-on to 09-27 item 1. In months TLT entered its final three down 2.9%+ (worst fifth), what did the
last two sessions do when the first of the three fell, as Monday's did (-0.88%)? Complete months only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

px = load_prices(["TLT", "IEF", "SPY"])
rows = []
for tk in ["TLT", "IEF"]:
    c = px[tk]["Close"].astype(float)
    idx = c.index
    per = pd.Series(idx.to_period("M"), index=idx)
    months = sorted(set(per.values))
    cur_m = idx[-1].to_period("M")
    for i, m in enumerate(months):
        if m == cur_m or i == 0 or i + 1 >= len(months):
            continue
        d = idx[(per == m).values]
        prev = idx[(per == months[i - 1]).values]
        nxt = idx[(per == months[i + 1]).values]
        if len(d) < 6 or len(prev) == 0 or len(nxt) < 2:
            continue
        a, s1, s2, s3 = d[-4], d[-3], d[-2], d[-1]
        rows.append({"tk": tk, "month": str(m), "a": a,
                     "mtd_a": c[a] / c[prev[-1]] - 1,
                     "r_s1": c[s1] / c[a] - 1, "r_s2": c[s2] / c[s1] - 1, "r_s3": c[s3] / c[s2] - 1,
                     "last2": c[s3] / c[s1] - 1, "final3": c[s3] / c[a] - 1,
                     "next2": c[nxt[1]] / c[s3] - 1})
df = pd.DataFrame(rows)
tl = df[df.tk == "TLT"].copy()
thr = tl["mtd_a"].quantile(0.2)
print(f"TLT months {len(tl)}, worst-fifth threshold {100 * thr:.2f}% (last night used -2.93%)")
tl["bad"] = tl["mtd_a"] <= -0.0293
out = []
for lab, sub in [("all months", tl), ("bad", tl[tl.bad]), ("bad, s1 down", tl[tl.bad & (tl.r_s1 < 0)]),
                 ("bad, s1 up", tl[tl.bad & (tl.r_s1 > 0)]), ("bad, s1 <= -0.5%", tl[tl.bad & (tl.r_s1 <= -0.005)]),
                 ("not bad, s1 down", tl[~tl.bad & (tl.r_s1 < 0)]), ("all, s1 down", tl[tl.r_s1 < 0]),
                 ("all, s1 <= -0.5%", tl[tl.r_s1 <= -0.005])]:
    for col in ["last2", "r_s2", "r_s3", "next2"]:
        s = summarize(sub[col].values, f"{lab}: {col}")
        s["up"] = int((sub[col] > 0).sum())
        out.append(s)
show(out, "TLT final two sessions by the first session's sign")

b = tl[tl.bad & (tl.r_s1 < 0)].copy()
v = b["last2"].values
print("bad & s1 down, last2 record:", int((v > 0).sum()), "of", len(v), "sign p", round(sign_test(int((v > 0).sum()), len(v)), 4))
print("vs all-months last2 up rate", round((tl.last2 > 0).mean(), 3), "->",
      round(sign_test(int((v > 0).sum()), len(v), float((tl.last2 > 0).mean())), 4))
show(era_split(pd.DatetimeIndex(b["a"]), v), "era, bad & s1 down, last2")
print(cluster_note(pd.DatetimeIndex(b["a"]), v))
bb = tl[tl.bad & (tl.r_s1 <= -0.005)]
print("bad & s1 <= -0.5%:", [(r.month, round(100 * r.mtd_a, 2), round(100 * r.r_s1, 2), round(100 * r.last2, 2), round(100 * r.next2, 2)) for r in bb.itertuples()])
v2 = b["r_s2"].values
print("bad & s1 down, 2nd-last session alone:", int((v2 > 0).sum()), "of", len(v2), f"mean {100 * v2.mean():.3f}%")
# Welch vs bad & s1 up
u = tl[tl.bad & (tl.r_s1 > 0)]["last2"].values
se = np.sqrt(v.var(ddof=1) / len(v) + u.var(ddof=1) / len(u))
print(f"Welch t (s1 down vs s1 up, bad months, last2): {(v.mean() - u.mean()) / se:.2f}")
# the final3 as a whole in bad & s1 down
print("bad & s1 down, final3:", round(100 * b.final3.mean(), 3), "up", int((b.final3 > 0).sum()), "of", len(b))

ie = df[df.tk == "IEF"].copy()
ie_thr = ie["mtd_a"].quantile(0.2)
ib = ie[(ie.mtd_a <= ie_thr) & (ie.r_s1 < 0)]
print(f"\nIEF worst-fifth ({100 * ie_thr:.2f}%) & s1 down: last2 {100 * ib.last2.mean():.3f}% up {int((ib.last2 > 0).sum())} of {len(ib)}")
