"""Does the TLT give-back in the new month's first two sessions hold across MTD thresholds, or only at
the worst-quintile cut drill 08 happened to use? Plus IEF since Wednesday for Thursday's footnote."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

px = close_panel(["TLT", "IEF", "SPY"])
idx = px["SPY"].dropna().index
px = px.reindex(idx)
tlt = px["TLT"]
ym = pd.Series(idx.year * 100 + idx.month, index=idx)
prev = tlt.shift(1).where(ym != ym.shift(1)).groupby(ym).transform("first")
mtd = tlt / prev - 1
rec = []
for m in sorted(ym.unique())[:-1]:
    sess = idx[(ym == m).values]
    nxt = idx[(ym > m).values]
    if len(sess) < 5 or len(nxt) < 2 or pd.isna(tlt.loc[sess[-4]]):
        continue
    a, last, n2 = sess[-4], sess[-1], nxt[1]
    rec.append({"a": a, "f3": tlt.loc[last] / tlt.loc[a] - 1, "n2": tlt.loc[n2] / tlt.loc[last] - 1, "mtd": mtd.loc[a]})
d = pd.DataFrame(rec).set_index("a")
for thr in (-0.01, -0.02, -0.025, -0.03, -0.035, -0.04):
    b = d[d.mtd <= thr]
    print(f"MTD <= {100 * thr:5.1f}%: n {len(b):3d} final3 {100 * b.f3.mean():6.3f} up {(b.f3 > 0).sum():3d} ({100 * (b.f3 > 0).mean():4.1f}%)"
          f" | next2 {100 * b.n2.mean():6.3f} up {(b.n2 > 0).sum():3d} ({100 * (b.n2 > 0).mean():4.1f}%)")
b = d[d.mtd > -0.01]
print(f"MTD >  -1.0%: n {len(b)} final3 {100 * b.f3.mean():.3f} ({100 * (b.f3 > 0).mean():.1f}%) | next2 {100 * b.n2.mean():.3f} ({100 * (b.n2 > 0).mean():.1f}%)")
ief = px["IEF"]
print("IEF since Wed 9/23 close:", round(100 * (ief.iloc[-1] / ief.loc["2026-09-23"] - 1), 2))
