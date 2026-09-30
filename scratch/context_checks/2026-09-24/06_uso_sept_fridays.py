"""USO on September Fridays (the engine's CL=F cell, -0.39% t -2.00, rerun on USO in 02:
34-53, -0.41%, t -2.21, but 15-37 before 2018 and 19-18 since). Is it September, or a
fall-Friday thing, and does it survive by era in the other months that look like it?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

px = close_panel(["USO", "^GSPC"])
idx = px["^GSPC"].dropna().index
uso = px["USO"].reindex(idx)
r = uso.pct_change().dropna()
fri = r[r.index.weekday == 4]

rows = []
for m in range(1, 13):
    v = fri[fri.index.month == m]
    pre, post = v[v.index < "2018"], v[v.index >= "2018"]
    rows.append(dict(month=m, n=len(v), mean=100 * v.mean(), up=int((v > 0).sum()), down=int((v < 0).sum()),
                     pre_mean=100 * pre.mean(), pre_rec=f"{int((pre > 0).sum())}-{int((pre < 0).sum())}",
                     post_mean=100 * post.mean(), post_rec=f"{int((post > 0).sum())}-{int((post < 0).sum())}"))
print(pd.DataFrame(rows).round(3).to_string(index=False))

sep = fri[fri.index.month == 9]
print("\nSep Fridays by year (sum, record):")
print(sep.groupby(sep.index.year).agg(lambda s: f"{100 * s.sum():+.2f} ({int((s > 0).sum())}-{int((s < 0).sum())})").to_string())
# all non-Friday September days, and September Fridays vs the Thursday before
sep_all = r[r.index.month == 9]
print("\nSep non-Fridays:", len(sep_all[sep_all.index.weekday != 4]), round(100 * sep_all[sep_all.index.weekday != 4].mean(), 3))
print("Sep Fridays after a USO up-2% Thursday:", end=" ")
prev = r.shift(1)
m = sep[(prev.reindex(sep.index) >= 0.02).values]
print(len(m), round(100 * m.mean(), 3) if len(m) else None, [(str(d.date()), round(100 * x, 2)) for d, x in m.items()])
# welch t: Sep Fridays vs all other Fridays
o = fri[fri.index.month != 9]
t = (sep.mean() - o.mean()) / np.sqrt(sep.var() / len(sep) + o.var() / len(o))
print("Welch t Sep Fridays vs other Fridays:", round(t, 2), "| other Fridays mean", round(100 * o.mean(), 3), "n", len(o))
