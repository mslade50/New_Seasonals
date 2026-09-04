"""Control for 01: is the NFP-kills-pre-holiday-drift result actually about NFP?

Every one of the 25 NFP-x-pre-holiday dates is a July 4th eve or a Labor Day
eve, because those are the only market holidays that can land on a payrolls
Friday. So the honest control is not "all pre-holidays" (which is mostly
Thanksgiving and Christmas) but "summer-holiday eves that are NOT payrolls".

If summer eves drift the same as every other eve, the NFP overlap is the
variable. If summer eves are already flat, NFP explains nothing.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

SUBJ = ["^GSPC", "SPY", "CL=F", "^VIX", "IWM"]
px = close_panel(SUBJ)
ref = px["^GSPC"].dropna().index
nfp = pd.DatetimeIndex(sorted(set(load_events(["nfp"])["date"]) & set(ref)))


def pre_holiday(idx):
    d = idx.to_numpy().astype("datetime64[D]").astype(np.int64)
    normal = np.where(idx.weekday.to_numpy() == 4, 3, 1)
    out = np.zeros(len(idx), dtype=bool)
    out[:-1] = (d[1:] - d[:-1]) > normal[:-1]
    return idx[out]


preh = pre_holiday(ref)
pos = {d: i for i, d in enumerate(ref)}
summer = pd.DatetimeIndex([d for d in preh if d.month in (6, 7, 8, 9)])
winter = pd.DatetimeIndex([d for d in preh if d.month not in (6, 7, 8, 9)])
sum_nfp = summer.intersection(nfp)
sum_non = summer.difference(nfp)

print("pre-holiday sessions by group")
print(f"  summer eves (Jun-Sep, = Jul4 / Labor Day): {len(summer)}")
print(f"    of which payrolls:                       {len(sum_nfp)}")
print(f"    of which NOT payrolls:                   {len(sum_non)}")
print(f"  all other eves:                            {len(winter)}")
print("  non-NFP summer eve dates:",
      [str(d.date()) for d in sum_non])
print()


def anchors_for(events):
    return pd.DatetimeIndex([ref[pos[d] - 1] for d in events if pos.get(d, 0) > 0])


def row(label, events, tick, h=1):
    a = anchors_for(events)
    f = fwd_ret(px[tick].dropna(), h).reindex(a).dropna()
    if len(f) < 3:
        print(f"    {label:26s} {tick:6s} (n<3)")
        return None
    r = summarize(f.to_numpy())
    up = int((f > 0).sum())
    p = sign_test(max(up, len(f) - up), len(f))
    print(f"    {label:26s} {tick:6s} n={r['n']:4d} mean={r['mean_pct']:+7.3f}% "
          f"med={r['median_pct']:+7.3f}% hit={r['hit']:5.1f}% t={r['t']:+6.2f} "
          f"{up}-{len(f)-up} up p={p:.4f}")
    return f


for tick in SUBJ:
    print(f"[{tick}] h1 from the eve's own anchor")
    row("all eves", preh, tick)
    row("winter eves", winter, tick)
    row("summer eves (all)", summer, tick)
    row("summer eves, NOT payrolls", sum_non, tick)
    row("summer eves, payrolls", sum_nfp, tick)
    print()

# Oil is the sharpest swing in 01. Where does its summer-eve mean live?
f = row("summer eves NOT payrolls", sum_non, "CL=F")
print(" ", cluster_note(f.index, f.to_numpy()))
f = row("summer eves payrolls", sum_nfp, "CL=F")
print(" ", cluster_note(f.index, f.to_numpy()))
print("  oil payroll-eve episodes:")
for d, v in f.items():
    print(f"    {d.date()} -> {ref[pos[d]+1].date()}  {100*v:+6.2f}%")
