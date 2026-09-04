"""Tighten 01b: restrict the control to the two holidays that can actually
collide with payrolls (July 4th and Labor Day), so the payrolls / no-payrolls
split is like-for-like on the same two calendar events.

01b's control set leaked in Juneteenth and the 2004 Reagan funeral closure.
Those are still holiday eves, but they are not the two holidays whose eve can
be a payrolls Friday, and this brief is about tomorrow specifically.

Also: era split, concentration, and the Labor-Day-eve-only cell.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

SUBJ = ["^GSPC", "SPY", "IWM", "^VIX", "CL=F", "TLT"]
px = close_panel(SUBJ)
ref = px["^GSPC"].dropna().index
nfp = pd.DatetimeIndex(sorted(set(load_events(["nfp"])["date"]) & set(ref)))
pos = {d: i for i, d in enumerate(ref)}


def pre_holiday(idx):
    d = idx.to_numpy().astype("datetime64[D]").astype(np.int64)
    normal = np.where(idx.weekday.to_numpy() == 4, 3, 1)
    out = np.zeros(len(idx), dtype=bool)
    out[:-1] = (d[1:] - d[:-1]) > normal[:-1]
    return idx[out]


def next_session(d):
    return ref[pos[d] + 1] if pos[d] + 1 < len(ref) else None


def holiday_after(eve):
    """The calendar date of the closure that follows this eve."""
    nxt = next_session(eve)
    return None if nxt is None else (eve + pd.Timedelta(days=1), nxt)


preh = pre_holiday(ref)
# July 4th eve: the closure gap contains a July 4 (or its observed Friday/Monday).
# Labor Day eve: the closure is the first Monday of September.
jul4, labor = [], []
for eve in preh:
    nxt = next_session(eve)
    if nxt is None:
        continue
    gap = pd.date_range(eve + pd.Timedelta(days=1), nxt - pd.Timedelta(days=1))
    if any(g.month == 7 and g.day in (3, 4, 5) for g in gap):
        jul4.append(eve)
    elif any(g.month == 9 and g.weekday() == 0 and g.day <= 7 for g in gap):
        labor.append(eve)
jul4, labor = pd.DatetimeIndex(jul4), pd.DatetimeIndex(labor)
tight = jul4.union(labor)

print(f"July-4 eves:    {len(jul4)}")
print(f"Labor-Day eves: {len(labor)}")
print(f"combined:       {len(tight)}  (payrolls {len(tight.intersection(nfp))}, "
      f"not {len(tight.difference(nfp))})")
print()


def anchors_for(events):
    return pd.DatetimeIndex([ref[pos[d] - 1] for d in events if pos.get(d, 0) > 0])


def row(label, events, tick, h=1, note=False):
    a = anchors_for(events)
    f = fwd_ret(px[tick].dropna(), h).reindex(a).dropna()
    if len(f) < 3:
        print(f"    {label:30s} {tick:6s} (n<3)")
        return None
    r = summarize(f.to_numpy())
    up = int((f > 0).sum())
    p = sign_test(max(up, len(f) - up), len(f))
    print(f"    {label:30s} {tick:6s} n={r['n']:3d} mean={r['mean_pct']:+7.3f}% "
          f"med={r['median_pct']:+7.3f}% hit={r['hit']:5.1f}% t={r['t']:+6.2f} "
          f"{up}-{len(f)-up} up p={p:.4f}")
    if note:
        print("      ", cluster_note(f.index, f.to_numpy()))
        for e in era_split(f.index, f.to_numpy()):
            print(f"       {e['label']:9s} n={e['n']:3d} "
                  f"mean={e['mean_pct']:+6.3f}% hit={e['hit']:5.1f}%")
    return f


for tick in SUBJ:
    print(f"[{tick}] h1 (the eve's own session move)")
    row("Jul4+Labor eve, NO payrolls", tight.difference(nfp), tick,
        note=(tick in ("^GSPC", "SPY")))
    row("Jul4+Labor eve, payrolls", tight.intersection(nfp), tick,
        note=(tick in ("^GSPC", "SPY")))
    print()

print("[Labor Day eve only]")
for tick in ("^GSPC", "SPY", "IWM", "^VIX"):
    row("Labor eve, NO payrolls", labor.difference(nfp), tick)
    row("Labor eve, payrolls", labor.intersection(nfp), tick)
print()

print("Labor-Day eves, payrolls or not, by year:")
for eve in labor:
    tag = "PAYROLLS" if eve in nfp else "        "
    a = ref[pos[eve] - 1]
    v = fwd_ret(px["^GSPC"].dropna(), 1).get(a, float("nan"))
    print(f"  {eve.date()}  {tag}  ^GSPC {100*v:+6.2f}%")
