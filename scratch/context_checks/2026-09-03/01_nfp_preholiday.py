"""Payrolls that are ALSO the last session before a market holiday.

Tomorrow (2026-09-04) is both: September NFP and the Friday before Labor Day.
Each parent cell is famous on its own, so the only interesting question is
whether the intersection is different from either parent.

Convention: anchor is the session BEFORE the event, so h1 IS the event session.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

SUBJ = ["^GSPC", "SPY", "TLT", "^VIX", "CL=F", "GC=F", "^TNX", "IWM"]
px = close_panel(SUBJ)
ref = px["^GSPC"].dropna().index
ev = load_events(["nfp"])
nfp = pd.DatetimeIndex(sorted(set(ev["date"]) & set(ref)))


def pre_holiday(idx):
    """Sessions whose NEXT session is separated by more than a normal gap."""
    d = idx.to_numpy().astype("datetime64[D]").astype(np.int64)
    wd = idx.weekday.to_numpy()
    normal = np.where(wd == 4, 3, 1)
    out = np.zeros(len(idx), dtype=bool)
    out[:-1] = (d[1:] - d[:-1]) > normal[:-1]
    return idx[out]


preh = pre_holiday(ref)
cross = nfp.intersection(preh)

pos = {d: i for i, d in enumerate(ref)}


def anchors_for(events):
    """The session before each event session."""
    keep = [ref[pos[d] - 1] for d in events if pos.get(d, 0) > 0]
    return pd.DatetimeIndex(keep)


def block(name, events, tick, h):
    a = anchors_for(events)
    if len(a) == 0:
        return None
    s = px[tick].dropna()
    f = fwd_ret(s, h).reindex(a).dropna()
    if len(f) < 3:
        return None
    r = summarize(f.to_numpy(), name)
    up = int((f > 0).sum())
    r.update(n_up=up, n_dn=len(f) - up,
             sign_p=sign_test(max(up, len(f) - up), len(f)),
             dates=f.index)
    return r


def show_row(r, tick, h):
    if r is None:
        print(f"  {tick:8s} h{h}  (empty)")
        return
    print(f"  {tick:8s} h{h}  n={r['n']:4d}  mean={r['mean_pct']:+7.3f}%  "
          f"med={r['median_pct']:+7.3f}%  hit={r['hit']:5.1f}%  "
          f"t={r['t']:+6.2f}  {r['n_up']}-{r['n_dn']} up  p={r['sign_p']:.4f}")


print(f"NFP sessions in the panel:            {len(nfp)}")
print(f"Pre-holiday sessions in the panel:    {len(preh)}")
print(f"NFP that is ALSO pre-holiday:         {len(cross)}")
print("dates:", [str(d.date()) for d in cross])
print()

for h in (1, 5):
    for label, events in (("ALL NFP", nfp), ("ALL PRE-HOLIDAY", preh),
                          ("NFP x PRE-HOLIDAY", cross)):
        print(f"[{label}] h{h}")
        for tick in SUBJ:
            show_row(block(label, events, tick, h), tick, h)
        print()

# The cross cell, episode by episode, for ^GSPC.
a = anchors_for(cross)
s = px["^GSPC"].dropna()
f1, f5 = fwd_ret(s, 1).reindex(a), fwd_ret(s, 5).reindex(a)
print("^GSPC cross cell, episode by episode (anchor -> event day):")
for d in a:
    ev_day = ref[pos[d] + 1]
    print(f"  anchor {d.date()}  event {ev_day.date()}  "
          f"h1={100*f1[d]:+6.2f}%  h5={100*f5[d]:+6.2f}%")
v = f1.dropna().to_numpy()
print(" ", cluster_note(f1.dropna().index, v))
print(" ", era_split(f1.dropna().index, v))

# Does the cross beat its own pre-holiday parent, or is it just the parent?
for tick in ("^GSPC", "CL=F", "^VIX"):
    par = block("par", preh, tick, 1)
    cr = block("cr", cross, tick, 1)
    if par and cr:
        print(f"{tick:8s} cross-minus-preholiday h1 mean: "
              f"{cr['mean_pct'] - par['mean_pct']:+.3f} pp "
              f"(cross n={cr['n']}, parent n={par['n']})")
