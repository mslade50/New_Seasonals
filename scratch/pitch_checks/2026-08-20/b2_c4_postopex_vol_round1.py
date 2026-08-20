"""C4 round 1: the August inversion of the post-opex short-vol cell.

Two objects, deliberately kept apart:
  (A) THE BOOK'S V4_POSTOPEX_VOL, exactly as event_sleeve.py specifies it:
      long SVXY, entry MOC on the OPEX close, exit MOC +3 sessions, every opex
      except September, stood down while V2 holds (Nov/Dec non-midterm).
      Question owed to McKinley: should August be excluded the way Sep is?
  (B) THE PITCH CANDIDATE: short vol entered MOC TODAY, i.e. at the opex-1
      close, held h sessions.

SVXY changed leverage -1x -> -0.5x on 2018-02-28 (registry 2026-08-14). Every
SVXY number below is reported pre-break / post-break separately. Spot ^VIX has
no such break and is the clean read on whether August genuinely inverts.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
warnings.filterwarnings("ignore")
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

BREAK = pd.Timestamp("2018-02-28")
TK = ["SVXY", "^VIX", "SPY", "^VIX3M"]
px = close_panel(TK)
d = px.index

ev = load_events(["opex"])
opex = pd.DatetimeIndex(sorted(set(ev["date"]) & set(d)))


def anchor_at(off: int, months=None, opex_dates=None) -> pd.DatetimeIndex:
    src = opex if opex_dates is None else opex_dates
    if months:
        src = pd.DatetimeIndex([x for x in src if x.month in months])
    pos = d.get_indexer(src) + off
    pos = pos[(pos >= 0) & (pos < len(d))]
    return d[pos]


svxy_first = px["SVXY"].dropna().index[0]
print(f"SVXY first bar {svxy_first.date()}; leverage break {BREAK.date()}")
print(f"opex events {len(opex)}  {opex.min().date()}..{opex.max().date()}")

# ---------- daily-sd sanity that the break is real, not asserted ------------
r = px["SVXY"].pct_change()
pre, post = r[r.index < BREAK].dropna(), r[r.index >= BREAK].dropna()
vr = px["^VIX"].pct_change()
for lbl, s in (("pre-break -1x", pre), ("post-break -0.5x", post)):
    j = s.index.intersection(vr.dropna().index)
    beta = np.polyfit(vr.loc[j].values, s.loc[j].values, 1)[0]
    print(f"  {lbl:18s} n={len(s):5d} daily sd {100*s.std():.2f}%  "
          f"worst day {100*s.min():.2f}%  VIX beta {beta:+.3f}")

# ===========================================================================
print("\n\n" + "=" * 78)
print("(A) THE BOOK'S V4: long SVXY, MOC on the OPEX close, exit MOC +3")
print("=" * 78)
H_V4 = 3
opex_all = pd.DatetimeIndex([x for x in opex])
# V2 holds Nov+Dec in NON-midterm years -> V4 stands down there
v4_ok = pd.DatetimeIndex([
    x for x in opex_all
    if x.month != 9 and not (x.month in (11, 12) and x.year % 4 != 2)])
print(f"V4-eligible opex dates: {len(v4_ok)} of {len(opex_all)} "
      f"(ex Sep, ex Nov/Dec non-midterm)")

sv3 = fwd_lag(px["SVXY"], H_V4, lag=0)
for era_lbl, sel in (("PRE-break (-1x)", lambda i: i < BREAK),
                     ("POST-break (-0.5x)", lambda i: i >= BREAK)):
    rows = []
    for lbl, dates in (("V4 all eligible", v4_ok),
                       ("V4 ex-August", pd.DatetimeIndex(
                           [x for x in v4_ok if x.month != 8])),
                       ("V4 AUGUST only", pd.DatetimeIndex(
                           [x for x in v4_ok if x.month == 8])),
                       ("September (excluded by spec)", pd.DatetimeIndex(
                           [x for x in opex_all if x.month == 9]))):
        dd = pd.DatetimeIndex([x for x in dates if sel(x)])
        vals = sv3.reindex(dd).dropna()
        rows.append(summarize(vals.values, f"{lbl} (N={len(vals)})"))
    show(rows, f"V4 h=3, {era_lbl}")

print("\n--- V4 August, post-break, YEAR BY YEAR (this is the whole sample) ---")
aug_v4 = pd.DatetimeIndex([x for x in v4_ok if x.month == 8 and x >= BREAK])
vals = sv3.reindex(aug_v4).dropna()
for dt, v in vals.items():
    print(f"   {dt.date()}  {100*v:+7.2f}%")
print(f"   mean {100*vals.mean():+.3f}%  median {100*np.median(vals):+.3f}%  "
      f"hit {100*(vals>0).mean():.0f}%  N={len(vals)}")
if len(vals) >= 3:
    print(f"   {cluster_note(vals.index, vals.values, k=1)}")

# same, pre-break, so the pooling error is visible
aug_v4p = pd.DatetimeIndex([x for x in v4_ok if x.month == 8 and x < BREAK])
valsp = sv3.reindex(aug_v4p).dropna()
print("\n--- V4 August, PRE-break -1x (the instrument that no longer exists) ---")
for dt, v in valsp.items():
    print(f"   {dt.date()}  {100*v:+7.2f}%")
print(f"   mean {100*valsp.mean():+.3f}%  N={len(valsp)}")

# ===========================================================================
print("\n\n" + "=" * 78)
print("(B) THE PITCH CANDIDATE: entry at the opex-1 close (TODAY), SVXY")
print("=" * 78)
for h in (2, 3, 5, 10):
    a = anchor_at(-1, months=[8])
    s = fwd_lag(px["SVXY"], h, lag=0).reindex(a).dropna()
    pre_ = s[s.index < BREAK]
    post_ = s[s.index >= BREAK]
    allm = fwd_lag(px["SVXY"], h, lag=0).reindex(anchor_at(-1)).dropna()
    allm_post = allm[allm.index >= BREAK]
    base_post = fwd_lag(px["SVXY"], h, lag=0).dropna()
    base_post = base_post[base_post.index >= BREAK]
    print(f"\nh={h}:  AUG all {100*s.mean():+.3f}% N={len(s)}  |  "
          f"AUG pre-break {100*pre_.mean():+.3f}% N={len(pre_)}  |  "
          f"AUG POST-break {100*post_.mean():+.3f}% N={len(post_)} "
          f"(hit {100*(post_>0).mean():.0f}%)")
    print(f"        pooled-all-months POST-break {100*allm_post.mean():+.3f}% "
          f"N={len(allm_post)}  |  SVXY all-days POST-break "
          f"{100*base_post.mean():+.3f}%  ->  August excess vs all-days = "
          f"{100*(post_.mean()-base_post.mean()):+.3f}pp")
    if len(post_):
        print("        post-break August values: " +
              ", ".join(f"{y}:{100*v:+.1f}%" for y, v in
                        zip(post_.index.year, post_.values)))

# ===========================================================================
print("\n\n" + "=" * 78)
print("(C) SPOT ^VIX — no instrument break, the clean read on August")
print("=" * 78)
for h in (1, 2, 3, 5, 10):
    a_aug = anchor_at(-1, months=[8])
    a_all = anchor_at(-1)
    v = fwd_lag(px["^VIX"], h, lag=0)
    base = v.dropna()
    print(f"h={h:2d}  AUG {100*v.reindex(a_aug).dropna().mean():+7.3f}% "
          f"(N={len(v.reindex(a_aug).dropna())}, up-rate "
          f"{100*(v.reindex(a_aug).dropna()>0).mean():.0f}%)   "
          f"pooled {100*v.reindex(a_all).dropna().mean():+7.3f}% "
          f"(N={len(v.reindex(a_all).dropna())})   all-days "
          f"{100*base.mean():+7.3f}%   AUG excess "
          f"{100*(v.reindex(a_aug).dropna().mean()-base.mean()):+.3f}pp")

print("\n--- ^VIX August cell by year, h=3 (entry opex-1 close) ---")
v3 = fwd_lag(px["^VIX"], 3, lag=0).reindex(anchor_at(-1, months=[8])).dropna()
print(pd.Series((100 * v3.values).round(2),
                index=v3.index.year).sort_values(ascending=False).to_string())
print(f"  mean {100*v3.mean():+.3f}%  median {100*np.median(v3):+.3f}%  "
      f"up-rate {100*(v3>0).mean():.0f}%")
print(f"  {cluster_note(v3.index, v3.values, k=2)}")
vv = np.sort(v3.values)[::-1]
print(f"  drop-1 {100*vv[1:].mean():+.3f}%  drop-2 {100*vv[2:].mean():+.3f}%  "
      f"drop-3 {100*vv[3:].mean():+.3f}%")

# ---------------------------------------------------- offset placebo ladder
print("\n\n" + "=" * 78)
print("OFFSET PLACEBO LADDER — AUGUST opex only, k=-10..+5, N held at 26")
print("=" * 78)
aug_opex = pd.DatetimeIndex([x for x in opex if x.month == 8])
print(f"August opex dates in the index: {len(aug_opex)}")
for tkr, hs in (("^VIX", (2, 3, 5)), ("SVXY", (3, 5))):
    for h in hs:
        rows = []
        for k in range(-10, 6):
            a = anchor_at(k, opex_dates=aug_opex)
            s = fwd_lag(px[tkr], h, lag=0).reindex(a).dropna()
            if tkr == "SVXY":
                s = s[s.index >= BREAK]
            if len(s) < 4:
                continue
            rows.append({"k": k, "n": len(s), "mean_pct": 100 * s.mean(),
                         "hit": 100 * (s > 0).mean(),
                         "true": "<== TRUE" if k == -1 else ""})
        df = pd.DataFrame(rows).sort_values("mean_pct", ascending=False)
        df["rank"] = range(1, len(df) + 1)
        print(f"\n--- {tkr} h={h} "
              f"{'(POST-BREAK ONLY)' if tkr == 'SVXY' else ''} ---")
        print(df.round(3).to_string(index=False))
        tr = df[df["k"] == -1]
        if len(tr):
            print(f"  >>> TRUE ANCHOR (k=-1) RANKS {int(tr['rank'].iloc[0])} "
                  f"of {len(df)}")

# ------------------------------------------------------- month-of-year ctrl
print("\n\n" + "=" * 78)
print("MONTH-OF-YEAR CONTROL at the opex-1 anchor")
print("=" * 78)
for tkr, h in (("^VIX", 3), ("SVXY", 3)):
    rows = []
    for mo in range(1, 13):
        a = anchor_at(-1, months=[mo])
        s = fwd_lag(px[tkr], h, lag=0).reindex(a).dropna()
        if tkr == "SVXY":
            s = s[s.index >= BREAK]
        rows.append({"month": mo, "n": len(s),
                     "mean_pct": 100 * s.mean() if len(s) else np.nan,
                     "hit": 100 * (s > 0).mean() if len(s) else np.nan})
    df = pd.DataFrame(rows).sort_values("mean_pct", ascending=False)
    df["rank"] = range(1, len(df) + 1)
    print(f"\n--- {tkr} h={h} by month "
          f"{'(post-break)' if tkr == 'SVXY' else ''} ---")
    print(df.round(3).to_string(index=False))
    a8 = df[df.month == 8]
    print(f"  August ranks {int(a8['rank'].iloc[0])} of {len(df)}")

# ------------------------------------------------------------ midterm split
print("\n\n" + "=" * 78)
print("MIDTERM SPLIT")
print("=" * 78)
for tkr, h in (("^VIX", 3), ("^VIX", 5), ("SVXY", 3)):
    a = anchor_at(-1, months=[8])
    s = fwd_lag(px[tkr], h, lag=0).reindex(a).dropna()
    if tkr == "SVXY":
        s = s[s.index >= BREAK]
    mid = s.index.year % 4 == 2
    print(f"{tkr} h={h}: midterm {100*s[mid].mean():+.3f}% "
          f"N={int(mid.sum())} {sorted(set(s.index[mid].year))}  |  "
          f"non-midterm {100*s[~mid].mean():+.3f}% N={int((~mid).sum())}")
