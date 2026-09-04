"""Two bond cells fired for tomorrow and they are different horizons.

  a) August Fridays, TLT 64-44 up (sign p 0.0335), IEF 65-43 (p 0.0214)  -> h1
  b) the Aug-14 trading-day anchor, TLT and IEF both 17-6 up over h5      -> h5

Same confound as the VIX drill: Fridays generally. Plus a state question that
matters more than either, since TLT closed 0.82% off a 52-week low and IEF
1.23% off its own: does the calendar cell survive when duration enters beaten
down, or is it a bull-market artefact?
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, era_split, fwd_ret,  # noqa: E402
                       sign_test, summarize, cluster_note)

px = close_panel(["TLT", "IEF", "^TNX"])


def rep(name, s, mask, h=1, quiet=False):
    r = fwd_ret(s, h)
    m = mask & r.notna().to_numpy()
    v = r.to_numpy()[m]
    if len(v) == 0:
        return None
    d = summarize(v, name)
    up = int((v > 0).sum())
    p = sign_test(up, len(v))
    if not quiet:
        print(f"{name:<46} n={len(v):>5}  mean={d['mean_pct']:+7.3f}%  "
              f"med={d['median_pct']:+7.3f}%  up={up}-{len(v) - up} "
              f"({100 * up / len(v):4.1f}%)  t={d['t']:+5.2f}  signp={p:.4f}")
    return {"v": v, "dates": s.index[m], "up": up, "n": len(v), "d": d, "p": p}


for tk in ("TLT", "IEF"):
    s = px[tk].dropna()
    s = s[s.index >= "1999-01-01"]
    idx = s.index
    dow, month = idx.dayofweek, idx.month
    aug_fri = (month == 8) & (dow == 3)
    not_aug_fri = (month != 8) & (dow == 3)
    all_fri = (dow == 3)

    print(f"\n================ {tk}  (h=1, the Friday's own move) ================")
    a = rep(f"{tk} August Fridays (the cell)", s, aug_fri)
    rep(f"{tk} Fridays outside August", s, not_aug_fri)
    rep(f"{tk} ALL Fridays", s, all_fri)
    rep(f"{tk} August, not Friday", s, (month == 8) & (dow != 3))
    rep(f"{tk} every session", s, np.ones(len(idx), dtype=bool))

    nf = rep(f"{tk} (again) Fridays outside August", s, not_aug_fri, quiet=True)
    p1, n1 = a["up"] / a["n"], a["n"]
    p2, n2 = nf["up"] / nf["n"], nf["n"]
    pp = (a["up"] + nf["up"]) / (n1 + n2)
    z = (p1 - p2) / np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    print(f"   two-proportion z, August Fridays vs other Fridays: {z:+.2f}")
    for e in era_split(a["dates"], a["v"]):
        print(f"   era {e['label']}: n={e['n']} mean={e['mean_pct']:+.3f}% "
              f"hit={e['hit']:.1f}% t={e['t']:+.2f}")
    print("   ", cluster_note(a["dates"], a["v"]))

    # --- the state question: duration entering near a 52-week low
    low52 = s.rolling(252).min()
    near_low = ((s / low52 - 1.0) <= 0.03).to_numpy()   # within 3% of a 52w low
    print(f"   -- conditioned on entering within 3% of a 52-week low "
          f"(tonight {tk} is {100 * (s.iloc[-1] / low52.iloc[-1] - 1):.2f}% above) --")
    rep(f"{tk} August Fridays, near a 52w low", s, aug_fri & near_low)
    rep(f"{tk} other Fridays, near a 52w low", s, not_aug_fri & near_low)
    rep(f"{tk} August Fridays, NOT near a 52w low", s, aug_fri & ~near_low)

    # --- the h5 seasonal leg, one anchor per year around Aug 14
    print(f"   -- the Aug-14 anchor over the following week (h=5) --")
    doy = (month == 8) & (np.abs(idx.day - 14) <= 2)
    seen, keep = set(), []
    for d in idx[doy]:
        if d.year not in seen:
            seen.add(d.year)
            keep.append(d)
    km = idx.isin(pd.DatetimeIndex(keep))
    k = rep(f"{tk} Aug-14 anchor, h5", s, km, h=5)
    if k:
        for e in era_split(k["dates"], k["v"]):
            print(f"      era {e['label']}: n={e['n']} mean={e['mean_pct']:+.3f}% "
                  f"hit={e['hit']:.1f}%")
        print("      ", cluster_note(k["dates"], k["v"]))
        yrs = pd.DatetimeIndex(k["dates"]).year
        print("      per-year:", {int(y): round(100 * float(v), 2)
                                  for y, v in zip(yrs, k["v"])})
    # rest-of-August control for the same horizon
    rep(f"{tk} any August session, h5", s, (month == 8), h=5)
    rep(f"{tk} every session, h5", s, np.ones(len(idx), dtype=bool), h=5)
