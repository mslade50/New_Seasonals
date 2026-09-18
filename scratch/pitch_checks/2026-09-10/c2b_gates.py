"""c2b — C2 gate attribution + the 12-month ladder.

c2_cost already returned the verdict (September roll drag -35.6 bp/session,
the worst month of the twelve, against a UNG cell that pays +24 bp gross at
h=5). This adds the two things that make the kill non-refutable rather than
cost-only:

  1. GATE ATTRIBUTION with the DISCARDED COMPLEMENT: September alone, the
     price condition alone, and the days each gate throws away.
  2. THE 12-MONTH LADDER with September's RANK, and the max-of-12 permutation
     scored AGAINST SEPTEMBER'S OWN STATISTIC (repo rule 1 — this product was
     corrected on 2026-09-08 for testing the wrong month's threshold).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = close_panel(["UNG", "NG=F"])
H = 5
NEAR = 0.06

for VEH in ("UNG", "NG=F"):
    s = px[VEH].dropna()
    idx = s.index
    lo = rolling_on_valid(s, lambda x: x.rolling(252).min())
    near = ((s / lo - 1.0) <= NEAR)
    ret = fwd_lag(s, H, 1)
    valid = ret.dropna().index

    def cellof(mask, label):
        t = pd.DatetimeIndex(idx[mask.fillna(False).values]).intersection(valid)
        if len(t) == 0:
            return {"label": label, "n": 0}
        e = declusters(t, max(H, 5), valid)
        v = ret.loc[e].values
        r = summarize(v, label)
        r["n_days"] = len(t)
        r["signp"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
        return r

    sep = pd.Series(idx.month == 9, index=idx)
    print("\n" + "=" * 80)
    print(f"1. GATE ATTRIBUTION — {VEH}, LONG, h={H}, entry lag=1")
    rows = [
        cellof(sep & near, "JOIN: September AND within 6% of 252d low (PITCHED)"),
        cellof(sep, "PARENT A: September alone (all price states)"),
        cellof(near, "PARENT B: within 6% of 252d low alone (all months)"),
        cellof(sep & ~near, "DISCARDED by price gate: Sept NOT near low"),
        cellof(~sep & near, "DISCARDED by month gate: near low, NOT Sept"),
        cellof(pd.Series(True, index=idx), "all days"),
    ]
    show(rows, f"   {VEH} gate attribution")
    j = rows[0]
    a, b = rows[1], rows[2]
    if j["n"]:
        print(f"   -> the JOIN adds {j['mean_pct']-a['mean_pct']:+.3f}pp over "
              f"September alone and {j['mean_pct']-b['mean_pct']:+.3f}pp over "
              f"the price state alone.")

    # ------------------------------------------------ 12-month ladder
    print(f"\n2. 12-MONTH LADDER — identical price construction, {VEH}, h={H}")
    rows, means = [], {}
    for m in range(1, 13):
        r = cellof(pd.Series(idx.month == m, index=idx) & near,
                   f"month {m:02d}" + ("  <-- SEPT" if m == 9 else ""))
        rows.append(r)
        means[m] = r.get("mean_pct", np.nan)
    show(rows, f"   {VEH}: 'within 6% of 252d low' in each calendar month")
    ok = {m: v for m, v in means.items() if not (v is None or np.isnan(v))}
    order = sorted(ok, key=lambda m: -ok[m])
    rank = order.index(9) + 1 if 9 in order else None
    print(f"   ** September RANK = {rank}/{len(ok)} "
          f"(mean {ok.get(9, float('nan')):+.3f}%; best month {order[0]} at "
          f"{ok[order[0]]:+.3f}%) **")

    # permutation charged against SEPTEMBER'S OWN statistic (rule 1)
    if 9 in ok:
        rng = np.random.default_rng(7)
        obs = ok[9]
        t_sep = pd.DatetimeIndex(idx[(sep & near).fillna(False).values]).intersection(valid)
        n_ep = len(declusters(t_sep, max(H, 5), valid))
        rv = ret.loc[valid].values
        null_sep, null_max = [], []
        for _ in range(4000):
            draws = rng.choice(rv, size=(12, max(n_ep, 1)), replace=True)
            mm = 100 * draws.mean(axis=1)
            null_sep.append(mm[8])       # the September slot, uncharged
            null_max.append(mm.max())    # max-of-12, charged
        null_sep, null_max = np.array(null_sep), np.array(null_max)
        print(f"   DEFENDED STATISTIC: September's own episode mean "
              f"({obs:+.3f}% on N={n_ep} episodes)")
        print(f"   UNCHARGED p (vs one random month-sized draw) = "
              f"{(null_sep >= obs).mean():.4f}")
        print(f"   CHARGED  p (vs max-of-12 null, scored against SEPTEMBER, "
              f"not against the best month) = {(null_max >= obs).mean():.4f}")
