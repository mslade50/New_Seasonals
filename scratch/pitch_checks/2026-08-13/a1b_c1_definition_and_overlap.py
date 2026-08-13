"""C1 round 2 - definition neighbours, calendar confound, event-sleeve overlap.

Round 1 (a1_c1_termspread.py) found the carry side positive post-cut
(+1.82% h=5, N=13 episodes) but the placebo anchor ladder put the REAL anchor
at rank 12/21 (h=5) and 14/21 (h=10), dominated by every k=-3..-10 offset.
This script asks the three questions that decide whether anything is left:

  1. Is the finding the DEFINITION? Today's spread is +27.4%, which satisfies
     BOTH "252d-pctile >= 98" and the plain "spread >= 25%". Do they agree?
  2. Is the finding DECEMBER? Post-cut episodes cluster in Nov/Dec, where the
     live event sleeve ALREADY holds long SVXY (V2 NOVDEC_VOL 5% NAV Nov->YE,
     V4 POSTOPEX_VOL 10%). Book overlap is a round-1 kill on its own.
  3. Is the trigger an EVENT or a REGIME MARKER? Compare k=0 against k=-5 as
     WHOLE variants (never a marginal-fill decomposition).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pitch_lab import (  # noqa: E402
    bootstrap_p_le0, cluster_note, declusters, load_prices, show, sign_test,
    summarize, fwd_lag,
)

LEV_CUT = pd.Timestamp("2018-03-01")
px = load_prices(["^VIX", "^VIX3M", "SVXY", "SPY"])
vx, v3 = px["^VIX"]["Close"].dropna(), px["^VIX3M"]["Close"].dropna()
svxy, spy = px["SVXY"]["Close"].dropna(), px["SPY"]["Close"].dropna()
spread = (v3 / vx - 1.0).dropna()
pv = pd.DataFrame({"SVXY": svxy, "SPY": spy}).dropna()


def rollpct(s, lb):
    return s.rolling(lb + 1).apply(lambda w: (w.iloc[-1] > w[:-1]).mean() * 100.0,
                                   raw=False)


defs = {
    "pctile252 >= 98 (the pitch)": rollpct(spread, 252) >= 98,
    "pctile252 >= 95": rollpct(spread, 252) >= 95,
    "pctile126 >= 98": rollpct(spread, 126) >= 98,
    "pctile504 >= 98": rollpct(spread, 504) >= 98,
    "abs spread >= 25% (today +27.4)": spread >= 0.25,
    "abs spread >= 22%": spread >= 0.22,
    "abs spread >= 20%": spread >= 0.20,
    "pctile252>=98 AND abs>=25%": (rollpct(spread, 252) >= 98) & (spread >= 0.25),
}
print("=== TODAY's value under each definition (is today in the population?) ===")
for lbl, m in defs.items():
    print(f"  {lbl:34s} live today: {bool(m.iloc[-1])}   hist days {int(m.sum())}")

print("\n=== 1+2. definition neighbours, POST-CUT only, long SVXY ===")
sub = pv[pv.index >= LEV_CUT]
for h in (5, 10):
    rows = []
    for lbl, m in defs.items():
        r = fwd_lag(sub["SVXY"], h, 1)
        ok = r.notna()
        t = sub.index[m.reindex(sub.index, fill_value=False).values & ok.values]
        if len(t) == 0:
            rows.append({"label": lbl, "n": 0})
            continue
        epi = declusters(t, 21, sub.index[ok.values])
        d = summarize(r.loc[epi].values, lbl)
        d["boot_p_le0"] = round(bootstrap_p_le0(r.loc[epi].values), 3)
        d["n_days"] = len(t)
        rows.append(d)
    base = fwd_lag(sub["SVXY"], h, 1).dropna()
    rows.append(summarize(base.values, "CTRL all post-cut days"))
    show(rows, f"definition neighbours h={h} (episodes, min_gap 21)")

print("\n=== 3. calendar confound: which MONTHS are the episodes? ===")
m98 = defs["pctile252 >= 98 (the pitch)"]
for era_lbl, s in (("FULL 2011+", pv), ("POST-CUT 2018-03+", sub)):
    for h in (5, 10):
        r = fwd_lag(s["SVXY"], h, 1)
        ok = r.notna()
        t = s.index[m98.reindex(s.index, fill_value=False).values & ok.values]
        epi = declusters(t, 21, s.index[ok.values])
        v = r.loc[epi].values
        mon = pd.DatetimeIndex(epi).month
        novdec = np.isin(mon, [11, 12])
        aug = mon == 8
        print(f"\n  {era_lbl} h={h}: N={len(epi)}   Nov/Dec {int(novdec.sum())}, "
              f"August {int(aug.sum())}")
        print(f"    months: {sorted(pd.Series(mon).value_counts().to_dict().items())}")
        rows = [summarize(v, f"all episodes (N={len(v)})"),
                summarize(v[novdec], f"Nov/Dec episodes (N={int(novdec.sum())})"),
                summarize(v[~novdec], f"ex-Nov/Dec (N={int((~novdec).sum())})")]
        show(rows)
        if (~novdec).sum() > 2:
            w = int((v[~novdec] > 0).sum())
            print(f"    ex-Nov/Dec bootstrap P(mean<=0) = "
                  f"{bootstrap_p_le0(v[~novdec]):.3f}  record {w}-{int((~novdec).sum())-w} "
                  f"sign p vs coin {sign_test(w, int((~novdec).sum())):.4f}")
        print(f"    {cluster_note(epi, v)}")

print("\n=== 4. event-sleeve overlap: V2 (Nov 1 -> year end) / V4 (opex +3td) ===")
ev = pd.read_csv(Path(__file__).resolve().parents[3] / "data/macro_events.csv",
                 parse_dates=["date"])
opex = ev[ev["event"] == "opex"]["date"]
idx = sub.index
pos = pd.Series(range(len(idx)), index=idx)
for h in (5, 10):
    r = fwd_lag(sub["SVXY"], h, 1)
    ok = r.notna()
    t = idx[m98.reindex(idx, fill_value=False).values & ok.values]
    epi = declusters(t, 21, idx[ok.values])
    flags = []
    for d in epi:
        in_v2 = d.month in (11, 12)                       # V2 window, non-midterm
        p = pos.get(d)
        nearest_opex = (opex - d).dt.days
        in_v4 = bool(((nearest_opex >= -1) & (nearest_opex <= 5)).any()) \
            and d.month != 9
        flags.append(in_v2 or in_v4)
    flags = np.array(flags)
    v = r.loc[epi].values
    print(f"  h={h}: {int(flags.sum())}/{len(epi)} episodes overlap a LIVE "
          f"event-sleeve short-vol window")
    show([summarize(v[flags], "overlapping sleeve (already owned)"),
          summarize(v[~flags], "NOT owned by the sleeve")])

print("\n=== 5. regime vs event: whole-variant comparison k=0 vs k=-5 vs k=-10 ===")
for h in (5, 10):
    r = fwd_lag(sub["SVXY"], h, 1)
    ok = r.notna()
    base_t = idx[m98.reindex(idx, fill_value=False).values]
    rows = []
    for k in (0, -3, -5, -10):
        sh = []
        for d in base_t:
            p = pos.get(d)
            if p is None:
                continue
            q = p + k
            if 0 <= q < len(idx) and ok.iloc[q]:
                sh.append(idx[q])
        s = pd.DatetimeIndex(sorted(set(sh)))
        epi = declusters(s, 21, idx[ok.values])
        d = summarize(r.loc[epi].values, f"anchor k={k}")
        d["boot_p_le0"] = round(bootstrap_p_le0(r.loc[epi].values), 3)
        rows.append(d)
    show(rows, f"whole-variant anchor comparison h={h} (post-cut)")

print("\n=== 6. what the trigger actually marks: state of the tape at k=0 ===")
r5 = spread.rolling(253).apply(lambda w: (w.iloc[-1] > w[:-1]).mean() * 100.0,
                               raw=False)
t = spread.index[(r5 >= 98).values]
tp = pd.DatetimeIndex(t)
tp = tp[tp >= LEV_CUT]
svr = svxy.pct_change(21)
print(f"  SVXY trailing-21d return on trigger days (post-cut): median "
      f"{100*svr.reindex(tp).median():+.2f}%  vs all post-cut days "
      f"{100*svr[svr.index >= LEV_CUT].median():+.2f}%")
vxl = vx.reindex(tp)
print(f"  VIX level on trigger days: median {vxl.median():.2f} vs all post-cut "
      f"{vx[vx.index >= LEV_CUT].median():.2f}   (today 14.55)")
