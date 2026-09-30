"""Sensitivity of drill 06's cell B to the VIX-level cut and the VIX-drop size,
plus the September-Wednesday VIX cell with FOMC-decision Wednesdays and the
Wednesday after Labor Day taken out (tomorrow is neither)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

cp = close_panel(["^VIX", "^GSPC"])
vix, spx = cp["^VIX"].dropna(), cp["^GSPC"].dropna()
idx = vix.index.intersection(spx.index)
vix, spx = vix.reindex(idx), spx.reindex(idx)
vr, sr = vix.pct_change(), spx.pct_change()
today = idx[-1]
f5 = fwd_ret(spx, 5)
v5 = vix.shift(-5) / vix - 1

rows = []
for lvl in (14, 15, 16, 17, 18, 20):
    for drop in (-0.03, -0.04, -0.05):
        m = (vr <= drop) & (sr <= 0) & (vix < lvl)
        trig = declusters(m[m].index, 5, idx)
        trig = trig[trig < today]
        v = f5.reindex(trig).dropna()
        s = summarize(v.values, f"VIX<{lvl} drop<={100*drop:.0f}%")
        s["ctrl_local"] = 100 * f5.reindex(local_control(idx, trig)).mean()
        s["vix5_up"] = 100 * (v5.reindex(trig).dropna() > 0).mean()
        s["sign_p_dn"] = sign_test(int((v < 0).sum()), len(v))
        rows.append(s)
show(rows, "S&P h5 after VIX drop on a flat-or-red S&P close, by level and drop size")

# per-episode listing for the headline cut
m = (vr <= -0.04) & (sr <= 0) & (vix < 16)
trig = declusters(m[m].index, 5, idx)
trig = trig[trig < today]
print("\nB episodes (date, VIX close, SPX h5 %, VIX h5 %):")
for d in trig:
    print(f"  {d.date()}  {vix[d]:.2f}  {100*f5[d]:+.2f}  {100*v5[d]:+.1f}")
print("B per-year counts:", pd.Series(trig.year).value_counts().sort_index().to_dict())

# September Wednesday VIX decomposition
ev = load_events(["fomc_decision"])
fomc = set(pd.to_datetime(ev["date"]).dt.normalize())
v1 = vix.pct_change()
wed = pd.Series(idx.weekday == 2, index=idx)
sep = pd.Series(idx.month == 9, index=idx)
is_fomc = pd.Series([d in fomc for d in idx], index=idx)
# Wednesday after Labor Day = first Wednesday of September that follows the first Monday
first_mon = {}
for y in sorted(set(idx.year)):
    d = pd.Timestamp(y, 9, 1)
    first_mon[y] = d + pd.Timedelta(days=(0 - d.weekday()) % 7)
post_ld = pd.Series([d.month == 9 and d.weekday() == 2 and 0 < (d - first_mon[d.year]).days <= 2
                     for d in idx], index=idx)
cells = {
    "Sep Wednesdays, all": wed & sep,
    "Sep Wed, FOMC decision days": wed & sep & is_fomc,
    "Sep Wed after Labor Day": wed & sep & post_ld,
    "Sep Wed, ex FOMC and ex post-Labor-Day": wed & sep & ~is_fomc & ~post_ld,
    "other-month Wednesdays ex FOMC": wed & ~sep & ~is_fomc,
}
out = []
for lab, mk in cells.items():
    v = v1[mk].dropna()
    s = summarize(v.values, lab)
    s["sign_p_dn"] = sign_test(int((v < 0).sum()), len(v))
    out.append(s)
show(out, "VIX same-session change on Wednesdays (the engine's k1 cell)")
