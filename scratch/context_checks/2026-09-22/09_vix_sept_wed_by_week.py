"""Drill 08 showed the September-Wednesday VIX drop survives removing FOMC days
and the post-Labor-Day Wednesday. Where in September does it sit, and what does
tomorrow's slot (the Wednesday after the quad-witching Friday) look like against
the same slot after the Mar/Jun/Dec expiries? Also the VIX-crush base rate."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

cp = close_panel(["^VIX", "^GSPC"])
vix = cp["^VIX"].dropna()
spx = cp["^GSPC"].reindex(vix.index)
idx = vix.index
v1 = vix.pct_change()
s1 = spx.pct_change()


def third_friday(y: int, m: int) -> pd.Timestamp:
    d = pd.Timestamp(y, m, 15)
    return d + pd.Timedelta(days=(4 - d.weekday()) % 7)


ev = load_events(["fomc_decision"])
fomc = set(pd.to_datetime(ev["date"]).dt.normalize())

# Sep Wednesdays by position relative to the quad-witching Friday
rows = []
for y in sorted(set(idx.year)):
    qf = third_friday(y, 9)
    for d in idx[(idx.year == y) & (idx.month == 9) & (idx.weekday == 2)]:
        rows.append(dict(date=d, rel_week=(d - qf).days // 7 + (1 if d > qf else 0),
                         fomc=d in fomc, v=v1[d], s=s1[d]))
df = pd.DataFrame(rows).dropna()
# rel_week: 0 = Wednesday before/of the expiry week, 1 = Wednesday after expiry, etc.
out = []
for k, g in df.groupby("rel_week"):
    s = summarize(g.v.values, f"rel_week {k}")
    s["n_fomc"] = int(g.fomc.sum())
    s["sign_p_dn"] = sign_test(int((g.v < 0).sum()), len(g))
    out.append(s)
show(out, "September Wednesdays by week relative to quad-witching Friday (VIX same-day)")

# tomorrow's slot: the Wednesday after the quad Friday (Fri + 5 calendar days), all 4 expiry months
out = []
slot = {}
for m in (9, 3, 6, 12):
    vals, spxv, dts = [], [], []
    for y in sorted(set(idx.year)):
        qf = third_friday(y, m)
        d = qf + pd.Timedelta(days=5)
        if d in idx and d < idx[-1]:
            vals.append(v1[d]); spxv.append(s1[d]); dts.append(d)
    vals, spxv = np.array(vals), np.array(spxv)
    s = summarize(vals, f"VIX, Wed after {m} expiry")
    s["sign_p_dn"] = sign_test(int((vals < 0).sum()), len(vals))
    s["spx_mean"] = 100 * spxv.mean()
    s["spx_hit"] = 100 * (spxv > 0).mean()
    out.append(s)
    slot[m] = (pd.DatetimeIndex(dts), vals)
show(out, "the Wednesday after quad witching, VIX same-session change")
d9, v9 = slot[9]
show(era_split(d9, v9), "Sep slot era")
print("  ", cluster_note(d9, v9))
print("   per-year:", {d.year: round(100 * v, 1) for d, v in zip(d9, v9)})
print("   fomc in Sep slot:", [d.date() for d in d9 if d in fomc])

# base rate for drill 06/08: VIX higher 5 sessions later, all days with VIX < 16
v5 = vix.shift(-5) / vix - 1
lo = vix < 16
print(f"\nBASE: VIX < 16 days, VIX higher 5 sessions later {100*(v5[lo].dropna()>0).mean():.1f}% "
      f"mean {100*v5[lo].mean():+.2f}% (n={int(lo.sum())})")
lo_dn = lo & (v1 <= -0.04)
print(f"      VIX < 16 AND VIX -4%+ (any S&P): up {100*(v5[lo_dn].dropna()>0).mean():.1f}% mean {100*v5[lo_dn].mean():+.2f}% n={int(lo_dn.sum())}")
f5 = fwd_ret(spx, 5)
print(f"      VIX < 16 all days: S&P h5 mean {100*f5[lo].mean():+.3f}% up {100*(f5[lo].dropna()>0).mean():.1f}%")
lo_dn_up = lo_dn & (s1 > 0)
print(f"      VIX < 16, VIX -4%+, S&P UP: S&P h5 mean {100*f5[lo_dn_up].mean():+.3f}% n={int(lo_dn_up.sum())}; "
      f"VIX h5 up {100*(v5[lo_dn_up].dropna()>0).mean():.1f}%")
