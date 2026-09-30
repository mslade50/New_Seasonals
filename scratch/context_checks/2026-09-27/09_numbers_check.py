"""Exact figures for every number the brief quotes, recomputed in one place."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
from scipy import stats

px = close_panel(["SPY", "TLT", "IEF", "^TNX", "USO", "UUP", "HYG", "QQQ", "IWM"])
idx = px["SPY"].dropna().index
px = px.reindex(idx)
ym = pd.Series(idx.year * 100 + idx.month, index=idx)
cur = ym.iloc[-1]

# --- TLT state
tlt = px["TLT"]
print("TLT close", round(tlt.iloc[-1], 2), "last close <= today:", str(tlt[tlt <= tlt.iloc[-1] + 1e-9].index[-2].date()))
prev = tlt.shift(1).where(ym != ym.shift(1)).groupby(ym).transform("first")
mtd = tlt / prev - 1
print("TLT MTD", round(100 * mtd.iloc[-1], 2))
tnx = px["^TNX"]
print("10y", round(tnx.iloc[-1], 3), "last close >= today:", str(tnx[tnx >= tnx.iloc[-1] - 1e-9].index[-2].date()),
      "21d bp", round(100 * (tnx.iloc[-1] - tnx.iloc[-22]), 1))

# --- one-per-month TLT blocks
rec = []
for m in sorted(ym.unique()):
    if m == cur:
        continue
    sess = idx[(ym == m).values]
    nxt = idx[(ym > m).values]
    if len(sess) < 5 or len(nxt) < 2 or pd.isna(tlt.loc[sess[-4]]):
        continue
    a, last, n2 = sess[-4], sess[-1], nxt[1]
    rec.append({"a": a, "f3": tlt.loc[last] / tlt.loc[a] - 1, "n2": tlt.loc[n2] / tlt.loc[last] - 1, "mtd": mtd.loc[a],
                "p2": tlt.loc[sess[-3]] / tlt.loc[a] - 1})
d = pd.DataFrame(rec).set_index("a")
q = d.mtd.quantile(0.2)
bad = d[d.mtd <= q]
print(f"\nTLT months {len(d)} from {d.index[0].date()}; all final3 up {(d.f3 > 0).sum()} ({100 * (d.f3 > 0).mean():.1f}%) mean {100 * d.f3.mean():.3f}")
print(f"worst quintile cut {100 * q:.2f}%, n {len(bad)}")
print(f"  final3 up {(bad.f3 > 0).sum()} mean {100 * bad.f3.mean():.3f} median {100 * bad.f3.median():.3f} t {summarize(bad.f3.values)['t']:.2f} sign p {sign_test(int((bad.f3 > 0).sum()), len(bad)):.4f}")
print(f"  next2 up {(bad.n2 > 0).sum()} down {(bad.n2 < 0).sum()} mean {100 * bad.n2.mean():.3f} t {summarize(bad.n2.values)['t']:.2f}"
      f" sign p (down, vs 50%) {sign_test(int((bad.n2 < 0).sum()), len(bad)):.4f}"
      f" sign p (down, vs all-month down-rate {100 * (d.n2 < 0).mean():.1f}%) {sign_test(int((bad.n2 < 0).sum()), len(bad), float((d.n2 < 0).mean())):.4f}")
for lab, m in [("pre-2018", bad.index < "2018-01-01"), ("2018+", bad.index >= "2018-01-01")]:
    b = bad[m]
    print(f"  {lab}: n {len(b)} final3 up {(b.f3 > 0).sum()} mean {100 * b.f3.mean():.3f} | next2 up {(b.n2 > 0).sum()} mean {100 * b.n2.mean():.3f}")
print("  net over the 5 sessions (bad):", round(100 * ((1 + bad.f3) * (1 + bad.n2) - 1).mean(), 3), "| all:", round(100 * ((1 + d.f3) * (1 + d.n2) - 1).mean(), 3))
print("  tomorrow's slot alone (3rd-last session) bad months: mean", round(100 * bad.p2.mean(), 3), "up", (bad.p2 > 0).sum(), "of", len(bad),
      "| all months:", round(100 * d.p2.mean(), 3), "up", (d.p2 > 0).sum(), "of", len(d))
print("  cluster final3:", cluster_note(bad.index, bad.f3.values))
print("  cluster next2:", cluster_note(bad.index, bad.n2.values))
print("  bad-month Septembers:", [(str(x.date()), round(100 * bad.mtd[x], 1), round(100 * bad.f3[x], 2), round(100 * bad.n2[x], 2)) for x in bad.index if x.month == 9])

# --- SPY turn window, one per month
spy = px["SPY"]
rec = []
for m in sorted(ym.unique()):
    if m == cur:
        continue
    sess = idx[(ym == m).values]
    nxt = idx[(ym > m).values]
    if len(sess) < 5 or len(nxt) < 2:
        continue
    a, last, n2 = sess[-4], sess[-1], nxt[1]
    rec.append({"a": a, "h5": spy.loc[n2] / spy.loc[a] - 1, "f3": spy.loc[last] / spy.loc[a] - 1, "mo": m % 100})
s = pd.DataFrame(rec).set_index("a")
sep, oth = s[s.mo == 9], s[s.mo != 9]
w = stats.ttest_ind(sep.h5, oth.h5, equal_var=False)
print(f"\nSPY turn h5 all: n {len(s)} up {(s.h5 > 0).sum()} ({100 * (s.h5 > 0).mean():.1f}%) mean {100 * s.h5.mean():.3f} t {summarize(s.h5.values)['t']:.2f}")
e18 = s[s.index >= "2018-01-01"]
print(f"   2018+: n {len(e18)} up {(e18.h5 > 0).sum()} mean {100 * e18.h5.mean():.3f}")
print(f"   Sep: n {len(sep)} up {(sep.h5 > 0).sum()} mean {100 * sep.h5.mean():.3f} median {100 * sep.h5.median():.3f} | other n {len(oth)} up {(oth.h5 > 0).sum()} mean {100 * oth.h5.mean():.3f} | Welch t {w.statistic:.2f}")
print(f"   Sep final3: up {(sep.f3 > 0).sum()} mean {100 * sep.f3.mean():.3f}")
by_mo = s.groupby("mo").h5.agg(["count", "mean", lambda x: (x > 0).sum()])
by_mo["mean"] *= 100
print(by_mo.round(3).to_string())
print("   Sep era:", [(e["label"], e["n"], round(e.get("mean_pct", np.nan), 3), round(e.get("hit", np.nan), 1)) for e in era_split(sep.index, sep.h5.values)])
print("   Sep cluster:", cluster_note(sep.index, sep.h5.values))
print("   Sep midterm years:", {x.year: round(100 * sep.h5[x], 2) for x in sep.index if x.year in (2002, 2006, 2010, 2014, 2018, 2022)})
r5 = (spy.shift(-5) / spy - 1).dropna()
print(f"   control: every 5-session window {100 * r5.mean():.3f} up {100 * (r5 > 0).mean():.1f}%")

# --- footnote resolution
r = px.pct_change()
print("\nFri 9/25: IEF", round(100 * r["IEF"].iloc[-1], 2), "USO", round(100 * r["USO"].iloc[-1], 2), "HYG", round(100 * r["HYG"].iloc[-1], 2), "UUP", round(100 * r["UUP"].iloc[-1], 2))
fr = r["USO"][(r.index.month == 9) & (r.index.year == 2026) & (r.index.weekday == 4)]
print("USO Sept 2026 Fridays:", fr.mul(100).round(2).to_dict(), "combined", round(100 * ((1 + fr).prod() - 1), 2))
print("10y bp Fri:", round(100 * tnx.diff().iloc[-1], 1), "| IEF since Thu 9/24 close:", round(100 * r["IEF"].iloc[-1], 2))
a = pd.Timestamp("2026-09-22")
print("since 9/22 close: SPY", round(100 * (spy.iloc[-1] / spy.loc[a] - 1), 2), "TLT", round(100 * (tlt.iloc[-1] / tlt.loc[a] - 1), 2))
