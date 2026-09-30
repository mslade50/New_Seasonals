"""One observation per month (the engine pools three overlapping anchors per month, inflating t).
Anchor = session before the final three (today's analogue). Split the 5-session turn into the final three
of the month and the first two of the next. Does TLT hand the month-end bid back in the new month's first
two sessions? Does the equity turn-of-month window hold one-per-month, in both eras, and for Sep->Oct?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

TICK = ["SPY", "QQQ", "IWM", "EEM", "TLT", "IEF", "HYG", "^GSPC"]
px = close_panel(TICK)
idx = px["SPY"].dropna().index
px = px.reindex(idx)
ym = pd.Series(idx.year * 100 + idx.month, index=idx)
pfe = ym.groupby(ym).cumcount(ascending=False)
pfs = ym.groupby(ym).cumcount()
cur = ym.iloc[-1]

rows = []
for m in sorted(ym.unique()):
    if m == cur:
        continue
    sess = idx[(ym == m).values]
    nxt = idx[(ym > m).values]
    if len(sess) < 5 or len(nxt) < 2:
        continue
    anc, last, n2 = sess[-4], sess[-1], nxt[1]
    rec = {"month": m, "mo": m % 100, "anchor": anc}
    for t in TICK:
        s = px[t]
        if pd.isna(s.loc[anc]) or pd.isna(s.loc[n2]):
            continue
        rec[f"{t}_f3"] = s.loc[last] / s.loc[anc] - 1
        rec[f"{t}_n2"] = s.loc[n2] / s.loc[last] - 1
        rec[f"{t}_h5"] = s.loc[n2] / s.loc[anc] - 1
    rows.append(rec)
df = pd.DataFrame(rows).set_index("anchor")
print("months:", len(df), "first", df.index[0].date(), "last", df.index[-1].date())

r5_all = {t: (px[t].shift(-5) / px[t] - 1).dropna() for t in TICK}
r3_all = {t: (px[t].shift(-3) / px[t] - 1).dropna() for t in TICK}
r2_all = {t: (px[t].shift(-2) / px[t] - 1).dropna() for t in TICK}


def line(v, lab):
    s = summarize(v.dropna().values)
    return f"{lab} n {s['n']} {s['mean_pct']:6.3f}/{s['hit']:4.1f}% t {s['t']:5.2f}"


for t in ["SPY", "QQQ", "IWM", "EEM", "TLT", "IEF", "HYG"]:
    c3, c2, c5 = f"{t}_f3", f"{t}_n2", f"{t}_h5"
    if c3 not in df:
        continue
    d = df[[c3, c2, c5, "mo"]].dropna()
    print(f"\n##### {t}  (all-days controls: 3d {100 * r3_all[t].mean():.3f}, 2d {100 * r2_all[t].mean():.3f}, 5d {100 * r5_all[t].mean():.3f})")
    print("  all  :", line(d[c3], "final3"), "|", line(d[c2], "next2"), "|", line(d[c5], "h5"))
    for lab, m in [("pre18", d.index < "2018-01-01"), ("2018+", d.index >= "2018-01-01"), ("Sep  ", d.mo == 9)]:
        e = d[m]
        print(f"  {lab}:", line(e[c3], "final3"), "|", line(e[c2], "next2"), "|", line(e[c5], "h5"))

# TLT: give-back after a big month-end bid, and after a bad month
d = df[["TLT_f3", "TLT_n2"]].dropna()
print("\nTLT next2 when final3 >= +0.5%:", line(d.TLT_n2[d.TLT_f3 >= 0.005], ""), "| when final3 < 0:", line(d.TLT_n2[d.TLT_f3 < 0], ""))
print("corr(final3, next2) TLT:", round(d.corr().iloc[0, 1], 3))
s = px["TLT"]
prev = s.shift(1).where(ym != ym.shift(1)).groupby(ym).transform("first")
mtd = (s / prev - 1).reindex(d.index)
q = mtd.quantile(0.2)
bad = mtd <= q
print(f"TLT worst-quintile MTD (<= {100 * q:.2f}%): final3 {line(d.TLT_f3[bad], '')} | next2 {line(d.TLT_n2[bad], '')}")
print("  era next2 (bad months):", [(e["label"], e["n"], round(e.get("mean_pct", np.nan), 3), round(e.get("hit", np.nan), 1)) for e in era_split(d.index[bad], d.TLT_n2[bad].values)])
print("  era next2 (all months):", [(e["label"], e["n"], round(e.get("mean_pct", np.nan), 3), round(e.get("hit", np.nan), 1)) for e in era_split(d.index, d.TLT_n2.values)])
# 10y in bp across the turn
tnx = close_panel(["^TNX"])["^TNX"].reindex(idx)
bp3, bp2 = [], []
for a in d.index:
    p = idx.get_loc(a)
    bp3.append((tnx.iloc[p + 3] - tnx.iloc[p]) * 100)
    bp2.append((tnx.iloc[p + 5] - tnx.iloc[p + 3]) * 100)
bp3, bp2 = pd.Series(bp3, index=d.index), pd.Series(bp2, index=d.index)
print(f"10y bp: final3 mean {bp3.mean():.2f} median {bp3.median():.2f} down {(bp3 < 0).sum()}/{len(bp3)} | next2 mean {bp2.mean():.2f} median {bp2.median():.2f} up {(bp2 > 0).sum()}/{len(bp2)}")
print(f"   2018+: final3 {bp3[bp3.index >= '2018-01-01'].mean():.2f} down {(bp3[bp3.index >= '2018-01-01'] < 0).sum()}/{(bp3.index >= '2018-01-01').sum()} | next2 {bp2[bp2.index >= '2018-01-01'].mean():.2f} up {(bp2[bp2.index >= '2018-01-01'] > 0).sum()}")
