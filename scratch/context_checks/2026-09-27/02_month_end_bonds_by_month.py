"""The engine's month-end bond bid (TLT 496-374, t 4.23, bh_pass) carries a September month_cell of
35 of 75 up. Is September really the hole, is it a quarter-end thing, and does TLT entering the final
three sessions at a 52-week low / after a bad month change anything?

Unit: the final 3 sessions of each month. Per-session rows (engine-style) and one row per month
(the 3-session sum from the anchor close, i.e. h3 from the Friday-before analogue = declustered)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

TICK = ["TLT", "IEF", "LQD", "HYG", "SPY", "^TNX", "NG=F", "UNG"]
px = close_panel(TICK)
idx = px["SPY"].dropna().index
px = px.reindex(idx)
ret = px.pct_change()
tnx_bp = px["^TNX"].diff() * 100

ym = pd.Series(idx.year * 100 + idx.month, index=idx)
pos_from_end = ym.groupby(ym).cumcount(ascending=False)  # 0 = last session of month
# September 2026 is incomplete: its last three sessions have not happened. Drop the month outright
# (the engine's month_window_anchors does not, so its cells carry Sep 23-25 as "month end").
pos_from_end[ym == 202609] = 99
last3 = pos_from_end <= 2
month_of = pd.Series(idx.month, index=idx)
# the anchor = session before the final three; h3 from it = the 3-session block
anchor = pos_from_end == 3
anchors = idx[anchor.values]
anchors = anchors[anchors < idx[-1]]


def block(t):
    s = px[t]
    return (s.shift(-3) / s - 1).reindex(anchors)


def tbl(t):
    rows = []
    r = ret[t][last3]
    for m in range(1, 13):
        v = r[month_of[last3] == m].dropna()
        b = block(t)[month_of.reindex(anchors).values == m].dropna()
        rows.append({"m": m, "n_sess": len(v), "sess_mean": 100 * v.mean(), "sess_hit": 100 * (v > 0).mean(),
                     "n_blk": len(b), "blk_mean": 100 * b.mean(), "blk_up": int((b > 0).sum())})
    df = pd.DataFrame(rows).round(3)
    print(f"\n=== {t}: final-3-session returns by calendar month ===")
    print(df.to_string(index=False))
    allv = r.dropna()
    ctl = ret[t].dropna()
    print(f"  all last-3 sessions: n {len(allv)} mean {100 * allv.mean():.3f} hit {100 * (allv > 0).mean():.1f};"
          f" all days mean {100 * ctl.mean():.3f} hit {100 * (ctl > 0).mean():.1f}")
    return df


for t in ["TLT", "IEF", "HYG", "LQD", "SPY"]:
    tbl(t)

# September vs the other eleven months, TLT, per session and per block
from scipy import stats  # noqa: E402

for t in ["TLT", "IEF", "HYG"]:
    r = ret[t][last3].dropna()
    sep = r[month_of.reindex(r.index) == 9]
    oth = r[month_of.reindex(r.index) != 9]
    w = stats.ttest_ind(sep, oth, equal_var=False)
    print(f"\n{t} Sep last-3 sessions: n {len(sep)} {100 * sep.mean():.3f}% hit {100 * (sep > 0).mean():.1f}"
          f" | other months n {len(oth)} {100 * oth.mean():.3f}% hit {100 * (oth > 0).mean():.1f} | Welch t {w.statistic:.2f} p {w.pvalue:.3f}")
    b = block(t).dropna()
    bs = b[month_of.reindex(b.index).values == 9]
    bo = b[month_of.reindex(b.index).values != 9]
    w2 = stats.ttest_ind(bs, bo, equal_var=False)
    print(f"   block: Sep n {len(bs)} {100 * bs.mean():.3f}% up {(bs > 0).sum()} | other n {len(bo)} {100 * bo.mean():.3f}% up {(bo > 0).sum()}"
          f" ({100 * (bo > 0).mean():.1f}%) | Welch t {w2.statistic:.2f}; sign p (Sep down vs other-month up-rate) "
          f"{sign_test(int((bs < 0).sum()), len(bs), 1 - (bo > 0).mean()):.4f}")
    print("   Sep blocks by year:", {d.year: round(100 * v, 2) for d, v in bs.items()})
    print("   era:", [(e["label"], e["n"], round(e.get("mean_pct", np.nan), 3), round(e.get("hit", np.nan), 1)) for e in era_split(bs.index, bs.values)])
    print("   cluster:", cluster_note(bs.index, bs.values))

# quarter-end months vs the rest, TLT blocks
b = block("TLT").dropna()
mo = month_of.reindex(b.index).values
for lab, m in [("Mar/Jun/Dec", np.isin(mo, [3, 6, 12])), ("Sep", mo == 9), ("non-quarter", ~np.isin(mo, [3, 6, 9, 12]))]:
    v = b[m]
    print(f"TLT block {lab}: n {len(v)} mean {100 * v.mean():.3f} up {(v > 0).sum()}")

# position within the final three: Monday is the 3rd-last
for t in ["TLT", "IEF", "HYG", "SPY"]:
    out = []
    for p in (2, 1, 0):
        v = ret[t][(pos_from_end == p)].dropna()
        vs = v[month_of.reindex(v.index) == 9]
        out.append(f"pos-from-end {p}: all {100 * v.mean():.3f}/{100 * (v > 0).mean():.1f}% (n {len(v)}) Sep {100 * vs.mean():.3f}/{100 * (vs > 0).mean():.1f}% (n {len(vs)})")
    print(t, " | ".join(out))

# conditioning on the state entering the block: TLT MTD return through the anchor, and 52w low
s = px["TLT"]
mstart = s.groupby(ym).transform(lambda x: x.iloc[0])
prev_close_month = s.shift(1).where(ym != ym.shift(1)).groupby(ym).transform("first")
mtd = s / prev_close_month - 1
low52 = s <= s.rolling(252, min_periods=200).min() + 1e-12
st = pd.DataFrame({"blk": block("TLT"), "mtd": mtd.reindex(anchors), "low": low52.reindex(anchors),
                   "m": month_of.reindex(anchors).values}).dropna()
print(f"\nTLT MTD at today's anchor: {100 * mtd.iloc[-1]:.2f}%, at 52w low: {bool(low52.iloc[-1])}")
for lab, m in [("MTD <= -2%", st.mtd <= -0.02), ("MTD <= -3%", st.mtd <= -0.03), ("MTD > -2%", st.mtd > -0.02),
               ("at 52w low", st.low.astype(bool)), ("MTD<=-2% & at 52w low", (st.mtd <= -0.02) & st.low.astype(bool))]:
    v = st.blk[m]
    print(f"  TLT block {lab}: n {len(v)} mean {100 * v.mean():.3f} up {(v > 0).sum()} ({100 * (v > 0).mean():.1f}%)"
          f" dates {[str(d.date()) for d in v.index[-8:]]}")
    vs = st.blk[m & (st.m == 9)]
    if len(vs):
        print(f"     of which Sep: {[(str(d.date()), round(100 * x, 2)) for d, x in vs.items()]}")

# 10y in bp over the September block
tb = (px["^TNX"].shift(-3) - px["^TNX"]).reindex(anchors) * 100
tbs = tb[month_of.reindex(anchors).values == 9].dropna()
tbo = tb[month_of.reindex(anchors).values != 9].dropna()
print(f"\n10y bp over final 3: Sep n {len(tbs)} mean {tbs.mean():.2f} median {tbs.median():.2f} up {(tbs > 0).sum()} | other mean {tbo.mean():.2f} up {(tbo > 0).sum()} of {len(tbo)}")

# seam check: NG=F month-end vs UNG month-end
for t in ["NG=F", "UNG"]:
    v = ret[t][last3].dropna()
    c = ret[t].dropna()
    print(f"{t} last-3 sessions n {len(v)} mean {100 * v.mean():.3f} hit {100 * (v > 0).mean():.1f} vs all days {100 * c.mean():.3f}")

# ---- the live cell: TLT entering the final three sessions after a bad month
print("\n##### TLT final-3 block by MTD state at the anchor (current month excluded)")
all_blk = st.blk
loc_ctl = block("TLT")
for thr in (-0.02, -0.03, -0.04):
    m = st.mtd <= thr
    v = all_blk[m]
    row = summarize(v.values, f"MTD<={thr}")
    rest = all_blk[~m]
    w = stats.ttest_ind(v, rest, equal_var=False)
    print(f"  MTD <= {100 * thr:.0f}%: n {row['n']} mean {row['mean_pct']:.3f} median {row['median_pct']:.3f} hit {row['hit']:.1f} t {row['t']:.2f}"
          f" | rest n {len(rest)} mean {100 * rest.mean():.3f} hit {100 * (rest > 0).mean():.1f} | Welch t {w.statistic:.2f}"
          f" | sign p vs rest hit {sign_test(int((v > 0).sum()), len(v), float((rest > 0).mean())):.4f}")
    print("     era:", [(e["label"], e["n"], round(e.get("mean_pct", np.nan), 3), round(e.get("hit", np.nan), 1)) for e in era_split(v.index, v.values)])
    print("     cluster:", cluster_note(v.index, v.values))
    qe = np.isin(st.m[m], [3, 6, 9, 12])
    print(f"     quarter-end months: n {qe.sum()} mean {100 * v[qe].mean():.3f} up {(v[qe] > 0).sum()} | other months n {(~qe).sum()} mean {100 * v[~qe].mean():.3f} up {(v[~qe] > 0).sum()}")
# every-session control over the same anchors' neighbourhood: TLT 3-session return on all days
r3 = px["TLT"].shift(-3) / px["TLT"] - 1
print(f"  control: TLT any-3-session return all days mean {100 * r3.mean():.3f} hit {100 * (r3 > 0).mean():.1f}")
# same state (MTD <= -3%) at a random non-month-end point: is it just mean reversion after a bad month?
mtd_all = mtd.copy()
mid = (pos_from_end >= 6) & (pos_from_end < 99)
v = r3[mid & (mtd_all <= -0.03)].dropna()
print(f"  mean-reversion control: MTD<=-3% on sessions 6+ from month end, 3-session fwd: n {len(v)} (overlapping) mean {100 * v.mean():.3f} hit {100 * (v > 0).mean():.1f}")
v2 = r3[mid & (mtd_all > -0.03)].dropna()
print(f"                          MTD>-3%  same sessions: n {len(v2)} mean {100 * v2.mean():.3f} hit {100 * (v2 > 0).mean():.1f}")
# the 3rd-last session alone (Monday's slot) under the state
p2 = ret["TLT"][pos_from_end == 2]
mtd_prev = mtd.shift(1).reindex(p2.index)
for lab, m in [("MTD<=-3% at anchor", mtd_prev <= -0.03), ("else", mtd_prev > -0.03)]:
    v = p2[m].dropna()
    print(f"  TLT 3rd-last session {lab}: n {len(v)} mean {100 * v.mean():.3f} hit {100 * (v > 0).mean():.1f}")
# IEF and 10y bp in the same state
blk_ief = block("IEF").reindex(st.index)
tb3 = tb.reindex(st.index)
m = st.mtd <= -0.03
print(f"  IEF block when TLT MTD<=-3%: n {m.sum()} mean {100 * blk_ief[m].mean():.3f} up {(blk_ief[m] > 0).sum()} | else mean {100 * blk_ief[~m].mean():.3f} hit {100 * (blk_ief[~m] > 0).mean():.1f}")
print(f"  10y bp when TLT MTD<=-3%: mean {tb3[m].mean():.2f} median {tb3[m].median():.2f} down {(tb3[m] < 0).sum()} of {m.sum()} | else mean {tb3[~m].mean():.2f}")
print("  episodes MTD<=-3%:", [(str(d.date()), round(100 * st.mtd[d], 1), round(100 * all_blk[d], 2)) for d in all_blk[m].index])
