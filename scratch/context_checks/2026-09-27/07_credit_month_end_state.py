"""HYG month-end is bh_pass (395-299, h5 +0.24 t 4.81). HYG enters the final three sessions -1.91% MTD
with a bottom-5% week (cap-dropped P5); LQD sits 0.06% above its 52-week low. Does the credit month-end
bid survive a bad month the way TLT's does (drill 02), and where does it sit?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
from scipy import stats

px = close_panel(["HYG", "LQD", "TLT", "IEF", "SPY"])
idx = px["SPY"].dropna().index
px = px.reindex(idx)
ym = pd.Series(idx.year * 100 + idx.month, index=idx)
pfe = ym.groupby(ym).cumcount(ascending=False)
pfe[ym == ym.iloc[-1]] = 99
anchors = idx[(pfe == 3).values]
anchors = anchors[anchors < idx[-1]]

for t in ["HYG", "LQD", "IEF", "TLT"]:
    s = px[t]
    prev = s.shift(1).where(ym != ym.shift(1)).groupby(ym).transform("first")
    mtd = s / prev - 1
    blk = (s.shift(-3) / s - 1).reindex(anchors)
    st = pd.DataFrame({"blk": blk, "mtd": mtd.reindex(anchors), "m": anchors.month}).dropna()
    today_mtd = mtd.iloc[-1]
    # rank today's MTD against the anchor-date MTD distribution
    pct = (st.mtd <= today_mtd).mean() * 100
    print(f"\n##### {t}: MTD at today's anchor {100 * today_mtd:.2f}% (percentile among month anchors {pct:.1f}); all blocks n {len(st)} mean {100 * st.blk.mean():.3f} hit {100 * (st.blk > 0).mean():.1f}")
    q = st.mtd.quantile([0.2]).iloc[0]
    for lab, m in [("worst-quintile MTD", st.mtd <= q), ("MTD <= today's", st.mtd <= today_mtd), ("rest (above worst quintile)", st.mtd > q)]:
        v = st.blk[m]
        s_ = summarize(v.values)
        print(f"  {lab:28s} (cut {100 * (q if 'quintile' in lab else today_mtd):.2f}%): n {s_['n']} mean {s_['mean_pct']:.3f} median {s_['median_pct']:.3f} hit {s_['hit']:.1f} t {s_['t']:.2f}")
        if lab == "worst-quintile MTD":
            print("     era:", [(e["label"], e["n"], round(e.get("mean_pct", np.nan), 3), round(e.get("hit", np.nan), 1)) for e in era_split(v.index, v.values)])
            print("     cluster:", cluster_note(v.index, v.values))
            rest = st.blk[~m]
            w = stats.ttest_ind(v, rest, equal_var=False)
            print(f"     Welch vs rest t {w.statistic:.2f}; sign p vs rest hit {sign_test(int((v > 0).sum()), len(v), float((rest > 0).mean())):.4f}")
