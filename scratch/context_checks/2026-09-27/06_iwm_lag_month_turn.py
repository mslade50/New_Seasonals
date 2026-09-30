"""IWM enters September's final three sessions 4.6pp behind SPY on the month (-3.82% vs +0.81%).
Does a lagging small-cap month catch up over the month turn (final 3 + first 2 of the next month),
and is that a month-end thing or plain mean reversion of a relative lag?
Engine base: E:month_end IWM h1 +0.071 t 1.55 era-unstable, h5 +0.31."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

px = close_panel(["SPY", "IWM"])
idx = px["SPY"].dropna().index
px = px.reindex(idx)
ym = pd.Series(idx.year * 100 + idx.month, index=idx)
pfe = ym.groupby(ym).cumcount(ascending=False)
pfe[ym == ym.iloc[-1]] = 99  # current month incomplete
prev_m_close = {}
for t in ["SPY", "IWM"]:
    s = px[t]
    first = s.shift(1).where(ym != ym.shift(1)).groupby(ym).transform("first")
    prev_m_close[t] = first
mtd = {t: px[t] / prev_m_close[t] - 1 for t in ["SPY", "IWM"]}
spread = mtd["IWM"] - mtd["SPY"]
print("today IWM MTD", round(100 * mtd["IWM"].iloc[-1], 2), "SPY MTD", round(100 * mtd["SPY"].iloc[-1], 2), "spread", round(100 * spread.iloc[-1], 2))


def rel(h):
    return (px["IWM"].shift(-h) / px["IWM"] - 1) - (px["SPY"].shift(-h) / px["SPY"] - 1)


anchors = idx[(pfe == 3).values]
st = pd.DataFrame({"spr": spread.reindex(anchors), "r1": rel(1).reindex(anchors), "r3": rel(3).reindex(anchors),
                   "r5": rel(5).reindex(anchors), "iwm5": fwd_ret(px["IWM"], 5).reindex(anchors)}).dropna()
print("month anchors:", len(st))
for lab, m in [("all months", st.spr > -9), ("spread <= -3pp", st.spr <= -0.03), ("spread <= -4pp", st.spr <= -0.04),
               ("spread > -3pp", st.spr > -0.03), ("spread >= +3pp", st.spr >= 0.03)]:
    d = st[m]
    out = []
    for c in ["r1", "r3", "r5", "iwm5"]:
        s = summarize(d[c].values, c)
        out.append(f"{c} {s['mean_pct']:6.2f}/{s['hit']:4.1f}% t {s['t']:5.2f}")
    print(f"  {lab:16s} n {len(d):3d} | " + " | ".join(out))

d = st[st.spr <= -0.04]
print("\nspread <= -4pp episodes:", [(str(x.date()), round(100 * d.spr[x], 1), round(100 * d.r5[x], 2)) for x in d.index])
print("era r5:", [(e["label"], e["n"], round(e.get("mean_pct", np.nan), 2), round(e.get("hit", np.nan), 1)) for e in era_split(d.index, d.r5.values)])
print("cluster r5:", cluster_note(d.index, d.r5.values))
print("sign p r5 up:", round(sign_test(int((d.r5 > 0).sum()), len(d)), 4),
      "vs all-month hit:", round(sign_test(int((d.r5 > 0).sum()), len(d), float((st.r5 > 0).mean())), 4))

# mean-reversion control: same lag, mid-month (8+ sessions from month end), 5-session relative forward
mid = (pfe >= 8) & (pfe < 99)
r5 = rel(5)
for thr in (-0.03, -0.04):
    v = r5[mid & (spread <= thr)].dropna()
    ep = declusters(v.index, 5, idx)
    vv = r5.reindex(ep).dropna()
    print(f"mid-month control spread <= {100 * thr:.0f}pp: declustered n {len(vv)} r5 mean {100 * vv.mean():.2f} hit {100 * (vv > 0).mean():.1f}")
