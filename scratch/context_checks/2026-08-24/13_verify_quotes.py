"""Every number that goes in tonight's brief, recomputed in one place."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import (load_prices, summarize, sign_test, fwd_ret, declusters,
                       local_control)  # noqa

ASOF = pd.Timestamp("2026-08-24")
px = load_prices(["ZC=F", "ZW=F", "GC=F", "SI=F", "QQQ", "^NYA", "SPY", "BTC-USD", "^GSPC"])

print("1. GRAIN GAPS")
for t in ("ZC=F", "ZW=F"):
    df = px[t]
    gap = (df["Open"] / df["Close"].shift(1) - 1).replace([np.inf, -np.inf], np.nan).dropna()
    aug = gap[gap.index.month == 8]
    print(f"   {t}: today gap {100*gap.loc[ASOF]:+.2f}%, session "
          f"{100*(df['Close'].loc[ASOF]/df['Close'].shift(1).loc[ASOF]-1):+.2f}%, "
          f"intraday {100*(df['Close'].loc[ASOF]/df['Open'].loc[ASOF]-1):+.2f}%; "
          f"August sessions {len(aug)}, of which gap>+3%: {int((aug>0.03).sum())}")
    allb = gap[gap > 0.03]
    print(f"      all >+3% gaps by month: {dict(pd.Series(allb.index.month).value_counts().sort_index())}")

print()
print("2. GOLD 21d >= 15%")
g = px["GC=F"]["Close"]
r21 = g.pct_change(21)
idx = g.index
dts = idx[(r21 >= 0.15).reindex(idx).fillna(False)]
dc = declusters(pd.DatetimeIndex([d for d in dts if d <= ASOF]), 21, idx)
print(f"   today 21d {100*r21.loc[ASOF]:.2f}%; raw sessions {len(dts)} of {r21.notna().sum()}; "
      f"episodes {len(dc)}: {[str(d.date()) for d in dc]}")
f21 = fwd_ret(g, 21).reindex(dc).dropna()
dn = int((f21.values < 0).sum())
st = summarize(f21.values, "")
ctrl = summarize(fwd_ret(g, 21).reindex(local_control(idx, dc, 126)).dropna().values, "")
alls = summarize(fwd_ret(g, 21).dropna().values, "")
print(f"   h21: n={st['n']} down {dn}/{st['n']} mean {st['mean_pct']:.2f}% t {st['t']:.2f} "
      f"sign p(down) {sign_test(dn, st['n']):.5f} worst {st['worst_pct']:.1f}% best {st['best_pct']:.1f}%")
print(f"   controls: local +/-126td {ctrl['mean_pct']:.2f}% (n={ctrl['n']}), all gold days {alls['mean_pct']:.2f}%")
f1 = fwd_ret(g, 1).reindex(dc).dropna()
print(f"   h1: n={len(f1)} {int((f1.values>0).sum())}-{int((f1.values<=0).sum())} up "
      f"mean {100*f1.mean():.2f}%")
s = px["SI=F"]["Close"]
print(f"   silver contrast: 21d>=15% sessions {int((s.pct_change(21)>=0.15).sum())} "
      f"of {s.pct_change(21).notna().sum()}; today {100*s.pct_change(21).loc[ASOF]:.2f}%")

print()
print("3. NARROW WEAKNESS")
pan = pd.concat({"QQQ": px["QQQ"]["Close"], "NYA": px["^NYA"]["Close"],
                 "SPY": px["SPY"]["Close"]}, axis=1).dropna().loc[:ASOF]
rq = pan["QQQ"].pct_change(5).rolling(252).apply(lambda w: (w[:-1] < w[-1]).mean()*100, raw=True)
nh = pan["NYA"] >= pan["NYA"].rolling(252).max()*0.995
dc3 = declusters(pan.index[((rq <= 10) & nh).fillna(False)], 10, pan.index)
print(f"   episodes {len(dc3)} since {pan.index[0].year}; by year "
      f"{dict(pd.Series(dc3.year).value_counts().sort_index())}")
print(f"   today: QQQ 5d {100*pan['QQQ'].pct_change(5).iloc[-1]:.2f}% (rank {rq.iloc[-1]:.1f}), "
      f"SPY 5d {100*pan['SPY'].pct_change(5).iloc[-1]:.2f}%, "
      f"NYA {100*(pan['NYA'].iloc[-1]/pan['NYA'].rolling(252).max().iloc[-1]-1):.2f}% from its 252d high")
for h in (10, 21):
    v = fwd_ret(pan["SPY"], h).reindex(dc3).dropna()
    st = summarize(v.values, "")
    a = summarize(fwd_ret(pan["SPY"], h).dropna().values, "")
    pre = v[v.index < "2018-01-01"]; post = v[v.index >= "2018-01-01"]
    print(f"   SPY h{h}: n={st['n']} mean {st['mean_pct']:.2f}% "
          f"{int((v.values>0).sum())}-{int((v.values<=0).sum())} up t {st['t']:.2f} | "
          f"all days {a['mean_pct']:.2f}% | pre-2018 {100*pre.mean():.2f}% (n={len(pre)}) "
          f"2018+ {100*post.mean():.2f}% (n={len(post)})")
sp = fwd_ret(pan["QQQ"], 21) - fwd_ret(pan["NYA"], 21)
v = sp.reindex(dc3).dropna()
print(f"   QQQ-minus-NYA h21: n={len(v)} mean {100*v.mean():.2f}pp vs all days "
      f"{100*sp.dropna().mean():.2f}pp")

print()
print("4. BITCOIN")
c = px["BTC-USD"]["Close"].loc[:ASOF]
z = c.pct_change(10) / (c.pct_change().rolling(21).std()*np.sqrt(10))
raw = c.index[(z >= 2).fillna(False)]
dcb = declusters(raw, 10, c.index)
print(f"   live z10 {z.iloc[-1]:.2f}; raw bars {len(raw)}, episodes {len(dcb)}")
for h in (1, 10):
    v = fwd_ret(c, h).reindex(raw).dropna(); vd = fwd_ret(c, h).reindex(dcb).dropna()
    sr, sd = summarize(v.values, ""), summarize(vd.values, "")
    ad = summarize(fwd_ret(c, h).dropna().values, "")
    pre = vd[vd.index < "2018-01-01"]; post = vd[vd.index >= "2018-01-01"]
    print(f"   h{h}: overlapping n={sr['n']} t {sr['t']:.2f} | declustered n={sd['n']} "
          f"mean {sd['mean_pct']:.2f}% {int((vd.values>0).sum())}-{int((vd.values<=0).sum())} up "
          f"sign p {sign_test(int((vd.values>0).sum()), len(vd)):.4f} t {sd['t']:.2f} | "
          f"all bars {ad['mean_pct']:.2f}% | pre-2018 {100*pre.mean():.2f}% 2018+ {100*post.mean():.2f}%")

print()
print("5. AUGUST LAST-5, S&P")
cg = px["^GSPC"]["Close"].dropna()
hi = cg.rolling(252).max()
last = cg.index[-1]
rows = []
for (y, m), grp in cg.groupby([cg.index.year, cg.index.month]):
    if m != 8 or len(grp) < 7 or grp.index[-1] == last:
        continue
    i = len(grp) - 6
    rows.append((y, grp.index[i], grp.iloc[-1]/grp.iloc[i] - 1, cg.loc[grp.index[i]]/hi.loc[grp.index[i]] - 1))
near = [r for r in rows if r[3] >= -0.03]; far = [r for r in rows if r[3] < -0.03]
for lab, sel in (("all", rows), ("entered within 3% of 52w high", near), ("entered >3% below", far)):
    v = np.array([r[2] for r in sel]); st = summarize(v, "")
    print(f"   {lab:<32} n={st['n']} mean {st['mean_pct']:.2f}% "
          f"{int((v>0).sum())}-{int((v<=0).sum())} up sign p {sign_test(int((v>0).sum()), len(v)):.4f} "
          f"worst {st['worst_pct']:.2f}%")
ctl = []
for m in range(1, 13):
    if m == 8: continue
    for (y, mm), grp in cg.groupby([cg.index.year, cg.index.month]):
        if mm != m or len(grp) < 7 or grp.index[-1] == last: continue
        i = len(grp) - 6
        ctl.append(grp.iloc[-1]/grp.iloc[i] - 1)
v = np.array(ctl); st = summarize(v, "")
print(f"   {'non-August last-5 control':<32} n={st['n']} mean {st['mean_pct']:.2f}% "
      f"{int((v>0).sum())}-{int((v<=0).sum())} up worst {st['worst_pct']:.2f}%")
print(f"   2026 enters at {100*(cg.iloc[-1]/hi.iloc[-1]-1):.2f}% from its 252d high")

print()
print("5b. AUGUST LAST-5 partition audit")
audit = [(r[0], round(100*r[3], 2) if r[3] == r[3] else None, round(100*r[2], 2)) for r in rows]
print("   (year, entry dist from 252d high %, window return %):")
print("  ", audit)
print(f"   near {len([r for r in rows if r[3] >= -0.03])}, "
      f"far {len([r for r in rows if r[3] < -0.03])}, "
      f"unclassified {len([r for r in rows if r[3] != r[3]])}, total {len(rows)}")
