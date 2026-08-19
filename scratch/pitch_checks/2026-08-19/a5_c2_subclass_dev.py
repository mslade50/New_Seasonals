"""C2 (long XLK / short XLV snap-back), decisive round-2/3 on TODAY'S subclass.

a4 established that in today's actual class -- calm tape (SPY Wilder-14 ATR
< 1.2% of price), index within 3% of its 52w high, XLV-XLK gap >= 3.0pp --
C2 is positive at every horizon (+0.46 / +0.71 / +0.89 / +0.75 / +0.88 pp,
h=1/2/3/5/10, N 17-22).  Never significant, always the same sign.  So it gets
the three tests that decide whether that is an edge or a costume:

  1. per-leg attribution INSIDE the subclass  (is the XLV short a drag?)
  2. the ignorant rule: long XLK alone after a big XLK DOWN day in the same
     calm/near-high tape, with NO rotation condition.  If that reproduces
     C2's number, the healthcare leg is decoration.
  3. beta-neutral subclass + year concentration + a midterm split.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

pd.set_option("display.width", 250)

TK = ["XLV", "XLK", "SPY", "QQQ"]
raw = load_prices(TK)
px = close_panel(TK)
r1 = px.pct_change()
vk = (r1["XLV"] - r1["XLK"]).reindex(px.index)
today = px.index[-1]

hi252 = px["SPY"].rolling(252).max()
dist = px["SPY"] / hi252 - 1.0
spy_atrp = (pd.Series(wilder_atr(raw["SPY"]["High"], raw["SPY"]["Low"],
                                 raw["SPY"]["Close"], 14),
                      index=raw["SPY"].index).reindex(px.index) / px["SPY"])
calm = (dist > -0.03) & (spy_atrp < 0.012)

SUB = ((vk >= 0.030) & calm).fillna(False)
print(f"subclass n_days={int(SUB.sum())}  FIRES TODAY: {bool(SUB.loc[today])}")
print(f"live: gap {100*vk.loc[today]:+.2f}pp  dist52wh {100*dist.loc[today]:+.2f}%  "
      f"SPY ATR%% {100*spy_atrp.loc[today]:.2f}%")

print("\n\n########## 1. PER-LEG ATTRIBUTION INSIDE THE SUBCLASS ##########")
print("C2 = long XLK, short XLV.  columns are the C2 SIDE of each leg.")
rows = []
for h in (1, 2, 3, 5, 10):
    ret_c2 = vehicle_ret(px, [("XLK", 1.0), ("XLV", -1.0)], h)
    valid = ret_c2.dropna().index
    epi = declusters(px.index[SUB.values].intersection(valid), h, valid)
    row = {"h": h, "n_epi": len(epi)}
    xlk = fwd_lag(px["XLK"], h, 1)
    xlv = fwd_lag(px["XLV"], h, 1)
    spy = fwd_lag(px["SPY"], h, 1)
    row["LONG_XLK_cond"] = round(100 * xlk.loc[epi].mean(), 3)
    row["LONG_XLK_exc"] = round(100 * (xlk.loc[epi].mean() - xlk.dropna().mean()), 3)
    row["SHORT_XLV_cond"] = round(-100 * xlv.loc[epi].mean(), 3)
    row["SHORT_XLV_exc"] = round(-100 * (xlv.loc[epi].mean() - xlv.dropna().mean()), 3)
    row["SPY_cond"] = round(100 * spy.loc[epi].mean(), 3)
    row["C2_total"] = round(100 * ret_c2.loc[epi].mean(), 3)
    rows.append(row)
att = pd.DataFrame(rows)
print(att.to_string(index=False))
print("\nshare of C2 carried by the LONG XLK leg:")
for _, r in att.iterrows():
    if r["C2_total"] != 0:
        print(f"  h={int(r['h']):2d}: {100*r['LONG_XLK_cond']/r['C2_total']:6.1f}%   "
              f"short-XLV leg contributes {r['SHORT_XLV_cond']:+.3f}pp "
              f"({'DRAG' if r['SHORT_XLV_cond'] < 0 else 'help'})")

print("\n\n########## 2. THE IGNORANT RULE ##########")
print("long XLK alone after a big XLK down day in the SAME calm/near-high tape,")
print("no healthcare condition at all.  matched on XLK's own move.")
xlk_thr = r1["XLK"].loc[px.index[SUB.values]].quantile(0.75)   # generous: as mild
print(f"  XLK 1d on subclass days: mean {100*r1['XLK'][SUB.values].mean():+.2f}% "
      f"median {100*r1['XLK'][SUB.values].median():+.2f}%  "
      f"75th {100*xlk_thr:+.2f}%   (today {100*r1['XLK'].loc[today]:+.2f}%)")
IGN = ((r1["XLK"] <= xlk_thr) & calm).fillna(False)
IGN_NOROT = (IGN & ~SUB).fillna(False)
print(f"  ignorant n_days={int(IGN.sum())}  (ex-subclass {int(IGN_NOROT.sum())})")
for h in (1, 2, 3, 5, 10):
    ret_c2 = vehicle_ret(px, [("XLK", 1.0), ("XLV", -1.0)], h)
    xlk_r = fwd_lag(px["XLK"], h, 1)
    valid = ret_c2.dropna().index
    e_sub = declusters(px.index[SUB.values].intersection(valid), h, valid)
    e_ign = declusters(px.index[IGN_NOROT.values].intersection(valid), h, valid)
    rows = [summarize(ret_c2.loc[e_sub].values, f"C2 pair, subclass (N={len(e_sub)})"),
            summarize(xlk_r.loc[e_sub].values, "  long XLK ALONE, same days"),
            summarize(xlk_r.loc[e_ign].values,
                      f"  IGNORANT long XLK, calm+down XLK, NO rotation (N={len(e_ign)})"),
            summarize(xlk_r.loc[valid].values, "  XLK all days"),
            summarize(ret_c2.loc[valid].values, "  C2 pair all days")]
    show(rows, f"h={h} ignorant-rule placebo")

print("\n\n########## 3a. BETA-NEUTRAL SUBCLASS ##########")
beta = r1["XLV"].rolling(252).cov(r1["XLK"]) / r1["XLK"].rolling(252).var()
print(f"live PIT beta(XLV~XLK) {beta.loc[today]:.3f}   subclass mean "
      f"{beta[SUB.values].mean():.3f}   hist median {beta.median():.3f}")
for h in (1, 3, 5, 10):
    ret_eq = vehicle_ret(px, [("XLK", 1.0), ("XLV", -1.0)], h)
    # C2 beta-neutral: long XLK, short beta-scaled... invert the XLV~XLK reg
    bkv = (r1["XLK"].rolling(252).cov(r1["XLV"]) / r1["XLV"].rolling(252).var())
    ret_bn = fwd_lag(px["XLK"], h, 1) - bkv * fwd_lag(px["XLV"], h, 1)
    valid = ret_eq.dropna().index.intersection(ret_bn.dropna().index)
    e = declusters(px.index[SUB.values].intersection(valid), h, valid)
    show([summarize(ret_eq.loc[e].values, f"h={h} C2 eq-$ subclass"),
          summarize(ret_eq.loc[valid].values, f"h={h} C2 eq-$ all days"),
          summarize(ret_bn.loc[e].values, f"h={h} C2 beta-neutral subclass"),
          summarize(ret_bn.loc[valid].values, f"h={h} C2 beta-neutral all days")],
         f"C2 beta-neutral h={h}  (beta(XLK~XLV) live {bkv.loc[today]:.3f})")

print("\n\n########## 3b. CONCENTRATION / YEAR / MIDTERM ##########")
for h in (3, 5):
    ret_c2 = vehicle_ret(px, [("XLK", 1.0), ("XLV", -1.0)], h)
    valid = ret_c2.dropna().index
    e = declusters(px.index[SUB.values].intersection(valid), h, valid)
    v = ret_c2.loc[e].values
    print(f"\nh={h}  N={len(e)}  mean {100*v.mean():+.3f}%  "
          f"record {int((v>0).sum())}-{int((v<=0).sum())}  "
          f"sign p={sign_test(int((v>0).sum()), len(v)):.4f}")
    print("  " + cluster_note(e, v))
    by = pd.Series(100 * v, index=e).groupby(e.year)
    print("  by year: " + ", ".join(f"{y}:n={len(g)} mean{g.mean():+.2f}"
                                    for y, g in by))
    mid = np.array([d.year % 4 == 2 for d in e])
    show([summarize(v[mid], f"midterm years (N={int(mid.sum())})  <-- 2026"),
          summarize(v[~mid], f"non-midterm (N={int((~mid).sum())})")],
         f"h={h} midterm split")
    show(era_split(e, v), f"h={h} era split")

print("\n\n########## 4. HORIZON SCAN + LOSING EPISODE PATHS ##########")
d = px.index[SUB.values]
show(horizon_scan(px, d, [("XLK", 1.0), ("XLV", -1.0)], hs=(1, 2, 3, 5, 7, 10)),
     "C2 pair, subclass")
show(horizon_scan(px, d, [("XLK", 1.0)], hs=(1, 2, 3, 5, 7, 10)),
     "long XLK alone, subclass")
paths = episode_paths(px, declusters(d, 5, px.index),
                      [("XLK", 1.0), ("XLV", -1.0)], 5)
print("\nC2 subclass episode paths (cum %, day 1..5):")
print((100 * paths).round(2).to_string())
losers = paths[paths[5] < 0]
print(f"\nLOSING episodes ({len(losers)} of {len(paths)}):")
print((100 * losers).round(2).to_string())
if len(losers):
    print(f"  by day 2 the median loser is already "
          f"{100*losers[2].median():+.2f}%")
