"""ROUND 2 items 2-6 (+ the gate-attribution question r1 raised).

r1 showed the UNGATED month-end anchor is the strongest thing in the grid
(TLT ungated N=288 +0.540% t=3.88 at h=9), so the first thing to settle is
whether the TLT-oversold gate adds anything at all. Then:
 (2) the fossil test - yield regime at the signal, the split that killed the
     August TLT seasonal on 2026-08-17
 (3) month-of-year control + month fixed effects + the W12 November overlap
 (4) era, LOYO, concentration, drop-best
 (5) which leg (TLT outright vs the TLT-SPY spread)
 (6) is today's TLT-at-52w-low state inside the trigger population's support,
     and does at-the-low poison the cell (the C2 finding)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np, pandas as pd

LAG, H = 1, 9
TK = ["SPY", "TLT", "IEF", "LQD"]
raw = load_prices(TK + ["^TNX"])
idx = raw["SPY"]["Close"].index
for t in TK[1:]:
    idx = idx.intersection(raw[t]["Close"].index)
px = pd.DataFrame({t: raw[t]["Close"].reindex(idx) for t in TK}).dropna()
idx = px.index
tnx = raw["^TNX"]["Close"].reindex(idx).ffill()

ymv = pd.Series(idx.year * 100 + idx.month, index=idx)
is_last = ymv.ne(ymv.shift(-1)).values; is_last[-1] = False
pos = pd.Series(range(len(idx)), index=idx)
T21 = px["TLT"].pct_change(21)


def anchor(h=H, k=0):
    t = pos.values + LAG + h + k
    m = np.zeros(len(idx), bool); ok = t < len(idx)
    m[ok] = is_last[t[ok]]
    return m


rT = fwd_lag(px["TLT"], H, LAG)
rS = fwd_lag(px["SPY"], H, LAG)
rI = fwd_lag(px["IEF"], H, LAG)
SP = rT - rS
A = anchor()
G = (T21 <= -0.025).fillna(False).values
vT = rT.notna().values
D = idx[A & G & vT]                     # the 59 trigger days
DA = idx[A & vT]                        # the 288 ungated anchor days
print("triggers N=%d | ungated anchor days N=%d" % (len(D), len(DA)))


def welch(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    return (x.mean() - y.mean()) / np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))


# ============================================================ 0. gate attribution
print("\n===== 0. DOES THE GATE FILTER? (anchor days only, TLT outright) =====")
on = rT.loc[D].values
off = rT.loc[idx[A & ~G & vT]].values
print("gate ON  N=%d %+.3f%% | gate OFF N=%d %+.3f%% | diff %+.3fpp welch t=%+.2f"
      % (len(on), 100 * on.mean(), len(off), 100 * off.mean(),
         100 * (on.mean() - off.mean()), welch(on, off)))
onS, offS = SP.loc[D].values, SP.loc[idx[A & ~G & SP.notna().values]].values
print("SPREAD ON N=%d %+.3f%% | OFF N=%d %+.3f%% | diff %+.3fpp welch t=%+.2f"
      % (len(onS), 100 * onS.mean(), len(offS), 100 * offS.mean(),
         100 * (onS.mean() - offS.mean()), welch(onS, offS)))
x = T21.loc[DA].values; y = rT.loc[DA].values
m = ~np.isnan(x) & ~np.isnan(y)
print("spearman(TLT 21d , fwd TLT) on the 288 anchor days = %+.3f  (pearson %+.3f)"
      % (pd.Series(x[m]).corr(pd.Series(y[m]), method="spearman"),
         np.corrcoef(x[m], y[m])[0, 1]))
q = pd.qcut(pd.Series(x[m]), 5, labels=False)
rows = [summarize(y[m][q.values == i], f"TLT21d quintile {i+1} "
                  f"[{100*np.nanmin(x[m][q.values==i]):+.1f}%,{100*np.nanmax(x[m][q.values==i]):+.1f}%]")
        for i in range(5)]
show(rows, "0b. is the gate MONOTONE inside the anchor? (quintiles of TLT 21d, fwd TLT h=9)")

# ============================================================ 2. FOSSIL TEST
print("\n===== 2. FOSSIL TEST - yield regime prevailing at the signal =====")
d21 = (tnx - tnx.shift(21))            # ^TNX points, +ve = yields rising
d63 = (tnx - tnx.shift(63))
tlt63 = px["TLT"].pct_change(63)
tlt126 = px["TLT"].pct_change(126)
print("LIVE: ^TNX 21d chg %+.3f pts, 63d chg %+.3f pts | TLT 63d %+.2f%% 126d %+.2f%%"
      % (d21.iloc[-1], d63.iloc[-1], 100 * tlt63.iloc[-1], 100 * tlt126.iloc[-1]))
for lab, sr, live in (("^TNX 21d chg", d21, d21.iloc[-1]),
                      ("^TNX 63d chg", d63, d63.iloc[-1]),
                      ("TLT 63d ret", -tlt63, -tlt63.iloc[-1]),
                      ("TLT 126d ret", -tlt126, -tlt126.iloc[-1])):
    s = sr.loc[D].values
    up = s > 0                              # "rising yields" half = the LIVE half
    rows = [summarize(rT.loc[D].values[up], f"{lab} RISING (live side) N={int(up.sum())}"),
            summarize(rT.loc[D].values[~up], f"{lab} FALLING N={int((~up).sum())}"),
            summarize(SP.loc[D].values[up], f"   spread RISING"),
            summarize(SP.loc[D].values[~up], f"   spread FALLING")]
    show(rows, f"2. regime split by {lab}  (live = {live:+.3f})")
    # same split on the UNGATED parent, to see if regime-flatness is a gate property
    sa = sr.loc[DA].values; ua = sa > 0
    print("   ungated anchor parent: RISING %+.3f%% (N=%d) vs FALLING %+.3f%% (N=%d)"
          % (100 * np.nanmean(rT.loc[DA].values[ua]), int(ua.sum()),
             100 * np.nanmean(rT.loc[DA].values[~ua]), int((~ua).sum())))

# ============================================================ 3. MONTH OF YEAR
print("\n===== 3. MONTH-OF-YEAR CONTROL =====")
mo = pd.Series(D.month)
print("trigger month histogram:", dict(mo.value_counts().sort_index()))
rows = []
for m_ in range(1, 13):
    sel = D[D.month == m_]
    if len(sel) == 0:
        continue
    r = summarize(rT.loc[sel].values, f"month {m_:02d} (N={len(sel)})")
    r["spread_pct"] = round(100 * SP.loc[sel].mean(), 3)
    rows.append(r)
show(rows, "3a. cell by trigger month, TLT outright")
# month fixed effect: demean by the month's unconditional h=9 lag-1 TLT mean
allm = rT.groupby(idx.month).transform("mean")
res = (rT - allm)
rows = [summarize(res.loc[D].values, f"TLT month-demeaned, triggers (N={len(D)})"),
        summarize(res.loc[DA].values, f"TLT month-demeaned, ungated anchor (N={len(DA)})"),
        summarize(res.dropna().values, "TLT month-demeaned, all days")]
sres = (SP - SP.groupby(idx.month).transform("mean"))
rows += [summarize(sres.loc[D].values, f"SPREAD month-demeaned, triggers"),
         summarize(sres.loc[DA].values, f"SPREAD month-demeaned, ungated anchor")]
show(rows, "3b. month fixed effect (subtract each calendar month's own h=9 mean)")
augD = D[D.month == 8]
print("3c. AUGUST triggers: %s" % ", ".join(str(d.date()) for d in augD))
print("    Aug TLT %+.3f%% (N=%d) | ex-Aug %+.3f%% (N=%d)"
      % (100 * rT.loc[augD].mean(), len(augD),
         100 * rT.loc[D.difference(augD)].mean(), len(D) - len(augD)))
# W12 overlap: November trading days 4-12
tdom = pd.Series(idx.to_period("M")).groupby(pd.Series(idx.to_period("M"))).cumcount().values + 1
tdom = pd.Series(tdom, index=idx)
w12 = ((idx.month == 11) & (tdom.values >= 4) & (tdom.values <= 12))
ov = np.isin(D, idx[w12])
print("3d. W12 overlap (Nov tdom 4-12): %d of %d trigger days (%.1f%%); trigger tdom range %d-%d, median %d"
      % (int(ov.sum()), len(D), 100 * ov.mean(),
         tdom.loc[D].min(), tdom.loc[D].max(), int(tdom.loc[D].median())))
print("    Nov triggers: %s" % ", ".join("%s(tdom %d)" % (d.date(), tdom.loc[d]) for d in D[D.month == 11]))
print("    ex-November cell: %+.3f%% N=%d | November only %+.3f%% N=%d"
      % (100 * rT.loc[D[D.month != 11]].mean(), int((D.month != 11).sum()),
         100 * rT.loc[D[D.month == 11]].mean(), int((D.month == 11).sum())))

# ============================================================ 4. ERA / LOYO
print("\n===== 4. ERA, LOYO, CONCENTRATION =====")
epi = declusters(D, 21, idx)
for lab, r in (("TLT", rT), ("SPREAD", SP)):
    ve = r.loc[epi].values
    rows = [summarize(r.loc[D].values, f"{lab} day-level all"),
            summarize(r.loc[D[D < pd.Timestamp('2018-01-01')]].values, f"{lab} pre-2018"),
            summarize(r.loc[D[D >= pd.Timestamp('2018-01-01')]].values, f"{lab} 2018+"),
            summarize(r.loc[D[D >= pd.Timestamp('2021-01-01')]].values, f"{lab} 2021+"),
            summarize(r.loc[D[D >= pd.Timestamp('2023-01-01')]].values, f"{lab} 2023+")]
    show(rows, f"4a. era, {lab}")
    print("   ", cluster_note(epi, ve))
    byy = pd.Series(ve, index=epi).groupby(epi.year)
    loyo = {}
    for y in sorted(set(epi.year)):
        k = ve[epi.year != y]
        loyo[y] = 100 * k.mean()
    lo = min(loyo, key=loyo.get)
    print("    LOYO floor: drop %d -> %+.3f%% (full %+.3f%%); worst 3: %s"
          % (lo, loyo[lo], 100 * ve.mean(),
             {y: round(loyo[y], 3) for y in sorted(loyo, key=loyo.get)[:3]}))
    yr_tot = pd.Series(ve, index=epi).groupby(epi.year).sum().sort_values(ascending=False)
    b1, b2 = yr_tot.index[0], yr_tot.index[1]
    k1 = ve[epi.year != b1]; k2 = ve[~np.isin(epi.year, [b1, b2])]
    print("    drop-best-year (%d): N=%d %+.3f%% | drop-2-best (%d,%d): N=%d %+.3f%%"
          % (b1, len(k1), 100 * k1.mean(), b1, b2, len(k2), 100 * k2.mean()))
    print("    per-year episode sums:", {int(y): round(100 * v, 2) for y, v in yr_tot.sort_index().items()})

# ============================================================ 5. WHICH LEG
print("\n===== 5. WHICH LEG =====")
a, b = rT.loc[D].values, rS.loc[D].values
print("corr(TLT,SPY) on triggers %+.3f | beta(TLT on SPY) %+.3f" % (np.corrcoef(a, b)[0, 1], np.polyfit(b, a, 1)[0]))
for lab, xx in (("TLT only", a), ("TLT-SPY", a - b), ("SHORT SPY only", -b), ("IEF only", rI.loc[D].values)):
    print("  %-14s mean %+.3f%% sd %.3f%% ratio %.3f hit %.1f%% worst %+.2f%% best %+.2f%%"
          % (lab, 100 * xx.mean(), 100 * xx.std(ddof=1), xx.mean() / xx.std(ddof=1),
             100 * (xx > 0).mean(), 100 * xx.min(), 100 * xx.max()))

# ============================================================ 6. SUPPORT
print("\n===== 6. IS TODAY INSIDE THE TRIGGER POPULATION'S SUPPORT? =====")
low52 = px["TLT"].rolling(252).min()
sma200 = px["TLT"].rolling(200).mean()
dist_low = px["TLT"] / low52 - 1.0          # 0 = AT the 52w low
dist_200 = px["TLT"] / sma200 - 1.0
live_low, live_200 = dist_low.iloc[-1], dist_200.iloc[-1]
print("LIVE 2026-08-17: TLT %.2f%% above its 52w low, %.2f%% vs 200d"
      % (100 * live_low, 100 * live_200))
dl = dist_low.loc[D].values; d2 = dist_200.loc[D].values
print("trigger population dist-to-52w-low: min %.2f%% p10 %.2f%% median %.2f%% p90 %.2f%% max %.2f%%"
      % tuple(100 * np.nanpercentile(dl, p) for p in (0, 10, 50, 90, 100)))
print("  today's %.2f%% sits at the %.1f percentile of that distribution; %d of %d triggers were within 1%% of the low"
      % (100 * live_low, 100 * (dl < live_low).mean(), int((dl <= 0.01).sum()), len(dl)))
print("trigger dist-to-200d: median %.2f%% ; today %.2f%% -> percentile %.1f"
      % (100 * np.nanmedian(d2), 100 * live_200, 100 * (d2 < live_200).mean()))
rows = []
for lab, sel in (("AT the low (<=1%% above 52wL)", dl <= 0.01),
                 ("<=3% above 52w low", dl <= 0.03),
                 (">3% above 52w low", dl > 0.03),
                 ("below 200d (live side)", d2 < 0),
                 ("above 200d", d2 >= 0)):
    s = D[sel & ~np.isnan(dl)] if "200d" not in lab else D[sel & ~np.isnan(d2)]
    if len(s) == 0:
        rows.append({"label": lab, "n": 0}); continue
    r = summarize(rT.loc[s].values, f"{lab} (N={len(s)})")
    r["spread_pct"] = round(100 * SP.loc[s].mean(), 3)
    rows.append(r)
show(rows, "6b. in-support subcells, TLT outright (+ spread column)")
