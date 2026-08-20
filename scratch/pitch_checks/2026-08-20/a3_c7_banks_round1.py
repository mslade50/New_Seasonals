"""C7 round 1: short the bank complex on a 5-day breadth washout inside an
intact 63-day uptrend.

The candidate is the LOSING half of watchlist W11 traded from the other side,
so the first thing measured is the gate-off parent (registry 2026-08-19: run
the cell gate-off BEFORE gate-on).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

pd.set_option("display.width", 240)

BANKS = ["JPM", "BAC", "C", "WFC", "GS", "MS", "USB", "KEY", "RF", "STT", "SCHW"]
raw = load_prices(BANKS + ["XLF", "KRE", "SPY"])
d = raw["SPY"]["Close"].dropna().index

close = pd.DataFrame({t: raw[t]["Close"].reindex(d) for t in raw})
r5 = pd.DataFrame({t: pct_rank(raw[t]["Close"].dropna(), 5).reindex(d) for t in BANKS})
r63 = pd.DataFrame({t: pct_rank(raw[t]["Close"].dropna(), 63).reindex(d) for t in BANKS})

nvalid = r5.notna().sum(axis=1)
breadth = (r5 <= 20).sum(axis=1) / nvalid.replace(0, np.nan)
med63 = r63.median(axis=1)
ok = nvalid >= 8

print("===== 0. today =====")
print(f" last bar {d[-1].date()}  names with a 5d rank: {int(nvalid.iloc[-1])}")
print(f" breadth (frac at 5d rank <= 20) = {breadth.iloc[-1]*100:.1f}%   median 63d rank = {med63.iloc[-1]:.1f}")
print(f" KRE 5d rank {pct_rank(raw['KRE']['Close'].dropna(),5).iloc[-1]:.1f}  "
      f"XLF 63d rank {pct_rank(raw['XLF']['Close'].dropna(),63).iloc[-1]:.1f}")

m_breadth = (breadth >= 0.70) & ok
m_intact = m_breadth & (med63 >= 70)
m_broken = m_breadth & (med63 < 70)
m_breadth = m_breadth.fillna(False).astype(bool)
m_intact = m_intact.fillna(False).astype(bool)
m_broken = m_broken.fillna(False).astype(bool)

for nm, mm in [("breadth ALONE (parent)", m_breadth), ("INTACT (the candidate)", m_intact),
               ("BROKEN (W11's live half)", m_broken)]:
    e = declusters(d[mm], 10, d)
    print(f" {nm:<26} days={int(mm.sum()):>5} episodes(gap10)={len(e):>4} "
          f"years={sorted(set(e.year))}")

# vehicles -------------------------------------------------------------------
px = close.copy()
EW = {"XLF short": [("XLF", -1.0)], "KRE short": [("KRE", -1.0)],
      "complex EW short": [(t, -1.0 / len(BANKS)) for t in BANKS]}

def cell_stats(mm, legs, h, gap=10, label=""):
    ret = vehicle_ret(px, legs, h)
    valid = ret.dropna().index
    e = declusters(pd.DatetimeIndex(d[mm]).intersection(valid), gap, valid)
    v = ret.loc[e].values
    base = ret.loc[valid]
    if len(v) == 0:
        return {"label": label, "n": 0}
    r = summarize(v, label)
    r["own_drift"] = round(100 * base.mean(), 3)
    r["edge_pp"] = round(r["mean_pct"] - 100 * base.mean(), 3)
    r["signp"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
    return r

print("\n\n===== 1. GATE ATTRIBUTION, short side, excess over the vehicle's own SHORT drift =====")
for vname, legs in EW.items():
    rows = []
    for h in (1, 3, 5, 10):
        for nm, mm in [("breadth ALONE", m_breadth), ("INTACT (cand)", m_intact),
                       ("BROKEN", m_broken)]:
            rows.append(cell_stats(mm, legs, h, label=f"{nm} h={h}"))
    show(rows, f"{vname}")

print("\n\n===== 2. round-1 battery on the candidate, XLF vehicle =====")
for h in (3, 5):
    battery(px, m_intact, EW["XLF short"], h,
            "C7 SHORT XLF, bank breadth washout inside an intact 63d trend",
            cost_bps=2.0, min_gap=10,
            variants={
                "breadth>=60 & med63>=70": ((breadth >= 0.60) & (med63 >= 70) & ok).fillna(False).astype(bool),
                "breadth>=80 & med63>=70": ((breadth >= 0.80) & (med63 >= 70) & ok).fillna(False).astype(bool),
                "breadth>=70 & med63>=80": ((breadth >= 0.70) & (med63 >= 80) & ok).fillna(False).astype(bool),
                "breadth>=70 & med63>=60": ((breadth >= 0.70) & (med63 >= 60) & ok).fillna(False).astype(bool),
                "breadth ALONE >=70": m_breadth,
                "BROKEN half": m_broken,
            })

# --------------------------------------------------------------- 3. crisis + tape
print("\n\n===== 3. crisis selection and tape over-selection =====")
spy = raw["SPY"]["Close"].dropna()
sma200 = rolling_on_valid(spy, lambda x: x.rolling(200).mean())
below = (spy < sma200).reindex(d)
base_rate = below.dropna().mean()
for nm, mm in [("INTACT (cand)", m_intact), ("breadth ALONE", m_breadth), ("BROKEN", m_broken)]:
    sub = below.reindex(d[mm]).dropna()
    print(f" {nm:<16} SPY below its 200d on {sub.mean()*100:.1f}% of trigger days "
          f"(base rate {base_rate*100:.1f}%), N={len(sub)}")

for h in (3, 5, 10):
    ret = vehicle_ret(px, EW["XLF short"], h)
    valid = ret.dropna().index
    e = declusters(pd.DatetimeIndex(d[m_intact]).intersection(valid), 10, valid)
    v = ret.loc[e]
    by = v.groupby(v.index.year).agg(["count", "mean", "sum"])
    by[["mean", "sum"]] = (by[["mean", "sum"]] * 100).round(2)
    print(f"\n h={h} by year (short XLF):")
    print(by.to_string())
    for drop in ([2008, 2009], [2008, 2009, 2020], [2008, 2009, 2011, 2020]):
        k = v[~v.index.year.isin(drop)]
        if len(k) == 0:
            print(f"   ex-{drop}: NOTHING LEFT")
            continue
        w = int((k > 0).sum())
        print(f"   ex-{drop}: N={len(k)} mean {100*k.mean():+.3f}% record {w}-{len(k)-w} "
              f"signp {sign_test(w, len(k)):.4f}")

# --------------------------------------------------------------- 4. the KRE form
print("\n\n===== 4. the KRE-specific recon form (KRE 5d rank<=10 & XLF 63d rank>=95) =====")
kre5 = pct_rank(raw["KRE"]["Close"].dropna(), 5).reindex(d)
xlf63 = pct_rank(raw["XLF"]["Close"].dropna(), 63).reindex(d)
mk = ((kre5 <= 10) & (xlf63 >= 95)).fillna(False).astype(bool)
ek = declusters(d[mk], 10, d)
print(f" days {int(mk.sum())} episodes {len(ek)} years {sorted(set(ek.year))}")
rows = []
for h in (1, 3, 5, 10):
    for vname, legs in EW.items():
        rows.append(cell_stats(mk, legs, h, label=f"{vname} h={h}"))
    rows.append(cell_stats((kre5 <= 10).fillna(False).astype(bool), EW["KRE short"], h,
                           label=f"KRE washout ALONE h={h}"))
show(rows, "KRE form + gate-off")
