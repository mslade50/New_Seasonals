"""C3 round 3 — the only TRADEABLE version: liquid names, today's geometry.

TJX prints 2026-08-19 (k=4 -> h=2), ROST 2026-08-20 (k=5 -> h=3). Both are
in LIQUID_PLUS_COMMODITIES. Everything below is restricted to that universe
because the $5-15 bucket that carries the full-panel result is neither
tradeable at pitch size nor free of the survivorship caveat.

Questions this settles:
  - horizon scan under the exit-before-the-print bound
  - the PRINT PREMIUM on liquid names (cell minus the same washout gate on
    the same names away from any print) -- that is what the earnings anchor
    is actually worth
  - cost at 5/10/20 bps single-name round trip against the >=5x bar
  - era + cycle stability, date-clustered
  - TJX's and ROST's own records (the reference-class question)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import strategy_config as sc  # noqa: E402
from pitch_lab import PRICES_PATH, show, sign_test, summarize  # noqa: E402

T0 = time.time()
ASOF = pd.Timestamp("2026-08-13")
WINS = 0.50

earn = pd.read_parquet("data/earnings_calendar.parquet", columns=["ticker", "date"])
earn["date"] = pd.to_datetime(earn["date"])
LIQ = sorted(set(sc.LIQUID_PLUS_COMMODITIES) & set(earn["ticker"].unique()))
print(f"liquid names with earnings coverage: {len(LIQ)}")

mp = pd.read_parquet(PRICES_PATH, columns=["ticker", "date", "Close"])
mp = mp[mp["ticker"].isin(LIQ)]
mp["date"] = pd.to_datetime(mp["date"])
mp = mp[(mp["date"] >= "1999-01-01") & (mp["date"] <= ASOF)]
close = mp.pivot_table(index="date", columns="ticker", values="Close", aggfunc="last").sort_index()
idx, cols = close.index, list(close.columns)
colpos = {t: i for i, t in enumerate(cols)}
C = close.values
RANK5 = (close.pct_change(5, fill_method=None).rolling(252, min_periods=252)
         .rank(pct=True) * 100).values
print(f"panel {close.shape}  ({time.time()-T0:.0f}s)")

earn = earn[earn["ticker"].isin(cols)]
ev_t, ev_p = [], []
for t, g in earn.groupby("ticker"):
    j = colpos[t]
    p = np.searchsorted(idx.values, g["date"].values, side="left")
    ok = (p > 0) & (p < len(idx))
    ev_t.append(np.full(ok.sum(), j))
    ev_p.append(p[ok])
EV_T, EV_P = np.concatenate(ev_t), np.concatenate(ev_p)
NEAR = np.zeros(C.shape, dtype=bool)
for j, p in zip(EV_T, EV_P):
    NEAR[max(0, p - 20):min(len(idx), p + 6), j] = True
WASH = RANK5 <= 10.0


def fwd(h):
    f = np.full(C.shape, np.nan)
    f[:-(1 + h)] = C[1 + h:] / C[1:-h] - 1.0
    return np.clip(f, -WINS, WINS)


print("\n" + "=" * 74)
print("1. HORIZON SCAN under the exit-BEFORE-the-print bound (h = k-2)")
print("=" * 74)
rows = []
store = {}
for k in range(3, 11):
    h = k - 2
    F = fwd(h)
    a = EV_P - k
    ok = (a >= 300) & (a + 1 + h < len(idx))
    a2, t2 = a[ok], EV_T[ok]
    r, s = F[a2, t2], RANK5[a2, t2]
    m = (s <= 10.0) & ~np.isnan(r)
    d = idx[a2[m]]
    row = summarize(r[m], f"k={k} h={h}")
    dfc = pd.DataFrame({"d": d, "r": r[m]}).groupby("d")["r"].mean()
    row["date_clust_pct"] = round(100 * dfc.mean(), 3)
    row["n_dates"] = len(dfc)
    # print premium: same gate, same h, same names, no print near
    offm = WASH & ~NEAR & ~np.isnan(F)
    row["away_from_print_pct"] = round(100 * F[offm].mean(), 3)
    row["PRINT_PREMIUM_pct"] = round(row["mean_pct"] - 100 * F[offm].mean(), 3)
    rows.append(row)
    store[k] = (r[m], d, t2[m], F, offm)
show(rows, "liquid universe only")

print("\n" + "=" * 74)
print("2. TODAY'S TWO GEOMETRIES IN DETAIL  (TJX k=4/h=2, ROST k=5/h=3)")
print("=" * 74)
for k in (4, 5):
    r, d, t, F, offm = store[k]
    h = k - 2
    dfc = pd.DataFrame({"d": d, "r": r}).groupby("d")["r"].mean()
    w = int((dfc.values > 0).sum())
    print(f"\n--- k={k}, h={h} ---")
    show([summarize(r, f"cell, event level (N={len(r)})"),
          summarize(dfc.values, f"cell, DATE-CLUSTERED (N={len(dfc)})"),
          summarize(F[offm], f"same gate, no print near (N={int(offm.sum())})"),
          summarize(F[~np.isnan(F)], "all liquid days")],
         f"k={k}")
    print(f"  date record {w}-{len(dfc)-w}, sign p = {sign_test(w, len(dfc)):.4f}")
    dd = pd.DatetimeIndex(dfc.index)
    v = dfc.values
    show([summarize(v[dd < pd.Timestamp("2010-01-01")], "pre-2010"),
          summarize(v[(dd >= pd.Timestamp("2010-01-01")) & (dd < pd.Timestamp("2018-01-01"))], "2010-2017"),
          summarize(v[dd >= pd.Timestamp("2018-01-01")], "2018+"),
          summarize(v[(dd.year % 4) == 2], "midterm"),
          summarize(v[dd.month == 8], "August")], f"splits, k={k} (date-clustered)")
    mm = dfc.values.mean()
    print("  cost vs the >=5x bar (date-clustered mean "
          f"{10000*mm:.1f} bps):  " +
          "  ".join(f"{rt}bps -> {10000*mm/rt:.1f}x" for rt in (5, 10, 20)))
    prem = 100 * r.mean() - 100 * F[offm].mean()
    print(f"  PRINT PREMIUM {100*prem:.1f} bps:  " +
          "  ".join(f"{rt}bps -> {100*prem/rt:.1f}x" for rt in (5, 10, 20)))

print("\n" + "=" * 74)
print("3. REFERENCE CLASS — are TJX/ROST special, or just members?")
print("=" * 74)
for k in (4, 5):
    r, d, t, F, offm = store[k]
    for name in ("TJX", "ROST", "TGT", "WMT"):
        if name not in colpos:
            continue
        sel = t == colpos[name]
        if sel.sum() == 0:
            print(f"  k={k} {name}: no qualifying events")
            continue
        v = r[sel]
        ww = int((v > 0).sum())
        print(f"  k={k} {name:5s}: N={len(v):3d} mean {100*v.mean():+.3f}% "
              f"record {ww}-{len(v)-ww} sign p={sign_test(ww, len(v)):.3f} "
              f"worst {100*v.min():+.2f}% best {100*v.max():+.2f}%")

print("\n" + "=" * 74)
print("4. LOSING EPISODES at k=4 (what kills it)")
print("=" * 74)
r, d, t, F, offm = store[4]
dfc = pd.DataFrame({"d": d, "r": r, "t": [cols[j] for j in t]})
worst = dfc.nsmallest(12, "r")
print(worst.assign(r=lambda x: (100 * x["r"]).round(2)).to_string(index=False))
byyr = dfc.groupby(dfc["d"].dt.year)["r"].mean() * 100
print("\n  yearly date-mean (pp):", {int(y): round(v, 2) for y, v in byyr.items()})
print(f"\n({time.time()-T0:.0f}s)")
