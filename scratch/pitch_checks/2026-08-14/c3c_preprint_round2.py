"""C3 round 2 — the reference class, cleanly.

Round 1 left one question open and one control polluted:
  - the "washout AWAY from a print" control came out of the raw 962-name
    panel and carried a best value of +26416% and an sd of 43.8%, i.e. it
    was measuring corrupt/penny bars, not a comparable trade. Rebuilt here
    with a price floor and winsorisation, matched to the SAME names.
  - the earnings ANCHOR never faced a placebo ladder. Shifting the print
    date backwards keeps the washout gate and the horizon identical and
    removes only the print, which is the one-line attribution test.
Plus: definition neighbours, price/liquidity buckets (the survivorship
channel), and the multiplicity charge on the retail-August cell.
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
START = pd.Timestamp("1999-01-01")
K, H = 4, 2          # today's geometry: anchor 08-13, print 08-19
PX_FLOOR = 5.0
WINS = 0.50          # winsorise the h-return at +/-50% (a 2d single-name move)

earn = pd.read_parquet("data/earnings_calendar.parquet", columns=["ticker", "date"])
earn["date"] = pd.to_datetime(earn["date"])
mp = pd.read_parquet(PRICES_PATH, columns=["ticker", "date", "Close"])
mp = mp[mp["ticker"].isin(earn["ticker"].unique())]
mp["date"] = pd.to_datetime(mp["date"])
mp = mp[(mp["date"] >= START) & (mp["date"] <= ASOF)]
close = mp.pivot_table(index="date", columns="ticker", values="Close", aggfunc="last").sort_index()
idx, cols = close.index, list(close.columns)
colpos = {t: i for i, t in enumerate(cols)}
C = close.values
RANK5 = (close.pct_change(5, fill_method=None).rolling(252, min_periods=252)
         .rank(pct=True) * 100).values
print(f"panel {close.shape}  ({time.time()-T0:.0f}s)")

ev_t, ev_p = [], []
for t, g in earn.groupby("ticker"):
    j = colpos.get(t)
    if j is None:
        continue
    p = np.searchsorted(idx.values, g["date"].values, side="left")
    ok = (p > 0) & (p < len(idx))
    ev_t.append(np.full(ok.sum(), j))
    ev_p.append(p[ok])
EV_T, EV_P = np.concatenate(ev_t), np.concatenate(ev_p)

# forward h-return from EVERY (day, name): entry i+1, exit i+1+H
FWD = np.full(C.shape, np.nan)
FWD[:-(1 + H)] = C[1 + H:] / C[1:-H] - 1.0
FWD = np.clip(FWD, -WINS, WINS)
PXOK = C >= PX_FLOOR
NEAR = np.zeros(C.shape, dtype=bool)          # +/- window around any print
for j, p in zip(EV_T, EV_P):
    NEAR[max(0, p - 15):min(len(idx), p + 6), j] = True
WASH = RANK5 <= 10.0


def ev_cell(shift: int):
    """the cell with the print date shifted by `shift` td (0 = true)."""
    pe = EV_P + shift
    a = pe - K
    ok = (a >= 300) & (a < len(idx)) & (a + 1 + H < len(idx))
    a, tt = a[ok], EV_T[ok]
    r = FWD[a, tt]
    s = RANK5[a, tt]
    p_ok = PXOK[a, tt]
    good = ~np.isnan(r) & ~np.isnan(s) & p_ok
    return r[good], idx[a[good]], tt[good], s[good]


r0, d0, t0_, s0 = ev_cell(0)
m0 = s0 <= 10.0

print("\n" + "=" * 74)
print("1. CLEAN REFERENCE CLASS (price >= $5, returns winsorised at +/-50%)")
print("=" * 74)
off = WASH & ~NEAR & ~np.isnan(FWD) & PXOK
allc = ~np.isnan(FWD) & PXOK
show([summarize(r0[m0], f"washout, {K} td BEFORE a print (N={int(m0.sum())})"),
      summarize(FWD[off], f"washout, no print anywhere near (N={int(off.sum())})"),
      summarize(FWD[allc], f"every day, every name (N={int(allc.sum())})")],
     f"h={H} long. does the PRINT add anything to the washout?")
print(f"  print premium over the same washout away from prints = "
      f"{100*(np.nanmean(r0[m0]) - np.nanmean(FWD[off])):+.3f}pp")
print(f"  washout premium over all days = "
      f"{100*(np.nanmean(FWD[off]) - np.nanmean(FWD[allc])):+.3f}pp")

print("\n" + "=" * 74)
print("2. PLACEBO ANCHOR LADDER — shift the print date back, keep everything")
print("=" * 74)
lad = []
for sh in range(-14, 1):
    r, d, t, s = ev_cell(sh)
    m = s <= 10.0
    if m.sum() < 50:
        continue
    row = summarize(r[m], f"print shifted {sh:+d} td" + ("  <-- TRUE" if sh == 0 else ""))
    dfc = pd.DataFrame({"d": d[m], "r": r[m]}).groupby("d")["r"].mean()
    row["date_clustered_pct"] = round(100 * dfc.mean(), 3)
    row["n_dates"] = len(dfc)
    lad.append(row)
show(lad, "ladder (day-level and date-clustered)")
real = [r for r in lad if r["label"].startswith("print shifted +0")][0]
rank = 1 + sum(1 for r in lad if r["mean_pct"] > real["mean_pct"])
rankc = 1 + sum(1 for r in lad if r["date_clustered_pct"] > real["date_clustered_pct"])
print(f"  TRUE anchor ranks {rank} of {len(lad)} on day-level mean, "
      f"{rankc} of {len(lad)} date-clustered. PLATEAU = the print is decoration.")

print("\n" + "=" * 74)
print("3. DEFINITION NEIGHBOURS")
print("=" * 74)
rows = []
for lbl, mm in [("rank5<=10 (pitched)", s0 <= 10),
                ("rank5<=5", s0 <= 5),
                ("rank5<=20", s0 <= 20),
                ("raw 5d ret <= -5%", None),
                ("raw 5d ret <= -8%", None)]:
    if mm is None:
        continue
    rows.append(summarize(r0[mm], lbl))
r5raw = close.pct_change(5, fill_method=None).values
a_all = EV_P - K
okk = (a_all >= 300) & (a_all < len(idx)) & (a_all + 1 + H < len(idx))
a_all, t_all = a_all[okk], EV_T[okk]
rr = FWD[a_all, t_all]
raw = r5raw[a_all, t_all]
pk = PXOK[a_all, t_all]
g = ~np.isnan(rr) & ~np.isnan(raw) & pk
for thr in (-0.05, -0.08, -0.12):
    rows.append(summarize(rr[g & (raw <= thr)], f"raw 5d ret <= {100*thr:.0f}%"))
show(rows, f"neighbours at k={K}, h={H}")

print("\n" + "=" * 74)
print("4. WHERE IT LIVES — price bucket (the survivorship / distress channel)")
print("=" * 74)
pxv = C[EV_P - K, EV_T]
pxv = pxv[okk]
rows = []
for lo, hi in [(5, 15), (15, 40), (40, 100), (100, 1e9)]:
    sel = g & (raw <= raw[g & (RANK5[a_all, t_all] <= 10)].max()) if False else None
    sel = (RANK5[a_all, t_all] <= 10) & (pxv >= lo) & (pxv < hi) & ~np.isnan(rr)
    rows.append(summarize(rr[sel], f"anchor price ${lo}-{hi if hi < 1e9 else '+'} (N={int(sel.sum())})"))
liq = np.array([cols[j] in sc.LIQUID_PLUS_COMMODITIES for j in t_all])
rows.append(summarize(rr[(RANK5[a_all, t_all] <= 10) & liq],
                      f"LIQUID_PLUS_COMMODITIES names only "
                      f"(N={int(((RANK5[a_all,t_all]<=10)&liq).sum())})"))
show(rows, "buckets")
print("  NOTE: master_prices holds TODAY'S universe. Every delisted loser is "
      "absent, which flatters a buy-the-washout cell by construction "
      "(CLAUDE.md ledger survivorship caveat).")

print("\n" + "=" * 74)
print("5. THE SEARCH — retail x August, and what it costs")
print("=" * 74)
RET = {"TJX", "ROST", "TGT", "WMT", "HD", "LOW", "DG", "DLTR", "BURL", "M", "KSS",
       "JWN", "BBY", "GPS", "ANF", "AEO", "URBN", "COST"}
is_ret = np.array([cols[j] in RET for j in t0_])
is_aug = pd.DatetimeIndex(d0).month == 8
cellsel = m0 & is_ret & is_aug
sub = r0[cellsel]
ww = int((sub > 0).sum())
p_raw = sign_test(ww, len(sub))
n_searched = 9 * 6 + 4 + 12      # k scan x threshold scan + subsets + months
print(f"  retail x August washout: N={len(sub)}  mean {100*sub.mean():+.3f}%  "
      f"record {ww}-{len(sub)-ww}  sign p = {p_raw:.4f}")
print(f"  cells this search touched (k x threshold x subset x month) ~ {n_searched}; "
      f"Bonferroni-adjusted p = {min(1.0, p_raw*n_searched):.3f}")
show([summarize(r0[m0 & is_ret], "retail, any month"),
      summarize(r0[m0 & is_aug], "August, any industry"),
      summarize(r0[m0 & ~is_ret & ~is_aug], "neither")], "decomposition")

print("\n" + "=" * 74)
print("6. COST")
print("=" * 74)
mm = np.nanmean(r0[m0])
for rt in (10, 20, 30):
    print(f"  single-name round trip {rt} bps -> {10000*mm/rt:.1f}x "
          f"(cell mean {100*mm:.3f}% = {10000*mm:.1f} bps)")
prem = 10000 * (np.nanmean(r0[m0]) - np.nanmean(FWD[off]))
for rt in (10, 20, 30):
    print(f"  ... and on the PRINT PREMIUM alone ({prem:.1f} bps): {prem/rt:.1f}x")
print(f"\n({time.time()-T0:.0f}s)")
