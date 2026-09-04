"""C3 — a washed-out name into its own scheduled print, EXITING BEFORE it.

New lane: the product has never anchored on an earnings date, so the
machinery is built and validated here before any cell is scored.

Definitions (fixed once, then scanned):
  E  = scheduled print date (data/earnings_calendar.parquet), pE = its td pos.
  A  = anchor, pE - k  (k = sessions before the print, SCANNED)
  signal at A: the name's 5d return is in the bottom decile of its OWN
       trailing-252d distribution (rank5 <= 10)
  entry = MOC at A+1 (lag=1 convention), exit = MOC at pE-1, the last close
       before the print lands in EITHER a BMO or an AMC world.  h = k-2.

Stage 0 validates that `date` really is an announcement date (the whole lane
is void otherwise). Stage 2 is the gate-attribution question the brief calls
mandatory: run the SAME anchor WITHOUT the washout gate, and run the SAME
washout gate AWAY from prints.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pitch_lab import PRICES_PATH, show, sign_test, summarize  # noqa: E402

T0 = time.time()
ASOF = pd.Timestamp("2026-08-13")
START = pd.Timestamp("1999-01-01")   # announcement-date era (see c0b probe)

earn = pd.read_parquet("data/earnings_calendar.parquet", columns=["ticker", "date"])
earn["date"] = pd.to_datetime(earn["date"])
tickers = sorted(earn["ticker"].unique())

mp = pd.read_parquet(PRICES_PATH, columns=["ticker", "date", "Open", "Close"])
mp = mp[mp["ticker"].isin(tickers)]
mp["date"] = pd.to_datetime(mp["date"])
mp = mp[(mp["date"] >= START) & (mp["date"] <= ASOF)]
close = mp.pivot_table(index="date", columns="ticker", values="Close", aggfunc="last")
open_ = mp.pivot_table(index="date", columns="ticker", values="Open", aggfunc="last")
close = close.sort_index()
open_ = open_.reindex(index=close.index, columns=close.columns)
idx = close.index
cols = list(close.columns)
colpos = {t: i for i, t in enumerate(cols)}
print(f"panel {close.shape} {idx[0].date()}..{idx[-1].date()}  ({time.time()-T0:.0f}s)")

C = close.values
O = open_.values
r5 = close.pct_change(5)
RANK5 = (r5.rolling(252, min_periods=252).rank(pct=True) * 100).values
print(f"rank5 built ({time.time()-T0:.0f}s)")

# ---- event table: (ticker col, td position of print)
earn = earn[(earn["date"] >= idx[0]) & (earn["date"] <= idx[-1] + pd.Timedelta(days=10))]
ev_t, ev_p, ev_d = [], [], []
ipos = idx.values
for t, g in earn.groupby("ticker"):
    j = colpos.get(t)
    if j is None:
        continue
    p = np.searchsorted(ipos, g["date"].values, side="left")
    ok = (p > 0) & (p < len(idx))
    ev_t.append(np.full(ok.sum(), j))
    ev_p.append(p[ok])
    ev_d.append(g["date"].values[ok])
EV_T = np.concatenate(ev_t)
EV_P = np.concatenate(ev_p)
EV_D = np.concatenate(ev_d)
print(f"events mapped: {len(EV_P)}  ({time.time()-T0:.0f}s)")

# ============================================================== 0. MACHINERY
print("\n" + "=" * 74)
print("0. IS THE ANCHOR REAL? |move| on the print day vs a baseline day")
print("=" * 74)
ret1 = np.abs(C[1:] / C[:-1] - 1.0)          # |1d ret| at position i+1
base = np.nanmedian(ret1)
for lbl, off in [("E-2", -2), ("E-1", -1), ("E (BMO lands here)", 0), ("E+1 (AMC lands here)", 1),
                 ("E+2", 2)]:
    p = EV_P + off
    ok = (p >= 1) & (p < len(idx))
    v = ret1[p[ok] - 1, EV_T[ok]]
    v = v[~np.isnan(v)]
    print(f"  {lbl:22s} median |ret| {100*np.median(v):.3f}%  mean {100*v.mean():.3f}%  "
          f"ratio to all-day median {np.median(v)/base:.2f}x   N={len(v)}")
print(f"  all-day median |ret| = {100*base:.3f}%")

# ================================================== 1. THE CELL, k SCANNED
print("\n" + "=" * 74)
print("1. THE CELL, k SCANNED. long, entry A+1 MOC, exit at pE-1 (before print)")
print("=" * 74)


def cell(k: int, thr: float = 10.0):
    """returns (event_ret, event_date, event_tickercol, all_ret_same_anchor)"""
    h = k - 2
    if h < 1:
        return None
    pa, pe, px_ = EV_P - k, EV_P - k + 1, EV_P - 1
    ok = (pa >= 300) & (px_ < len(idx)) & (pe < px_)
    pa, pe, px_, tt = pa[ok], pe[ok], px_[ok], EV_T[ok]
    dd = idx[pa]
    r = C[px_, tt] / C[pe, tt] - 1.0
    s = RANK5[pa, tt]
    good = ~np.isnan(r) & ~np.isnan(s)
    return r[good], dd[good], tt[good], s[good], h


rows, gate_rows = [], []
for k in range(3, 12):
    out = cell(k)
    if out is None:
        continue
    r, d, t, s, h = out
    m = s <= 10.0
    rows.append({**summarize(r[m], f"k={k} (h={h}) washout"), "k": k, "h": h})
    gate_rows.append({**summarize(r, f"k={k} (h={h}) ALL pre-print, gate OFF"),
                      "k": k, "h": h})
show(rows, "washout cell by k")
show(gate_rows, "2. GATE ATTRIBUTION: the same anchor with NO washout gate")
print("  read the two tables together: if the gate row and the no-gate row "
      "agree, the washout filter does not filter.")

# =========================== pick the k that matches today and go deeper
K = 4          # TJX: anchor 2026-08-13, print 2026-08-19 -> 4 td; h = 2
r, d, t, s, h = cell(K)
m = s <= 10.0
print("\n" + "=" * 74)
print(f"3. DEPTH AT k={K} (h={h}) — TODAY'S GEOMETRY (TJX prints 2026-08-19)")
print("=" * 74)
show([summarize(r[m], f"washout rank5<=10 (N={int(m.sum())})"),
      summarize(r[(s > 10) & (s <= 30)], "rank5 10-30"),
      summarize(r[(s > 30) & (s <= 70)], "rank5 30-70 (middle)"),
      summarize(r[s > 90], "rank5 >90 (strong into the print)"),
      summarize(r, "ALL pre-print anchors (gate OFF)")],
     f"dose response in the washout variable, k={K}")

# threshold sensitivity
show([summarize(r[s <= th], f"rank5<={th:g}") for th in (2, 5, 10, 15, 20, 30)],
     "threshold sensitivity")

# ---- CROSS-SECTIONAL CLUSTERING: dates are the real unit
print("\n  cross-sectional clustering — a bad week washes out MANY names at once")
dfc = pd.DataFrame({"d": d[m], "r": r[m]})
per_date = dfc.groupby("d")["r"].mean()
print(f"    event-level N={int(m.sum())} collapses to {len(per_date)} distinct anchor "
      f"dates; max names on one date = {dfc.groupby('d').size().max()}")
show([summarize(per_date.values, f"date-clustered equal-weight (N={len(per_date)})")],
     "the honest unit")
w = int((per_date.values > 0).sum())
print(f"    date record {w}-{len(per_date)-w}, sign p = {sign_test(w, len(per_date)):.4f}")

# ---- era / month / cycle
dd = pd.DatetimeIndex(per_date.index)
v = per_date.values
show([summarize(v[dd < pd.Timestamp("2010-01-01")], "pre-2010"),
      summarize(v[(dd >= pd.Timestamp("2010-01-01")) & (dd < pd.Timestamp("2018-01-01"))], "2010-2017"),
      summarize(v[dd >= pd.Timestamp("2018-01-01")], "2018+"),
      summarize(v[(dd.year % 4) == 2], "midterm years"),
      summarize(v[dd.month == 8], "August only")],
     "era / cycle / month splits (date-clustered)")

# =============================================== 4. REFERENCE CLASS
print("\n" + "=" * 74)
print("4. REFERENCE CLASS — is it the WASHOUT or the PRINT?")
print("=" * 74)
# same washout gate, same h, on days NOT in a pre-print window
near = np.zeros(C.shape, dtype=bool)
for j, p in zip(EV_T, EV_P):
    near[max(0, p - 15):min(len(idx), p + 6), j] = True
sig_all = RANK5 <= 10.0
fwd_h = np.full(C.shape, np.nan)
fwd_h[:-(1 + h)] = C[1 + h:] / C[1:-h] - 1.0     # entry at i+1, exit at i+1+h
mask_off = sig_all & ~near & ~np.isnan(fwd_h)
mask_on = sig_all & near & ~np.isnan(fwd_h)
allday = ~np.isnan(fwd_h)
show([summarize(r[m], f"washout k={K}, pre-print (N={int(m.sum())})"),
      summarize(fwd_h[mask_off], f"washout AWAY from any print (N={int(mask_off.sum())})"),
      summarize(fwd_h[allday], f"ALL days, all names, h={h} (N={int(allday.sum())})")],
     f"the same washout gate with and without a print ahead of it, h={h}")

# =============================================== 5. RETAIL / AUGUST SUBSET
print("\n" + "=" * 74)
print("5. THE SUBSET THE IDEA CAME FROM (charge yourself for this search)")
print("=" * 74)
RET = ["TJX", "ROST", "TGT", "WMT", "HD", "LOW", "DG", "DLTR", "BURL", "M", "KSS",
       "JWN", "BBY", "GPS", "ANF", "AEO", "URBN", "COST", "TGT"]
retcols = {colpos[x] for x in RET if x in colpos}
is_ret = np.array([x in retcols for x in t])
is_aug = pd.DatetimeIndex(d).month == 8
show([summarize(r[m & is_ret], f"retail names, washout (N={int((m&is_ret).sum())})"),
      summarize(r[m & is_aug], f"August, washout (N={int((m&is_aug).sum())})"),
      summarize(r[m & is_ret & is_aug], f"retail AND August (N={int((m&is_ret&is_aug).sum())})"),
      summarize(r[m & ~is_ret], "everything else, washout")],
     f"subsets at k={K}")
sub = r[m & is_ret & is_aug]
if len(sub):
    ww = int((sub > 0).sum())
    print(f"  retail+August record {ww}-{len(sub)-ww}, sign p = {sign_test(ww, len(sub)):.4f}")

# =============================================== 6. WHERE DOES IT ACCRUE?
print("\n" + "=" * 74)
print("6. GAP SHARE — does a pre-print drift accrue overnight or in-session?")
print("=" * 74)
pa, pe, px_ = EV_P - K, EV_P - K + 1, EV_P - 1
ok = (pa >= 300) & (px_ < len(idx)) & (pe < px_)
pa, pe, px_, tt = pa[ok], pe[ok], px_[ok], EV_T[ok]
sg = RANK5[pa, tt]
sel = (sg <= 10.0)
gap_tot = np.zeros(sel.sum())
ins_tot = np.zeros(sel.sum())
pe_s, px_s, tt_s = pe[sel], px_[sel], tt[sel]
for n, (a_, b_, j) in enumerate(zip(pe_s, px_s, tt_s)):
    cs = C[a_:b_ + 1, j]
    os_ = O[a_ + 1:b_ + 1, j]
    if np.isnan(cs).any() or np.isnan(os_).any():
        gap_tot[n] = ins_tot[n] = np.nan
        continue
    gap_tot[n] = np.sum(os_ / cs[:-1] - 1.0)
    ins_tot[n] = np.sum(cs[1:] / os_ - 1.0)
gm = np.nanmean(gap_tot)
im = np.nanmean(ins_tot)
print(f"  overnight (gap) component  = {100*gm:+.3f}%")
print(f"  in-session component       = {100*im:+.3f}%")
print(f"  total (approx, additive)   = {100*(gm+im):+.3f}%   "
      f"gap share = {100*gm/(gm+im) if (gm+im)!=0 else float('nan'):.0f}%")
print(f"\n({time.time()-T0:.0f}s)")
