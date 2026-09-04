"""Adversarial verification of the proxy-falsification study.

Independent recompute from raw inputs (does not reuse researcher code or
their cached joined frame). Claims verified:
 1. frag>=50 split baseline (monthly-clustered)
 2. best simple proxy: rv21 3y-pctile calm tail, matched-N
 3. vix_ma10 low-tail matched-N + tuned high-VIX sweep
 4. rank correlations frag vs proxies; days_since_5dd has no R signal
 5. frag survives union control (any of 6 proxies in aligned tail)
 6. frag LOYO within rv-calm stratum
 7. toxic cell frag>=50 & rv-elevated
 8. dist200<=-5% zero overlap with frag>=50; excl -> frag t=-3.20
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

ROOT = r"C:\Users\McKinley Slade\dev\New_Seasonals"

# ---------------------------------------------------------------- proxies
px = pd.read_parquet(ROOT + r"\data\master_prices.parquet",
                     filters=[("ticker", "in", ["SPY", "^VIX"])])
px["date"] = pd.to_datetime(px["date"])
spy = (px[px.ticker == "SPY"].set_index("date")["Close"].sort_index())
vix = (px[px.ticker == "^VIX"].set_index("date")["Close"].sort_index())

proxy = pd.DataFrame(index=spy.index)
proxy["vix"] = vix.reindex(spy.index)
proxy["vix_ma10"] = proxy["vix"].rolling(10).mean()
proxy["dist200"] = spy / spy.rolling(200).mean() - 1.0
ret = spy.pct_change()
rv21 = ret.rolling(21).std() * np.sqrt(252)
proxy["rv21_pct"] = rv21.rolling(756).rank(pct=True)
roll_high = spy.rolling(252).max()
proxy["dd252"] = spy / roll_high - 1.0
# trading days since SPY last closed >=5% below its 252d high
in_dd = proxy["dd252"] <= -0.05
idx_arr = np.arange(len(proxy))
last_dd = pd.Series(np.where(in_dd, idx_arr, np.nan), index=proxy.index).ffill()
proxy["days_since_5dd"] = idx_arr - last_dd  # NaN if never yet in dd

PROXIES = ["vix", "vix_ma10", "dist200", "rv21_pct", "dd252", "days_since_5dd"]
# direction correlated with HIGH fragility (frag-aligned tail):
#   low vix, low vix_ma10, high dist200, low rv, shallow (high) dd, long days
ALIGN_HIGH = {"vix": False, "vix_ma10": False, "dist200": True,
              "rv21_pct": False, "dd252": True, "days_since_5dd": True}

# ---------------------------------------------------------------- frag basis
frag = pd.read_parquet(ROOT + r"\data\rd2_fragility.parquet")
frag_live = frag["63d"].rolling(10, min_periods=1).mean()

# ---------------------------------------------------------------- trades
tr = pd.read_parquet(ROOT + r"\data\backtest_trades_full.parquet")
tr = tr[tr["Strategy"] != "Overbot Vol Spike"].copy()
tr["Signal Date"] = pd.to_datetime(tr["Signal Date"])
tr = tr[tr["Signal Date"] >= "2016-08-01"].sort_values("Signal Date")

fr = frag_live.reset_index()
fr.columns = ["Date", "frag"]
tr = pd.merge_asof(tr, fr, left_on="Signal Date", right_on="Date",
                   direction="backward", tolerance=pd.Timedelta(days=5))
pxr = proxy.reset_index().rename(columns={proxy.index.name or "index": "Date"})
tr = pd.merge_asof(tr, pxr, left_on="Signal Date", right_on="Date",
                   direction="backward", tolerance=pd.Timedelta(days=5),
                   suffixes=("", "_px"))
n_before = len(tr)
tr = tr.dropna(subset=["frag", "R_Multiple"] + PROXIES).copy()
print(f"non-OVS trades 2016-08+: {n_before}; after dropping missing: {len(tr)}")

tr["month"] = tr["Signal Date"].dt.to_period("M")
tr["year"] = tr["Signal Date"].dt.year
R = "R_Multiple"


def clus_t(df: pd.DataFrame, mask_hi: pd.Series) -> dict:
    """Monthly-clustered Welch t: per-month mean R within each group."""
    hi = df[mask_hi]
    lo = df[~mask_hi]
    mh = hi.groupby("month")[R].mean()
    ml = lo.groupby("month")[R].mean()
    if len(mh) < 2 or len(ml) < 2:
        return dict(n_hi=len(hi), n_lo=len(lo), avg_hi=hi[R].mean(),
                    avg_lo=lo[R].mean(), t=np.nan, p=np.nan,
                    m_hi=len(mh), m_lo=len(ml))
    t, p = stats.ttest_ind(mh, ml, equal_var=False)
    return dict(n_hi=len(hi), n_lo=len(lo), avg_hi=hi[R].mean(),
                avg_lo=lo[R].mean(), t=t, p=p, m_hi=len(mh), m_lo=len(ml))


def fmt(tag: str, r: dict) -> None:
    print(f"{tag:52s} N={r['n_hi']:4d}/{r['n_lo']:4d} "
          f"avgR={r['avg_hi']:+.3f}/{r['avg_lo']:+.3f} "
          f"t={r['t']:+.2f} p={r['p']:.3f} (months {r['m_hi']}/{r['m_lo']})")


# ============================================================ CLAIM 1
print("\n=== Claim 1: frag>=50 baseline ===")
frag_hi = tr["frag"] >= 50
fmt("frag>=50 vs <50", clus_t(tr, frag_hi))
frac_hi = frag_hi.mean()
print(f"frag>=50 share: {frac_hi:.3f}")

# matched-N thresholds per proxy (aligned tail, same share as frag>=50)
tails = {}
for p_ in PROXIES:
    if ALIGN_HIGH[p_]:
        thr = tr[p_].quantile(1 - frac_hi)
        tails[p_] = tr[p_] >= thr
    else:
        thr = tr[p_].quantile(frac_hi)
        tails[p_] = tr[p_] <= thr
    tails[p_ + "_thr"] = thr

# ============================================================ CLAIMS 2/3/4b: solo proxies
print("\n=== Claims 2-4: solo proxy matched-N aligned tails ===")
for p_ in PROXIES:
    r = clus_t(tr, tails[p_])
    fmt(f"{p_} aligned tail (thr={tails[p_ + '_thr']:.3f})", r)

print("\nTuned high-VIX-MA sweep (vix_ma10 >= X):")
for x in [15, 17, 19, 21, 23, 25, 28, 30]:
    m = tr["vix_ma10"] >= x
    if m.sum() < 20:
        continue
    r = clus_t(tr, m)
    fmt(f"vix_ma10>={x}", r)

# ============================================================ CLAIM 4: rank corrs
print("\n=== Claim 4: Spearman rho(frag, proxy) at trade signal dates ===")
for p_ in PROXIES:
    rho, pv = stats.spearmanr(tr["frag"], tr[p_])
    print(f"  {p_:16s} rho={rho:+.3f} (p={pv:.3g})")

# ============================================================ CLAIM 5: union control
print("\n=== Claim 5: frag within any-proxy-aligned-tail union ===")
union = np.zeros(len(tr), dtype=bool)
for p_ in PROXIES:
    union |= tails[p_].to_numpy()
union = pd.Series(union, index=tr.index)
print(f"union share: {union.mean():.3f}")
fmt("within union-HIGH: frag>=50 vs <50", clus_t(tr[union], frag_hi[union]))
fmt("within union-LOW:  frag>=50 vs <50", clus_t(tr[~union], frag_hi[~union]))

# ============================================================ CLAIM 7: toxic 2x2 (rv elevated = high tail matched-N)
print("\n=== Claim 7: 2x2 frag x rv21_pct (elevated = top tail matched-N) ===")
rv_hi_thr = tr["rv21_pct"].quantile(1 - frac_hi)
rv_elev = tr["rv21_pct"] >= rv_hi_thr
print(f"rv elevated threshold: {rv_hi_thr:.3f}, N elevated={rv_elev.sum()}")
for f_, r_, tag in [(False, False, "frag<50, rv-calm"),
                    (False, True, "frag<50, rv-elev"),
                    (True, False, "frag>=50, rv-calm"),
                    (True, True, "frag>=50, rv-elev")]:
    cell = tr[(frag_hi == f_) & (rv_elev == r_)]
    win = (cell[R] > 0).mean() * 100
    print(f"  {tag:20s} N={len(cell):4d} avgR={cell[R].mean():+.3f} "
          f"med={cell[R].median():+.3f} win={win:.0f}% "
          f"months={cell['month'].nunique()}")
tox = tr[frag_hi & rv_elev]
print("  toxic cell year counts:", tox["year"].value_counts().to_dict())
sub = tr[rv_elev]
fmt("within rv-ELEV: frag split", clus_t(sub, frag_hi[rv_elev]))
# Spearman(frag,R) within strata
rho_e, pe = stats.spearmanr(sub["frag"], sub[R])
sub_c = tr[~rv_elev]
rho_c, pc = stats.spearmanr(sub_c["frag"], sub_c[R])
print(f"  Spearman(frag,R) rv-elev: {rho_e:+.3f} (p={pe:.3f}); "
      f"rv-calm: {rho_c:+.3f} (p={pc:.3f})")
# LOYO in rv-elev dropping 2021 / 2024
for yr in [2021, 2024]:
    s2 = sub[sub["year"] != yr]
    fmt(f"  rv-ELEV frag split, drop {yr}", clus_t(s2, s2["frag"] >= 50))

# ============================================================ CLAIM 6: LOYO within rv-calm
print("\n=== Claim 6: LOYO frag split within rv-calm stratum ===")
calm = tr[~rv_elev]
fmt("rv-calm all years", clus_t(calm, calm["frag"] >= 50))
for yr in sorted(calm["year"].unique()):
    s2 = calm[calm["year"] != yr]
    fmt(f"  drop {yr}", clus_t(s2, s2["frag"] >= 50))

# ============================================================ CLAIM 8: dist200 <= -5%
print("\n=== Claim 8: deep-correction rule dist200<=-5% ===")
deep = tr["dist200"] <= -0.05
print(f"N deep={deep.sum()}, overlap with frag>=50: {(deep & frag_hi).sum()}")
fmt("deep vs rest", clus_t(tr, deep))
excl = tr[~deep]
fmt("frag split excluding deep trades", clus_t(excl, excl["frag"] >= 50))

# ============================================================ extra skeptic checks
print("\n=== Skeptic extras ===")
# double sorts: frag within each proxy stratum (12 cells direction check)
same_dir = 0
cells = 0
for p_ in PROXIES:
    for side, m in [("HI", tails[p_]), ("LO", ~tails[p_])]:
        s2 = tr[m]
        r = clus_t(s2, s2["frag"] >= 50)
        cells += 1
        d = r["avg_hi"] - r["avg_lo"]
        if d < 0:
            same_dir += 1
        print(f"  within {p_}-{side}: frag gap {d:+.3f} t={r['t']:+.2f} "
              f"p={r['p']:.3f} (N_hi={r['n_hi']})")
print(f"cells with frag pointing negative: {same_dir}/{cells}")

# episode concentration of frag>=50
hi_months = sorted(str(m) for m in tr.loc[frag_hi, "month"].unique())
print(f"\nfrag>=50 distinct months: {len(hi_months)}")
print(hi_months)
