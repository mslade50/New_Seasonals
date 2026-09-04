"""Is the fragility score just a proxy for something simpler?

Builds five simple point-in-time market-state variables from master_prices
(SPY, ^VIX), joins them to non-OVS trades at signal date alongside the live
fragility basis (63d score, 10d MA, as-of ffill<=5d), and tests:
  1. Does each simple proxy's own top tail reproduce the frag>=50 R-degradation?
  2. Rank correlation between frag and each proxy at signal dates.
  3. Double-sorts: does frag add discrimination after controlling for the best
     proxy, and vice versa?
All significance via monthly-clustered t-tests (never raw per-trade).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------- market data
mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                     columns=["ticker", "date", "Close"])
spy = (mp[mp.ticker == "SPY"].set_index("date")["Close"].sort_index())
vix = (mp[mp.ticker == "^VIX"].set_index("date")["Close"].sort_index())
spy.index = pd.to_datetime(spy.index).normalize()
vix.index = pd.to_datetime(vix.index).normalize()
spy = spy[~spy.index.duplicated(keep="last")]
vix = vix[~vix.index.duplicated(keep="last")]
print(f"SPY {spy.index.min().date()}..{spy.index.max().date()} n={len(spy)}; "
      f"VIX {vix.index.min().date()}..{vix.index.max().date()} n={len(vix)}")

# ------------------------------------------------------------ simple proxies
px = pd.DataFrame(index=spy.index)
px["vix"] = vix.reindex(spy.index).ffill(limit=5)
px["vix_ma10"] = px["vix"].rolling(10, min_periods=5).mean()

sma200 = spy.rolling(200).mean()
px["dist200"] = spy / sma200 - 1.0          # + = extended above, - = below

logret = np.log(spy / spy.shift(1))
rv21 = logret.rolling(21).std() * np.sqrt(252)
px["rv21_pct"] = rv21.rolling(756, min_periods=252).rank(pct=True)  # 3y pctile

roll_hi = spy.rolling(252, min_periods=60).max()
px["dd252"] = spy / roll_hi - 1.0           # <= 0, deeper = more negative

in_dd5 = px["dd252"] <= -0.05
# days since last close in a >=5% drawdown (point-in-time)
last_dd = pd.Series(np.where(in_dd5, np.arange(len(px)), np.nan),
                    index=px.index).ffill()
px["days_since_5dd"] = np.arange(len(px)) - last_dd

PROXIES = ["vix", "vix_ma10", "dist200", "rv21_pct", "dd252", "days_since_5dd"]

# ------------------------------------------------------------------ fragility
frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
frag_ma = frag["63d"].dropna().rolling(10, min_periods=1).mean()
frag_ma.index = pd.to_datetime(frag_ma.index).normalize()

# ------------------------------------------------------------------- trades
trades = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
trades["Signal Date"] = pd.to_datetime(trades["Signal Date"]).dt.normalize()
start = frag_ma.index.min() + pd.Timedelta(days=20)
t = trades[trades["Signal Date"] >= start].copy().sort_values("Signal Date")

t["frag"] = pd.merge_asof(
    t[["Signal Date"]], frag_ma.rename("frag").reset_index(),
    left_on="Signal Date", right_on="Date",
    tolerance=pd.Timedelta(days=5),
)["frag"].values

pxr = px.reset_index().rename(columns={"index": "date"})
pxr.columns = ["date"] + PROXIES
t = pd.merge_asof(t, pxr, left_on="Signal Date", right_on="date",
                  tolerance=pd.Timedelta(days=5))

is_ovs = t["Strategy"].str.contains("Overbot Vol|OVS", case=False, na=False)
nb = t[~is_ovs].dropna(subset=["frag", "R_Multiple"] + PROXIES).copy()
nb["ym"] = nb["Signal Date"].dt.to_period("M")
nb["yr"] = nb["Signal Date"].dt.year
print(f"\nnon-OVS trades joined: N={len(nb)} "
      f"({nb['Signal Date'].min().date()}..{nb['Signal Date'].max().date()})")

frag_hi = nb.frag >= 50
frac_hi = frag_hi.mean()
print(f"frag>=50: N={frag_hi.sum()} ({frac_hi:.1%} of trades)  "
      f"avgR {nb.loc[frag_hi,'R_Multiple'].mean():+.3f} vs "
      f"{nb.loc[~frag_hi,'R_Multiple'].mean():+.3f} below")


def clustered_t(df_hi: pd.DataFrame, df_lo: pd.DataFrame):
    hm = df_hi.groupby("ym")["R_Multiple"].mean()
    lm = df_lo.groupby("ym")["R_Multiple"].mean()
    if len(hm) < 3 or len(lm) < 3:
        return np.nan, np.nan, hm, lm
    tt = stats.ttest_ind(hm, lm, equal_var=False)
    return tt.statistic, tt.pvalue, hm, lm


# ------------------------------------------------ 1. rank correlations w/ frag
print("\n=== Spearman rank correlation with frag (at non-OVS signal dates) ===")
corr_rows = []
for pxn in PROXIES:
    rho, p = stats.spearmanr(nb.frag, nb[pxn])
    corr_rows.append({"proxy": pxn, "rho_vs_frag": round(rho, 3), "p": f"{p:.1e}"})
print(pd.DataFrame(corr_rows).to_string(index=False))

# daily-basis correlation too (not trade-weighted)
daily = px.join(frag_ma.rename("frag"), how="inner").dropna()
print("\ndaily-basis Spearman with frag (2016-07..):")
for pxn in PROXIES:
    rho, _ = stats.spearmanr(daily.frag, daily[pxn])
    print(f"  {pxn:>15}: {rho:+.3f}")

# --------------------------------------- 2. per-proxy conditioning (solo test)
# direction aligned with high fragility = sign of trade-level Spearman
DIR = {r["proxy"]: np.sign(stats.spearmanr(nb.frag, nb[r["proxy"]])[0])
       for r in corr_rows}

print("\n=== Solo conditioning: does each proxy's own high tail reproduce the "
      "split? ===")
print(f"(threshold A = matched-N: top {frac_hi:.1%} of trade-date values in "
      f"frag-aligned direction; threshold B = top decile)")
solo_rows = []
masks = {}
for pxn in PROXIES:
    v = nb[pxn] * DIR[pxn]          # flip so high = frag-aligned
    for lab, q in [("matchN", 1 - frac_hi), ("top10%", 0.90)]:
        thr = v.quantile(q)
        hi = v >= thr
        tstat, pval, hm, lm = clustered_t(nb[hi], nb[~hi])
        solo_rows.append({
            "proxy": pxn, "thr_def": lab,
            "dir": "+" if DIR[pxn] > 0 else "-",
            "cut_raw": round(thr * DIR[pxn], 3),
            "N_hi": int(hi.sum()),
            "avgR_hi": round(nb.loc[hi, "R_Multiple"].mean(), 3),
            "avgR_lo": round(nb.loc[~hi, "R_Multiple"].mean(), 3),
            "t_clust": round(tstat, 2), "p": round(pval, 3),
            "mo_hi": len(hm),
        })
        if lab == "matchN":
            masks[pxn] = hi
# frag itself for reference
tstat, pval, hm, lm = clustered_t(nb[frag_hi], nb[~frag_hi])
solo_rows.append({"proxy": "FRAG (63d 10dMA)", "thr_def": ">=50", "dir": "+",
                  "cut_raw": 50, "N_hi": int(frag_hi.sum()),
                  "avgR_hi": round(nb.loc[frag_hi, "R_Multiple"].mean(), 3),
                  "avgR_lo": round(nb.loc[~frag_hi, "R_Multiple"].mean(), 3),
                  "t_clust": round(tstat, 2), "p": round(pval, 3),
                  "mo_hi": len(hm)})
solo = pd.DataFrame(solo_rows)
print(solo.to_string(index=False))

# ------------------------------------------------------- 3. double sorts
# best proxy = most negative clustered t at matched-N
best = (solo[(solo.thr_def == "matchN")]
        .sort_values("t_clust").iloc[0]["proxy"])
print(f"\nbest solo proxy (matched-N): {best}")

print("\n=== Double sorts (matched-N proxy threshold), monthly-clustered ===")
ds_rows = []
for pxn in PROXIES:
    phi = masks[pxn]
    # (a) frag discrimination WITHIN proxy-high and WITHIN proxy-low
    for cond, m in [("proxyHI", phi), ("proxyLO", ~phi)]:
        sub = nb[m]
        f_hi = sub.frag >= 50
        tstat, pval, hm, lm = clustered_t(sub[f_hi], sub[~f_hi])
        ds_rows.append({
            "proxy": pxn, "within": cond, "split": "frag>=50 vs <50",
            "N_hi": int(f_hi.sum()), "N_lo": int((~f_hi).sum()),
            "avgR_hi": round(sub.loc[f_hi, "R_Multiple"].mean(), 3) if f_hi.sum() else np.nan,
            "avgR_lo": round(sub.loc[~f_hi, "R_Multiple"].mean(), 3) if (~f_hi).sum() else np.nan,
            "t_clust": round(tstat, 2) if tstat == tstat else np.nan,
            "p": round(pval, 3) if pval == pval else np.nan,
        })
    # (b) proxy discrimination WITHIN frag-high and WITHIN frag-low
    for cond, m in [("fragHI", frag_hi), ("fragLO", ~frag_hi)]:
        sub, psub = nb[m], phi[m]
        tstat, pval, hm, lm = clustered_t(sub[psub], sub[~psub])
        ds_rows.append({
            "proxy": pxn, "within": cond, "split": "proxyHI vs LO",
            "N_hi": int(psub.sum()), "N_lo": int((~psub).sum()),
            "avgR_hi": round(sub.loc[psub, "R_Multiple"].mean(), 3) if psub.sum() else np.nan,
            "avgR_lo": round(sub.loc[~psub, "R_Multiple"].mean(), 3) if (~psub).sum() else np.nan,
            "t_clust": round(tstat, 2) if tstat == tstat else np.nan,
            "p": round(pval, 3) if pval == pval else np.nan,
        })
ds = pd.DataFrame(ds_rows)
print(ds.to_string(index=False))

# 2x2 cell table for the best proxy
print(f"\n=== 2x2 cells: frag>=50 x {best} (matched-N) ===")
phi = masks[best]
for fl, fm in [("frag<50", ~frag_hi), ("frag>=50", frag_hi)]:
    for pl, pm in [(f"{best}_LO", ~phi), (f"{best}_HI", phi)]:
        sub = nb[fm & pm]
        print(f"  {fl:>9} x {pl:>18}: N={len(sub):4d}  "
              f"avgR {sub.R_Multiple.mean():+.3f}  "
              f"medR {sub.R_Multiple.median():+.3f}  "
              f"win% {(sub.R_Multiple > 0).mean()*100:.0f}  "
              f"months {sub.ym.nunique()}")

# overlap: how many frag>=50 trades are also proxy-high?
print("\noverlap of frag>=50 with each matched-N proxy-high set (Jaccard):")
for pxn in PROXIES:
    inter = (frag_hi & masks[pxn]).sum()
    union = (frag_hi | masks[pxn]).sum()
    print(f"  {pxn:>15}: inter={inter:4d}  jaccard={inter/union:.2f}  "
          f"share of frag-hi covered={inter/frag_hi.sum():.2f}")

# ------------------------------------- 4. LOYO on frag-within-proxyHI residual
print(f"\n=== LOYO: frag>=50 vs <50 WITHIN {best}-high (matched-N) ===")
sub_all = nb[masks[best]]
years = sorted(sub_all.yr.unique())
for drop in [None] + years:
    sub = sub_all if drop is None else sub_all[sub_all.yr != drop]
    f_hi = sub.frag >= 50
    tstat, pval, hm, lm = clustered_t(sub[f_hi], sub[~f_hi])
    print(f"  drop {str(drop):>4}: hi {sub.loc[f_hi,'R_Multiple'].mean():+.3f} "
          f"(N={f_hi.sum():3d}) lo {sub.loc[~f_hi,'R_Multiple'].mean():+.3f} "
          f"(N={(~f_hi).sum():3d})  t={tstat:+.2f} p={pval:.3f}")

# also: does frag survive controlling for ALL proxies at once? crude control:
# any-proxy-high indicator (union of matched-N sets)
any_hi = np.logical_or.reduce([masks[p].values for p in PROXIES])
any_hi = pd.Series(any_hi, index=nb.index)
print(f"\nany-proxy-high union: N={any_hi.sum()} ({any_hi.mean():.1%})")
for cond, m in [("anyProxyHI", any_hi), ("allProxyLO", ~any_hi)]:
    sub = nb[m]
    f_hi = sub.frag >= 50
    tstat, pval, hm, lm = clustered_t(sub[f_hi], sub[~f_hi])
    print(f"  within {cond}: frag>=50 avgR "
          f"{sub.loc[f_hi,'R_Multiple'].mean():+.3f} (N={f_hi.sum()}) vs "
          f"{sub.loc[~f_hi,'R_Multiple'].mean():+.3f} (N={(~f_hi).sum()})  "
          f"t={tstat if tstat==tstat else float('nan'):+.2f} "
          f"p={pval if pval==pval else float('nan'):.3f}")

nb.to_parquet(ROOT / "scratch" / "ultracode_research" / "proxy_joined.parquet")
print("\nsaved joined frame -> proxy_joined.parquet")
