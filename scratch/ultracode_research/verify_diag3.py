"""Corrected-label recompute of claims 6/7 + N-sensitivity checks.

Key insight: proxy_supplement.py's masks['rv21_pct'] is the LOW-rv (ultra-calm)
tail, but the md labels it 'rv-elevated'. Recompute both ways, independently.
"""
import numpy as np
import pandas as pd
from scipy import stats

ROOT = r"C:\Users\McKinley Slade\dev\New_Seasonals"

# rebuild independent frame (same as verify_proxy-falsification.py, condensed)
px = pd.read_parquet(ROOT + r"\data\master_prices.parquet",
                     filters=[("ticker", "in", ["SPY", "^VIX"])])
px["date"] = pd.to_datetime(px["date"])
spy = px[px.ticker == "SPY"].set_index("date")["Close"].sort_index()
ret = spy.pct_change()
rv21 = ret.rolling(21).std() * np.sqrt(252)
rv_pct = rv21.rolling(756).rank(pct=True)

frag = pd.read_parquet(ROOT + r"\data\rd2_fragility.parquet")
frag_live = frag["63d"].rolling(10, min_periods=1).mean()

tr = pd.read_parquet(ROOT + r"\data\backtest_trades_full.parquet")
tr = tr[tr["Strategy"] != "Overbot Vol Spike"].copy()
tr["Signal Date"] = pd.to_datetime(tr["Signal Date"])
tr = tr[tr["Signal Date"] >= "2016-08-01"].sort_values("Signal Date")
fr = frag_live.reset_index(); fr.columns = ["Date", "frag"]
tr = pd.merge_asof(tr, fr, left_on="Signal Date", right_on="Date",
                   direction="backward", tolerance=pd.Timedelta(days=5))
rvf = rv_pct.reset_index(); rvf.columns = ["D2", "rv21_pct"]
tr = pd.merge_asof(tr, rvf, left_on="Signal Date", right_on="D2",
                   direction="backward", tolerance=pd.Timedelta(days=5))
tr = tr.dropna(subset=["frag", "R_Multiple", "rv21_pct"]).copy()
tr["month"] = tr["Signal Date"].dt.to_period("M")
tr["year"] = tr["Signal Date"].dt.year
R = "R_Multiple"
print("N =", len(tr))


def clus_t(df, mask_hi):
    mh = df[mask_hi].groupby("month")[R].mean()
    ml = df[~mask_hi].groupby("month")[R].mean()
    t, p = stats.ttest_ind(mh, ml, equal_var=False)
    return dict(n_hi=int(mask_hi.sum()), n_lo=int((~mask_hi).sum()),
                avg_hi=df[mask_hi][R].mean(), avg_lo=df[~mask_hi][R].mean(),
                t=t, p=p)


def fmt(tag, r):
    print(f"{tag:44s} N={r['n_hi']:4d}/{r['n_lo']:4d} "
          f"avgR={r['avg_hi']:+.3f}/{r['avg_lo']:+.3f} t={r['t']:+.2f} p={r['p']:.3f}")


frag_hi = tr["frag"] >= 50
frac = frag_hi.mean()
calm_thr = tr["rv21_pct"].quantile(frac)       # LOW tail = ultra-calm (their mask)
print(f"calm-tail thr = {calm_thr:.3f}")
ultra_calm = tr["rv21_pct"] <= calm_thr

print("\n--- CORRECT-LABEL toxic cell: frag>=50 & rv ULTRA-CALM (bottom ~21%) ---")
tox = tr[frag_hi & ultra_calm]
print(f"N={len(tox)} avgR={tox[R].mean():+.3f} med={tox[R].median():+.3f} "
      f"win={(tox[R]>0).mean()*100:.0f}% months={tox['month'].nunique()}")
print("years:", tox["year"].value_counts().to_dict())
print("strategies:", tox.groupby("Strategy")[R].agg(["size", "mean"]).round(2).to_dict("index"))

sub = tr[ultra_calm]
fmt("within ULTRA-CALM: frag split", clus_t(sub, sub["frag"] >= 50))
for yr in [2021, 2024]:
    s2 = sub[sub["year"] != yr]
    fmt(f"  drop {yr}", clus_t(s2, s2["frag"] >= 50))

print("\n--- LOYO: frag split within rv NOT-ultra-calm (their 'rv-calm' claim 6) ---")
rest = tr[~ultra_calm]
fmt("all years", clus_t(rest, rest["frag"] >= 50))
ts = []
for yr in sorted(rest["year"].unique()):
    s2 = rest[rest["year"] != yr]
    r = clus_t(s2, s2["frag"] >= 50)
    ts.append(r["t"])
    fmt(f"  drop {yr}", r)
print(f"LOYO t range: [{min(ts):.2f}, {max(ts):.2f}]")

print("\n--- Spearman(frag,R) within strata ---")
for lab, m in [("ultra-calm", ultra_calm), ("rest", ~ultra_calm)]:
    s = tr[m]
    rho, p = stats.spearmanr(s["frag"], s[R])
    print(f"  {lab}: rho={rho:+.3f} p={p:.3f} N={len(s)}")

print("\n--- N-sensitivity: claim-1/2 on full N=1153 and their cached frame ---")
tr2 = pd.read_parquet(ROOT + r"\data\backtest_trades_full.parquet")
tr2 = tr2[tr2["Strategy"] != "Overbot Vol Spike"].copy()
tr2["Signal Date"] = pd.to_datetime(tr2["Signal Date"])
tr2 = tr2[tr2["Signal Date"] >= "2016-08-01"].sort_values("Signal Date")
tr2 = pd.merge_asof(tr2, fr, left_on="Signal Date", right_on="Date",
                    direction="backward", tolerance=pd.Timedelta(days=5))
tr2 = tr2.dropna(subset=["frag", "R_Multiple"])
tr2["month"] = tr2["Signal Date"].dt.to_period("M")
fmt("frag>=50 on full non-OVS join", clus_t(tr2, tr2["frag"] >= 50))

j = pd.read_parquet(ROOT + r"\scratch\ultracode_research\proxy_joined.parquet")
j["month"] = pd.to_datetime(j["Signal Date"]).dt.to_period("M")
fmt("frag>=50 on THEIR cached frame", clus_t(j, j["frag"] >= 50))
cal_j = j["rv21_pct"] <= j["rv21_pct"].quantile((j["frag"] >= 50).mean())
fmt("rv calm-tail rule on THEIR frame", clus_t(j, cal_j))
