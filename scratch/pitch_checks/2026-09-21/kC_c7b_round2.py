"""c7 round 2 on the UNGATED election-year parent (the pitched gate was an
anti-filter in round 1): (1) offset placebo ladder k=-5..+5 around the Sep opex+1
entry at fixed h=22 (to ~Oct VIX expiry) and h=19 (Oct opex), election years;
(2) convexity-adjusted SPY residual: regress every 22-session short-SVXY window
on SPY's window return AND its downside part, then read the election Octobers'
residual; (3) term structure at entry (VIX/VIX3M; live 0.812) by year; (4) the
live-state slice: VIX distance from its 252 low at the signal."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kC_common import *  # noqa

BREAK = pd.Timestamp("2018-02-28")
px = nyse_panel(["SVXY", "SPY", "^VIX", "^VIX3M"])
idx = px.index
r = dret(px["SVXY"])
r_adj = r.where(idx >= BREAK, 0.5 * r)
S = (1 + r_adj.fillna(0)).cumprod()
S[idx < px["SVXY"].first_valid_index()] = np.nan
pos = pd.Series(range(len(idx)), index=idx)
opex = to_sessions(load_events(["opex"])["date"], idx)
vix, v3 = px["^VIX"], px["^VIX3M"]
vlow = rolling_on_valid(vix, lambda x: x.rolling(252).min())

H = 22
shortS = -(S.shift(-H) / S - 1)            # entry at this close, exit +H
spyH = px["SPY"].shift(-H) / px["SPY"] - 1
vixH = vix.shift(-H) / vix - 1
ok = shortS.notna() & spyH.notna() & (idx >= "2012-01-03")
X = pd.DataFrame({"y": shortS[ok], "spy": spyH[ok], "dn": np.minimum(spyH[ok], 0),
                  "post": (idx[ok] >= BREAK).astype(float)})
A = np.column_stack([np.ones(len(X)), X.spy, X.dn, X.post, X.post * X.spy, X.post * X.dn])
coef, *_ = np.linalg.lstsq(A, X.y.values, rcond=None)
print("all-window fit short_S(22) = a + b*spy + c*min(spy,0) (+ post-break interactions):",
      np.round(coef, 3))
fit = pd.Series(A @ coef, index=X.index)
resid = X.y - fit
A1 = np.column_stack([np.ones(len(X)), X.spy, X.post, X.post * X.spy])
c1, *_ = np.linalg.lstsq(A1, X.y.values, rcond=None)
resid_lin = X.y - pd.Series(A1 @ c1, index=X.index)

sep_entries = {}
for d in opex:
    if d.month == 9 and 2012 <= d.year <= 2025:
        sep_entries[d.year] = pos[d] + 1


def cell(vals, label):
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]
    s = summarize(v, label)
    if len(v):
        w = int((v > 0).sum())
        s["rec"] = f"{w}-{len(v)-w}"
        s["sign_p"] = round(sign_test(w, len(v)), 4)
    return s


# (1) offset ladder
for hh in (22, 19, 10):
    fr = -(S.shift(-hh) / S - 1)
    out = []
    for k in range(-5, 6):
        ve = [fr.iloc[p + k] for y, p in sep_entries.items() if y % 2 == 0]
        vo = [fr.iloc[p + k] for y, p in sep_entries.items() if y % 2 == 1]
        ce, co = cell(ve, ""), cell(vo, "")
        out.append({"k": k, "elect_mean": round(ce["mean_pct"], 2), "elect_rec": ce["rec"],
                    "odd_mean": round(co["mean_pct"], 2), "odd_rec": co["rec"],
                    "diff_pp": round(ce["mean_pct"] - co["mean_pct"], 2)})
    df = pd.DataFrame(out)
    df["rank_elect"] = df.elect_mean.rank(ascending=False).astype(int)
    df["rank_diff"] = df.diff_pp.rank(ascending=False).astype(int)
    print(f"\n=== (1) offset ladder, short synthetic SVXY, fixed h={hh}, entry = Sep opex+1+k ===")
    print(df.to_string(index=False))

# VIX spot ladder 2000-2025 (long history), h=22
fr = vixH
out = []
sep_all = {d.year: pos[d] + 1 for d in opex if d.month == 9 and d.year <= 2025}
for k in range(-5, 6):
    ve = [fr.iloc[p + k] for y, p in sep_all.items() if y % 2 == 0]
    vo = [fr.iloc[p + k] for y, p in sep_all.items() if y % 2 == 1]
    ce, co = cell(ve, ""), cell(vo, "")
    out.append({"k": k, "elect_mean": round(ce["mean_pct"], 2), "elect_med": round(ce["median_pct"], 2),
                "elect_rec": ce["rec"], "odd_mean": round(co["mean_pct"], 2), "odd_rec": co["rec"]})
df = pd.DataFrame(out)
df["rank_elect"] = df.elect_mean.rank(ascending=False).astype(int)
print("\n=== VIX spot ladder 2000-2025, h=22 ===")
print(df.to_string(index=False))

# (2) residuals for the Sep opex+1 windows, per year
rows = []
for y, p in sep_entries.items():
    d = idx[p]
    rows.append({"year": y, "elect": y % 2 == 0, "shortS": 100 * shortS.iloc[p], "spy": 100 * spyH.iloc[p],
                 "resid_lin": 100 * resid_lin.get(d, np.nan), "resid_convex": 100 * resid.get(d, np.nan),
                 "vix_sig": vix.iloc[p - 1], "vix_vs_low_pct": 100 * (vix.iloc[p - 1] / vlow.iloc[p - 1] - 1),
                 "vix_vix3m_sig": vix.iloc[p - 1] / v3.iloc[p - 1]})
R = pd.DataFrame(rows).round(3)
print("\n=== (2)/(3) per-year Sep opex+1 -> +22: raw short, linear- and convexity-adjusted residual, entry state ===")
print(R.to_string(index=False))
out = []
for lbl, m in [("election", R.elect), ("odd", ~R.elect), ("midterm", R.year % 4 == 2),
               ("election post-break", R.elect & (R.year >= 2018)),
               ("election VIX/VIX3M <= 0.85", R.elect & (R.vix_vix3m_sig <= 0.85)),
               ("ANY yr VIX/VIX3M <= 0.85", R.vix_vix3m_sig <= 0.85),
               ("ANY yr VIX <= 20% above low", R.vix_vs_low_pct <= 20)]:
    for c in ["shortS", "resid_lin", "resid_convex"]:
        s = cell(R.loc[m, c].values / 100, f"{lbl} {c}")
        out.append(s)
show(out, "splits (h=22 from Sep opex+1)")
print(f"\nresidual all-window mean (by construction ~0): linear {100*resid_lin.mean():+.3f}%, convex {100*resid.mean():+.3f}%")
print(f"LIVE entry state: VIX/VIX3M {vix.iloc[-1]/v3.iloc[-1]:.3f}, VIX {100*(vix.iloc[-1]/vlow.iloc[-1]-1):.1f}% above low")
# distribution of VIX/VIX3M at Sep opex for context
q = (vix / v3).dropna()
print(f"VIX/VIX3M all-days 2012+ pctile of 0.812: {100*(q[q.index>='2012-01-01'] <= 0.812).mean():.1f}")
