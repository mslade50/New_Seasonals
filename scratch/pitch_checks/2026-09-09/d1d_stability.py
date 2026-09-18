"""C13 steps 6 + kill battery: era split, midterm split, concentration,
definition neighbours (threshold / lookback / ranking-window nudges), the
mandatory SVXY-vs-SPY residual (kill #9), and the registry-collision check.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change, rolling_on_valid
import numpy as np, pandas as pd

HERE = Path(__file__).parent
st = pd.read_parquet(HERE / "d1_state.parquet")
px = close_panel(["SPY", "QQQ", "TLT", "IEF", "SVXY", "^VIX", "^MOVE"])
st = st.reindex(px.index)
s = st["sprA"]
THR, H = 60.0, 10
mask = (s >= THR).fillna(False)

r = fwd_lag(px["SPY"], H, 1)
d = px.index[mask.values & r.notna().values]
epi = declusters(pd.DatetimeIndex(d), H, r.dropna().index)
v = r.loc[epi].values
print(f"SPY h={H} spread>={THR:.0f}: N={len(epi)} mean {100*v.mean():+.3f}% t {summarize(v)['t']:+.2f}")

# --- concentration ---------------------------------------------------------
print("\n6a. CONCENTRATION")
print("  ", cluster_note(epi, v, k=2))
by_yr = pd.Series(v).groupby(pd.DatetimeIndex(epi).year.values)
tot = v.sum()
print("   per-year totals (pp):",
      {int(y): round(100*g.sum(), 2) for y, g in by_yr})
print("   per-year N:", {int(y): int(g.size) for y, g in by_yr})
for k in (1, 2, 3):
    order = np.argsort(-np.abs(v))[:k]
    keep = np.setdiff1d(np.arange(len(v)), order)
    print(f"   drop-best-{k} episodes: mean {100*v[keep].mean():+.3f}% (N={len(keep)})")
yr = pd.DatetimeIndex(epi).year.values
worst_yr = max(set(yr), key=lambda y: v[yr == y].sum())
kp = yr != worst_yr
print(f"   drop-best-YEAR ({worst_yr}, N={int((~kp).sum())}): "
      f"mean {100*v[kp].mean():+.3f}% t {summarize(v[kp])['t']:+.2f} (N={int(kp.sum())})")

# --- era + midterm ---------------------------------------------------------
show(era_split(epi, v), "6b. era split (pre-2018 / 2018+)")
mid = (pd.DatetimeIndex(epi).year % 4 == 2)
show([summarize(v[mid], f"MIDTERM years (N={int(mid.sum())})"),
      summarize(v[~mid], f"non-midterm (N={int((~mid).sum())})")],
     "6c. midterm split (2026 IS midterm)")
print("   midterm episode dates:", [str(x.date()) for x in pd.DatetimeIndex(epi)[mid]])

# --- definition neighbours -------------------------------------------------
print("\n7. DEFINITION NEIGHBOURS (SPY h=10, episodes)")
rows = []
for t_ in (45, 50, 55, 60, 65, 70, 75):
    m = (s >= t_).fillna(False)
    dd = px.index[m.values & r.notna().values]
    e = declusters(pd.DatetimeIndex(dd), H, r.dropna().index)
    rr = summarize(r.loc[e].values, f"spread>={t_}")
    rr["n_days"] = int(m.sum())
    rows.append(rr)
show(rows, "  threshold nudge")

# lookback nudge on both component percentiles
mv = px["^MOVE"].dropna()
spy = px["SPY"].dropna()
rr1 = spy.pct_change()
rows = []
for lb in (126, 189, 252, 378, 504):
    mp = rolling_on_valid(mv, lambda x: x.rolling(lb).rank(pct=True)*100).reindex(px.index)
    rv = (rr1.rolling(21).std()*np.sqrt(252)*100)
    rp = rolling_on_valid(rv, lambda x: x.rolling(lb).rank(pct=True)*100).reindex(px.index)
    sp = mp - rp
    live = sp.dropna().iloc[-1]
    m = (sp >= THR).fillna(False)
    dd = px.index[m.values & r.notna().values]
    e = declusters(pd.DatetimeIndex(dd), H, r.dropna().index)
    x = summarize(r.loc[e].values, f"lookback {lb}")
    x["live_spread"] = round(live, 1)
    x["n_days"] = int(m.sum())
    rows.append(x)
show(rows, "  percentile-lookback nudge (live spread shown)")

# realized-vol window nudge
rows = []
for w in (10, 15, 21, 30, 42):
    rv = (rr1.rolling(w).std()*np.sqrt(252)*100)
    rp = rolling_on_valid(rv, lambda x: x.rolling(252).rank(pct=True)*100).reindex(px.index)
    sp = st["move_pct"] - rp
    live = sp.dropna().iloc[-1]
    m = (sp >= THR).fillna(False)
    dd = px.index[m.values & r.notna().values]
    e = declusters(pd.DatetimeIndex(dd), H, r.dropna().index)
    x = summarize(r.loc[e].values, f"rvol window {w}d")
    x["live_spread"] = round(live, 1)
    x["n_days"] = int(m.sum())
    rows.append(x)
show(rows, "  realized-vol window nudge")

# spread B (MOVE - VIX) as the alternative pre-spec
sB = st["sprB"]
rows = []
for t_ in (50, 60, 70):
    m = (sB >= t_).fillna(False)
    dd = px.index[m.values & r.notna().values]
    e = declusters(pd.DatetimeIndex(dd), H, r.dropna().index)
    rows.append(summarize(r.loc[e].values, f"sprB(MOVE-VIX)>={t_}"))
show(rows, "  spread B (MOVE minus VIX level pctile)")

# --- 9. SVXY residual against SPY -----------------------------------------
print("\n9. SVXY RESIDUAL AGAINST SPY (kill #9)")
for h in (5, 10):
    rs = fwd_lag(px["SVXY"], h, 1)
    rp = fwd_lag(px["SPY"], h, 1)
    ok = rs.notna() & rp.notna()
    X = rp[ok].values; Y = rs[ok].values
    b, a = np.polyfit(X, Y, 1)
    resid = pd.Series(np.nan, index=px.index)
    resid[ok] = Y - (a + b * X)
    r2 = np.corrcoef(X, a + b*X)[0, 1]**2
    dd = px.index[mask.values & ok.values]
    e = declusters(pd.DatetimeIndex(dd), h, px.index[ok.values])
    rv_ = resid.loc[e].values
    print(f"   h={h}: SVXY = {100*a:+.3f}% + {b:.2f}*SPY, R2 {r2:.3f}, N {ok.sum()}")
    show([summarize(rv_, f"SVXY residual COND (N={len(e)})"),
          summarize(resid.dropna().values, "SVXY residual ALL days")], "")
    # concentration of the residual by year
    yy = pd.DatetimeIndex(e).year.values
    tt = rv_.sum()
    byy = {int(y): round(100*rv_[yy == y].sum(), 2) for y in sorted(set(yy))}
    print("   residual per-year (pp):", byy, f" total {100*tt:+.2f}pp")
