"""c7 round 2: exposure ordering, beta-sign regime, per-session increments,
signed concentration, horizon neighbours.

A. REGIME: the ITB-on-TLT beta was NEGATIVE 2008-2020 and is +1.54 live. In the
   negative-beta era 'ITB below beta x TLT' is a risk-off crash day, a different
   object. Split the cell by sign of the PIT beta at the signal (live = positive).
B. EXPOSURE ORDERING: on every TLT >= +1% day (no other gate), each of 10 sector
   ETFs' next-session (lag0 h1) and tradeable (lag1 h1/h3/h5) excess vs own drift
   and vs SPY-beta residual, against its mean PIT TLT beta on those days. Done
   in the full sample and in the positive-stock-bond-beta regime (2022+).
C. where the h=5 return lands: per-session increments from the entry close.
D. signed concentration + by-year, horizon neighbours h=4/6/7, midterm.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kC_common import *  # noqa
from scipy.stats import spearmanr

GAP = 5
SECT = ["ITB", "XHB", "XLRE", "KRE", "XLU", "XLF", "XLE", "XLK", "XLB", "XLP"]
px = panel(["SPY", "TLT"] + SECT, "SPY")
r1 = {t: dret(px[t]) for t in px.columns}
tlt1 = r1["TLT"]
beta_t = {t: pit_beta(r1[t], tlt1) for t in SECT}
beta_s = {t: pit_beta(r1[t], r1["SPY"]) for t in SECT}
nearlow = {t: (px[t] / rolling_on_valid(px[t], lambda x: x.rolling(252).min()) - 1.0) for t in SECT}
L = [("ITB", 1.0)]
par = (tlt1 >= 0.01).fillna(False)
und = (r1["ITB"] < beta_t["ITB"] * tlt1).fillna(False)
low = (nearlow["ITB"] <= 0.10).fillna(False)
cell = par & und & low
bpos = (beta_t["ITB"] > 0).fillna(False)

print("=" * 100)
print("A. REGIME split by sign of ITB's PIT TLT beta at the signal (live beta +1.54 = positive)")
rows = []
for lbl, m in [("cell, beta>0 (live regime)", cell & bpos), ("cell, beta<=0", cell & ~bpos),
               ("parent TLT>=1%, beta>0", par & bpos), ("parent+underreact, beta>0", par & und & bpos),
               ("parent+near-low, beta>0", par & low & bpos),
               ("near-low NOT underreact, beta>0", par & low & ~und & bpos)]:
    for h in (1, 3, 5):
        for lag in (0, 1):
            s, epi, vals = cellstats(px, m, L, h, f"{lbl} h={h} lag={lag}", GAP, lag)
            rows.append({k: s.get(k) for k in ("label", "n", "mean_pct", "ctl_pct", "edge_pp", "rec", "p_coin", "p_base")})
show(rows)
s, epi, vals = cellstats(px, cell & bpos, L, 5, "", GAP)
print("  beta>0 cell episodes h=5:", [(str(d.date()), round(100 * v, 2)) for d, v in zip(epi, vals)])

print("\n" + "=" * 100)
print("B. EXPOSURE ORDERING on TLT >= +1% days (no sector gate); excess = episode mean - own all-days mean")
for reg_lbl, reg in [("FULL sample", pd.Series(True, index=px.index)),
                     ("2022+ (positive stock-bond beta regime)", pd.Series(px.index >= "2022-01-01", index=px.index))]:
    rows = []
    for t in SECT:
        m = (par & reg).fillna(False)
        row = {"etf": t}
        days = px.index[m.values & px[t].notna().values]
        row["mean_beta_TLT"] = round(float(beta_t[t].reindex(days).mean()), 3)
        row["mean_beta_SPY"] = round(float(beta_s[t].reindex(days).mean()), 3)
        for h, lag in ((1, 0), (1, 1), (3, 1), (5, 1)):
            s, epi, vals = cellstats(px, m, [(t, 1.0)], h, "", GAP, lag)
            row[f"x_h{h}L{lag}"] = round(s.get("edge_pp", np.nan), 3)
            rt = vehicle_ret(px, [(t, 1.0)], h, lag)
            rs = vehicle_ret(px, [("SPY", 1.0)], h, lag)
            res = (rt - beta_s[t] * rs)
            row[f"res_h{h}L{lag}"] = round(100 * float(res.reindex(epi).mean()), 3)
        row["n"] = s.get("n", 0)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("mean_beta_TLT", ascending=False)
    print(f"\n--- {reg_lbl} ---")
    print(df.to_string(index=False))
    for c in [c for c in df.columns if c.startswith("x_") or c.startswith("res_")]:
        rho, p = spearmanr(df["mean_beta_TLT"], df[c])
        print(f"  spearman(beta_TLT, {c}) = {rho:+.2f}  (p {p:.2f})")

print("\n" + "=" * 100)
print("C. WHERE THE h=5 RETURN LANDS: per-session increments from the entry close (cell, all / beta>0)")
for lbl, m in [("cell all", cell), ("cell beta>0", cell & bpos)]:
    days = px.index[m.values]
    epi = declusters(days, GAP, px.index)
    paths = episode_paths(px, epi, L, 7)
    inc = paths.diff(axis=1)
    inc[1] = paths[1]
    print(f"  {lbl} (N={len(paths)}):")
    for k in paths.columns:
        v = inc[k].dropna().values
        print(f"    session +{k}: mean {100*v.mean():+.3f}pp hit {100*(v>0).mean():.1f}%  cum {100*paths[k].mean():+.3f}%")

print("\n" + "=" * 100)
print("D. SIGNED CONCENTRATION, by-year, horizon neighbours, midterm (cell, lag=1)")
for h in (4, 5, 6, 7):
    s, epi, vals = cellstats(px, cell, L, h, "", GAP)
    print(f"  h={h}: N={s['n']} mean {s['mean_pct']:+.3f}% edge {s['edge_pp']:+.3f}pp rec {s['rec']} p_base {s['p_base']}")
s, epi, vals = cellstats(px, cell, L, 5, "", GAP)
print("  ", signed_conc(epi, vals, "h=5"))
yrs = pd.Series(vals, index=pd.DatetimeIndex(epi).year)
print("   by year (n, sum pp):", {int(y): (int(g.size), round(100 * g.sum(), 2)) for y, g in yrs.groupby(level=0)})
mt = midterm_mask(px.index)
for lbl, m in [("midterm", cell & mt), ("non-midterm", cell & ~mt)]:
    s, _, _ = cellstats(px, m, L, 5, "", GAP)
    print(f"  {lbl} h=5: N={s.get('n')} mean {s.get('mean_pct', np.nan):+.3f}% rec {s.get('rec')}")
