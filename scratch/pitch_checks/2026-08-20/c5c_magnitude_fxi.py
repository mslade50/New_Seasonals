"""C5 attacks 2, 3, 5: the magnitude form, the longer-history vehicle, cost.

2. The trigger is a rank artifact. Re-run KWEB on the MAGNITUDE form and on
   today's exact magnitude.
3. KWEB starts 2013. If the mechanism (dollar easing the EM funding
   constraint) is real it must show in FXI, which has 2004+ history and 36
   episodes rather than 18.
5. Cost.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = close_panel(["DX-Y.NYB", "KWEB", "FXI", "EEM", "YINN", "SPY"])
d = px.index
dx = px["DX-Y.NYB"].dropna()
r21 = (dx / dx.shift(21) - 1.0).reindex(d)
rk21 = pct_rank(px["DX-Y.NYB"], 21)
TODAY = float(r21.dropna().iloc[-1])

forms = {
    "RANK <=2 (pitched)": (rk21 <= 2),
    "MAG <= -2.32% (today)": (r21 <= TODAY),
    "MAG <= -3%": (r21 <= -0.03),
    "MAG <= -4%": (r21 <= -0.04),
    "RANK<=2 AND MAG<=-4% (deep half)": (rk21 <= 2) & (r21 <= -0.04),
    "RANK<=2 AND MAG>-3.6% (today's half)": (rk21 <= 2) & (r21 > -0.036),
}
forms = {k: v.reindex(d).fillna(False) for k, v in forms.items()}

for tkr in ["KWEB", "FXI", "EEM", "YINN"]:
    print(f"\n\n################ {tkr} ################")
    rows = []
    for lbl, m in forms.items():
        for h in (5, 10):
            r = vehicle_ret(px, [(tkr, 1.0)], h)
            s = pd.DatetimeIndex([x for x in d[m.values]
                                  if not np.isnan(r.get(x, np.nan))])
            if len(s) == 0:
                continue
            e = declusters(s, 21, d)
            v = r.reindex(e).dropna()
            c = r.dropna()
            if len(v) < 4:
                continue
            rows.append({"form": lbl, "h": h, "N_ep": len(v),
                         "mean_pct": round(100 * v.mean(), 3),
                         "drift_pct": round(100 * c.mean(), 3),
                         "excess_pp": round(100 * (v.mean() - c.mean()), 3),
                         "hit": round(100 * (v > 0).mean(), 1),
                         "signp": round(sign_test(int((v > 0).sum()), len(v)), 3),
                         "first": str(v.index[0].date())})
    show(rows, f"{tkr}: rank form vs magnitude form")

# ---- FXI on the FULL history (the mechanism's longer test)
print("\n\n######## FXI, FULL HISTORY, rank<=2 (36-episode set) ########")
m = (rk21 <= 2).reindex(d).fillna(False)
for h in (3, 5, 10):
    r = vehicle_ret(px, [("FXI", 1.0)], h)
    s = pd.DatetimeIndex([x for x in d[m.values] if not np.isnan(r.get(x, np.nan))])
    e = declusters(s, 21, d)
    v = r.reindex(e).dropna()
    c = r.dropna()
    print(f"  h={h:<3} N={len(v)} span {v.index[0].date()}..{v.index[-1].date()}  "
          f"mean {100*v.mean():+.3f}%  drift {100*c.mean():+.3f}%  "
          f"excess {100*(v.mean()-c.mean()):+.3f}pp  hit {100*(v>0).mean():.1f}%  "
          f"t {v.mean()/(v.std(ddof=1)/np.sqrt(len(v))):+.2f}  "
          f"sign p {sign_test(int((v>0).sum()), len(v)):.3f}")
    print("      ", cluster_note(v.index, v.values))
    show(era_split(v.index, v.values), "")

# ---- the KWEB-era slice of FXI vs the pre-KWEB slice
print("\n\n######## FXI: pre-KWEB era vs KWEB era on the same rule ########")
k0 = px["KWEB"].dropna().index[0]
for h in (5, 10):
    r = vehicle_ret(px, [("FXI", 1.0)], h)
    s = pd.DatetimeIndex([x for x in d[m.values] if not np.isnan(r.get(x, np.nan))])
    e = declusters(s, 21, d)
    v = r.reindex(e).dropna()
    pre, post = v[v.index < k0], v[v.index >= k0]
    print(f"  h={h:<3} pre-2013 N={len(pre)} {100*pre.mean():+.3f}% hit "
          f"{100*(pre>0).mean():.0f}%  |  2013+ N={len(post)} {100*post.mean():+.3f}% "
          f"hit {100*(post>0).mean():.0f}%")

# ---- cost
print("\n\n######## COST ########")
print("  KWEB: ~$8bn AUM, typical quoted spread ~2-4 bps, so ~6 bps round trip"
      " plus ADR/HK-session basis. Long only, so borrow is not a factor.")
r5 = vehicle_ret(px, [("KWEB", 1.0)], 5)
e = declusters(pd.DatetimeIndex([x for x in d[m.values]
                                 if not np.isnan(r5.get(x, np.nan))]), 21, d)
v = r5.reindex(e).dropna()
loc = local_control(d[r5.notna().values], pd.DatetimeIndex(
    [x for x in d[m.values] if not np.isnan(r5.get(x, np.nan))]))
print(f"  h=5 episode mean {100*v.mean():+.3f}% = {10000*v.mean():.0f} bps "
      f"-> {10000*v.mean()/6:.0f}x a 6 bps round trip.")
print(f"  BUT vs the LOCAL +/-126td control ({100*r5.loc[loc].mean():+.3f}%) the "
      f"excess is only {100*(v.mean()-r5.loc[loc].mean()):+.3f}pp = "
      f"{10000*(v.mean()-r5.loc[loc].mean())/6:.1f}x cost.")
r10 = vehicle_ret(px, [("KWEB", 1.0)], 10)
e10 = declusters(pd.DatetimeIndex([x for x in d[m.values]
                                   if not np.isnan(r10.get(x, np.nan))]), 21, d)
v10 = r10.reindex(e10).dropna()
loc10 = local_control(d[r10.notna().values], pd.DatetimeIndex(
    [x for x in d[m.values] if not np.isnan(r10.get(x, np.nan))]))
print(f"  h=10 episode mean {100*v10.mean():+.3f}% vs LOCAL "
      f"{100*r10.loc[loc10].mean():+.3f}% -> excess "
      f"{100*(v10.mean()-r10.loc[loc10].mean()):+.3f}pp = "
      f"{10000*(v10.mean()-r10.loc[loc10].mean())/6:.1f}x cost.")
