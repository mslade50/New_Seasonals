"""C6 round 1+2: SHORT SVXY (= long vol) across the PPI print session.

Claim: at the PPI anchor k=2 (entry MOC on the eve, exit MOC on the print),
SVXY pays -0.365% excess over own drift at a 55% hit, N=177 -- the loudest
class reading on the PPI eve, and the mirror of the +0.436% at k=1.

The registry has killed this construction twice already, so the burden is
entirely on this script to show it is NOT the same corpse:
  A. SPY-beta residual. The registry rule is explicit: regress the VEHICLE on
     the market and quote the RESIDUAL. Full sample and 2018+.
  B. the 2018-02 leverage cut (-1x -> -0.5x). State which side the sample sits
     on and re-measure post-cut only.
  C. the MANDATED placebo anchor ladder k=-8..+12.
  D. does it compose with, or IS it, the SVXY-at-52w-high state that is live
     today (-0.317/-0.638 at h=3/5)?
  E. index vs vehicle: does ^VIX itself rise across the PPI print?
  F. cost of a SHORT SVXY: borrow, and the carry it fights.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

LEV_CUT = pd.Timestamp("2018-02-28")   # SVXY -1x -> -0.5x

px = close_panel(["SVXY", "SPY", "UVXY", "^VIX"])
idx = px.index
ev = load_events(["ppi", "cpi"])
PPI = pd.DatetimeIndex(sorted(ev.loc[ev.event == "ppi", "date"].unique()))
CPI = set(ev.loc[ev.event == "cpi", "date"])


def anchors(dates, k):
    out = []
    for d in dates:
        loc = idx.searchsorted(pd.Timestamp(d))
        if loc >= len(idx):
            continue
        p = loc - k
        if 0 <= p < len(idx):
            out.append(idx[p])
    return pd.DatetimeIndex(sorted(set(out)))


def mask_from(dts):
    return pd.Series(True, index=dts).reindex(idx, fill_value=False)


A2 = anchors(PPI, 2)
r_svxy = fwd_lag(px["SVXY"], 1, 1)
r_spy = fwd_lag(px["SPY"], 1, 1)
r_uvxy = fwd_lag(px["UVXY"], 1, 1)
r_vix = fwd_lag(px["^VIX"], 1, 1)

trig = A2.intersection(r_svxy.dropna().index)
print(f"PPI anchors k=2 with SVXY data: N={len(trig)}  "
      f"{trig[0].date()}..{trig[-1].date()}")

# --------------------------------------------------------- B. leverage eras
pre = trig[trig < LEV_CUT]
post = trig[trig >= LEV_CUT]
print(f"\n=== B. 2018-02 leverage cut (-1x -> -0.5x) ===")
print(f"  trigger days PRE-cut (-1x SVXY): {len(pre)} = "
      f"{100*len(pre)/len(trig):.1f}% of the sample")
print(f"  trigger days POST-cut (-0.5x)  : {len(post)}")
rows = []
for lbl, t in [("ALL", trig), ("pre-2018-02 (-1x)", pre), ("post-2018-02 (-0.5x)", post)]:
    v = r_svxy.loc[t].dropna()
    span = (idx >= v.index[0]) & (idx <= v.index[-1])
    base = r_svxy[span].dropna()
    rows.append({"seg": lbl, "n": len(v), "cond_pct": round(100*v.mean(), 3),
                 "own_drift_pct": round(100*base.mean(), 3),
                 "excess_pct": round(100*(v.mean()-base.mean()), 3),
                 "hit_long": round(100*(v > 0).mean(), 1),
                 "short_wins": int((v < 0).sum()),
                 "sign_p_short": round(sign_test(int((v < 0).sum()), len(v),
                                                 float((base < 0).mean())), 4)})
show(rows, "SVXY on the PPI eve by leverage era (LONG basis; short pays -1x)")

# ------------------------------------------------------- A. SPY-beta residual
print("\n=== A. SPY-beta residual (the registry's own rule) ===")
for lbl, sl in [("full", slice(None)), ("2018-03+", slice(LEV_CUT, None))]:
    d = pd.concat([r_svxy, r_spy], axis=1, keys=["s", "m"]).dropna().loc[sl]
    beta = np.polyfit(d["m"], d["s"], 1)
    resid = d["s"] - (beta[0] * d["m"] + beta[1])
    r2 = np.corrcoef(d["m"], d["s"])[0, 1] ** 2
    t = d.index.intersection(trig)
    rv = resid.loc[t]
    print(f"  [{lbl}] beta={beta[0]:.3f}  R^2={r2:.3f}  N_all={len(d)}")
    print(f"    raw SVXY on trigger  : {100*d['s'].loc[t].mean():+.3f}%  "
          f"(N={len(t)})")
    print(f"    SPY on trigger       : {100*d['m'].loc[t].mean():+.3f}%")
    print(f"    RESIDUAL on trigger  : {100*rv.mean():+.3f}%  "
          f"hit(long) {100*(rv > 0).mean():.1f}%  t={rv.mean()/(rv.std(ddof=1)/np.sqrt(len(rv))):+.2f}")
    print(f"    -> beta explains {100*(1 - abs(rv.mean())/abs(d['s'].loc[t].mean())):.0f}% "
          f"of the raw cell")

# ------------------------------------------------------------- E. index vs vehicle
print("\n=== E. index vs vehicle: does ^VIX itself rise on the print? ===")
for lbl, s in [("^VIX", r_vix), ("SVXY", r_svxy), ("UVXY", r_uvxy),
               ("SPY", r_spy)]:
    v = s.loc[s.index.intersection(trig)].dropna()
    if len(v) < 10:
        continue
    span = (idx >= v.index[0]) & (idx <= v.index[-1])
    base = s[span].dropna()
    print(f"  {lbl:6s} cond {100*v.mean():+.3f}%  drift {100*base.mean():+.3f}%"
          f"  excess {100*(v.mean()-base.mean()):+.3f}%  hit {100*(v>0).mean():.1f}%"
          f"  N={len(v)}")

# -------------------------------------------------------------- C. placebo
print("\n=== C. PLACEBO ANCHOR LADDER k=-8..+12 (SVXY excess, LONG basis; "
      "the short pays the negative) ===")
lad = []
for k in range(-8, 13):
    a = anchors(PPI, k)
    v = r_svxy.loc[r_svxy.index.intersection(a)].dropna()
    if len(v) < 10:
        continue
    span = (idx >= v.index[0]) & (idx <= v.index[-1])
    base = r_svxy[span].dropna()
    # beta-neutral version too
    d = pd.concat([r_svxy, r_spy], axis=1, keys=["s", "m"]).dropna()
    b = np.polyfit(d["m"], d["s"], 1)
    res = (d["s"] - (b[0]*d["m"] + b[1])).loc[d.index.intersection(a)]
    lad.append({"k": k, "n": len(v),
                "svxy_excess_pct": round(100*(v.mean()-base.mean()), 3),
                "resid_pct": round(100*res.mean(), 3),
                "real": "<<<< REAL" if k == 2 else ""})
show(lad, "placebo ladder")
col = pd.Series({r["k"]: r["svxy_excess_pct"] for r in lad})
rescol = pd.Series({r["k"]: r["resid_pct"] for r in lad})
print(f"  raw   : real k=2 {col.get(2):+.3f}%. nonsense anchors MORE negative: "
      f"{dict(col[col < col.get(2)].round(3))}")
print(f"  resid : real k=2 {rescol.get(2):+.3f}%. nonsense anchors MORE "
      f"negative: {dict(rescol[rescol < rescol.get(2)].round(3))}")

# ------------------------------------------- D. compose with SVXY-at-52w-high
print("\n=== D. does the PPI cell compose with today's SVXY-at-52w-high "
      "state, or IS it that state? ===")
hi52 = px["SVXY"] >= px["SVXY"].rolling(252).max() * 0.999
print(f"  SVXY at a 52w high today (2026-08-11 bar): {bool(hi52.iloc[-1])}")
both = trig[hi52.reindex(trig, fill_value=False).values]
only_ppi = trig[~hi52.reindex(trig, fill_value=False).values]
hi_days = idx[hi52.fillna(False).values]
rows = []
for lbl, t in [("PPI-eve ALL", trig), ("PPI-eve AND 52wh (today)", both),
               ("PPI-eve, NOT 52wh", only_ppi),
               ("52wh alone, any day", hi_days)]:
    v = r_svxy.loc[r_svxy.index.intersection(t)].dropna()
    if len(v) < 3:
        rows.append({"label": lbl, "n": len(v)})
        continue
    span = (idx >= v.index[0]) & (idx <= v.index[-1])
    base = r_svxy[span].dropna()
    rows.append({"label": lbl, "n": len(v),
                 "mean_pct": round(100*v.mean(), 3),
                 "excess_pct": round(100*(v.mean()-base.mean()), 3),
                 "hit": round(100*(v > 0).mean(), 1)})
show(rows, "joint state")

# ------------------------------------------------------------------ battery
variants = {f"k={k} anchor": mask_from(anchors(PPI, k)) for k in (1, 2, 3)}
variants["k=2, post-2018 leverage only"] = mask_from(post)
variants["k=2, CPI on entry (today)"] = mask_from(
    pd.DatetimeIndex([d for d in A2
                      if idx[min(idx.searchsorted(d)+1, len(idx)-1)] in CPI]))
battery(px, mask_from(A2), [("SVXY", -1.0)], 1,
        "C6 SHORT SVXY (long vol) anchor k=2 -> exit on the PPI print",
        cost_bps=25.0, variants=variants, min_gap=5, event_kinds=("cpi",))

# also the legal long-vol expression
battery(px, mask_from(A2), [("UVXY", 1.0)], 1,
        "C6-alt LONG UVXY anchor k=2 -> exit on the PPI print",
        cost_bps=15.0, variants=None, min_gap=5, event_kinds=("cpi",))

# ---------------------------------------------------------------- F. carry
print("\n=== F. cost of carrying a SHORT SVXY ===")
sv = px["SVXY"].pct_change().dropna()
sv_post = sv[sv.index >= LEV_CUT]
print(f"  SVXY unconditional daily drift post-2018: {100*sv_post.mean():+.3f}%/day "
      f"({100*sv_post.mean()*252:+.1f}%/yr) -- a short pays this")
print(f"  SVXY daily sd post-2018: {100*sv_post.std():.3f}%  "
      f"worst 1d for a short: {100*sv_post.max():+.2f}% on "
      f"{sv_post.idxmax().date()}")
print(f"  worst 1d for a short since 2019: "
      f"{100*sv_post[sv_post.index>='2019-01-01'].max():+.2f}%")
uv = px["UVXY"].pct_change().dropna()
uv_post = uv[uv.index >= "2018-01-01"]
print(f"  UVXY (the legal long-vol leg) drift: {100*uv_post.mean():+.3f}%/day "
      f"({100*uv_post.mean()*252:+.1f}%/yr) -- a LONG pays this")
