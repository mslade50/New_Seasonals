"""C1 round 1 - short-vol carry / complacency fade at a 98.4th-pctile VIX3M/VIX
term spread.

Round-1 obligations answered here:
 1. does it exist, vs all-days + own-drift + local +/-126td controls
 2. N, worst window, era stability (pre/post 2018), + the 2018-02 SVXY
    leverage cut (-1x -> -0.5x) stated and re-run post-cut only
 3. registry collisions: pre-expiry SVXY carry (dead 08-07), post-CPI vol
    crush (dead after 2018), SVXY as pre-FOMC leg, naked short UVXY
 4. book overlap
 5. cost (SVXY round trip ~8-10 bps)
 6. tail risk inside the window (vix_expiry 08-19 +4td, opex 08-21 +6td)
BOTH directions tested: long SVXY (carry) and short SVXY (complacency fade).
Mandatory: beta-neutral SPY residual on the vehicle leg.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pitch_lab import (  # noqa: E402
    battery, bootstrap_p_le0, cluster_note, declusters, load_prices,
    local_control, show, sign_test, summarize, vehicle_ret, fwd_lag,
)

LEV_CUT = pd.Timestamp("2018-03-01")   # SVXY -1x -> -0.5x (Feb 2018 event)
TK = ["^VIX", "^VIX3M", "SVXY", "SPY", "UVXY", "VXX"]
px_raw = load_prices(TK)

print("=== 0. data spans (single-instrument series, never a panel) ===")
for t in TK:
    if t in px_raw:
        c = px_raw[t]["Close"].dropna()
        print(f"  {t:8s} {c.index[0].date()} .. {c.index[-1].date()}  n={len(c)}")

vx = px_raw["^VIX"]["Close"].dropna()
v3 = px_raw["^VIX3M"]["Close"].dropna()
spread = (v3 / vx - 1.0).dropna()

# rolling trailing-252d percentile of the SPREAD LEVEL (matches 00_map_facts)
pct = spread.rolling(253).apply(lambda w: (w.iloc[-1] > w[:-1]).mean() * 100.0,
                                raw=False)
print(f"\nTODAY check: spread {spread.iloc[-1]*100:.1f}%  pctile "
      f"{pct.iloc[-1]:.1f}  on {spread.index[-1].date()}"
      f"   (map asserts +27.4% / 98.4)")

# ---- trigger definitions -------------------------------------------------
def mk(thr, lb=253, s=spread):
    p = s.rolling(lb).apply(lambda w: (w.iloc[-1] > w[:-1]).mean() * 100.0,
                            raw=False)
    return p >= thr

trig98 = mk(98.0)
print(f"trigger days (pctile>=98, full VIX3M history): {int(trig98.sum())}"
      f"  first {trig98[trig98].index[0].date()}")

# ------------------------------------------------------------------------
# A. does the pattern exist in the INDEX at all (mechanism, untradeable)
# ------------------------------------------------------------------------
print("\n=== A. mechanism in the index (^VIX level change, lag=1) ===")
rows = []
for h in (5, 10):
    rv = fwd_lag(vx, h, 1)
    t = vx.index[trig98.reindex(vx.index, fill_value=False).values
                 & rv.notna().values]
    epi = declusters(t, 21, vx.index)
    rows.append(summarize(rv.loc[epi].values, f"VIX chg h={h} COND epi (N={len(epi)})"))
    rows.append(summarize(rv.dropna().values, f"VIX chg h={h} all days"))
show(rows, "VIX level forward change at extreme contango")
print("  (VIX UP after extreme contango => fade side; DOWN => carry side)")

# spread mean-reversion sanity: does the spread itself compress?
for h in (5, 10):
    sd = spread.shift(-(1 + h)) - spread.shift(-1)
    t = spread.index[trig98.values & sd.notna().values]
    epi = declusters(t, 21, spread.index)
    print(f"  spread change h={h}: cond epi mean {100*sd.loc[epi].mean():+.2f}pp "
          f"(N={len(epi)}) vs all days {100*sd.dropna().mean():+.2f}pp")

# ------------------------------------------------------------------------
# B. the vehicle: SVXY both directions, full sample then post-cut
# ------------------------------------------------------------------------
svxy = px_raw["SVXY"]["Close"].dropna()
spy = px_raw["SPY"]["Close"].dropna()
pxv = pd.DataFrame({"SVXY": svxy, "SPY": spy}).dropna()

for era_lbl, sub in (("FULL 2011+", pxv), ("POST-CUT 2018-03+", pxv[pxv.index >= LEV_CUT])):
    for h in (5, 10):
        for side, legs in (("LONG SVXY (carry)", [("SVXY", 1.0)]),
                           ("SHORT SVXY (fade)", [("SVXY", -1.0)])):
            m = trig98.reindex(sub.index, fill_value=False)
            battery(sub, m, legs, h,
                    f"B. {side}  {era_lbl}", cost_bps=9.0,
                    variants={
                        "pctile>=90": mk(90.0).reindex(sub.index, fill_value=False),
                        "pctile>=95": mk(95.0).reindex(sub.index, fill_value=False),
                        "pctile>=97": mk(97.0).reindex(sub.index, fill_value=False),
                        "pctile>=99": mk(99.0).reindex(sub.index, fill_value=False),
                        "lb=126,>=98": mk(98.0, 127).reindex(sub.index, fill_value=False),
                        "lb=504,>=98": mk(98.0, 505).reindex(sub.index, fill_value=False),
                        "abs spread>=25%": (spread >= 0.25).reindex(sub.index, fill_value=False),
                    },
                    min_gap=21, event_kinds=("vix_expiry",))

# ------------------------------------------------------------------------
# C. MANDATORY beta-neutral residual of the SVXY leg vs SPY
# ------------------------------------------------------------------------
print("\n\n=== C. beta-neutral residual (SVXY vs SPY), the 08-11 killer ===")
res_rows = []
for era_lbl, sub in (("FULL 2011+", pxv), ("POST-CUT 2018-03+", pxv[pxv.index >= LEV_CUT])):
    for h in (5, 10):
        rs = fwd_lag(sub["SVXY"], h, 1)
        rm = fwd_lag(sub["SPY"], h, 1)
        ok = rs.notna() & rm.notna()
        trg = sub.index[trig98.reindex(sub.index, fill_value=False).values & ok.values]
        epi = declusters(trg, 21, sub.index[ok.values])
        # fit on NON-trigger days only (control-day fit)
        ctl = sub.index[ok.values].difference(trg)
        b = np.polyfit(rm.loc[ctl].values, rs.loc[ctl].values, 1)
        beta, alpha = b[0], b[1]
        resid = rs - (alpha + beta * rm)
        r2 = np.corrcoef(rm.loc[ctl].values, rs.loc[ctl].values)[0, 1] ** 2
        base_hit = float((resid.loc[ctl] > 0).mean())
        ep = resid.loc[epi].values
        wins = int((ep > 0).sum())
        res_rows.append({
            "era": era_lbl, "h": h, "N_epi": len(epi), "beta": round(beta, 3),
            "R2": round(r2, 3),
            "raw_pct": round(100 * rs.loc[epi].mean(), 3),
            "spy_pct": round(100 * rm.loc[epi].mean(), 3),
            "resid_pct": round(100 * ep.mean(), 3),
            "resid_hit": round(100 * (ep > 0).mean(), 1),
            "base_hit": round(100 * base_hit, 1),
            "sign_p_vs_base": round(sign_test(wins, len(ep), base_hit), 4),
            "worst_resid": round(100 * ep.min(), 2),
            "boot_p_le0": round(bootstrap_p_le0(ep), 3),
        })
show(res_rows, "C. residual table (long-SVXY sign; flip sign for the fade)")

# ------------------------------------------------------------------------
# D. MANDATORY placebo anchor ladder (offsets around the trigger)
# ------------------------------------------------------------------------
print("\n=== D. placebo anchor ladder: shift the trigger set +/- k sessions ===")
for era_lbl, sub in (("FULL 2011+", pxv), ("POST-CUT 2018-03+", pxv[pxv.index >= LEV_CUT])):
    for h in (5, 10):
        ret = vehicle_ret(sub, [("SVXY", 1.0)], h, 1)
        ok = ret.notna()
        idx = sub.index
        pos = pd.Series(range(len(idx)), index=idx)
        base_trg = idx[trig98.reindex(idx, fill_value=False).values]
        lad = []
        for k in range(-10, 11):
            shifted = []
            for d in base_trg:
                p = pos.get(d)
                if p is None:
                    continue
                q = p + k
                if 0 <= q < len(idx) and ok.iloc[q]:
                    shifted.append(idx[q])
            if not shifted:
                continue
            s = pd.DatetimeIndex(sorted(set(shifted)))
            epi = declusters(s, 21, idx[ok.values])
            v = ret.loc[epi].values
            lad.append({"k": k, "n_epi": len(epi),
                        "long_svxy_pct": round(100 * np.nanmean(v), 3),
                        "hit": round(100 * float(np.nanmean(v > 0)), 1)})
        df = pd.DataFrame(lad)
        real = df[df["k"] == 0]["long_svxy_pct"].iloc[0]
        rank = int((df["long_svxy_pct"] >= real).sum())
        print(f"\n  {era_lbl} h={h}: real k=0 = {real:+.3f}%   rank {rank} of "
              f"{len(df)} offsets (empirical p {rank/len(df):.3f})")
        print(df.to_string(index=False))

# ------------------------------------------------------------------------
# E. tomorrow-specific: what happened at the two nearest structural events
# ------------------------------------------------------------------------
print("\n=== E. tail risk: vix_expiry / opex inside the hold (post-cut) ===")
sub = pxv[pxv.index >= LEV_CUT]
for h in (5, 10):
    ret = vehicle_ret(sub, [("SVXY", 1.0)], h, 1)
    trg = sub.index[trig98.reindex(sub.index, fill_value=False).values & ret.notna().values]
    epi = declusters(trg, 21, sub.index)
    v = ret.loc[epi].values
    n_dd = int((v < -0.05).sum())
    print(f"  h={h}: N={len(epi)} episodes, {n_dd} worse than -5%, "
          f"worst {100*np.nanmin(v):.2f}%, 5th pctile {100*np.nanpercentile(v,5):.2f}%")
    print(f"    concentration: {cluster_note(epi, v)}")
