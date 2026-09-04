"""C9 round 1: EWZ thrust into a payrolls print.

Registry constraint that governs this: 2026-08-20 killed "long KWEB on a dollar
washout" on the REFERENCE CLASS -- permuting the identical rule across 11 EM /
international vehicles gave P(max name excess >= the pitched name) = 0.283 at
h=5 and 0.641 at h=10, and the pitched name sat BELOW the null's median
best-of-11. So the reference class runs FIRST here, before anything is credited
to Brazil.

Also owed:
  - EWZ is a currency-plus-equity compound: report the beta against EEM and the
    residual. An EM-beta trade with a Brazil label is not a Brazil idea.
  - the two co-resident z10 definitions disagree by 0.27 on EWZ today
    (+2.23 tape / +1.96 pitch_lab). Print both (2026-09-02 registry entry,
    which used EWZ as its own example).

Live geometry: k=-2 NFP anchor 2026-09-02, entry today's close (lag=1).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import (anchor_positions, close_panel, declusters, load_events,
                       load_prices, local_control, pct_rank, show, sign_test,
                       summarize, vehicle_ret, zscore, cluster_note,
                       bootstrap_p_le0, era_split)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)

CLASS = ["EWZ", "EEM", "FXI", "EWW", "EWY", "EWT", "EWJ", "INDA", "RSX", "EFA"]
PITCHED = "EWZ"
LIVE_K = -2
THRUST = 90.0     # 5d return rank; EWZ is at 96.0

print("=" * 78)
print("C9  EM thrust into payrolls -- REFERENCE CLASS FIRST")
print("=" * 78)

raw = load_prices(CLASS + ["UUP", "SPY"])
nfp = load_events(["nfp"])["date"]

# ---- the two z10 definitions, on EWZ, as the registry demands -------------
ewz = raw["EWZ"]["Close"].dropna()
z_lab = zscore(ewz, 10).iloc[-1]
r10 = ewz.pct_change(10)
d1 = ewz.pct_change()
z_tape = (r10 / (d1.rolling(21).std() * np.sqrt(10))).iloc[-1]
print(f"\nEWZ z10  pitch_lab.zscore = {z_lab:+.2f}   tape/_metrics_for form = "
      f"{z_tape:+.2f}   spread {abs(z_lab - z_tape):.2f}")
print(f"EWZ r5 rank(252) = {pct_rank(ewz, 5, 252).iloc[-1]:.1f}   "
      f"1d {100*(ewz.iloc[-1]/ewz.iloc[-2]-1):+.2f}%")

# ---- EWZ vs EEM: how much of this is EM beta with a Brazil label? ---------
pan = close_panel(["EWZ", "EEM", "SPY", "UUP"])
rr = pan.pct_change().dropna()
b_eem = float(np.polyfit(rr["EEM"], rr["EWZ"], 1)[0])
c_eem = float(rr[["EWZ", "EEM"]].corr().iloc[0, 1])
resid = rr["EWZ"] - b_eem * rr["EEM"]
b_spy = float(np.polyfit(rr["SPY"], rr["EWZ"], 1)[0])
print(f"\nEWZ on EEM: beta {b_eem:.3f}  corr {c_eem:.3f}  "
      f"R2 {c_eem**2:.3f}  -> {100*c_eem**2:.0f}% of EWZ daily variance is EEM")
print(f"EWZ on SPY: beta {b_spy:.3f}  corr {float(rr[['EWZ','SPY']].corr().iloc[0,1]):.3f}")
print(f"residual (EWZ - {b_eem:.2f}*EEM) ann vol {100*resid.std()*np.sqrt(252):.1f}% "
      f"vs EWZ ann vol {100*rr['EWZ'].std()*np.sqrt(252):.1f}%")

# ---- THE REFERENCE CLASS --------------------------------------------------
def cell(tkr: str, h: int, lag: int = 1):
    s = raw[tkr]["Close"].dropna()
    cal = s.index
    px1 = pd.DataFrame({tkr: s})
    ret = vehicle_ret(px1, [(tkr, 1.0)], h, lag)
    r5 = pct_rank(s, 5, 252)
    pos, _ = anchor_positions(cal, nfp, LIVE_K)
    anch = pd.DatetimeIndex([cal[i] for i in pos])
    m = (r5.reindex(anch) >= THRUST).fillna(False).values
    d = anch[m]
    d = d[ret.reindex(d).notna().values]
    epi = declusters(d, max(h, 5), cal)
    r = summarize(ret.reindex(epi).values, tkr)
    base = ret.dropna()
    r["ctl_all_pct"] = round(100 * base.mean(), 3)
    r["excess_pp"] = round(r["mean_pct"] - 100 * base.mean(), 3) if r["n"] else np.nan
    # local control, the honest one
    loc = local_control(cal[ret.notna().values], d)
    r["local_ctl_pct"] = round(100 * ret.reindex(loc).mean(), 3)
    r["excess_local_pp"] = round(r["mean_pct"] - 100 * ret.reindex(loc).mean(), 3) if r["n"] else np.nan
    r["dates"] = epi
    r["vals"] = ret.reindex(epi).values
    return r


for h in (2, 3, 5, 10):
    print("\n" + "#" * 78)
    print(f"# REFERENCE CLASS, h={h}: r5 rank >= {THRUST} on the k={LIVE_K} NFP anchor, "
          f"long, entry lag=1")
    print("#" * 78)
    rows = [cell(t, h) for t in CLASS]
    disp = [{k: v for k, v in r.items() if k not in ("dates", "vals")} for r in rows]
    show(disp)
    ok = [r for r in rows if r["n"] >= 3 and not np.isnan(r["excess_pp"])]
    if not ok:
        print("  no vehicle with n>=3")
        continue
    exc = np.array([r["excess_pp"] for r in ok])
    names = [r["label"] for r in ok]
    order = np.argsort(-exc)
    pitched = [r for r in ok if r["label"] == PITCHED]
    print(f"\n  ranking by excess over own all-days drift: "
          + ", ".join(f"{names[i]} {exc[i]:+.3f}" for i in order))
    if pitched:
        p = pitched[0]["excess_pp"]
        rank = int((exc > p).sum()) + 1
        frac = float((exc >= p).mean())
        print(f"  {PITCHED} excess {p:+.3f}pp ranks {rank} of {len(ok)}; "
              f"share of the class at least as good = {frac:.2f}")
        print(f"  class median excess {np.median(exc):+.3f}pp, class max "
              f"{exc.max():+.3f}pp ({names[int(np.argmax(exc))]})")
        print(f"  --> the 2026-08-20 test: is the pitched name the best of the class? "
              f"{'YES' if rank == 1 else 'NO'}")
        # the null the registry used: how often does the BEST of the class beat
        # the pitched name's excess purely by permuting which name we picked?
        print(f"  P(some other class member >= pitched) = "
              f"{float((np.delete(exc, names.index(PITCHED)) >= p).mean()):.3f}")
    e = pitched[0] if pitched else None
    if e is not None and e["n"] > 1:
        v = e["vals"]
        w = int((v > 0).sum())
        print(f"  {PITCHED} episodes: {e['n']}, record {w}-{e['n']-w}, "
              f"sign p {sign_test(w, e['n']):.4f}, "
              f"bootstrap P(mean<=0) {bootstrap_p_le0(v):.3f}")
        print(f"  {PITCHED} concentration: {cluster_note(e['dates'], v)}")
        print(f"  {PITCHED} episode dates: "
              + ", ".join(str(d.date()) for d in e['dates']))

# ---- does the NFP anchor add anything over the bare thrust? ---------------
print("\n" + "=" * 78)
print("GATE ATTRIBUTION -- the same thrust WITHOUT the payrolls anchor")
print("=" * 78)
for h in (3, 5):
    rows = []
    for t in (PITCHED, "EEM"):
        s = raw[t]["Close"].dropna()
        cal = s.index
        px1 = pd.DataFrame({t: s})
        ret = vehicle_ret(px1, [(t, 1.0)], h, 1)
        r5 = pct_rank(s, 5, 252)
        pos, _ = anchor_positions(cal, nfp, LIVE_K)
        anch = pd.DatetimeIndex([cal[i] for i in pos])
        d_ev = anch[(r5.reindex(anch) >= THRUST).fillna(False).values]
        d_all = cal[(r5 >= THRUST).fillna(False).values & ret.notna().values]
        d_no = pd.DatetimeIndex(sorted(set(d_all) - set(d_ev)))
        rows.append(summarize(ret.reindex(declusters(d_ev, max(h, 5), cal)).values,
                              f"{t} thrust AND NFP k={LIVE_K}"))
        rows.append(summarize(ret.reindex(declusters(d_no, max(h, 5), cal)).values,
                              f"{t} thrust, NO event anchor"))
        rows.append(summarize(ret.dropna().values, f"{t} all days"))
    show(rows, f"NFP-anchor attribution, h={h}")

# ---- the dollar leg the mechanism claims ----------------------------------
print("\n" + "=" * 78)
print("MECHANISM PROBE: 'a soft print is dollar-negative and EM-positive'")
print("=" * 78)
uup = raw["UUP"]["Close"].dropna()
cal = uup.index
pos, _ = anchor_positions(cal, nfp, 0)
anch = pd.DatetimeIndex([cal[i] for i in pos])
# same-day dollar move on the print, and EWZ's same-day move; do they co-move?
du = uup.pct_change().reindex(anch).dropna()
pan2 = close_panel(["EWZ", "UUP"])
de = pan2["EWZ"].pct_change().reindex(du.index)
both = pd.concat([du.rename("UUP"), de.rename("EWZ")], axis=1).dropna()
print(f"on NFP print days (n={len(both)}): corr(UUP 1d, EWZ 1d) = "
      f"{float(both.corr().iloc[0,1]):+.3f}   "
      f"beta {float(np.polyfit(both['UUP'], both['EWZ'], 1)[0]):+.2f}")
allb = pd.concat([uup.pct_change().rename("UUP"),
                  pan2["EWZ"].pct_change().rename("EWZ")], axis=1).dropna()
print(f"on ALL days (n={len(allb)}): corr = {float(allb.corr().iloc[0,1]):+.3f} "
      f"-- if the print-day corr is not more negative, the channel is not special")
soft = both[both["UUP"] < 0]
hard = both[both["UUP"] >= 0]
print(f"  dollar DOWN on the print (n={len(soft)}): EWZ same day "
      f"{100*soft['EWZ'].mean():+.3f}%   dollar UP (n={len(hard)}): "
      f"{100*hard['EWZ'].mean():+.3f}%")

print("\nDONE C9")
