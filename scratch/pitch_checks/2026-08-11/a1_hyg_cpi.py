"""C9 kill attempt: long HYG into the CPI print, exit +3 td.

Recon: h=3 +0.136% vs a tdom-matched control of +0.062, excess +0.074 (7.4
bps), hit 62.4%, sign p 0.0001, N=229.  h=1 excess +0.008, h=5 -0.023, h=10
-0.159.

Four attacks:
  N. cost.  7.4 bps of excess against an HYG round trip.  The >=5x bar.
  O. is it credit, or is it duration?  Residual of the HYG window return
     against IEF (the 2026-08-10 LQD/HYG lesson: "the credit story is
     decoration on a duration trade").
  P. is it credit, or is it equity beta?  Same residual against SPY, and
     against BOTH at once.
  Q. sign stability: horizon profile, era split, declustered episodes, the
     year histogram and the worst window.

Run: python scratch/pitch_checks/2026-08-11/a1_hyg_cpi.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1_lab import anchor_dates, loyo, tdom_control, tdom_of  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, cluster_note, declusters, fwd_lag, load_events, show,
    sign_test, summarize,
)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

OFFSET = 2
HORIZONS = (1, 2, 3, 5, 10)
px = close_panel(["HYG", "IEF", "SPY", "LQD", "TLT"])
all_dates = px.index
pos = pd.Series(np.arange(len(all_dates)), index=all_dates)
TDOM = tdom_of(all_dates)
ev = load_events(["cpi"])
hyg = px["HYG"].dropna()
anch = declusters(anchor_dates(ev, "cpi", OFFSET, all_dates), 5, all_dates)
anch = anch[anch.isin(hyg.index)]
print(f"HYG series {hyg.index[0].date()} .. {hyg.index[-1].date()};  "
      f"CPI anchors N={len(anch)}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("N. reproduce, and put it against cost")
print("=" * 78)
rows = []
for h in HORIZONS:
    f = fwd_lag(hyg, h, lag=1)
    v = f.reindex(anch).dropna()
    ctl = tdom_control(f, anch, TDOM, all_dates, pos)
    st = summarize(v.values, f"HYG h={h}")
    st["ctl_pct"] = 100 * ctl.mean()
    st["excess_bps"] = 1e4 * (st["mean_pct"] - st["ctl_pct"]) / 100
    st["signp"] = sign_test(int((v.values > 0).sum()), len(v))
    rows.append(st)
show(rows, "N1. horizon profile, excess in bps over the tdom control")
print("""
  HYG round trip: ~1 c on a ~$80 tape = ~1.2 bps half-spread, so ~2.5 bps of
  spread, plus commission and the fact that a retail-sized MOC in HYG prints
  at the closing auction.  Call the honest all-in 4-6 bps.
  The pitch bar is >=5x cost, i.e. the excess must clear ~20-30 bps.""")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("O/P. is it credit, or duration, or equity beta?")
print("=" * 78)
for h in (1, 3, 5):
    fh = fwd_lag(hyg, h, lag=1)
    fi = fwd_lag(px["IEF"], h, lag=1)
    fs = fwd_lag(px["SPY"], h, lag=1)
    j = pd.concat([fh.rename("y"), fi.rename("ief"), fs.rename("spy")],
                  axis=1).dropna()
    a = anch[anch.isin(j.index)]
    print(f"\n-- h={h} td, N_cond={len(a)} --")
    for name, cols in (("vs IEF (duration)", ["ief"]),
                       ("vs SPY (equity)", ["spy"]),
                       ("vs IEF+SPY (both)", ["ief", "spy"])):
        X = np.column_stack([np.ones(len(j))] + [j[c].values for c in cols])
        b, *_ = np.linalg.lstsq(X, j["y"].values, rcond=None)
        r = pd.Series(j["y"].values - X @ b, index=j.index)
        rc = r.reindex(a).dropna()
        raw = j["y"].reindex(a).dropna()
        betas = "  ".join(f"b_{c}={bb:+.2f}" for c, bb in zip(cols, b[1:]))
        print(f"   {name:<20} {betas:<28} RAW {1e4*raw.mean():+6.1f}bps -> "
              f"RESID {1e4*rc.mean():+6.1f}bps t={rc.mean()/(rc.std(ddof=1)/np.sqrt(len(rc))):+.2f} "
              f"hit={100*(rc>0).mean():.1f}% signp={sign_test(int((rc>0).sum()), len(rc)):.4f}")
    for c in ("ief", "spy"):
        print(f"   {c.upper()} on the same days {1e4*j[c].reindex(a).dropna().mean():+6.1f} "
              f"bps vs all days {1e4*j[c].mean():+6.1f} bps")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("Q. sign stability: eras, years, tails")
print("=" * 78)
f3 = fwd_lag(hyg, 3, lag=1)
v3 = f3.reindex(anch).dropna()
rows = []
for lbl, lo, hi in (("full", pd.Timestamp("2000-01-01"), pd.Timestamp("2030-01-01")),
                    ("pre-2018", pd.Timestamp("2000-01-01"), pd.Timestamp("2018-01-01")),
                    ("2018+", pd.Timestamp("2018-01-01"), pd.Timestamp("2030-01-01")),
                    ("2021+", pd.Timestamp("2021-01-01"), pd.Timestamp("2030-01-01"))):
    fa = f3.dropna()
    fa = fa[(fa.index >= lo) & (fa.index < hi)]
    a = anch[(anch >= lo) & (anch < hi)]
    a = a[a.isin(fa.index)]
    if len(a) < 5:
        continue
    ent = TDOM.reindex(all_dates)[pos[a].values + 1].values
    m = TDOM.reindex(fa.index).isin(set(ent.tolist())) & ~fa.index.isin(a)
    ctl = fa[m]
    v = fa.reindex(a).dropna()
    st = summarize(v.values, f"h=3 {lbl}")
    st["ctl_pct"] = 100 * ctl.mean()
    st["excess_bps"] = 1e4 * (st["mean_pct"] - st["ctl_pct"]) / 100
    st["signp"] = sign_test(int((v.values > 0).sum()), len(v))
    rows.append(st)
show(rows, "Q1. era-matched split (control drawn from the same era)")

print("\nQ2. concentration:", cluster_note(v3.index, v3.values, k=2))
yr = pd.DataFrame({"sum_bps": 1e4 * v3.groupby(v3.index.year).sum(),
                   "n": v3.groupby(v3.index.year).size(),
                   "mean_bps": 1e4 * v3.groupby(v3.index.year).mean()})
print(yr.round(1).to_string())
print(f"\nQ3. worst window {100*v3.min():+.2f}% on {v3.idxmin().date()}; "
      f"sd {100*v3.std(ddof=1):.2f}%  -> the 7 bps mean is "
      f"{100*v3.mean()/v3.std(ddof=1):.3f} of one sd")
lo_ = loyo(v3.index, v3.values)
i = lo_["mean_pct"].idxmin()
print(f"Q4. LOYO floor: dropping {int(lo_.loc[i,'drop_year'])} leaves "
      f"{1e2*lo_.loc[i,'mean_pct']:.1f} bps of RAW mean "
      f"(control is ~{1e2*100*0.00062:.1f} bps)")
print(lo_.round(3).to_string(index=False))
