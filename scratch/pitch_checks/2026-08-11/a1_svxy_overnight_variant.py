"""C1 postscript: the ONE piece of the window that is alive post-2018.

B4/D1 found that after the leverage break the only segment of C1's hold with a
pulse is the OVERNIGHT into the print (MOC the session before -> MOO on the
print session): +32.5 bps 2018+, +44.6 bps 2021+, against a tdom-matched
non-CPI SVXY overnight of ~+6 bps.  That is a DIFFERENT trade from C1 (h=1
close-to-close, hold +3 td), so it cannot rescue the candidate - but a
near-miss owes the number that would turn it on, and it owes the same beta
test that killed the parent.

Checks:
  K. SPY's own overnight into CPI, and the SVXY overnight residual after SPY.
  L. era stability, LOYO and the worst overnight.
  M. cost arithmetic for an overnight-only SVXY round trip.

Run: python scratch/pitch_checks/2026-08-11/a1_svxy_overnight_variant.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1_lab import SVXY_LEV_BREAK, event_sessions, loyo, tdom_of  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, load_events, load_prices, show, sign_test, summarize,
)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

px = close_panel(["SVXY", "SPY"])
all_dates = px.index
TDOM = tdom_of(all_dates)
ev = load_events(["cpi"])
cpi_sess = event_sessions(ev, "cpi", all_dates)

pr = load_prices(["SVXY", "SPY"])
on = {t: (pr[t]["Open"] / pr[t]["Close"].shift(1) - 1.0).dropna()
      for t in ("SVXY", "SPY")}

print("=" * 78)
print("K. SPY overnight into CPI, and the SVXY overnight residual after SPY")
print("=" * 78)
rows = []
for t in ("SPY", "SVXY"):
    s = on[t]
    for lbl, lo, hi in (("full", pd.Timestamp("2000-01-01"), pd.Timestamp("2030-01-01")),
                        ("-1x/pre-2018", pd.Timestamp("2000-01-01"), SVXY_LEV_BREAK),
                        ("2018+", SVXY_LEV_BREAK, pd.Timestamp("2030-01-01")),
                        ("2021+", pd.Timestamp("2021-01-01"), pd.Timestamp("2030-01-01"))):
        sub = s[(s.index >= lo) & (s.index < hi)]
        ce = sub.reindex(cpi_sess).dropna()
        if len(ce) < 5:
            continue
        non = sub[~sub.index.isin(cpi_sess)]
        ct = set(TDOM.reindex(ce.index).dropna().astype(int).tolist())
        non_t = non[TDOM.reindex(non.index).isin(ct)]
        st = summarize(ce.values, f"{t} overnight {lbl}")
        st["ctl_tdom_pct"] = 100 * non_t.mean()
        st["excess_bps"] = 1e4 * (ce.mean() - non_t.mean())
        st["signp"] = sign_test(int((ce.values > 0).sum()), len(ce))
        rows.append(st)
show(rows, "K1. SVXY vs SPY, close->open into the 08:30 print")

print("\nK2. regress the SVXY overnight on the SAME overnight in SPY "
      "(fit on all days in the era), then look at the CPI residual")
for lbl, lo, hi in (("pre-2018 (-1x)", pd.Timestamp("2000-01-01"), SVXY_LEV_BREAK),
                    ("2018+ (-0.5x, LIVE)", SVXY_LEV_BREAK, pd.Timestamp("2030-01-01")),
                    ("2021+", pd.Timestamp("2021-01-01"), pd.Timestamp("2030-01-01"))):
    j = pd.concat([on["SVXY"].rename("y"), on["SPY"].rename("x")], axis=1).dropna()
    j = j[(j.index >= lo) & (j.index < hi)]
    if len(j) < 60:
        continue
    X = np.column_stack([np.ones(len(j)), j["x"].values])
    b, *_ = np.linalg.lstsq(X, j["y"].values, rcond=None)
    r = pd.Series(j["y"].values - X @ b, index=j.index)
    rc = r.reindex(cpi_sess).dropna()
    raw = j["y"].reindex(cpi_sess).dropna()
    spy_c = j["x"].reindex(cpi_sess).dropna()
    print(f"  {lbl:<22} N={len(rc):<4} beta_SPY={b[1]:+.2f}  "
          f"RAW {1e4*raw.mean():+6.1f}bps -> RESIDUAL {1e4*rc.mean():+6.1f}bps "
          f"t={rc.mean()/(rc.std(ddof=1)/np.sqrt(len(rc))):+.2f} "
          f"hit={100*(rc>0).mean():.1f}% signp={sign_test(int((rc>0).sum()), len(rc)):.4f}")
    print(f"                         SPY overnight on the same days "
          f"{1e4*spy_c.mean():+6.1f} bps vs all days {1e4*j['x'].mean():+6.1f} bps")

print("\n" + "=" * 78)
print("L. era stability, LOYO and the tail of the overnight")
print("=" * 78)
s = on["SVXY"]
post = s.reindex(cpi_sess).dropna()
post = post[post.index >= SVXY_LEV_BREAK]
lo_ = loyo(post.index, post.values)
print(lo_.round(3).to_string(index=False))
i = lo_["mean_pct"].idxmin()
print(f"  LOYO FLOOR (2018+): dropping {int(lo_.loc[i,'drop_year'])} leaves "
      f"{1e2*lo_.loc[i,'mean_pct']:.1f} bps")
print(f"  worst overnight 2018+: {1e4*post.min():+.0f} bps on "
      f"{post.idxmin().date()};  worst 5: "
      f"{[f'{d.date()} {1e4*v:+.0f}bps' for d, v in post.nsmallest(5).items()]}")
yr = pd.DataFrame({"sum_bps": 1e4 * post.groupby(post.index.year).sum(),
                   "n": post.groupby(post.index.year).size(),
                   "mean_bps": 1e4 * post.groupby(post.index.year).mean()})
print("\n  year histogram (2018+):")
print(yr.round(1).to_string())

print("\n" + "=" * 78)
print("M. cost for an overnight-only round trip")
print("=" * 78)
print("""  SVXY round trip ~6 bps (2-4 bps half-spread each way on a ~$50 tape).
  An MOO exit adds auction slippage; call the honest all-in 8-10 bps.
  2021+ excess over the tdom-matched non-CPI overnight: see K1.
  The >=5x-cost bar therefore needs the excess above ~40-50 bps.""")
