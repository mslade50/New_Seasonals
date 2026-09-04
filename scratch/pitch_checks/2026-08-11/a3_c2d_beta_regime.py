"""C2 last kill attempt: the BETA REGIME.

Two things fell out of a3_c2_dev.py that deserve their own test.

1. The five worst sessions in the whole 311-observation history are
   2000-11-15, 2001-01-12, 2001-07-17, 2001-09-17 and 2002-02-19 -- every one
   of them with a PIT beta between 1.77 and 2.39. TODAY'S beta is 1.482, well
   above the 1.233 full-sample mean and in the top decile of the sample. If
   the cell only works when beta is near 1 and blows up when beta is high,
   today is the wrong day for it.

2. The V4 limit variant in the dev script prints a 90.9% hit rate at 0.25 ATR.
   That number is NOT to be believed and this script says why in code: the
   joint-fill condition (QQQ trades 0.25 ATR below the prior close AND SPY
   trades 0.25 ATR above it, same session) is itself an intraday-reversal
   filter, and the two legs' fills are not simultaneous. Test: measure the
   SAME both-touch condition on NON-CPI days. If it prints the same absurd
   hit rate there, the limit variant is measuring the touch, not the print.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    load_prices, close_panel, load_events, declusters, summarize, sign_test,
    bootstrap_p_le0, wilder_atr,
)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)

OHLC = load_prices(["SPY", "QQQ"])
C = close_panel(["SPY", "QQQ"]).dropna()
all_dates = C.index
pos = pd.Series(np.arange(len(all_dates)), index=all_dates)
O = pd.DataFrame({t: OHLC[t]["Open"] for t in OHLC}).reindex(all_dates)
Hh = pd.DataFrame({t: OHLC[t]["High"] for t in OHLC}).reindex(all_dates)
L = pd.DataFrame({t: OHLC[t]["Low"] for t in OHLC}).reindex(all_dates)
ATR = pd.DataFrame({t: wilder_atr(Hh[t], L[t], C[t]) for t in ("SPY", "QQQ")},
                   index=all_dates)
rq, rs = C["QQQ"].pct_change(), C["SPY"].pct_change()
BETA = rq.rolling(126).cov(rs) / rs.rolling(126).var()

ev = load_events(["cpi"])
anch = []
for d in pd.DatetimeIndex(sorted(ev["date"].unique())):
    loc = all_dates.searchsorted(d)
    if loc >= len(all_dates):
        continue
    j = loc - 2
    if 0 <= j < len(all_dates):
        anch.append(all_dates[j])
anch = declusters(pd.DatetimeIndex(sorted(set(anch))), 5, all_dates)
p_a = pos.reindex(anch).dropna().astype(int)
p_a = p_a[p_a + 2 < len(all_dates)]
i_e, i_p = p_a.values + 1, p_a.values + 2
b = BETA.values[p_a.values]
cq, cs = C["QQQ"].values, C["SPY"].values
v1 = (cq[i_p] / cq[i_e] - 1) - b * (cs[i_p] / cs[i_e] - 1)

print("=" * 105)
print("1. BETA REGIME. today's PIT 126d beta = %.3f" % BETA.iloc[-1])
print("=" * 105)
q = pd.Series(BETA.dropna()).rank(pct=True)
print(f"   that sits at the {100*q.loc[all_dates[-1]]:.0f}th percentile of the "
      f"whole 2000-2026 beta history.")
D = pd.DataFrame({"date": all_dates[i_e], "r": v1, "beta": b}).dropna()
bins = [("beta < 1.0", D.beta < 1.0), ("1.0 <= beta < 1.2", (D.beta >= 1.0) & (D.beta < 1.2)),
        ("1.2 <= beta < 1.4", (D.beta >= 1.2) & (D.beta < 1.4)),
        ("beta >= 1.4 (TODAY 1.48)", D.beta >= 1.4),
        ("beta >= 1.4, ex 2000-2003", (D.beta >= 1.4) & (D.date.dt.year > 2003))]
rows = []
for lbl, m in bins:
    v = D[m]["r"].values
    st = summarize(v)
    if not st["n"]:
        continue
    w = int((v > 0).sum())
    rows.append(dict(bucket=lbl, N=st["n"], mean=round(st["mean_pct"], 3),
                     hit=round(st["hit"], 1), t=round(st["t"], 2),
                     signp=round(sign_test(w, st["n"]), 4),
                     worst=round(st["worst_pct"], 2),
                     bootP=round(bootstrap_p_le0(v), 3),
                     yrs=f"{int(D[m].date.dt.year.min())}-{int(D[m].date.dt.year.max())}"))
print(pd.DataFrame(rows).to_string(index=False))
print("\n   correlation between PIT beta and the realised spread return: "
      f"{np.corrcoef(D.beta, D.r)[0, 1]:+.3f}")
print("   sd of the spread by beta bucket (the tail question):")
for lbl, m in bins[:4]:
    v = D[m]["r"].values
    print(f"     {lbl:<26} sd {100*v.std(ddof=1):.2f}%  worst {100*v.min():+.2f}%")

print("\n" + "=" * 105)
print("2. IS THE V4 LIMIT VARIANT REAL? measure the same both-touch condition")
print("   on NON-CPI days. If it prints the same hit rate, V4 is measuring the")
print("   touch, not the print, and it is not a variant of this idea at all.")
print("=" * 105)
lq, hs_ = L["QQQ"].values, Hh["SPY"].values
aq, as_ = ATR["QQQ"].values, ATR["SPY"].values
cpi_entry = set(i_e.tolist())
rows = []
for k in (0.10, 0.25, 0.50):
    for lbl, idx in (("CPI print sessions", i_p),
                     ("ALL sessions (the null)",
                      np.array([j for j in range(200, len(all_dates) - 1)
                                if (j - 1) not in cpi_entry]))):
        ie = idx - 1
        lim_q = cq[ie] - k * aq[ie]
        lim_s = cs[ie] + k * as_[ie]
        fill = (lq[idx] <= lim_q) & (hs_[idx] >= lim_s)
        bb = BETA.values[ie - 1]
        r = (cq[idx] / lim_q - 1) - bb * (cs[idx] / lim_s - 1)
        r = r[fill & np.isfinite(r)]
        st = summarize(r)
        if not st["n"]:
            continue
        w = int((r > 0).sum())
        rows.append(dict(k_atr=k, sample=lbl, fill_pct=round(100 * np.nanmean(fill), 1),
                         N=st["n"], mean=round(st["mean_pct"], 3),
                         hit=round(st["hit"], 1), t=round(st["t"], 2),
                         signp=round(sign_test(w, st["n"]), 5)))
print(pd.DataFrame(rows).to_string(index=False))
print("""
   If the ALL-sessions rows look like the CPI rows, V4's 90.9% hit rate is a
   property of the both-touch FILTER, not of the CPI print: requiring QQQ to
   trade a quarter-ATR below the prior close while SPY simultaneously trades a
   quarter-ATR above it selects intraday spread dislocations, and measuring
   from those prices to the close books the snap-back. On daily bars the two
   legs' touches are also not simultaneous, so a real order would sit naked in
   one index for an unknown part of the session. V4 is NOT a usable variant.""")
