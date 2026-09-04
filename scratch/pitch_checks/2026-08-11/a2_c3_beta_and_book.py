"""C3 kills 3, 4 and 5.

3. MECHANISM: is XLE's cell just levered crude? Regress the XLE cell return on
   the SAME-WINDOW USO return and read the intercept. If alpha is ~0 the
   "equity underreaction" thesis is false and the trade is a crude proxy.
   Also compare XLE against a vol-matched USO position.
4. DIRECTION vs the live book: has the systematic book historically been SHORT
   energy names on these exact days? Read data/backtest_trades_full.parquet.
5. ANTI-RIP-OFF: is this materially Sector BO or LT Trend ST OS on XLE?
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import close_panel, fwd_lag, declusters, summarize, sign_test  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
ROOT = Path(__file__).resolve().parents[3]

px = close_panel(["USO", "XLE", "SPY", "DBC", "XOP", "OIH"])
uso_1d = px["USO"].pct_change()
s = px["XLE"].dropna()
base = (uso_1d >= 0.05).reindex(s.index).fillna(False)
epi = declusters(s.index[base.values], 5, s.index)

H = 3
xle_f = fwd_lag(px["XLE"], H, lag=1)
uso_f = fwd_lag(px["USO"], H, lag=1)
spy_f = fwd_lag(px["SPY"], H, lag=1)
ok = xle_f.notna() & uso_f.notna() & spy_f.notna()
epi = pd.DatetimeIndex([d for d in epi if bool(ok.get(d, False))])
print(f"episodes used: {len(epi)}  (h={H}, lag=1)")

# ---------------------------------------------------------------------------
# 3. mechanism -- crude beta vs producer alpha
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("3. IS XLE JUST LEVERED CRUDE? regress cell XLE return on same-window USO")
print("=" * 100)


def ols(y, X, names):
    X = np.column_stack([np.ones(len(y))] + list(X))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    r2 = 1 - (resid @ resid) / (((y - y.mean()) ** 2).sum())
    print(f"  n={len(y)}  R2={r2:.3f}")
    for nm, b, e in zip(["alpha"] + names, beta, se):
        print(f"    {nm:<16} {100*b if nm=='alpha' else b:+8.4f}"
              f"{'%' if nm=='alpha' else ' '}  se {100*e if nm=='alpha' else e:.4f}"
              f"  t {b/e:+6.2f}")
    return beta, se, resid


y = xle_f.loc[epi].values
u = uso_f.loc[epi].values
sp = spy_f.loc[epi].values

print("\n(a) UNCONDITIONAL relationship, all days in span (what the beta IS):")
allmask = ok & (px.index >= epi[0]) & (px.index <= epi[-1])
ya, ua, spa = xle_f[allmask].values, uso_f[allmask].values, spy_f[allmask].values
b_all, _, _ = ols(ya, [ua], ["beta_USO"])
b_all2, _, _ = ols(ya, [ua, spa], ["beta_USO", "beta_SPY"])

print("\n(b) CONDITIONAL cell, XLE ~ USO over the SAME h=3 window:")
b_c, se_c, res_c = ols(y, [u], ["beta_USO"])
print(f"  --> alpha = {100*b_c[0]:+.3f}% per episode with t = {b_c[0]/se_c[0]:+.2f}")

print("\n(c) CONDITIONAL cell, XLE ~ USO + SPY (strip the market too):")
b_c2, se_c2, _ = ols(y, [u, sp], ["beta_USO", "beta_SPY"])
print(f"  --> alpha = {100*b_c2[0]:+.3f}% with t = {b_c2[0]/se_c2[0]:+.2f}")

print("\n(d) RESIDUAL cell: XLE minus its unconditional-beta crude exposure")
resid_cell = y - (b_all[0] + b_all[1] * u)
st = summarize(resid_cell)
print(f"  residual mean {st['mean_pct']:+.3f}%  hit {st['hit']:.1f}  t {st['t']:+.2f}  "
      f"sign p {sign_test(int((resid_cell>0).sum()), len(resid_cell)):.4f}")

print("\n(e) USO's OWN cell, and XLE vs a VOL-MATCHED USO position:")
st_x = summarize(y)
st_u = summarize(u)
own_x = xle_f[ok].mean() * 100
own_u = uso_f[ok].mean() * 100
print(f"  XLE cell: mean {st_x['mean_pct']:+.3f}% sd {st_x['sd_pct']:.2f} "
      f"excess {st_x['mean_pct']-own_x:+.3f}% hit {st_x['hit']:.1f}")
print(f"  USO cell: mean {st_u['mean_pct']:+.3f}% sd {st_u['sd_pct']:.2f} "
      f"excess {st_u['mean_pct']-own_u:+.3f}% hit {st_u['hit']:.1f}")
k = st_x["sd_pct"] / st_u["sd_pct"]
print(f"  vol-match: scale USO by {k:.3f} -> mean {k*st_u['mean_pct']:+.3f}% "
      f"excess {k*(st_u['mean_pct']-own_u):+.3f}% at XLE's risk")
print(f"  per unit of risk (mean/sd): XLE {st_x['mean_pct']/st_x['sd_pct']:.3f} "
      f"vs USO {st_u['mean_pct']/st_u['sd_pct']:.3f}")

# ---------------------------------------------------------------------------
# 4. direction vs the live systematic book
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("4. DIRECTION CHECK -- what has the systematic book DONE on these days?")
print("=" * 100)
led = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
led["Signal Date"] = pd.to_datetime(led["Signal Date"])
led["Entry Date"] = pd.to_datetime(led["Entry Date"])

ENERGY = {"XLE", "XOP", "OIH", "USO", "UNG", "DBC", "XVE", "ERX", "ERY", "GUSH", "DRIP",
          "CVX", "XOM", "COP", "EOG", "SLB", "HAL", "OXY", "PSX", "VLO", "MPC", "KMI",
          "WMB", "OKE", "BKR", "DVN", "FANG", "HES", "MRO", "APA", "PXD", "NOV", "FTI"}
en = led[led["Ticker"].isin(ENERGY)].copy()
print(f"energy trades in the 23y ledger: {len(en)}")
print(en.groupby(["Strategy", "Direction"]).size().to_string())

# trades whose SIGNAL DATE is a trigger day (day-level, not just episodes)
trig_days = set(pd.DatetimeIndex(s.index[base.values]))
on_trig = en[en["Signal Date"].isin(trig_days)]
print(f"\nenergy trades SIGNALLED on a USO>=+5% day: {len(on_trig)}")
if len(on_trig):
    print(on_trig.groupby(["Strategy", "Direction"]).agg(
        n=("R_Multiple", "size"), avgR=("R_Multiple", "mean")).round(2).to_string())
    print(on_trig[["Strategy", "Ticker", "Direction", "Signal Date", "Entry Date",
                   "R_Multiple"]].to_string(index=False))

# trades OPEN across the 3-day hold that follows the entry
print("\nenergy trades whose HOLD overlaps the C3 window (signal D -> exit D+4):")
rows = []
idx = px.index
pos = pd.Series(range(len(idx)), index=idx)
for d in epi:
    p = pos[d]
    lo, hi = idx[p + 1], idx[min(p + 1 + H, len(idx) - 1)]
    o = en[(en["Entry Date"] <= hi) & (pd.to_datetime(en["Exit Date"]) >= lo)]
    for _, r_ in o.iterrows():
        rows.append({"trigger": d.date(), "strat": r_["Strategy"], "tkr": r_["Ticker"],
                     "dir": r_["Direction"], "entry": r_["Entry Date"].date(),
                     "R": round(r_["R_Multiple"], 2)})
ov = pd.DataFrame(rows)
if len(ov):
    print(f"  {len(ov)} overlapping energy positions")
    print(ov.groupby(["strat", "dir"]).agg(n=("R", "size"), avgR=("R", "mean")).round(2).to_string())
    print(f"  SHORT overlaps: {int((ov['dir'].str.lower()=='short').sum())}  "
          f"LONG overlaps: {int((ov['dir'].str.lower()=='long').sum())}")
else:
    print("  none")

# ---------------------------------------------------------------------------
# 5. anti-rip-off -- Sector BO / LT Trend ST OS on XLE
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("5. ANTI-RIP-OFF -- Sector BO / LT Trend ST OS filters vs this trigger")
print("=" * 100)
sys.path.insert(0, str(ROOT))
import strategy_config as sc  # noqa: E402

for want in ("Sector BO", "LT Trend ST OS"):
    for cfg in sc.STRATEGY_BOOK:
        if cfg.get("name") == want:
            print(f"\n--- {want} ---")
            print(f"  id: {cfg.get('id')}")
            su = cfg.get("setup", {})
            print(f"  thesis: {su.get('thesis')}")
            print(f"  key_filters: {su.get('key_filters')}")
            print(f"  direction: {cfg.get('direction', cfg.get('settings', {}).get('direction'))}")
            uni = cfg.get("universe_tickers") or []
            print(f"  XLE in universe: {'XLE' in uni}   universe size {len(uni)}")
            ex = cfg.get("exit_summary", {})
            print(f"  exits: {ex.get('primary_exit')}")

# do those two ever fire on XLE at all, and ever on a trigger day?
for want in ("Sector BO", "LT Trend ST OS", "52wh Breakout", "ATR Extended Gap Up",
             "Overbot Vol Spike", "3x ETF Overbot Fade"):
    sub = led[(led["Strategy"] == want)]
    xle_hits = sub[sub["Ticker"] == "XLE"]
    trig_hits = sub[sub["Signal Date"].isin(trig_days)]
    print(f"\n{want:<24} total {len(sub):<5} on XLE {len(xle_hits):<4} "
          f"on a USO>=5% day {len(trig_hits):<4} "
          f"({'LONG' if len(sub) and str(sub['Direction'].mode()[0]).lower()=='long' else 'SHORT'} book)")
    if len(trig_hits):
        print("   ", trig_hits.groupby("Direction").agg(
            n=("R_Multiple", "size"), avgR=("R_Multiple", "mean")).round(2).to_dict())
