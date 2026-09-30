"""kA N1 round 1: UNG volume-thrust RE-FIRE. The 09-23 rule (UNG 1d >= +5% AND
volume >= 3x its 63d mean INCLUDING today, kC_c3 convention) fires again 2
sessions after its first firing (09-22 -> 09-24). Pre-specified: LONG UNG h=2
from the next close (lag=1), continuation. Kill question: do re-firings (a
qualifying day 1..5 sessions after a prior qualifying day) pay like first
firings, or does a second thrust mark exhaustion? Split by Thursday-in-hold."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

d = load_prices(["UNG"])["UNG"].astype(float)
c, v = d["Close"], d["Volume"]
px = pd.DataFrame({"UNG": c})
idx = px.index
r1 = c.pct_change()
vr = v / v.rolling(63).mean()
trig = ((r1 >= 0.05) & (vr >= 3)).fillna(False)

print("LIVE STATE (09-23 convention: vol / 63d mean incl today)")
print(pd.DataFrame({"close": c, "r1_pct": 100 * r1, "vol_x63": vr, "trig": trig}).tail(5).round(3).to_string())
lo252 = c.rolling(252).min().iloc[-1]
hi252 = c.rolling(252).max().iloc[-1]
print(f"UNG {c.iloc[-1]:.2f}: {100*(c.iloc[-1]/lo252-1):+.1f}% over 252 low {lo252:.2f}, "
      f"{100*(c.iloc[-1]/hi252-1):+.1f}% vs 252 high {hi252:.2f}; r21 {100*(c.iloc[-1]/c.iloc[-22]-1):+.2f}%")
a = wilder_atr(d["High"].to_numpy(), d["Low"].to_numpy(), c.to_numpy())
print(f"Wilder-14 ATR {a[-1]:.3f} ({100*a[-1]/c.iloc[-1]:.2f}%)")

pos = pd.Series(range(len(idx)), index=idx)
tdays = idx[trig.values]
gap_prev = {}
last = None
for t in tdays:
    p = pos[t]
    gap_prev[t] = (p - last) if last is not None else 9999
    last = p
gp = pd.Series(gap_prev)

rets = {h: vehicle_ret(px, [("UNG", 1.0)], h) for h in (1, 2, 3, 5, 10)}
h2 = rets[2]


def thu_in_hold(dd, h=2, lag=1):
    p = pos[dd]
    return any(p + lag + k < len(idx) and idx[p + lag + k].weekday() == 3 for k in range(1, h + 1))


rows = []
tab = []
for t in tdays:
    tab.append({"date": t.date(), "gap_prev": int(gp[t]) if gp[t] < 9999 else None,
                "r1_pct": round(100 * r1[t], 2), "vol_x": round(vr[t], 2),
                "h1": round(100 * rets[1].get(t, np.nan), 2), "h2": round(100 * h2.get(t, np.nan), 2),
                "h5": round(100 * rets[5].get(t, np.nan), 2), "h10": round(100 * rets[10].get(t, np.nan), 2),
                "thu": thu_in_hold(t)})
T = pd.DataFrame(tab)
print("\nALL TRIGGER DAYS (h = lag-1 forward % ; gap_prev in td to previous trigger)")
print(T.to_string(index=False))

first10 = gp >= 10
refire5 = (gp >= 1) & (gp <= 5)
refire_in_hold = (gp >= 1) & (gp <= 3)  # inside prior firing's lag-1 h=2 hold (entry p0+1, exit p0+3)
gap2 = gp == 2
mid = (gp > 5) & (gp < 10)
print(f"\ncounts: first(gap>=10) {int(first10.sum())}, refire 1..5 {int(refire5.sum())}, "
      f"in-hold 1..3 {int(refire_in_hold.sum())}, gap==2 {int(gap2.sum())}, gap 6..9 {int(mid.sum())}")


def summ(mask, h, lab):
    ds = gp.index[mask.values]
    x = rets[h].reindex(ds).dropna().values
    r = summarize(x, lab)
    if r["n"]:
        w = int((x > 0).sum())
        r["rec"] = f"{w}-{len(x)-w}"
        r["sign_p"] = round(sign_test(w, len(x)), 4)
    return r


for h in (1, 2, 3, 5, 10):
    show([summ(first10, h, f"h={h} FIRST firings (gap>=10)"),
          summ(gp < 10, h, f"h={h} NON-first (gap<10)"),
          summ(refire5, h, f"h={h} RE-FIRE gap 1..5"),
          summ(refire_in_hold, h, f"h={h} RE-FIRE in prior hold (gap 1..3)"),
          summ(gap2, h, f"h={h} RE-FIRE gap==2 (today's shape)"),
          summ(pd.Series(True, index=gp.index), h, f"h={h} ALL trigger days")],
         f"h={h} first vs re-fire (day-level, lag 1)")

ctrl = rets[2].dropna()
print(f"\nCTRL UNG own drift h=2 all days: {100*ctrl.mean():+.3f}% (N={len(ctrl)}); "
      f"2011+ {100*ctrl[ctrl.index >= '2011-01-01'].mean():+.3f}%")
loc = local_control(ctrl.index, gp.index[refire5.values])
print(f"CTRL local +/-126td around re-fires h=2: {100*ctrl.reindex(loc).mean():+.3f}%")

# Thursday split for re-fires
ds = gp.index[refire5.values]
x = h2.reindex(ds)
th = np.array([thu_in_hold(dd) for dd in ds])
show([summarize(x.values[th], f"re-fire h=2 Thu IN hold N={int(th.sum())}"),
      summarize(x.values[~th], f"re-fire h=2 Thu OUT N={int((~th).sum())}")], "re-fire Thursday split")
ds1 = gp.index[first10.values]
x1 = h2.reindex(ds1)
th1 = np.array([thu_in_hold(dd) for dd in ds1])
show([summarize(x1.values[th1], f"first h=2 Thu IN N={int(th1.sum())}"),
      summarize(x1.values[~th1], f"first h=2 Thu OUT N={int((~th1).sum())}")], "first-firing Thursday split")

# exhaustion check: cumulative path from the FIRST firing's entry through the re-fire
print("\nEXHAUSTION: for each re-fire, its prior firing's entry->re-fire close, then re-fire lag1 h=2/5")
for dd in ds:
    p = pos[dd]
    p0 = p - int(gp[dd])
    run = c.iloc[p] / c.iloc[p0 + 1] - 1 if p0 + 1 <= p else np.nan
    print(f"  {dd.date()} prior {idx[p0].date()} run-in(prior entry->refire close) {100*run:+.2f}%  "
          f"h2 {100*h2.get(dd, np.nan):+.2f}  h5 {100*rets[5].get(dd, np.nan):+.2f}")

# NG=F cross-check, same dates, seams not a concern for the historical dates (report only)
ng = close_panel(["NG=F"]).dropna()
rng = vehicle_ret(ng, [("NG=F", 1.0)], 2)
show([summarize(rng.reindex(gp.index[first10.values]).dropna().values, "NG=F h=2 on FIRST firings"),
      summarize(rng.reindex(ds).dropna().values, "NG=F h=2 on RE-FIRES 1..5"),
      summarize(rng.dropna().values, "NG=F h=2 all days")], "NG=F front cross-check")
