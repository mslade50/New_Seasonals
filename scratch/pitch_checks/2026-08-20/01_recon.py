"""Reconnaissance: quantify today's headline states with real data before the map.
No trade claims here -- this exists so the surface map's verdicts are numbers."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

pd.set_option("display.width", 200)

# ---------------------------------------------------------------- 1. opex anchors
ev = load_events(["opex"])
px = close_panel(["SPY", "QQQ", "IWM", "TLT", "GLD", "UUP", "EEM", "XLV", "XLK", "SVXY"])
dates = px.index
print("panel", dates.min().date(), "->", dates.max().date(), len(dates))

opex = pd.DatetimeIndex(sorted(set(ev["date"]) & set(dates)))
print("\nopex anchors in panel:", len(opex), opex.min().date(), "->", opex.max().date())
# the session BEFORE opex is the entry close for a trade pitched the morning of opex-1
pos = dates.get_indexer(opex)
pre = pos - 1
pre = pre[pre >= 0]
pre_dates = dates[pre]
print("opex-1 anchors:", len(pre_dates), " last:", pre_dates[-1].date())

def wind(entry_idx, h, col="SPY"):
    """return from close at entry_idx to close at entry_idx+h"""
    s = px[col]
    out, ds = [], []
    for i in entry_idx:
        if i + h < len(dates):
            out.append(s.iloc[i + h] / s.iloc[i] - 1.0)
            ds.append(dates[i])
    return pd.DatetimeIndex(ds), np.array(out)

print("\n=== SPY from the opex-1 CLOSE, h sessions forward (all months) ===")
for h in (1, 2, 3, 5, 7, 10):
    d, v = wind(pre, h)
    base = px["SPY"].pct_change(h).shift(-h).dropna()
    print(f" h={h:>2}  N={len(v):>3}  mean={v.mean()*100:+.3f}%  hit={np.mean(v>0)*100:.1f}%   "
          f"all-days mean={base.mean()*100:+.3f}%")

print("\n=== same, by MONTH, h=5 ===")
d5, v5 = wind(pre, 5)
bym = pd.DataFrame({"m": d5.month, "r": v5})
tb = bym.groupby("m")["r"].agg(["count", "mean", lambda x: (x > 0).mean()])
tb.columns = ["N", "mean", "hit"]
tb["mean"] = (tb["mean"] * 100).round(3)
tb["hit"] = (tb["hit"] * 100).round(1)
print(tb)

print("\n=== AUGUST opex-1 anchors, horizon ladder, SPY ===")
for h in (1, 2, 3, 5, 7, 10):
    d, v = wind(pre, h)
    m = d.month == 8
    vv = v[m]
    print(f" h={h:>2}  N={len(vv):>3}  mean={vv.mean()*100:+.3f}%  hit={np.mean(vv>0)*100:.1f}%  "
          f"years={sorted(set(d[m].year))[:3]}..{sorted(set(d[m].year))[-1]}")

print("\n=== AUGUST opex-1, other vehicles h=5 ===")
for c in ["QQQ", "IWM", "TLT", "GLD", "UUP", "EEM", "XLV", "XLK"]:
    d, v = wind(pre, 5, c)
    m = d.month == 8
    if m.sum() < 3:
        continue
    print(f" {c:<5} N={m.sum():>3}  mean={v[m].mean()*100:+.3f}%  hit={np.mean(v[m]>0)*100:.1f}%")

# ---------------------------------------------------------------- 2. dispersion state
print("\n\n=== 2. cross-sectional dispersion vs index move ===")
import json
tape = json.load(open(ROOT / "data/pitch_tape.json"))["tickers"]
univ = [t for t in tape if not t.startswith("^") and "=" not in t and "-" not in t]
pan = close_panel(univ)
rets = pan.pct_change()
# restrict to names with data
cs_sd = rets.std(axis=1, skipna=True)
n_ok = rets.notna().sum(axis=1)
cs_sd = cs_sd[n_ok >= 120]
spy = px["SPY"].pct_change().reindex(cs_sd.index)
ratio = cs_sd.abs() / spy.abs().replace(0, np.nan)
print("today cross-sectional sd of daily returns:", round(cs_sd.iloc[-1] * 100, 3), "%",
      " pctile(252d):", round((cs_sd.iloc[-252:] < cs_sd.iloc[-1]).mean() * 100, 1),
      " pctile(full):", round((cs_sd < cs_sd.iloc[-1]).mean() * 100, 1))
print("today |SPY|:", round(abs(spy.iloc[-1]) * 100, 3), "%   ratio sd/|SPY| =", round(ratio.iloc[-1], 2),
      " pctile(full):", round((ratio < ratio.iloc[-1]).mean() * 100, 1))

# ---------------------------------------------------------------- 3. dollar magnitude
print("\n\n=== 3. dollar: rank vs magnitude ===")
dx = close_panel(["DX-Y.NYB"])["DX-Y.NYB"].dropna()
r21 = dx.pct_change(21)
print("DXY 21d return today:", round(r21.iloc[-1] * 100, 3), "%")
print("  pctile of 21d returns, trailing 252d:", round((r21.iloc[-252:] < r21.iloc[-1]).mean() * 100, 1))
print("  pctile of 21d returns, FULL history :", round((r21.dropna() < r21.iloc[-1]).mean() * 100, 1))
print("  21d return sd (full):", round(r21.std() * 100, 3), "%  -> today =",
      round(r21.iloc[-1] / r21.std(), 2), "sd")
print("  dist from 52w low:", round((dx.iloc[-1] / dx.iloc[-252:].min() - 1) * 100, 2), "%")
r21_tr = pct_rank(dx, 21)
print("  trailing-252 rank of 21d ret (pitch_lab pct_rank):", round(r21_tr.iloc[-1], 1))
cell = (r21_tr <= 2)
print("  days with 21d rank <=2 :", int(cell.sum()), " episodes(min_gap 21):",
      len(declusters(dx.index[cell], 21, dx.index)))

# ---------------------------------------------------------------- 4. TLT thrust from the low
print("\n\n=== 4. TLT one-day thrust near a 52w low ===")
tlt = close_panel(["TLT"])["TLT"].dropna()
d1 = tlt.pct_change()
low52 = tlt.rolling(252).min()
distlow = tlt / low52 - 1.0
print("today TLT 1d:", round(d1.iloc[-1] * 100, 2), "%  dist 52w low:", round(distlow.iloc[-1] * 100, 2), "%")
m = (d1 >= 0.015) & (distlow <= 0.04)
print("  state days:", int(m.sum()), " episodes(min_gap 10):", len(declusters(tlt.index[m], 10, tlt.index)))

# ---------------------------------------------------------------- 5. healthcare simultaneity
print("\n\n=== 5. healthcare complex simultaneity ===")
hc = close_panel(["XLV", "IBB", "XBI", "IHI", "SPY"])
r63 = hc.pct_change(63)
rk = {c: pct_rank(hc[c], 63) for c in ["XLV", "IBB", "XBI", "IHI"]}
hi52 = {c: hc[c] / hc[c].rolling(252).max() - 1.0 for c in ["XLV", "IBB", "XBI"]}
joint = (rk["XLV"] >= 97) & (rk["IBB"] >= 97) & (rk["XBI"] >= 95) & (rk["IHI"] >= 95)
print("  today ranks:", {c: round(rk[c].iloc[-1], 1) for c in rk})
print("  joint days:", int(joint.sum()), " episodes(min_gap 21):",
      len(declusters(hc.index[joint.fillna(False)], 21, hc.index)))
