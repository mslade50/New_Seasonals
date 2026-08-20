"""C2 round 1+2: short TLT on a >=1.5% up day from inside the 52-week low zone.

Everything is computed on TLT's OWN series (never a panel column) per the
2026-08-12 / 2026-08-19 close_panel traps. Vehicle is the SHORT, so every
mean/hit below reads for the side actually being pitched.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

pd.set_option("display.width", 220)

raw = load_prices(["TLT", "IEF", "^TNX"])
tlt = raw["TLT"]["Close"].dropna()
d = tlt.index
px = pd.DataFrame({"TLT": tlt, "IEF": raw["IEF"]["Close"].reindex(d),
                   "TNX": raw["^TNX"]["Close"].reindex(d)})
SHORT = [("TLT", -1.0)]

d1 = tlt.pct_change(fill_method=None)
low52 = tlt.rolling(252).min()
dist = tlt / low52 - 1.0

print("===== 0. today =====")
print(f" TLT last bar {d[-1].date()}  1d {d1.iloc[-1]*100:+.3f}%  dist above 52w low {dist.iloc[-1]*100:.2f}%")
print(f" gate: 1d >= +1.5% AND dist <= 4.0%  ->  fires today? "
      f"{bool(d1.iloc[-1] >= 0.015 and dist.iloc[-1] <= 0.04)}")

m = (d1 >= 0.015) & (dist <= 0.04)
m = m.fillna(False).astype(bool)
epi = declusters(d[m], 10, d)
print(f" state days {int(m.sum())}, episodes(gap10) {len(epi)}")
print(" episode dates:", ", ".join(str(x.date()) for x in epi))
print(" year histogram:", pd.Series(epi.year).value_counts().sort_index().to_dict())

m_thrust = (d1 >= 0.015).fillna(False).astype(bool)
m_low = (dist <= 0.04).fillna(False).astype(bool)

variants = {
    "thrust>=1.0%, low<=4%": ((d1 >= 0.010) & (dist <= 0.04)).fillna(False).astype(bool),
    "thrust>=1.25%, low<=4%": ((d1 >= 0.0125) & (dist <= 0.04)).fillna(False).astype(bool),
    "thrust>=2.0%, low<=4%": ((d1 >= 0.020) & (dist <= 0.04)).fillna(False).astype(bool),
    "thrust>=1.5%, low<=2%": ((d1 >= 0.015) & (dist <= 0.02)).fillna(False).astype(bool),
    "thrust>=1.5%, low<=3%": ((d1 >= 0.015) & (dist <= 0.03)).fillna(False).astype(bool),
    "thrust>=1.5%, low<=6%": ((d1 >= 0.015) & (dist <= 0.06)).fillna(False).astype(bool),
    "thrust>=1.5%, low<=8%": ((d1 >= 0.015) & (dist <= 0.08)).fillna(False).astype(bool),
    "THRUST ALONE (no low gate)": m_thrust,
    "NEAR-LOW ALONE (no thrust)": m_low,
}

for h in (2, 5, 10):
    battery(px, m, SHORT, h, "C2 SHORT TLT after a >=1.5% day near the 52w low",
            cost_bps=3.0, variants=variants if h == 2 else None, min_gap=10,
            event_kinds=("jackson_hole", "fomc_decision"))

# ------------------------------------------------------------------ gate attribution
print("\n\n===== GATE ATTRIBUTION (short TLT, episodes gap10) =====")
rows = []
for nm, mm in [("JOINT thrust & near-low", m),
               ("THRUST alone", m_thrust),
               ("NEAR-LOW alone", m_low),
               ("thrust & NOT near-low", m_thrust & ~m_low)]:
    for h in (1, 2, 3, 5, 10):
        ret = vehicle_ret(px, SHORT, h)
        valid = ret.dropna().index
        e = declusters(pd.DatetimeIndex(d[mm]).intersection(valid), 10, valid)
        r = summarize(ret.loc[e].values, f"{nm} h={h}")
        base = ret.loc[valid]
        loc = local_control(valid, pd.DatetimeIndex(d[mm]).intersection(valid))
        r["ctrl_all"] = round(100 * base.mean(), 3)
        r["ctrl_local"] = round(100 * ret.loc[loc].mean(), 3)
        r["edge_vs_local"] = round(r.get("mean_pct", np.nan) - 100 * ret.loc[loc].mean(), 3)
        rows.append(r)
show(rows, "gate attribution + local control")

# ------------------------------------------------------------------ yield regime
print("\n\n===== YIELD REGIME (bond-bull fossil test, run in reverse) =====")
tnx = px["TNX"]
tnx_252 = tnx - tnx.shift(252)
rising = (tnx_252 > 0)
for h in (2, 10):
    ret = vehicle_ret(px, SHORT, h)
    valid = ret.dropna().index
    e = pd.DatetimeIndex(epi).intersection(valid)
    rr = rising.reindex(e).fillna(False).values
    show([summarize(ret.loc[e[rr]].values, f"RISING-yield triggers h={h} (N={rr.sum()})"),
          summarize(ret.loc[e[~rr]].values, f"FALLING-yield triggers h={h} (N={(~rr).sum()})")],
         f"episodes split by trailing-252d TNX change, h={h}")
    # era-matched control: all days in the same regime
    vr = rising.reindex(valid).fillna(False).values
    print(f"   era-matched control h={h}: rising-regime all-days short "
          f"{100*ret.loc[valid[vr]].mean():+.3f}%   falling {100*ret.loc[valid[~vr]].mean():+.3f}%")

print("\n episode-year x regime:")
rr_all = rising.reindex(epi).fillna(False)
print(pd.DataFrame({"year": epi.year, "rising": rr_all.values}).groupby(["year", "rising"]).size().to_string())

# ------------------------------------------------------------------ multiplicity
print("\n\n===== MULTIPLICITY: the sign came out of a recon that asked 'long or short?' =====")
from math import sqrt
for h in (1, 2, 3, 5, 10):
    ret = vehicle_ret(px, SHORT, h)
    e = pd.DatetimeIndex(epi).intersection(ret.dropna().index)
    v = ret.loc[e].values
    tstat = v.mean() / (v.std(ddof=1) / sqrt(len(v)))
    # two-sided nominal p from a normal approx, then Sidak over 2 signs x 5 horizons
    from math import erf
    p2 = 2 * (1 - 0.5 * (1 + erf(abs(tstat) / sqrt(2))))
    k = 10
    sidak = 1 - (1 - p2) ** k
    print(f" h={h:>2}  N={len(v)}  mean {100*v.mean():+.3f}%  t={tstat:+.2f}  "
          f"nominal two-sided p={p2:.4f}  Sidak over {k} looks = {sidak:.4f}  "
          f"(bonferroni alpha for 0.05 = {0.05/k:.4f})")

# ------------------------------------------------------------------ gradient
print("\n\n===== DISTANCE-FROM-THE-EXTREME GRADIENT (today = 2.05% above the low) =====")
for h in (2, 10):
    ret = vehicle_ret(px, SHORT, h)
    e = pd.DatetimeIndex(epi).intersection(ret.dropna().index)
    x = dist.reindex(e).values * 100
    y = ret.loc[e].values * 100
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    se = np.sqrt((resid ** 2).sum() / (len(x) - 2) / ((x - x.mean()) ** 2).sum())
    print(f" h={h}: slope {b:+.4f} pp per 1% off the low (t={b/se:+.2f}), N={len(x)}; "
          f"fitted at today's 2.05% = {a + b*2.05:+.3f}% (cell mean {y.mean():+.3f}%)")
    print(f"    trigger dist distribution: {np.round(np.percentile(x, [0, 25, 50, 75, 100]), 2)}; "
          f"today's pctile within triggers {100*(x < 2.05).mean():.0f}")
    # magnitude of the thrust today vs the trigger set
mag = d1.reindex(epi) * 100
print(f"\n today's thrust +1.67% vs trigger-set thrust distribution: "
      f"{np.round(np.percentile(mag.dropna(), [0, 25, 50, 75, 100]), 2)}; "
      f"today's pctile within triggers {100*(mag < 1.67).mean():.0f}")

# ------------------------------------------------------------------ cost
print("\n\n===== COST + TAIL =====")
for h in (2, 5, 10):
    ret = vehicle_ret(px, SHORT, h)
    e = pd.DatetimeIndex(epi).intersection(ret.dropna().index)
    v = ret.loc[e].values
    print(f" h={h:>2}: mean {100*v.mean()*100:+.1f} bps vs a ~3 bps TLT round trip + borrow "
          f"-> {100*v.mean()*100/3:.1f}x ; worst episode {100*v.min():+.2f}% on "
          f"{e[int(np.argmin(v))].date()}; best {100*v.max():+.2f}%")
    print(f"        top-2 |episodes|: {cluster_note(e, v)}")
