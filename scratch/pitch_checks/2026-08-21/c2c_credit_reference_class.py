"""C2 round 2b: the reference class for the surviving DEFINITION.

The pitched form (HYG within 0.5% of its 52w high) is dead: the gate is worth
-0.022pp. The RETURN form (HYG 5d >= -0.5% while SPY 5d rank <= 10) pays
+1.122% at h=5 over 70 episodes. Before crediting the word "credit", swap HYG
for every other liquid non-equity and equity vehicle and re-run the identical
gate. If "SPY fell and X did not" works for every X, the credit label carries
nothing -- the industry-label finding transposed to asset class.

Also: gate threshold walk, bull-tape over-selection, multiplicity.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change

pd.set_option("display.width", 240)

GATES = ["HYG", "LQD", "IEF", "TLT", "GLD", "IWM", "EFA", "EEM", "XLU", "XLP",
         "XLK", "QQQ", "UUP", "DBC"]
px = close_panel(["SPY"] + GATES)
spy5r = pct_rank(px["SPY"], 5)
EQ = spy5r <= 10
H = 5


def cellstats(mask, h, gap=None):
    r = fwd_lag(px["SPY"], h, 1)
    v = r.notna()
    dd = px.index[mask.reindex(px.index, fill_value=False).values & v.values]
    if len(dd) < 3:
        return None
    e = declusters(dd, gap or h, px.index[v.values])
    vals = r.loc[e].values
    return e, vals, float(r.loc[v].mean())


print("===== A. REFERENCE CLASS: 'SPY 5d rank<=10 and X 5d >= -0.5%' =====")
print("   (each gate restricted to the SAME window so the samples are comparable)")
start = px.index[px[GATES].notna().all(axis=1)][0]
print("   common window starts", start.date())
sub = px.loc[start:]
spy5r_s = pct_rank(sub["SPY"], 5)
EQ_s = spy5r_s <= 10
rows = []
for h in (3, 5, 10):
    r = fwd_lag(sub["SPY"], h, 1)
    v = r.notna()
    base = float(r.loc[v].mean())
    dd0 = sub.index[EQ_s.reindex(sub.index, fill_value=False).values & v.values]
    e0 = declusters(dd0, h, sub.index[v.values])
    rows.append({"gate": "(none) EQ alone", "h": h, "n": len(e0),
                 "mean_pct": round(100 * r.loc[e0].mean(), 3),
                 "edge_vs_alldays_pp": round(100 * (r.loc[e0].mean() - base), 3),
                 "gate_value_pp": 0.0, "hit": round(100 * (r.loc[e0] > 0).mean(), 1)})
    for g in GATES:
        g5 = _valid_pct_change(sub[g], 5)
        m = EQ_s & (g5 >= -0.005)
        dd = sub.index[m.reindex(sub.index, fill_value=False).values & v.values]
        if len(dd) < 5:
            continue
        e = declusters(dd, h, sub.index[v.values])
        mu = float(r.loc[e].mean())
        rows.append({"gate": f"{g} 5d >= -0.5%", "h": h, "n": len(e),
                     "mean_pct": round(100 * mu, 3),
                     "edge_vs_alldays_pp": round(100 * (mu - base), 3),
                     "gate_value_pp": round(100 * (mu - r.loc[e0].mean()), 3),
                     "hit": round(100 * (r.loc[e] > 0).mean(), 1)})
show(rows, "gate reference class (episodes, common window)")
for h in (3, 5, 10):
    sl = [x for x in rows if x["h"] == h and x["gate"] != "(none) EQ alone"]
    gv = np.array([x["gate_value_pp"] for x in sl])
    hyg = [x for x in sl if x["gate"].startswith("HYG")][0]["gate_value_pp"]
    print(f"  h={h}: {len(gv)} candidate gates, mean gate value {gv.mean():+.3f}pp, "
          f"median {np.median(gv):+.3f}pp, HYG {hyg:+.3f}pp, "
          f"HYG rank {int((gv >= hyg).sum())} of {len(gv)}, "
          f"positive gates {int((gv > 0).sum())}/{len(gv)}")

print("\n===== B. threshold walk on the HYG 5d gate (full HYG history) =====")
hyg5 = _valid_pct_change(px["HYG"], 5)
pxh = px.loc[px["HYG"].notna()]
rows = []
for thr in (0.01, 0.005, 0.0, -0.0025, -0.005, -0.01, -0.015, -0.02):
    for h in (3, 5, 10):
        out = cellstats(EQ & (hyg5 >= thr), h)
        if out is None:
            continue
        e, vals, base = out
        rows.append({"HYG 5d >=": f"{100*thr:+.2f}%", "h": h, "n": len(vals),
                     "mean_pct": round(100 * vals.mean(), 3),
                     "median_pct": round(100 * np.median(vals), 3),
                     "hit": round(100 * (vals > 0).mean(), 1),
                     "edge_pp": round(100 * (vals.mean() - base), 3),
                     "t": round(vals.mean() / (vals.std(ddof=1) / np.sqrt(len(vals))), 2)})
show(rows, "threshold walk")

print("\n===== C. the SPREAD form: HYG 5d minus SPY 5d =====")
spy5 = _valid_pct_change(px["SPY"], 5)
sp = hyg5 - spy5
rows = []
for thr in (0.0, 0.01, 0.015, 0.02, 0.025, 0.03):
    for h in (3, 5, 10):
        out = cellstats(EQ & (sp >= thr), h)
        if out is None:
            continue
        e, vals, base = out
        rows.append({"HYG-SPY 5d >=": f"{100*thr:+.1f}pp", "h": h, "n": len(vals),
                     "mean_pct": round(100 * vals.mean(), 3), "hit": round(100 * (vals > 0).mean(), 1),
                     "edge_pp": round(100 * (vals.mean() - base), 3)})
show(rows, "spread form (today's spread = %.2fpp)" % (100 * (hyg5.iloc[-1] - spy5.iloc[-1])))

print("\n===== D. bull-tape over-selection, return form =====")
sma200 = rolling_on_valid(px["SPY"], lambda x: x.rolling(200).mean())
abv = px["SPY"] > sma200
r5 = fwd_lag(px["SPY"], 5, 1)
v5 = r5.notna()
alld = px.index[v5.values]
d_ret = px.index[(EQ & (hyg5 >= -0.005)).reindex(px.index, fill_value=False).values & v5.values]
d_eq = px.index[EQ.reindex(px.index, fill_value=False).values & v5.values]
print(f" base above-200d {100*abv.loc[alld].mean():.1f}% | EQ alone {100*abv.loc[d_eq].mean():.1f}%"
      f" | EQ+HYG-return {100*abv.loc[d_ret].mean():.1f}%")

print("\n===== E. multiplicity: what grid did this morning walk? =====")
print(" C2 neighbour table: 4 HYG-distance tolerances x 1 + 3 SPY rank cuts + 4 SPY")
print(" magnitude cuts + 3 HYG-return cuts + 2 combos = 16 cells at h=5, plus this")
print(" script's 14-gate reference class x 3 horizons and an 8-step threshold walk.")
print(" A single cell clearing t=2.6 out of ~16 head cells is ~0.4 expected by chance")
print(" at p<0.025 one-sided; the reference class above is the test that matters.")
