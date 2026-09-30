"""c6 round-1 follow-up: (a) the candidate's RAW gap definition (EFA - SPY <= -1pp,
no beta); (b) gate attribution: the UNGATED post-expiry EFA/EWJ residual (all
quads, non-quad opex), its offset placebo ladder k=-5..+5, era / September / midterm
split, and whether the residual's post-quad drift is lagged SPY beta
(non-synchronous close) rather than an international effect."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kC_common import *  # noqa

px = nyse_panel(["EFA", "SPY", "DX-Y.NYB", "EEM", "EWJ"], ffill=("DX-Y.NYB",))
idx = px.index
rdx = dret(px["DX-Y.NYB"])
rs = dret(px["SPY"])
for leg in ["EFA", "EEM", "EWJ"]:
    ri, rr, b = resid_index(px, leg)
    px[f"R_{leg}"] = ri
    px[f"gap_{leg}"] = rr
opex = to_sessions(load_events(["opex"])["date"], idx)
quad = to_sessions(load_events(["quad_witching"])["date"], idx)
is_opex = pd.Series(idx.isin(opex), index=idx)
is_quad = pd.Series(idx.isin(quad), index=idx)
dxflat = rdx.abs() <= 0.002

# (a) raw gap
rows = []
for leg in ["EFA", "EWJ"]:
    raw = dret(px[leg]) - rs
    for h in (1, 3, 5):
        for lag in (0, 1):
            for lbl, m in [("quad & rawgap<=-1", is_quad & (raw <= -0.01)),
                           ("opex & rawgap<=-1", is_opex & (raw <= -0.01)),
                           ("opex & rawgap<=-1 & dxflat", is_opex & (raw <= -0.01) & dxflat),
                           ("NONEXP & rawgap<=-1 & dxflat", ~is_opex & (raw <= -0.01) & dxflat)]:
                s, _, _ = cellstats(px, m, [(f"R_{leg}", 1.0)], h, f"{leg} {lbl} h={h} lag={lag}", lag=lag)
                rows.append(s)
show(rows, "(a) candidate's raw gap definition, residual vehicle")

# (b) ungated parent: offset ladder around quads and around non-quad opex
pos = pd.Series(range(len(idx)), index=idx)
nq_opex = opex.difference(quad)


def ladder(anchor_dates, leg, h=3, lag=1):
    out = []
    R = px[f"R_{leg}"]
    fr = vehicle_ret(px, [(f"R_{leg}", 1.0)], h, lag)
    for k in range(-5, 6):
        ps = [pos[d] + k for d in anchor_dates if d in pos.index and 0 <= pos[d] + k < len(idx)]
        ds = idx[ps]
        v = fr.loc[ds].dropna().values
        w = int((v > 0).sum())
        out.append({"k": k, "n": len(v), "mean_pct": round(100 * v.mean(), 3),
                    "hit": round(100 * (v > 0).mean(), 1), "rec": f"{w}-{len(v)-w}",
                    "sign_p": round(sign_test(w, len(v)), 4)})
    df = pd.DataFrame(out)
    df["rank"] = df["mean_pct"].rank(ascending=False).astype(int)
    return df


for leg in ["EFA", "EWJ"]:
    for nm, A in [("quads", quad), ("non-quad opex", nq_opex)]:
        print(f"\n=== offset ladder, UNGATED {nm}, long {leg} vs beta-SPY, h=3 lag=1 (k=0 is the expiry close) ===")
        print(ladder(A, leg).to_string(index=False))

# era / month / midterm split for the ungated quad parent, EFA and EWJ, h=3 lag=1
rows = []
for leg in ["EFA", "EWJ"]:
    fr = vehicle_ret(px, [(f"R_{leg}", 1.0)], 3, 1)
    q = pd.DatetimeIndex([d for d in quad if d in fr.dropna().index])
    v = fr.loc[q]
    for lbl, m in [("all", np.ones(len(q), bool)), ("pre-2018", q.year < 2018), ("2018+", q.year >= 2018),
                   ("Sep", q.month == 9), ("Mar", q.month == 3), ("Jun", q.month == 6), ("Dec", q.month == 12),
                   ("midterm yrs", q.year % 4 == 2), ("midterm Sep", (q.year % 4 == 2) & (q.month == 9))]:
        vv = v[m].values
        w = int((vv > 0).sum())
        s = summarize(vv, f"{leg} quad parent {lbl}")
        s["rec"] = f"{w}-{len(vv)-w}"
        s["sign_p"] = round(sign_test(w, len(vv)), 4)
        rows.append(s)
show(rows, "ungated quad parent h=3 lag=1 splits")

# (c) is the post-quad residual just lagged SPY beta? regress window residual on
# SPY's return over the window shifted one session earlier (non-synchronous close)
fr = vehicle_ret(px, [("R_EFA", 1.0)], 3, 1)
spy_prev = vehicle_ret(px, [("SPY", 1.0)], 3, 0)  # SPY over the window one day earlier
spy_same = vehicle_ret(px, [("SPY", 1.0)], 3, 1)
q = pd.DatetimeIndex([d for d in quad if d in fr.dropna().index])
df = pd.DataFrame({"res": fr.loc[q], "spy_prev": spy_prev.loc[q], "spy_same": spy_same.loc[q]}).dropna()
allw = pd.DataFrame({"res": fr, "spy_prev": spy_prev}).dropna()
bl = np.polyfit(allw.spy_prev, allw.res, 1)[0]
print(f"\n(c) all-days slope of EFA 3d residual on SPY 3d return one session earlier: {bl:+.3f}")
print(f"    post-quad windows: mean SPY prev-window {100*df.spy_prev.mean():+.3f}%, same-window {100*df.spy_same.mean():+.3f}%")
adj = df.res - bl * df.spy_prev
w = int((adj > 0).sum())
print(f"    post-quad residual {100*df.res.mean():+.3f}% -> net of lagged-SPY loading {100*adj.mean():+.3f}% "
      f"({w}-{len(adj)-w}, sign p {sign_test(w, len(adj)):.4f})")
