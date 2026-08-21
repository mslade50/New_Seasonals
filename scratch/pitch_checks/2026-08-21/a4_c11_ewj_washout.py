"""C11 round 1: long Japan on a five-day washout while developed intl holds.

Live: EWJ 5d rank 3.6 (-4.27% over 5 sessions) while EFA is -1.23% at a 5d
rank of 17.9. Gap -3.04pp.

Standing family caveat, applied not ignored: the country-decoupling family has
died FOUR times (EWZ twice, FXI, KWEB) to the SAME argument -- the trigger
selects tape the whole class shares, the outright is class beta with a country
label, and permuting the identical rule across the peer set shows the country
is a routine draw from the no-label null.

Order: (0) live state + history, (1) outright vs controls, (2) beta-neutral
residual against EFA at the measured beta, (3) REFERENCE CLASS -- both forms:
same-episode peer excess, and the identical rule permuted onto every peer,
(4) tape over-selection check (the specific thing that killed EWZ/FXI/SMH).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change  # noqa

import warnings
warnings.filterwarnings("ignore")

ASOF = pd.Timestamp("2026-08-20")
H = 5
PEERS = ["EWJ", "EWT", "EWW", "EWY", "EWZ", "EEM", "FXI", "INDA", "KWEB", "VGK", "RSX"]

px = close_panel(PEERS + ["EFA", "SPY"]).loc[:ASOF]
idx = px.index
print("history starts:", {c: str(px[c].dropna().index[0].date()) for c in px.columns})

r5 = {c: _valid_pct_change(px[c], 5) for c in px.columns}
rk5 = {c: pct_rank(px[c], 5) for c in px.columns}

print("\n" + "=" * 100)
print("C11-0  LIVE STATE", ASOF.date())
print("=" * 100)
for c in ["EWJ", "EFA", "EEM", "VGK", "SPY"]:
    print(f"  {c:6s} 5d ret {100*r5[c].loc[ASOF]:7.2f}%  5d rank {rk5[c].loc[ASOF]:5.1f}")
gap_live = r5["EWJ"].loc[ASOF] - r5["EFA"].loc[ASOF]
print(f"  EWJ - EFA 5d gap = {100*gap_live:.2f}pp")

gap = r5["EWJ"] - r5["EFA"]
mask = ((rk5["EWJ"] <= 5) & (gap <= -0.025)).fillna(False)
print(f"  trigger = EWJ 5d rank<=5 AND (EWJ5d - EFA5d) <= -2.5pp")
print(f"  trigger days = {int(mask.sum())}   fires today = {bool(mask.loc[ASOF])}")

leg = fwd_lag(px["EWJ"], H, 1)
epi = declusters(idx[(mask & leg.notna()).values], 5, idx)

variants = {}
for rk in (3, 5, 10, 15):
    variants[f"EWJ rank5<={rk}"] = ((rk5["EWJ"] <= rk) & (gap <= -0.025)).fillna(False)
for g in (-0.015, -0.025, -0.035, -0.05):
    variants[f"gap<={100*g:.1f}pp"] = ((rk5["EWJ"] <= 5) & (gap <= g)).fillna(False)

battery(px, mask, [("EWJ", 1.0)], H, "C11  long EWJ | 5d washout, EFA holding",
        cost_bps=3.0, variants=variants, min_gap=5, event_kinds=("cpi", "nfp"))

# ------------------------------------------------- 2. beta-neutral vs EFA
print("\n" + "=" * 100)
print("C11-2  BETA(EWJ on EFA) and the residual")
print("=" * 100)
d = px[["EWJ", "EFA"]].pct_change().dropna()
B = float(np.polyfit(d["EFA"], d["EWJ"], 1)[0])
B252 = float(np.polyfit(d["EFA"].iloc[-252:], d["EWJ"].iloc[-252:], 1)[0])
print(f"  full-sample beta {B:.3f}  (corr {d['EWJ'].corr(d['EFA']):.3f})   trailing-252d beta {B252:.3f}")
rows = []
for lbl, legs in (("EWJ outright", [("EWJ", 1.0)]),
                  ("EFA outright (the class)", [("EFA", 1.0)]),
                  ("equal-dollar EWJ - EFA", [("EWJ", 1.0), ("EFA", -1.0)]),
                  (f"beta-neutral EWJ - {B:.2f}*EFA", [("EWJ", 1.0), ("EFA", -B)])):
    r = vehicle_ret(px, legs, H, 1)
    b = r.dropna()
    v = r.loc[epi].values
    s = summarize(v, lbl)
    s["own_drift"] = round(100 * b.mean(), 3)
    s["excess_pct"] = round(s["mean_pct"] - 100 * b.mean(), 3)
    s["signp"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
    rows.append(s)
show(rows, f"h={H}, {len(epi)} episodes")

# ------------------------------------------------- 3a. same-episode peers
print("\n" + "=" * 100)
print("C11-3a  REFERENCE CLASS form 1: on EWJ's OWN trigger episodes, what does")
print("        every peer do?  If the class moves together, this is class beta.")
print("=" * 100)
rows = []
for c in PEERS + ["EFA", "SPY"]:
    r = fwd_lag(px[c], H, 1)
    e = pd.DatetimeIndex([x for x in epi if not np.isnan(r.get(x, np.nan))])
    if len(e) < 5:
        rows.append({"label": c, "n": len(e)})
        continue
    b = r.dropna()
    v = r.loc[e].values
    s = summarize(v, c)
    s["own_drift"] = round(100 * b.mean(), 3)
    s["excess_pct"] = round(s["mean_pct"] - 100 * b.mean(), 3)
    rows.append(s)
df = pd.DataFrame(rows).sort_values("excess_pct", ascending=False)
show(df.to_dict("records"), "peer excess on EWJ's episodes (h=5)")

# ------------------------------------------------- 3b. rule permuted
print("\n" + "=" * 100)
print("C11-3b  REFERENCE CLASS form 2: permute the IDENTICAL RULE onto every peer")
print("        (own 5d rank<=5 AND own 5d ret - EFA 5d ret <= -2.5pp)")
print("=" * 100)
res = []
for c in PEERS:
    g = r5[c] - r5["EFA"]
    m = ((rk5[c] <= 5) & (g <= -0.025)).fillna(False)
    r = fwd_lag(px[c], H, 1)
    e = declusters(idx[(m & r.notna()).values], 5, idx)
    if len(e) < 5:
        res.append({"name": c, "n": len(e), "excess_pct": np.nan})
        continue
    b = r.dropna()
    v = r.loc[e].values
    # beta-neutral residual against EFA, own beta
    dd = px[[c, "EFA"]].pct_change().dropna()
    bb = float(np.polyfit(dd["EFA"], dd[c], 1)[0])
    rr = vehicle_ret(px, [(c, 1.0), ("EFA", -bb)], H, 1)
    res.append({"name": c, "n": len(e),
                "mean_pct": round(100 * v.mean(), 3),
                "excess_pct": round(100 * v.mean() - 100 * b.mean(), 3),
                "hit": round(100 * (v > 0).mean(), 1),
                "beta": round(bb, 2),
                "resid_pct": round(100 * rr.loc[e].mean(), 3),
                "signp": round(sign_test(int((v > 0).sum()), len(v)), 4)})
rdf = pd.DataFrame(res).sort_values("excess_pct", ascending=False)
show(rdf.to_dict("records"), "identical rule on every peer, h=5 episodes")
ok = rdf.dropna(subset=["excess_pct"])
ewj = ok[ok["name"] == "EWJ"]
if len(ewj):
    rank = int((ok["excess_pct"] > float(ewj["excess_pct"].iloc[0])).sum()) + 1
    print(f"  EWJ ranks {rank} of {len(ok)} peers by excess; median peer excess "
          f"{ok['excess_pct'].median():+.3f}%, positive in {int((ok['excess_pct']>0).sum())}/{len(ok)}")
    rk2 = int((ok["resid_pct"] > float(ewj["resid_pct"].iloc[0])).sum()) + 1
    print(f"  EWJ ranks {rk2} of {len(ok)} by beta-neutral residual; median residual "
          f"{ok['resid_pct'].median():+.3f}%, positive in {int((ok['resid_pct']>0).sum())}/{len(ok)}")

# ------------------------------------------------- 4. tape over-selection
print("\n" + "=" * 100)
print("C11-4  TAPE OVER-SELECTION (what killed EWZ, FXI and SMH/QQQ)")
print("=" * 100)
sma200 = rolling_on_valid(px["SPY"], lambda x: x.rolling(200).mean())
above = (px["SPY"] > sma200)
base_days = above.reindex(idx).dropna()
print(f"  SPY above its 200d on {100*base_days.mean():.1f}% of all days (base rate)")
print(f"  SPY above its 200d on {100*above.loc[epi].mean():.1f}% of the {len(epi)} trigger episodes")
print(f"  TODAY SPY above 200d = {bool(above.loc[ASOF])}")
efa200 = rolling_on_valid(px["EFA"], lambda x: x.rolling(200).mean())
ea = (px["EFA"] > efa200)
print(f"  EFA above its 200d: base {100*ea.reindex(idx).dropna().mean():.1f}%  "
      f"trigger episodes {100*ea.loc[epi].mean():.1f}%  today {bool(ea.loc[ASOF])}")
