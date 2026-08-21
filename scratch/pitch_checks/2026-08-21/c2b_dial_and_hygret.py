"""C2 round 2: (a) the coordinator's fragility-dial conditioning -- does the
cell's edge live in the CALM half, which is the book's own finding; (b) the
one definition neighbour that worked (HYG's own 5d RETURN rather than its
distance from a 52w high), tested for era stability and for whether it is
just a shallower-SPY-drawdown selector.

Dial series: data/rd2_fragility.parquet, column '63d', 10d MA, PIT-append
since 2026-07-02 and a RECOMPUTE vintage before that (stated per CLAUDE.md's
vintage rule). Series starts 2016, so the dial split covers a subsample.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change

pd.set_option("display.width", 240)

px = close_panel(["SPY", "HYG"])
px = px.loc[px["HYG"].notna()]
spy5 = _valid_pct_change(px["SPY"], 5)
spy5r = pct_rank(px["SPY"], 5)
hyg_off = px["HYG"] / rolling_on_valid(px["HYG"], lambda x: x.rolling(252).max()) - 1.0
hyg5 = _valid_pct_change(px["HYG"], 5)

EQ = spy5r <= 10
CR = hyg_off >= -0.005
CRR = hyg5 >= -0.005          # the RETURN form of the credit gate

frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
frag.index = pd.to_datetime(frag.index)
dial = frag["63d"].rolling(10).mean()
dial = dial.reindex(px.index).ffill(limit=3)
print("dial series", frag.index[0].date(), "..", frag.index[-1].date(),
      " live ma10(63d) =", round(float(dial.iloc[-1]), 1))
print("VINTAGE NOTE: PIT-append only from 2026-07-02; everything earlier is the "
      "recompute vintage (drift up to ~7 pts on the 63d dial).")


def stats(mask, h, label, gap=None):
    r = fwd_lag(px["SPY"], h, 1)
    v = r.notna()
    dd = px.index[mask.reindex(px.index, fill_value=False).values & v.values]
    if len(dd) < 2:
        return {"label": label, "n": len(dd)}
    e = declusters(dd, gap or h, px.index[v.values])
    s = summarize(r.loc[e].values, label)
    s["n_days"] = len(dd)
    s["edge_pp"] = round(s["mean_pct"] - 100 * r.loc[v].mean(), 3)
    return s


# ---------------------------------------------------------- A. dial conditioning
print("\n===== A. FRAGILITY-DIAL CONDITIONING (the book's own dip-buy finding) =====")
has = dial.notna()
for h in (3, 5, 10):
    rows = [stats(EQ & CR & has, h, f"gated cell, dial-era only h={h}"),
            stats(EQ & CR & (dial < 50), h, f"  dial ma10(63d) < 50  h={h}"),
            stats(EQ & CR & (dial >= 50), h, f"  dial >= 50 (TODAY 89.1) h={h}"),
            stats(EQ & CR & (dial >= 70), h, f"  dial >= 70            h={h}"),
            stats(EQ & has, h, f"EQ ALONE, dial era  h={h}"),
            stats(EQ & (dial < 50), h, f"  EQ alone dial<50    h={h}"),
            stats(EQ & (dial >= 50), h, f"  EQ alone dial>=50   h={h}")]
    show(rows, f"h={h}")

# ---------------------------------------------------------- B. HYG-return form
print("\n===== B. the HYG 5d-RETURN form (the one neighbour that worked) =====")
rows = []
for h in (1, 2, 3, 5, 10):
    rows.append(stats(EQ & CRR, h, f"spy5rank<=10 & HYG5d>=-0.5%  h={h}"))
    rows.append(stats(EQ & ~CRR, h, f"   complement HYG5d<-0.5%   h={h}"))
    rows.append(stats(EQ, h, f"   EQ alone                  h={h}"))
show(rows, "horizon profile, episodes")

# era + controls on the h=5 return form
r5 = fwd_lag(px["SPY"], 5, 1)
v5 = r5.notna()
dd = px.index[(EQ & CRR).reindex(px.index, fill_value=False).values & v5.values]
e5 = declusters(dd, 5, px.index[v5.values])
vals = r5.loc[e5].values
loc = local_control(px.index[v5.values], dd)
show([summarize(vals, f"COND episodes (N={len(e5)})"),
      summarize(r5.loc[loc].values, "CTRL-c local +/-126td ex-trigger"),
      summarize(r5[v5].values, "CTRL-b all days")], "B2. controls, HYG-return form h=5")
show(era_split(e5, vals), "B3. era split")
print(" concentration:", cluster_note(e5, vals))
print(f" bootstrap P(mean<=0) {bootstrap_p_le0(vals):.3f}  record "
      f"{int((vals>0).sum())}-{int((vals<=0).sum())} sign p "
      f"{sign_test(int((vals>0).sum()), len(vals)):.4f}")

# is it just a shallower SPY drawdown?
print("\n===== C. is 'HYG did not fall' just 'SPY did not fall much'? =====")
allsig = px.index[EQ.reindex(px.index, fill_value=False).values & v5.values]
sub = pd.DataFrame({"spy5": spy5.loc[allsig], "hyg5": hyg5.loc[allsig],
                    "fwd": r5.loc[allsig]})
print(f" corr(SPY 5d, HYG 5d) on EQ days = {sub['spy5'].corr(sub['hyg5']):.3f}")
print(f" mean SPY 5d when HYG5d>=-0.5% : {100*sub.loc[sub.hyg5>=-0.005,'spy5'].mean():.2f}%"
      f"   when HYG5d<-0.5%: {100*sub.loc[sub.hyg5<-0.005,'spy5'].mean():.2f}%")
# depth-matched: within the SHALLOW half of SPY 5d moves, does the credit gate add?
med = sub["spy5"].median()
for nm, m in [("SPY-5d shallow half", sub["spy5"] >= med), ("SPY-5d deep half", sub["spy5"] < med)]:
    a = sub.loc[m & (sub.hyg5 >= -0.005), "fwd"]
    b = sub.loc[m & (sub.hyg5 < -0.005), "fwd"]
    print(f" {nm:<22} HYG-ok {100*a.mean():+.3f}% (n={len(a)})  HYG-fell "
          f"{100*b.mean():+.3f}% (n={len(b)})  gate {100*(a.mean()-b.mean()):+.3f}pp")
# dial split on the return form too
print("\n dial split, HYG-return form h=5:")
show([stats(EQ & CRR & has, 5, "return form, dial era"),
      stats(EQ & CRR & (dial < 50), 5, "  dial < 50"),
      stats(EQ & CRR & (dial >= 50), 5, "  dial >= 50 (today)")], "")
