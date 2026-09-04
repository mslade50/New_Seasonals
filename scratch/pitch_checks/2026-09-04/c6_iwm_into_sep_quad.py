"""C6 - the small-cap laggard into September quad witching.

Anchor: the session 9 trading days before September quad witching (today's
analogue). Entry lag=1 (the next close), exit at the quad-witching close, so
h=8. Gate: IWM 63d return rank <= 10 (today 9.1).

Mandatory blockers run here: placebo anchor ladder k=-8..+8, the reference
class across index and industry ETFs (Cochran Q / I^2 / rank), gate
attribution (ungated parent vs gated cell vs discarded complement), the
episode year histogram, the midterm cross, cost and concentration.
"""
import sys
from math import erf, sqrt
from pathlib import Path

ROOTP = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOTP))
import numpy as np
import pandas as pd
from pitch_lab import *  # noqa

CLASS = ["SPY", "QQQ", "DIA", "IWM", "XLI", "XLF", "XLK", "XLY", "XLP",
         "XLV", "XLU", "XLB", "XLE", "IYT", "SMH", "EFA", "EEM"]
px = close_panel(CLASS)
cal = px["SPY"].dropna().index          # master US equity calendar
H = 8                                   # entry close +8 -> quad close
LAG = 1
OFF = -9                                # today sits 9 sessions before the quad

ev = load_events(["quad_witching"])
sep = ev[(ev["date"].dt.month == 9) & (ev["date"] <= cal[-1])]["date"]
print(f"September quad witchings in range: {len(sep)}  "
      f"{sep.min().date()}..{sep.max().date()}")

pos = pd.Series(range(len(cal)), index=cal)


def anchors(offset):
    """Calendar dates `offset` sessions from each September quad close."""
    out = []
    for d in sep:
        loc = int(cal.searchsorted(d))
        if loc >= len(cal) or cal[loc] != d:
            continue
        p = loc + offset
        if 0 <= p < len(cal):
            out.append(cal[p])
    return pd.DatetimeIndex(out)


anc = anchors(OFF)
print(f"anchors at qw{OFF:+d}: N={len(anc)}  most recent "
      f"{', '.join(str(d.date()) for d in anc[-4:])}")
# sanity: 2026-09-04 must be the live analogue
qw26 = pd.Timestamp("2026-09-18")
print(f"  sanity: 2026 anchor would be "
      f"{cal[int(cal.searchsorted(pd.Timestamp('2026-09-03')))]} (cache ends "
      f"2026-09-03; today 2026-09-04 is qw-9 by session count)")

r63 = pct_rank(px["IWM"], 63)


def cell(tkr_legs, dates, h=H, lag=LAG):
    ret = vehicle_ret(px, tkr_legs, h, lag)
    d = pd.DatetimeIndex(dates).intersection(ret.dropna().index)
    return d, ret.loc[d].values, ret


# ---------------------------------------------------- gate attribution
print("\n=== BLOCKER 3: gate attribution (ungated parent / gated / complement) ===")
for legs, lbl, cost in [([("IWM", 1.0)], "LONG IWM", 3.0),
                        ([("IWM", 1.0), ("SPY", -1.0)], "LONG IWM / SHORT SPY", 5.0)]:
    d_all, v_all, ret = cell(legs, anc)
    g = r63.reindex(d_all)
    gated = d_all[(g <= 10).values]
    comp = d_all[(g > 10).values]
    base = ret.dropna()
    rows = [summarize(v_all, f"UNGATED parent, all Sep quads (N={len(d_all)})"),
            summarize(ret.loc[gated].values, f"GATED r63<=10 (N={len(gated)})"),
            summarize(ret.loc[comp].values, f"DISCARDED complement r63>10 (N={len(comp)})"),
            summarize(base.values, "CTRL-b all days, full history"),
            summarize(ret.loc[local_control(base.index, d_all)].values,
                      "CTRL-c local +/-126td ex-anchor")]
    show(rows, f"{lbl}  h={H} entry lag={LAG}")
    gm = rows[1]["mean_pct"] if rows[1]["n"] else np.nan
    um = rows[0]["mean_pct"]
    print(f"  GATE WORTH = {gm - um:+.3f}pp (gated minus ungated parent); "
          f"gated minus all-days = {gm - rows[3]['mean_pct']:+.3f}pp")
    if rows[1]["n"]:
        w = int((ret.loc[gated].values > 0).sum())
        n = len(gated)
        print(f"  gated record {w}-{n-w}, sign p = {sign_test(w, n):.4f}; "
              f"gated dates: {', '.join(str(x.date()) for x in gated)}")
        print(f"  gated values %: "
              f"{np.round(100*ret.loc[gated].values, 2).tolist()}")
        print(f"  concentration: {cluster_note(gated, ret.loc[gated].values)}")
        edge_bps = 100 * ret.loc[gated].mean() * 100
        print(f"  cost {cost} bps round trip -> {edge_bps/cost:.1f}x "
              f"(episode mean {100*ret.loc[gated].mean():.3f}%)")
        mid = gated[gated.year % 4 == 2]
        non = gated[gated.year % 4 != 2]
        print(f"  midterm N={len(mid)} {100*ret.loc[mid].mean() if len(mid) else np.nan:+.3f}% "
              f"vs non-midterm N={len(non)} "
              f"{100*ret.loc[non].mean() if len(non) else np.nan:+.3f}%")
        yrs = sorted(gated.year.tolist())
        print(f"  year histogram: {yrs}")

# ---------------------------------------------------- placebo anchor ladder
print("\n=== BLOCKER 1: placebo anchor ladder, k = -8..+8 around qw-9 ===")
lad = []
for k in range(-8, 9):
    a = anchors(OFF + k)
    for legs, lbl in [([("IWM", 1.0)], "IWM"),
                      ([("IWM", 1.0), ("SPY", -1.0)], "IWM-SPY")]:
        d, v, ret = cell(legs, a)
        g = r63.reindex(d)
        gd = d[(g <= 10).values]
        lad.append({"k": k, "veh": lbl, "n_ungated": len(d),
                    "ungated_pct": round(100 * np.mean(v), 3),
                    "n_gated": len(gd),
                    "gated_pct": round(100 * ret.loc[gd].mean(), 3) if len(gd) else np.nan})
L = pd.DataFrame(lad)
for veh in ("IWM", "IWM-SPY"):
    sub = L[L["veh"] == veh].copy()
    print(f"\n-- {veh} --")
    print(sub.to_string(index=False))
    for col in ("ungated_pct", "gated_pct"):
        s = sub.dropna(subset=[col])
        true = s.loc[s["k"] == 0, col].iloc[0]
        rk = int((s[col] > true).sum()) + 1
        print(f"  {col}: true k=0 = {true:+.3f}% ranks {rk} of {len(s)}  "
              f"(best {s[col].max():+.3f}% at k={int(s.loc[s[col].idxmax(),'k'])}, "
              f"median {s[col].median():+.3f}%)")

# ---------------------------------------------------- reference class
print("\n=== BLOCKER 2: reference class, identical rule across index/industry ETFs ===")
print("Identical rule: long the member from qw-9 (lag 1) to the quad close, "
      "gated on the MEMBER'S OWN 63d rank <= 10.")
rows, eff, var = [], [], []
for t in CLASS:
    ret = vehicle_ret(px, [(t, 1.0)], H, LAG)
    d = anc.intersection(ret.dropna().index)
    rk = pct_rank(px[t], 63).reindex(d)
    gd = d[(rk <= 10).values]
    if len(gd) < 3:
        rows.append({"tkr": t, "n": len(gd)})
        continue
    s = summarize(ret.loc[gd].values, t)
    se = s["sd_pct"] / np.sqrt(s["n"])
    rows.append({"tkr": t, "n": s["n"], "mean_pct": round(s["mean_pct"], 3),
                 "hit": round(s["hit"], 1), "se": round(se, 3),
                 "worst": round(s["worst_pct"], 2)})
    eff.append(s["mean_pct"])
    var.append(se ** 2)
R = pd.DataFrame(rows).sort_values("mean_pct", ascending=False)
print(R.to_string(index=False))
eff = np.array(eff)
var = np.array(var)
w = 1 / var
mu = (w * eff).sum() / w.sum()
Q = float((w * (eff - mu) ** 2).sum())
dfree = len(eff) - 1
I2 = max(0.0, (Q - dfree) / Q) * 100 if Q > 0 else 0.0
pQ = 1 - 0.5 * (1 + erf((Q - dfree) / sqrt(2 * 2 * dfree)))
iw = R.loc[R["tkr"] == "IWM", "mean_pct"]
rank = int((R["mean_pct"] > iw.iloc[0]).sum()) + 1 if len(iw) else np.nan
print(f"  pooled {mu:.3f}%  Cochran Q={Q:.2f} on {dfree} df (normal-approx "
      f"p~{pQ:.3f})  I^2={I2:.1f}%  IWM ranks {rank} of "
      f"{int(R['mean_pct'].notna().sum())}")

# ---------------------------------------------------- book overlap
print("\n=== BLOCKER 10: book overlap with the 23-year ledger ===")
led = pd.read_parquet(ROOTP / "data" / "backtest_trades_full.parquet")
dcol = "Signal_Date" if "Signal_Date" in led.columns else led.columns[0]
led[dcol] = pd.to_datetime(led[dcol])
win = set()
for a in anc:
    p = int(pos.get(a))
    for j in range(p + LAG, min(len(cal), p + LAG + H + 1)):
        win.add(cal[j])
inwin = led[led[dcol].isin(win)]
share = len(win) / len(cal)
print(f"  window sessions {len(win)} of {len(cal)} = {100*share:.2f}% of the calendar")
print(f"  ledger rows in window {len(inwin)} of {len(led)} = "
      f"{100*len(inwin)/len(led):.2f}%  -> concentration ratio "
      f"{(len(inwin)/len(led))/share:.2f}x")
if len(inwin):
    print("  by strategy:",
          inwin["Strategy_Name"].value_counts().head(8).to_dict()
          if "Strategy_Name" in inwin.columns else "n/a")
