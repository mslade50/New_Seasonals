"""C1 -- the month-end rebalance OVERNIGHT.

Enter MOC at the month's LAST trading close (ME-0), exit MOO at the FIRST
session of the next month (ME+1 open). Signal read at ME-1 close, so lag=1.

Mechanism claimed: month-end index/pension rebalancing prints in the closing
auction and the market-impact of that print reverses at the next open. That
predicts an OVERNIGHT, not a multi-day drift.

Every closed form of this anchor in the registry is CLOSE-TO-CLOSE. This is
the decomposition.

Kill tests, in order:
 1. overnight vs the SAME-SPAN close-to-close vs the intraday leg
 2. control against the UNCONDITIONAL overnight (the overnight premium)
 3. offset placebo ladder ME-5..ME+5 on the close-to-next-open trade
 4. era split pre/post 2013 (arbitraged-calendar signature) and pre/post 2018
 5. midterm split + August-turn split
 6. cost (TWO auctions crossed): 4-6 bps equity, 5 bps TLT, need >= 5x
 7. concentration, sign test, worst episode
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

VEH = ["SPY", "QQQ", "IWM", "TLT", "DIA"]
COST = {"SPY": 4.0, "QQQ": 4.0, "IWM": 6.0, "TLT": 5.0, "DIA": 6.0}

px = load_prices(VEH)

print("NOTE: master_prices is the ADJUSTED basis, so the overnight leg")
print("Open[t+1]/Close[t]-1 already nets out the ex-dividend open gap. That is")
print("the tradeable total-return overnight and it is the SAME basis used for")
print("the unconditional overnight control, so the comparison is like-for-like.")


def month_end_positions(idx):
    """Positions of the LAST trading session of each calendar month."""
    per = pd.Series(range(len(idx)), index=idx)
    grp = per.groupby([idx.year, idx.month]).max()
    return sorted(grp.values.tolist())


def overnight(o, c, p):
    """Close[p] -> Open[p+1]."""
    return o[p + 1] / c[p] - 1.0


rows_main = []
per_vehicle = {}

for t in VEH:
    df = px[t]
    idx = df.index
    o, c = df["Open"].values, df["Close"].values
    me = [p for p in month_end_positions(idx) if p + 1 < len(idx)]
    dates = pd.DatetimeIndex([idx[p] for p in me])

    on = np.array([overnight(o, c, p) for p in me])                  # ME0c -> ME1o
    cc = np.array([c[p + 1] / c[p] - 1.0 for p in me])               # ME0c -> ME1c
    intr = np.array([c[p + 1] / o[p + 1] - 1.0 for p in me])         # ME1o -> ME1c

    lo, hi = me[0], me[-1]
    all_on = o[lo + 1:hi + 2] / c[lo:hi + 1] - 1.0
    all_cc = c[lo + 1:hi + 2] / c[lo:hi + 1] - 1.0

    per_vehicle[t] = dict(idx=idx, o=o, c=c, me=me, dates=dates, on=on,
                          cc=cc, intr=intr, all_on=all_on, all_cc=all_cc)

    show([summarize(on, t + " ME0c->ME1o OVERNIGHT"),
          summarize(cc, t + " ME0c->ME1c close-to-close (same span)"),
          summarize(intr, t + " ME1o->ME1c intraday leg"),
          summarize(all_on, t + " CTRL uncond OVERNIGHT all days"),
          summarize(all_cc, t + " CTRL uncond close-to-close all days")],
         "1+2. " + t + ": decomposition and the unconditional-overnight control")

    edge = on.mean() - all_on.mean()
    se = np.sqrt(on.var(ddof=1) / len(on) + all_on.var(ddof=1) / len(all_on))
    wins = int((on > 0).sum())
    base = float((all_on > 0).mean())
    frac_cc = 100 * on.mean() / cc.mean() if cc.mean() != 0 else float("nan")
    print("  %s: overnight EXCESS over uncond overnight = %+.4fpp  welch t = %+.2f"
          % (t, 100 * edge, edge / se))
    print("      overnight %+.4f%% vs same-span c2c %+.4f%% -> overnight is %.0f%% of the c2c"
          % (100 * on.mean(), 100 * cc.mean(), frac_cc))
    print("      record %d-%d (%.1f%% hit) vs uncond overnight up-rate %.1f%%, sign p vs that base = %.4f"
          % (wins, len(on) - wins, 100 * wins / len(on), 100 * base,
             sign_test(wins, len(on), base)))
    print("      cost %.1f bps rt (2 auctions) -> %.2fx on the RAW mean, %.2fx on the EXCESS (need >=5x)"
          % (COST[t], 100 * 100 * on.mean() / COST[t], 100 * 100 * edge / COST[t]))
    rows_main.append({"ticker": t, "n": len(on),
                      "on_pct": round(100 * on.mean(), 4),
                      "uncond_on_pct": round(100 * all_on.mean(), 4),
                      "excess_pp": round(100 * edge, 4),
                      "t_excess": round(edge / se, 2),
                      "c2c_pct": round(100 * cc.mean(), 4),
                      "intraday_pct": round(100 * intr.mean(), 4),
                      "hit": round(100 * wins / len(on), 1),
                      "uncond_hit": round(100 * base, 1),
                      "x_cost_excess": round(100 * 100 * edge / COST[t], 2)})

show(rows_main, "SUMMARY 1+2: overnight vs unconditional overnight, per vehicle")

# ---------------------------------------------------------------------------
# 3. offset placebo ladder ME-5 .. ME+5 (the same close-to-next-open trade)
# ---------------------------------------------------------------------------
ladder = []
for t in VEH:
    d = per_vehicle[t]
    idx, o, c, me = d["idx"], d["o"], d["c"], d["me"]
    row = {"ticker": t}
    for k in range(-5, 6):
        vals = [overnight(o, c, p + k) for p in me
                if 0 <= p + k and p + k + 1 < len(idx)]
        row["ME%+d" % k] = round(100 * float(np.mean(vals)), 4)
    ladder.append(row)
show(ladder, "3. offset placebo ladder: close->next-open mean % at ME-5..ME+5")

for t in VEH:
    row = [x for x in ladder if x["ticker"] == t][0]
    vals = {k: row[k] for k in row if k != "ticker"}
    order = sorted(vals.items(), key=lambda kv: -kv[1])
    rank0 = [k for k, _ in order].index("ME+0") + 1
    print("  %s: ME+0 ranks %d of 11 on the ladder; top3 %s"
          % (t, rank0, order[:3]))

# ---------------------------------------------------------------------------
# 4. era splits
# ---------------------------------------------------------------------------
for cut in ("2013-01-01", "2018-01-01"):
    rows = []
    for t in VEH:
        d = per_vehicle[t]
        m = d["dates"] < pd.Timestamp(cut)
        a = summarize(d["on"][m], "%s pre-%s" % (t, cut[:4]))
        b = summarize(d["on"][~m], "%s %s+" % (t, cut[:4]))
        idx = d["idx"]
        allpos = np.arange(d["me"][0], d["me"][-1] + 1)
        adates = idx[allpos]
        am = adates < pd.Timestamp(cut)
        if a.get("n"):
            a["ctrl_uncond_pct"] = round(100 * d["all_on"][am].mean(), 4)
            a["excess_pp"] = round(a["mean_pct"] - a["ctrl_uncond_pct"], 4)
        if b.get("n"):
            b["ctrl_uncond_pct"] = round(100 * d["all_on"][~am].mean(), 4)
            b["excess_pp"] = round(b["mean_pct"] - b["ctrl_uncond_pct"], 4)
        rows += [a, b]
    show(rows, "4. era split at %s (overnight, with matched uncond control)" % cut)

# ---------------------------------------------------------------------------
# 5. midterm + August-turn splits
# ---------------------------------------------------------------------------
rows = []
for t in VEH:
    d = per_vehicle[t]
    yr, mo = d["dates"].year, d["dates"].month
    mid = (yr % 4) == 2
    aug = mo == 8
    rows += [summarize(d["on"][mid], t + " midterm"),
             summarize(d["on"][~mid], t + " non-midterm"),
             summarize(d["on"][aug], t + " AUGUST turn"),
             summarize(d["on"][~aug], t + " non-August")]
show(rows, "5. midterm and August-turn splits (overnight)")

rows = []
for t in VEH:
    d = per_vehicle[t]
    yr, mo = d["dates"].year, d["dates"].month
    both = ((yr % 4) == 2) & (mo == 8)
    r = summarize(d["on"][both], t + " AUGUST turn in a MIDTERM year")
    r["dates"] = ", ".join(str(x.date()) for x in d["dates"][both])
    rows.append(r)
show(rows, "5b. today's exact cell: August turn, midterm year")

# ---------------------------------------------------------------------------
# 7. concentration / sign test / worst
# ---------------------------------------------------------------------------
print("\n7. concentration + worst episode (overnight)")
for t in VEH:
    d = per_vehicle[t]
    print("  %s: %s" % (t, cluster_note(d["dates"], d["on"], k=3)))
    i = int(np.argmin(d["on"]))
    print("      worst %.2f%% on %s; best %.2f%%"
          % (100 * d["on"][i], d["dates"][i].date(), 100 * d["on"].max()))
