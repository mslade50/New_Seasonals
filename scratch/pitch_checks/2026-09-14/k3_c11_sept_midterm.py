import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

TK = ["^GSPC", "SPY", "IWM"]
px = close_panel(TK)
for t in TK:
    s = px[t].dropna()
    print(t, "first", s.index[0].date(), "last", s.index[-1].date())

cal = px["^GSPC"].dropna().index
fomc = pd.DatetimeIndex(load_events(["fomc_decision"])["date"])

# trading-day-of-month on the ^GSPC calendar
df = pd.DataFrame(index=cal)
df["ym"] = cal.year * 100 + cal.month
df["tdom"] = df.groupby("ym").cumcount() + 1
df["pos"] = np.arange(len(cal))
last_pos = df.groupby("ym")["pos"].max()


def cell(tkr, entry_tdom, h, month=None, years=None, to_month_end=False):
    """short return entering at close of entry_tdom, exiting h sessions later
    (or at the last session of the month). One observation per month."""
    s = px[tkr].reindex(cal)
    rows = []
    for ym, g in df.groupby("ym"):
        y, m = divmod(ym, 100)
        if month is not None and m != month:
            continue
        if years is not None and not years(y):
            continue
        e = g[g["tdom"] == entry_tdom]
        if e.empty:
            continue
        p0 = int(e["pos"].iloc[0])
        p1 = int(last_pos[ym]) if to_month_end else p0 + h
        if to_month_end and p1 <= p0:
            continue
        if p1 >= len(cal) or ym == 202609:
            continue
        a, b = s.iloc[p0], s.iloc[p1]
        if np.isnan(a) or np.isnan(b):
            continue
        fin = bool(((fomc > cal[p0]) & (fomc <= cal[p1])).any())
        rows.append((cal[p0], y, m, -(b / a - 1.0), fin, p1 - p0))
    return pd.DataFrame(rows, columns=["date", "y", "m", "short", "fomc_in", "held"])


mid = lambda y: y % 4 == 2
nonmid = lambda y: y % 4 != 2


def rec(d, label):
    v = d["short"].values
    w = int((v > 0).sum())
    r = summarize(v, label)
    r["rec"] = f"{w}-{len(v)-w}"
    r["sign_p"] = round(sign_test(w, len(v)), 4) if len(v) else np.nan
    return r


for tkr in ["^GSPC", "SPY", "IWM"]:
    rows = []
    for h in [5, 8, 10, "ME"]:
        me = h == "ME"
        hh = 0 if me else h
        a = cell(tkr, 9, hh, month=9, years=mid, to_month_end=me)
        b = cell(tkr, 9, hh, month=9, years=nonmid, to_month_end=me)
        c = cell(tkr, 9, hh, month=None, years=None, to_month_end=me)
        dmid = cell(tkr, 9, hh, month=None, years=mid, to_month_end=me)
        rows += [rec(a, f"h={h} Sep MIDTERM"), rec(b, f"h={h} Sep non-mid"),
                 rec(c, f"h={h} all months all yrs"), rec(dmid, f"h={h} all months midterm")]
    show(rows, f"{tkr}: SHORT from TDOM9 close (returns = short P&L)")

# detail: midterm Septembers month-end
for tkr in ["^GSPC", "IWM"]:
    a = cell(tkr, 9, 0, month=9, years=mid, to_month_end=True)
    print(f"\n{tkr} midterm Sep TDOM9->month-end detail:")
    for _, r in a.iterrows():
        print(f"  {r['date'].date()} short {100*r['short']:+.2f}%  fomc_in={r['fomc_in']} held={r['held']}")
    a10 = cell(tkr, 9, 10, month=9, years=mid)
    print(f"{tkr} midterm Sep h=10 detail:", ", ".join(f"{r['y']}:{100*r['short']:+.2f}" for _, r in a10.iterrows()))
    allsep = cell(tkr, 9, 0, month=9, to_month_end=True)
    show([rec(allsep[allsep.fomc_in], "all Sep FOMC in window"),
          rec(allsep[~allsep.fomc_in], "all Sep FOMC NOT in window")], f"{tkr} Sep FOMC split (all years, ME)")

# placebo ladder over months (midterm years, TDOM9 -> month end)
for tkr in ["^GSPC", "IWM"]:
    rows = []
    for m in range(1, 13):
        a = cell(tkr, 9, 0, month=m, years=mid, to_month_end=True)
        rows.append(rec(a, f"month {m}"))
    t = pd.DataFrame(rows).sort_values("mean_pct", ascending=False)
    t["rank"] = range(1, 13)
    print(f"\n{tkr} placebo ladder over MONTHS (midterm, TDOM9->ME, short P&L, best first)")
    print(t[["label", "n", "mean_pct", "hit", "rec", "sign_p", "rank"]].round(3).to_string(index=False))
    # entry-day ladder in midterm September, fixed h=10
    rows = []
    for e in range(3, 15):
        a = cell(tkr, e, 10, month=9, years=mid)
        rows.append(rec(a, f"entry TDOM {e}"))
    t = pd.DataFrame(rows).sort_values("mean_pct", ascending=False)
    t["rank"] = range(1, len(t) + 1)
    print(f"\n{tkr} placebo ladder over ENTRY TDOM (midterm Sep, h=10, short P&L, best first)")
    print(t[["label", "n", "mean_pct", "hit", "rec", "sign_p", "rank"]].round(3).to_string(index=False))
    # month x cycle grid (48 cells) rank for TDOM9 -> ME
    grid = []
    for m in range(1, 13):
        for cyc in range(4):
            a = cell(tkr, 9, 0, month=m, years=(lambda y, c=cyc: y % 4 == c), to_month_end=True)
            grid.append((m, cyc, a["short"].mean() * 100, len(a)))
    g = pd.DataFrame(grid, columns=["m", "cyc", "mean", "n"]).sort_values("mean", ascending=False).reset_index(drop=True)
    r = int(g.index[(g.m == 9) & (g.cyc == 2)][0]) + 1
    print(f"{tkr} 48-cell month x cycle grid, Sep-midterm short rank {r} of 48 ; top5:\n", g.head(5).round(3).to_string(index=False))
