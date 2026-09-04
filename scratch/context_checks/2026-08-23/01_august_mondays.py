"""Mondays in August: is the QQQ 76-41 record an August fact or a Monday fact?

Engine cell E:weekday_month|QQQ: n=117 h1 +0.140% hit 65.0 t=1.12,
record 76-41 up, sign p 0.0008, BH pass, era stable.

ANCHOR NOTE: the engine anchors on the session BEFORE the target weekday
(`anchors_before`), so h=1 is the August Monday's OWN close-to-close return.
This script masks the Monday itself and reads its own return, which is the
same number. The mean is unremarkable and the record is not, so the whole
question is which control the record beats.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import load_prices, load_events, summarize, sign_test, era_split, cluster_note

TK = ["QQQ", "SPY", "IWM", "^VIX"]
px = load_prices(TK)
ASOF = pd.Timestamp("2026-08-21")
RET = {t: px[t]["Close"].astype(float).loc[:ASOF].pct_change(fill_method=None).dropna()
       for t in TK}


def row(t, mask, label):
    v = RET[t][mask].values
    r = summarize(v, label)
    if r["n"]:
        up = int((v > 0).sum())
        r["record"] = f"{up}-{r['n'] - up}"
        r["sign_p"] = round(sign_test(up, r["n"]), 4)
    return r


def show(rows):
    df = pd.DataFrame(rows)
    keep = [c for c in ["label", "n", "mean_pct", "median_pct", "hit", "t", "record", "sign_p"] if c in df]
    print(df[keep].round(3).to_string(index=False))


for t in TK:
    i = RET[t].index
    print(f"\n########## {t}: the session's OWN return ##########")
    show([row(t, np.ones(len(i), bool),                    "all days"),
          row(t, i.dayofweek == 0,                          "all Mondays"),
          row(t, i.month == 8,                              "all August days"),
          row(t, (i.month == 8) & (i.dayofweek == 0),       "August Mondays"),
          row(t, (i.month != 8) & (i.dayofweek == 0),       "Mondays, not August"),
          row(t, (i.month == 8) & (i.dayofweek != 0),       "August, not Monday")])

print("\n########## QQQ August Mondays: era and concentration ##########")
i = RET["QQQ"].index
m = (i.month == 8) & (i.dayofweek == 0)
ser = RET["QQQ"][m]
cut = pd.Timestamp("2018-01-01")
for lab, v in [("pre-2018", ser[ser.index < cut].values), ("2018+", ser[ser.index >= cut].values)]:
    up = int((v > 0).sum())
    st = summarize(v, lab)
    print(f"  {lab:9s} n={st['n']:3d} mean={st['mean_pct']:+.3f}% med={st['median_pct']:+.3f}% "
          f"record {up}-{st['n']-up} sign_p={sign_test(up, st['n']):.4f}")
print("  concentration:", cluster_note(ser.index, ser.values, k=2))
print("  worst/best: %.2f%% / %.2f%%" % (100*ser.min(), 100*ser.max()))
yr = pd.Series(ser.values, index=ser.index).groupby(ser.index.year)
print("  years with a losing majority:",
      [int(y) for y, x in yr if (x > 0).sum() <= (x <= 0).sum()])

print("\n########## Is it Monday, or is it the first session of the week? ##########")
for t in ["QQQ", "SPY"]:
    i = RET[t].index
    first_of_week = pd.Series(i, index=i).groupby([i.year, i.isocalendar().week]).transform("min").values == i.values
    show([row(t, (i.month == 8) & first_of_week, f"{t} August week-openers"),
          row(t, (i.month == 8) & (i.dayofweek == 0), f"{t} August Mondays")])

print("\n########## August Mondays split by post-opex ##########")
opex = pd.DatetimeIndex(load_events(["opex"])["date"])
for t in ["QQQ", "SPY", "^VIX"]:
    i = RET[t].index
    pos = pd.Series(range(len(i)), index=i)
    ox = np.array([pos[d] for d in opex if d in pos.index])
    prev = np.searchsorted(ox, pos.values, side="right") - 1
    td_since = np.where(prev >= 0, pos.values - ox[np.clip(prev, 0, None)], 10**6)
    augmon = (i.month == 8) & (i.dayofweek == 0)
    print(f"\n--- {t} ---")
    show([row(t, augmon & (td_since == 1), "Aug Mon, 1st session after opex"),
          row(t, augmon & (td_since != 1), "Aug Mon, every other")])
    print("   post-opex Aug Mondays:", [str(d.date()) for d in i[augmon & (td_since == 1)]])
