import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

TK = ["^GSPC", "SPY", "IWM"]
px = close_panel(TK)
cal = px["^GSPC"].dropna().index
px = px.reindex(cal)
df = pd.DataFrame(index=cal)
df["ym"] = cal.year * 100 + cal.month
df["tdom"] = df.groupby("ym").cumcount() + 1
df["pos"] = np.arange(len(cal))
last_pos = df.groupby("ym")["pos"].max()

sma200 = {t: rolling_on_valid(px[t], lambda x: x.rolling(200, min_periods=180).mean()) for t in TK}
z10 = {t: zscore(px[t], 10) for t in TK}
r63 = {t: pct_rank(px[t], 63) for t in TK}
live = pd.Timestamp("2026-09-11")
for t in TK:
    print(f"live {t}: above200 {100*(px[t].loc[live]/sma200[t].loc[live]-1):+.2f}%  z10 {z10[t].loc[live]:+.2f}  r63 {r63[t].loc[live]:.1f}")


def obs(tkr, entry_tdom=9, month=9, exit_h=None):
    s = px[tkr]
    rows = []
    for ym, g in df.groupby("ym"):
        y, m = divmod(ym, 100)
        if month is not None and m != month or ym == 202609:
            continue
        e = g[g["tdom"] == entry_tdom]
        if e.empty:
            continue
        p0 = int(e["pos"].iloc[0])
        p1 = int(last_pos[ym]) if exit_h is None else p0 + exit_h
        if p1 >= len(cal) or p1 <= p0:
            continue
        a, b = s.iloc[p0], s.iloc[p1]
        if np.isnan(a) or np.isnan(b):
            continue
        sig = cal[p0 - 1]  # signal close = TDOM8 (lag-1 entry at TDOM9 close)
        rows.append(dict(date=cal[p0], y=y, m=m, short=-(b / a - 1), mid=(y % 4 == 2),
                         above200=bool(s.loc[sig] > sma200[tkr].loc[sig]),
                         z10=z10[tkr].loc[sig], r63=r63[tkr].loc[sig]))
    return pd.DataFrame(rows)


def rec(d, label):
    v = d["short"].values
    w = int((v > 0).sum())
    r = summarize(v, label)
    r["rec"] = f"{w}-{len(v)-w}"
    r["sign_p"] = round(sign_test(w, len(v)), 4) if len(v) else np.nan
    return r


for tkr in ["^GSPC", "IWM"]:
    a = obs(tkr)
    rows = [rec(a, "ALL Sep TDOM9->ME"), rec(a[a.mid], "midterm"), rec(a[~a.mid], "non-midterm"),
            rec(a[a.above200], "entry ABOVE 200d (live state)"), rec(a[~a.above200], "entry BELOW 200d"),
            rec(a[a.mid & a.above200], "midterm & above 200d"), rec(a[a.mid & ~a.above200], "midterm & below 200d"),
            rec(a[a.z10 < -1], "z10<-1 at signal (live)"), rec(a[a.z10 >= -1], "z10>=-1"),
            rec(a[a.date < "2018-01-01"], "pre-2018"), rec(a[a.date >= "2018-01-01"], "2018+")]
    show(rows, f"{tkr}: Sep TDOM9 -> month-end SHORT, splits")
    print(a[["date", "short", "mid", "above200", "z10", "r63"]].round(3).to_string(index=False))
    # all-years month ladder
    rows = []
    for m in range(1, 13):
        rows.append(rec(obs(tkr, month=m), f"month {m}"))
    t = pd.DataFrame(rows).sort_values("mean_pct", ascending=False)
    t["rank"] = range(1, 13)
    print(f"\n{tkr} ALL-YEARS month ladder TDOM9->ME short (best first)")
    print(t[["label", "n", "mean_pct", "median_pct", "hit", "rec", "sign_p", "rank"]].round(3).to_string(index=False))
    # all months, above-200d-only drift for the regime control
    allm = pd.concat([obs(tkr, month=m) for m in range(1, 13)])
    show([rec(allm[allm.above200], "ALL months, entry above 200d"),
          rec(allm[~allm.above200], "ALL months, entry below 200d"),
          rec(allm[allm.above200 & (allm.z10 < -1)], "ALL months, above 200d & z10<-1")],
         f"{tkr} regime control across all months")

# permutation: 48-cell month x cycle grid, P(max cell >= observed Sep-midterm), shuffle cycle labels across years
rng = np.random.default_rng(7)
for tkr in ["^GSPC", "IWM"]:
    allm = pd.concat([obs(tkr, month=m) for m in range(1, 13)])
    piv = allm.pivot_table(index="y", columns="m", values="short")
    years = piv.index.values
    obs_val = piv.loc[years % 4 == 2, 9].mean()
    cnt = 0
    B = 2000
    for _ in range(B):
        cyc = rng.permutation(years % 4)
        best = max(piv.loc[cyc == c, m].mean() for c in range(4) for m in range(1, 13))
        cnt += best >= obs_val
    print(f"{tkr}: Sep-midterm {100*obs_val:+.3f}%; P(max of 48 cells >= obs | shuffled cycle labels) = {cnt/B:.3f}")
