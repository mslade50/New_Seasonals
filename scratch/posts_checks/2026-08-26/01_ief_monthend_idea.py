"""Idea check: long duration (IEF / TLT) over the last two sessions of August.

Tonight (Wed 2026-08-26) is the fourth-to-last session of August. The context
brief's month-end cell says sessions INSIDE a month's final three pay bonds
the next day (IEF 574-385, t 8.96), with or without Jackson Hole nearby. The
tradeable shape from here, in pitch convention (signal day D = tonight, entry
at the close of D+1 = MOC Thursday 08-27, exit h sessions later):

    signal: dist-to-month-end == 3   (tonight)
    entry:  MOC on ME-3 (Thu 08-27)
    exit:   MOC on ME-1 (Mon 08-31)  -> h = 2

So the trade captures exactly the two forward sessions whose anchors sit
inside the final-three window (Fri anchored on Thu, Mon anchored on Fri).

Kill attempts: all-months full history vs all-days and local controls, era
split at 2018 (the brief says the effect is decaying), midterm years, August
only (every August case is Jackson Hole adjacent, so this is the confound
split), h=1 / h=3 horizon neighbours, concentration, TLT as the second
vehicle, and a naive lag-0 replication of the brief's own cell so the two
conventions tie out.

Second section: the August ME-3 session itself in SPY / IWM (tomorrow's
session, close-to-close from tonight), split by cycle year, because the
brief only printed the midterm split for SPY.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    cluster_note, era_split, fwd_lag, fwd_ret, load_events, load_prices,
    local_control, sign_test, summarize, wilder_atr,
)

px = load_prices(["IEF", "TLT", "SPY", "IWM"])
closes = {k: v["Close"].dropna() for k, v in px.items()}
for k in ("IEF", "TLT"):
    f = px[k]
    atr = pd.Series(wilder_atr(f["High"], f["Low"], f["Close"]), index=f.index)
    print(f"tonight {k}: close {closes[k].iloc[-1]:.2f} | Wilder-14 ATR {atr.iloc[-1]:.4f} "
          f"({atr.iloc[-1] / closes[k].iloc[-1] * 100:.2f}%) | bar {f.index[-1].date()}")


def dist_to_month_end(idx: pd.DatetimeIndex) -> pd.Series:
    """Sessions remaining after each date inside its month (0 = last session).
    The final month on file is open, so its rows are dropped."""
    ser = pd.Series(np.arange(len(idx)), index=idx)
    ym = idx.to_period("M")
    last_pos = ser.groupby(ym).transform("max")
    d = last_pos - ser
    # open month: unknown distance
    d[ym == ym[-1]] = np.nan
    return d


jh = pd.to_datetime(sorted(load_events(["jackson_hole"])["date"].unique()))


def near_jh(dates: pd.DatetimeIndex, days: int = 7) -> np.ndarray:
    out = np.zeros(len(dates), dtype=bool)
    for j in jh:
        out |= np.abs((dates - j).days) <= days
    return out


def block(name: str, r: pd.Series, s: pd.Series, h: int, lag: int = 1) -> None:
    v = r.values
    st = summarize(v)
    nup = int((r > 0).sum())
    allr = fwd_lag(s, h, lag).dropna()
    loc = allr.reindex(local_control(s.index, r.index, 126)).dropna()
    print(f"  {name:<28} n={st['n']:<4} mean={st['mean_pct']:+.3f}%  med={st['median_pct']:+.3f}%  "
          f"{nup}-{len(r)-nup} up ({st['hit']:.1f}%)  t={st['t']:+.2f}  sign_p={sign_test(nup, len(r)):.4f}  "
          f"| all {100*allr.mean():+.3f}%  local {100*loc.mean():+.3f}%  "
          f"| worst {st['worst_pct']:+.2f}% ({r.idxmin().date()})  best {st['best_pct']:+.2f}% ({r.idxmax().date()})")


def splits(r: pd.Series) -> None:
    v = r.values
    print("    era:", [(e["label"], e["n"], round(e.get("mean_pct", np.nan), 3),
                        round(e.get("hit", np.nan), 1), round(e.get("t", np.nan), 2))
                       for e in era_split(r.index, v)])
    print("    concentration:", cluster_note(r.index, v))
    mid = r[[d.year % 4 == 2 for d in r.index]]
    non = r[[d.year % 4 != 2 for d in r.index]]
    for lab, sub in (("midterm", mid), ("non-midterm", non)):
        if len(sub):
            su = summarize(sub.values)
            nu = int((sub > 0).sum())
            print(f"    {lab:<12} n={su['n']:<4} mean={su['mean_pct']:+.3f}%  {nu}-{len(sub)-nu} up  "
                  f"sign_p={sign_test(nu, len(sub)):.4f}  worst={su['worst_pct']:+.2f}%")


print("\n" + "=" * 90)
print("SECTION 1: signal = fourth-to-last session of ANY month; entry MOC next session; hold h")
print("=" * 90)
for tkr in ("IEF", "TLT"):
    s = closes[tkr]
    dist = dist_to_month_end(s.index)
    sig = s.index[(dist == 3).values]
    print(f"\n--- {tkr}  signals {len(sig)}  {sig[0].date()} .. {sig[-1].date()} ---")
    for h in (1, 2, 3):
        r = fwd_lag(s, h, 1).reindex(sig).dropna()
        block(f"h={h} (MOC ME-3 -> ME-{3-h})", r, s, h)
    r2 = fwd_lag(s, 2, 1).reindex(sig).dropna()
    splits(r2)
    aug = r2[[d.month == 8 for d in r2.index]]
    nonaug = r2[[d.month != 8 for d in r2.index]]
    for lab, sub in (("august (all JH-adjacent)", aug), ("other 11 months", nonaug)):
        su = summarize(sub.values)
        nu = int((sub > 0).sum())
        print(f"    {lab:<26} n={su['n']:<4} mean={su['mean_pct']:+.3f}%  {nu}-{len(sub)-nu} up  "
              f"sign_p={sign_test(nu, len(sub)):.4f}  t={su['t']:+.2f}  worst={su['worst_pct']:+.2f}%")
    print("    august by year:", [(d.year, round(100 * x, 2)) for d, x in aug.items()])
    aug_mid = aug[[d.year % 4 == 2 for d in aug.index]]
    print(f"    august midterms: n={len(aug_mid)} mean={100*aug_mid.mean():+.3f}% "
          f"{int((aug_mid>0).sum())}-{int((aug_mid<=0).sum())}")
    # 2018+ only, the decaying half
    late = r2[r2.index >= "2018-01-01"]
    print("    2018+ by year mean:", {y: round(100 * g.mean(), 2) for y, g in late.groupby(late.index.year)})
    # entry-day cost check: what does the ME-3 session itself do (lag 0, h1)?
    r0 = fwd_ret(s, 1).reindex(sig).dropna()
    print(f"    the entry session itself (ME-4 close -> ME-3 close): {100*r0.mean():+.3f}%  "
          f"{int((r0>0).sum())}-{int((r0<=0).sum())}")

print("\n--- replication of the brief's cell (lag 0: any session with dist<=2, h=1) ---")
for tkr in ("IEF", "TLT"):
    s = closes[tkr]
    dist = dist_to_month_end(s.index)
    anc = s.index[(dist <= 2).values]
    r = fwd_ret(s, 1).reindex(anc).dropna()
    nj = near_jh(r.index)
    for lab, sub in (("all final-3 anchors", r), ("JH week removed", r[~nj])):
        su = summarize(sub.values)
        nu = int((sub > 0).sum())
        print(f"  {tkr} {lab:<22} n={su['n']:<4} mean={su['mean_pct']:+.3f}%  {nu}-{len(sub)-nu}  t={su['t']:+.2f}")

print("\n" + "=" * 90)
print("SECTION 2: the August third-to-last session itself (tonight's close -> tomorrow's close)")
print("=" * 90)
for tkr in ("SPY", "IWM"):
    s = closes[tkr]
    dist = dist_to_month_end(s.index)
    anc = s.index[((dist == 3) & (s.index.month == 8)).values]
    r = fwd_ret(s, 1).reindex(anc).dropna()
    st = summarize(r.values)
    nup = int((r > 0).sum())
    other = fwd_ret(s, 1).reindex(s.index[((dist == 3) & (s.index.month != 8)).values]).dropna()
    print(f"\n  {tkr} august ME-3 session: n={st['n']} mean={st['mean_pct']:+.3f}% {nup}-{len(r)-nup} "
          f"t={st['t']:+.2f} sign_p={sign_test(nup, len(r)):.4f} | same slot other months "
          f"{100*other.mean():+.3f}% {int((other>0).sum())}-{int((other<=0).sum())}")
    splits(r)
    print("    by year:", [(d.year, round(100 * x, 2)) for d, x in r.items()])
