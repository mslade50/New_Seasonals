"""Posts check (2026-09-17): the week after September quad witching, and four
secondary cells for tomorrow's session.

Tomorrow (2026-09-18) is monthly opex AND September quad witching; yesterday was
a scheduled FOMC decision; today SPY +1.13%, QQQ +1.73%, IWM +0.53%, VIX -12.8%.

A. IWM entered MOC on the September quad-witching close, held 5 sessions.
   Controls: the same 5-session window off every OTHER September Friday close,
   and the 5 sessions after the DECEMBER quad-witching close. Plus the
   IWM-minus-SPY spread and the by-year table.
B. Horizon scan h=1..10 on the same anchors (is 5 cherry-picked?).
C. Split A by IWM's 21d return pct_rank (252d lookback) on the quad-witching
   EVE (lag-1 basis, the number you can see before the close) < 25 vs >= 25.
D. SPY 5 sessions after a session that is (i) the day after a scheduled FOMC
   decision and (ii) itself closed >= +1%. Also h=1. Era split at 2018.
E. VIX close-to-close ON the monthly opex session, after VIX closed <= -8% on
   the session before opex. Control: all other opex days.
F. UUP close-to-close on the September quad-witching session, 2007+.
   Control: all other Fridays.

Everything is close-to-close and anchored on a close that is already printed, so
entry is lag=0 from the anchor (MOC that session), matching how the post reads.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa: E402,F401,F403

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ASOF = pd.Timestamp("2026-09-17")
ERA = "2018-01-01"
TK = ["IWM", "SPY", "UUP", "^VIX"]

px = load_prices(TK)
nyse = px["SPY"]["Close"].dropna().index
nyse = pd.DatetimeIndex(nyse[nyse <= ASOF])
C: dict[str, pd.Series] = {t: px[t]["Close"].astype(float).reindex(nyse) for t in TK}
POS = pd.Series(range(len(nyse)), index=nyse)

print(f"NYSE calendar (SPY): {nyse[0].date()} .. {nyse[-1].date()}  n={len(nyse)}")
for t in TK:
    v = C[t].dropna()
    print(f"  {t}: first bar {v.index[0].date()}  last {v.index[-1].date()}  n={len(v)}")


def rec(v: np.ndarray) -> tuple[int, int, int]:
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return int((v > 0).sum()), int((v < 0).sum()), len(v)


def line(label: str, s: pd.Series, dates: pd.DatetimeIndex,
         ctrl: pd.Series | None = None) -> pd.Series:
    v = s.reindex(pd.DatetimeIndex(dates)).dropna()
    if len(v) == 0:
        print(f"  {label}: n=0")
        return v
    up, dn, n = rec(v.values)
    sm = summarize(v.values)
    extra = ""
    if ctrl is not None:
        cv = ctrl.dropna()
        cu, cd, cn = rec(cv.values)
        extra = (f"\n      ctrl: n={cn} {cu}-{cd} mean {100*cv.mean():+.3f}% "
                 f"med {100*float(np.median(cv.values)):+.3f}% hit {100*(cv>0).mean():.1f}%")
    print(f"  {label}: n={n} {up}-{dn} mean {sm['mean_pct']:+.3f}% med {sm['median_pct']:+.3f}% "
          f"hit {sm['hit']:.1f}% t {sm['t']:+.2f} signp_up {sign_test(up, n):.4f} "
          f"signp_dn {sign_test(dn, n):.4f} worst {sm['worst_pct']:+.2f}% "
          f"({v.idxmin().date()}) best {sm['best_pct']:+.2f}% ({v.idxmax().date()}){extra}")
    return v


def third_friday(year: int, month: int) -> pd.Timestamp:
    days = pd.date_range(f"{year}-{month:02d}-01", periods=31, freq="D")
    fri = [d for d in days if d.month == month and d.weekday() == 4]
    return fri[2]


def fwd(s: pd.Series, h: int) -> pd.Series:
    """Close of anchor -> close h sessions later, aligned to the anchor."""
    return s.shift(-h) / s - 1.0


# ---------------------------------------------------------------- event dates
qw = load_events(["quad_witching"])
qw_all = pd.DatetimeIndex(pd.to_datetime(qw["date"].unique()))
sep_qw_cal = pd.DatetimeIndex([d for d in qw_all if d.month == 9 and d <= ASOF])
dec_qw_cal = pd.DatetimeIndex([d for d in qw_all if d.month == 12 and d <= ASOF])

print("\n=== calendar integrity: quad_witching vs third Friday ===")
bad = [d for d in qw_all if d != third_friday(d.year, d.month)]
print(f"  quad_witching rows 2000-2027: {len(qw_all)}; NOT a third Friday: {len(bad)}"
      + (f" -> {[str(x.date()) for x in bad]}" if bad else ""))
print(f"  September quad dates <= asof: {len(sep_qw_cal)} "
      f"({sep_qw_cal[0].date()} .. {sep_qw_cal[-1].date()})")
nontrading = [d for d in sep_qw_cal if d not in nyse]
print(f"  September quad dates not in the NYSE calendar: {[str(x.date()) for x in nontrading]}")

sep_qw = pd.DatetimeIndex([d for d in sep_qw_cal if d in nyse])
dec_qw = pd.DatetimeIndex([d for d in dec_qw_cal if d in nyse])

H = 5
iwm5, spy5 = fwd(C["IWM"], H), fwd(C["SPY"], H)
print("\n=== A. coverage: which September years survive (IWM, h=5) ===")
kept = []
for d in sep_qw:
    p = int(POS[d])
    why = []
    if np.isnan(C["IWM"].iloc[p]):
        why.append("IWM has no bar on the quad close (pre-inception 2000-05-26 or missing)")
    if p + H >= len(nyse):
        why.append(f"only {len(nyse)-1-p} sessions of history after it (need {H})")
    elif np.isnan(iwm5.iloc[p]):
        why.append("IWM exit bar missing")
    if why:
        print(f"  DROP {d.date()} ({d.year}): " + "; ".join(why))
    else:
        kept.append(d)
kept = pd.DatetimeIndex(kept)
print(f"  kept {len(kept)} September quad anchors: {kept[0].year}..{kept[-1].year}")

other_sep_fri = pd.DatetimeIndex(
    [d for d in nyse if d.month == 9 and d.weekday() == 4 and d not in sep_qw])

print("\n=== A. IWM, 5 sessions from the September quad-witching close ===")
a_iwm = line("SEPT quad close -> +5", iwm5, kept, ctrl=iwm5.reindex(other_sep_fri))
print("   (control above = every OTHER September Friday close, same 5-session window)")
line("DEC quad close -> +5 (control 2)", iwm5, dec_qw)
a_spy = line("SPY, same September anchors", spy5, kept)
spread = (iwm5 - spy5)
line("IWM minus SPY spread, same anchors", spread, kept)
pre, post = a_iwm[a_iwm.index < ERA], a_iwm[a_iwm.index >= ERA]
print(f"   era: pre-2018 {rec(pre.values)[:2]} mean {100*pre.mean():+.3f}% | "
      f"2018+ {rec(post.values)[:2]} mean {100*post.mean():+.3f}%")

print("\n   by-year (September quad anchor, 5 sessions, close-to-close):")
print("   year  date        IWM_5d%   SPY_5d%   spread_pp")
for d in kept:
    print(f"   {d.year}  {d.date()}  {100*iwm5.loc[d]:+7.2f}  {100*spy5.loc[d]:+7.2f}  "
          f"{100*(iwm5.loc[d]-spy5.loc[d]):+8.2f}")

print("\n=== B. horizon scan h=1..10, IWM off the September quad close ===")
rows = []
for h in range(1, 11):
    s = fwd(C["IWM"], h)
    v = s.reindex(kept).dropna()
    r = summarize(v.values, f"h={h}")
    up, dn, n = rec(v.values)
    r["rec"] = f"{up}-{dn}"
    r["signp_dn"] = round(sign_test(dn, n), 4)
    base = s.dropna()
    r["ctl_all_days_pct"] = round(100 * base.mean(), 3)
    r["edge_pct"] = round(r["mean_pct"] - 100 * base.mean(), 3)
    rows.append(r)
show(rows, "IWM September quad, horizon scan")

print("\n=== C. split by IWM 21d return pct_rank (252d) on the quad EVE ===")
rank21 = pct_rank(C["IWM"], 21, 252)
eve = {}
for d in kept:
    p = int(POS[d])
    eve[d] = rank21.iloc[p - 1] if p >= 1 else np.nan
eve_s = pd.Series(eve)
print("   eve ranks:", ", ".join(f"{d.year}:{'nan' if np.isnan(v) else round(v,1)}"
                                 for d, v in eve_s.items()))
n_nan = int(eve_s.isna().sum())
if n_nan:
    print(f"   NOTE: {n_nan} anchors have no eve rank (need 21+252 valid IWM sessions); "
          "they appear in NEITHER bucket")
lo = pd.DatetimeIndex([d for d, v in eve_s.items() if not np.isnan(v) and v < 25])
hi = pd.DatetimeIndex([d for d, v in eve_s.items() if not np.isnan(v) and v >= 25])
line(f"eve 21d rank < 25  ({[d.year for d in lo]})", iwm5, lo)
line(f"eve 21d rank >= 25 ({[d.year for d in hi]})", iwm5, hi)
print(f"   today's IWM 21d rank (2026-09-17 close) = {rank21.iloc[-1]:.1f} "
      f"(eve of tomorrow's quad witching)")

print("\n=== D. SPY, 5 sessions after a +1% day-after-FOMC session ===")
dec_ev = load_events(["fomc_decision"])
decs = pd.DatetimeIndex([d for d in pd.to_datetime(dec_ev["date"].unique())
                         if d in nyse and d <= ASOF])
spy_dd = C["SPY"] / C["SPY"].shift(1) - 1.0
after, skipped = [], 0
for d in decs:
    p = int(POS[d])
    if p + 1 >= len(nyse):
        skipped += 1
        continue
    after.append(nyse[p + 1])
after = pd.DatetimeIndex(after)
cond = pd.DatetimeIndex([d for d in after if spy_dd.loc[d] >= 0.01])
print(f"   scheduled decisions in calendar {len(decs)} ({decs[0].date()}..{decs[-1].date()}), "
      f"day-after sessions {len(after)} (skipped {skipped} with no next session)")
print(f"   of those, day-after closed >= +1%: {len(cond)}")
print("   dates:", ", ".join(f"{d.date()} ({100*spy_dd.loc[d]:+.2f}%)" for d in cond))
d5 = line("h=5 from that close", fwd(C["SPY"], 5), cond,
          ctrl=fwd(C["SPY"], 5).reindex(after))
print("   (control above = ALL day-after-decision sessions, unconditional)")
line("h=1 from that close", fwd(C["SPY"], 1), cond,
     ctrl=fwd(C["SPY"], 1).reindex(after))
for lbl, s in (("h=5", fwd(C["SPY"], 5)), ("h=1", fwd(C["SPY"], 1))):
    v = s.reindex(cond).dropna()
    a, b = v[v.index < ERA], v[v.index >= ERA]
    print(f"   {lbl} era: pre-2018 {rec(a.values)[:2]} mean {100*a.mean():+.3f}% med "
          f"{100*np.median(a.values) if len(a) else float('nan'):+.3f}% | "
          f"2018+ {rec(b.values)[:2]} mean {100*b.mean():+.3f}% med "
          f"{100*np.median(b.values) if len(b) else float('nan'):+.3f}%")

print("\n=== E. VIX on the monthly opex session, after a <= -8% pre-opex day ===")
opex = load_events(["opex"])
opex_d = pd.DatetimeIndex([d for d in pd.to_datetime(opex["date"].unique())
                           if d in nyse and d <= ASOF])
vix_dd = C["^VIX"] / C["^VIX"].shift(1) - 1.0
cond_ox, other_ox = [], []
for d in opex_d:
    p = int(POS[d])
    prev_ret = vix_dd.iloc[p - 1] if p >= 1 else np.nan
    if not np.isnan(prev_ret) and prev_ret <= -0.08:
        cond_ox.append(d)
    else:
        other_ox.append(d)
cond_ox, other_ox = pd.DatetimeIndex(cond_ox), pd.DatetimeIndex(other_ox)
print(f"   opex sessions on the calendar and tradeable: {len(opex_d)} "
      f"({opex_d[0].date()}..{opex_d[-1].date()})")
v_ox = line("VIX c2c on opex, prior day <= -8%", vix_dd, cond_ox,
            ctrl=vix_dd.reindex(other_ox))
print("   (control above = VIX c2c on all OTHER opex days)")
if len(v_ox):
    print("   episodes:", ", ".join(f"{d.date()} {100*x:+.1f}%" for d, x in v_ox.items()))
    print(f"   today's VIX move -12.8% qualifies; tomorrow is the opex session")

print("\n=== F. UUP close-to-close ON the September quad session, 2007+ ===")
uup_sep = pd.DatetimeIndex([d for d in sep_qw if d.year >= 2007])
uup_dd = C["UUP"] / C["UUP"].shift(1) - 1.0
other_fri = pd.DatetimeIndex([d for d in nyse if d.weekday() == 4 and d not in sep_qw])
line("UUP on Sept quad day", uup_dd, uup_sep, ctrl=uup_dd.reindex(other_fri))
print("   (control above = UUP c2c on every other Friday)")
vv = uup_dd.reindex(uup_sep).dropna()
if len(vv):
    print("   by-year:", ", ".join(f"{d.year}:{100*x:+.2f}%" for d, x in vv.items()))
    miss = [d.year for d in uup_sep if np.isnan(uup_dd.loc[d])]
    if miss:
        print(f"   missing UUP bars: {miss}")

print(f"\nFROZEN closes on {nyse[-1].date()}: "
      + "  ".join(f"{t} {C[t].iloc[-1]:.2f}" for t in TK))
