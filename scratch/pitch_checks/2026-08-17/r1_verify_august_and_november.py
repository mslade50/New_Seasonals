"""Red-team verification of the two numbers the morning turns on.

(1) The C1 kill: is the August TLT month-position effect really era-dead, or did
    the checker's era cut manufacture it? Independent re-derivation.
(2) The one thing worth parking: November is the same effect's live sibling.
    Confirm it, and CHARGE it for the 12-month scan it came out of.

Entry convention: signal on close D, entry MOC on close D+1 (lag=1), so a
"tdom 11 entry" means the signal bar is tdom 10 and we buy the tdom 11 close --
which is exactly today, 2026-08-17.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = load_prices(["TLT", "IEF"])
tlt = px["TLT"]["Close"].dropna()
ief = px["IEF"]["Close"].dropna()
ALL = tlt.index

TDOM_LO, TDOM_HI = 4, 12          # the plateau the checker reported
H = 10


def tdom(idx: pd.DatetimeIndex) -> pd.Series:
    """1-based trading day of month for every date in the index."""
    s = pd.Series(idx, index=idx)
    return s.groupby([idx.year, idx.month]).cumcount() + 1


TD = tdom(ALL)


def cell(month: int, lo: int = TDOM_LO, hi: int = TDOM_HI) -> pd.DatetimeIndex:
    """Signal days in `month` whose ENTRY (D+1) lands in the tdom band."""
    m = pd.Series(ALL.month == month, index=ALL) & (TD.shift(-1) >= lo) & (TD.shift(-1) <= hi)
    return ALL[m.fillna(False)]


def stats(dates, s=tlt, h=H):
    v = fwd_lag(s, h, 1).reindex(dates).dropna()
    if not len(v):
        return None
    yr = pd.Series(v.values, index=v.index).groupby(v.index.year).mean()
    return {"N": len(v), "mean_pct": v.mean() * 100, "med_pct": v.median() * 100,
            "yrs": len(yr), "yr_up": int((yr > 0).sum()),
            "sign_p": sign_test(int((yr > 0).sum()), len(yr)),
            "vals": v}


print("=" * 96)
print(f"MONTH-OF-YEAR at MATCHED tdom {TDOM_LO}-{TDOM_HI}, TLT h={H}, lag-1 entry")
print("=" * 96)
print(f"{'mon':<5} {'N':>5} {'mean%':>8} {'med%':>8} {'yrs':>5} {'up':>4} {'signp':>8}"
      f" | {'2018+ mean%':>12} {'2018+ up/yrs':>13} | {'2013+ mean%':>12}")
rows = {}
for m in range(1, 13):
    d = cell(m)
    a = stats(d)
    v = a["vals"]
    v18 = v[v.index >= "2018-01-01"]
    v13 = v[v.index >= "2013-01-01"]
    y18 = pd.Series(v18.values, index=v18.index).groupby(v18.index.year).mean() if len(v18) else pd.Series(dtype=float)
    rows[m] = a
    print(f"{m:<5} {a['N']:>5} {a['mean_pct']:>8.3f} {a['med_pct']:>8.3f} {a['yrs']:>5} "
          f"{a['yr_up']:>4} {a['sign_p']:>8.4f} | {v18.mean()*100 if len(v18) else float('nan'):>12.3f}"
          f" {f'{int((y18>0).sum())}/{len(y18)}':>13} | "
          f"{v13.mean()*100 if len(v13) else float('nan'):>12.3f}")

print("\n" + "=" * 96)
print("AUGUST ERA LADDER (the C1 kill) -- year-mean basis, non-overlapping by year")
print("=" * 96)
aug = cell(8)
va = fwd_lag(tlt, H, 1).reindex(aug).dropna()
ya = pd.Series(va.values, index=va.index).groupby(va.index.year).mean()
for lo, hi in [(2002, 2012), (2013, 2017), (2018, 2020), (2021, 2025), (2013, 2025), (2018, 2025)]:
    w = ya[(ya.index >= lo) & (ya.index <= hi)]
    print(f"  {lo}-{hi}: yrs {len(w):>2}  mean {w.mean()*100:+7.3f}%  "
          f"up {int((w>0).sum())}/{len(w)}  sign p {sign_test(int((w>0).sum()), len(w)):.4f}")

print("\nNOVEMBER ERA LADDER (the park candidate)")
nov = cell(11)
vn = fwd_lag(tlt, H, 1).reindex(nov).dropna()
yn = pd.Series(vn.values, index=vn.index).groupby(vn.index.year).mean()
for lo, hi in [(2002, 2012), (2013, 2017), (2018, 2020), (2021, 2025), (2013, 2025), (2018, 2025)]:
    w = yn[(yn.index >= lo) & (yn.index <= hi)]
    print(f"  {lo}-{hi}: yrs {len(w):>2}  mean {w.mean()*100:+7.3f}%  "
          f"up {int((w>0).sum())}/{len(w)}  sign p {sign_test(int((w>0).sum()), len(w)):.4f}")

print("\n" + "=" * 96)
print("CHARGING NOVEMBER FOR THE 12-MONTH SCAN IT CAME OUT OF")
print("=" * 96)
best = max(rows, key=lambda m: rows[m]["mean_pct"])
print(f"best month by mean = {best} ({rows[best]['mean_pct']:+.3f}%), "
      f"its per-year sign p = {rows[best]['sign_p']:.5f}")
print(f"Bonferroni over 12 months: {min(1.0, rows[best]['sign_p']*12):.5f}")
print(f"August rank by mean: {sorted(rows, key=lambda m: -rows[m]['mean_pct']).index(8)+1} of 12")
print(f"Nov rank by mean:    {sorted(rows, key=lambda m: -rows[m]['mean_pct']).index(11)+1} of 12")

print("\nIEF-neutral residual (duration-neutral), h=10, per month")
beta = float(np.polyfit(ief.pct_change().reindex(ALL).fillna(0.0),
                        tlt.pct_change().reindex(ALL).fillna(0.0), 1)[0])
print(f"  measured TLT~IEF daily beta = {beta:.3f}")
for m in (8, 11):
    d = cell(m)
    r = (fwd_lag(tlt, H, 1) - beta * fwd_lag(ief, H, 1)).reindex(d).dropna()
    r18 = r[r.index >= "2018-01-01"]
    print(f"  month {m}: residual {r.mean()*100:+7.3f}%  (2018+ {r18.mean()*100:+7.3f}%)")

print("\n" + "=" * 96)
print("TODAY'S RATE REGIME (the live conditioner the checker flagged)")
print("=" * 96)
BAR = pd.Timestamp("2026-08-14")
lvl = px["TLT"]["Close"]
chg63 = (lvl.pct_change(63) * 100).loc[BAR]
print(f"  TLT 63d price change {chg63:+.2f}%  -> a RISING-yield regime")
print(f"  TLT sits {(lvl.loc[BAR]/lvl.rolling(252).min().loc[BAR]-1)*100:.2f}% off its 52w low")
