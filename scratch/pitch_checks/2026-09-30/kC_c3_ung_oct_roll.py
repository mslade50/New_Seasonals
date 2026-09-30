"""C3 round 1: short UNG from the September month-end close across the October roll.

Pre-specified: short UNG from the Sep ME close, h=1..10 plus whole October (to the Oct
ME close). The question is NOT whether short UNG pays (it bleeds every month) but
whether the OCTOBER window beats the short's all-month bleed over same-length windows.
Controls: all ME anchors (every month), own unconditional drift (all days), and the
12-month ladder. NG=F front leg measured on the same windows (seam-checked: the NG
expiry sits ~ME-3, so ME-anchored windows of <= 10 sessions are seam-free).
Calendar anchor: entry = the anchor close (lag=0).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

P = load_prices(["UNG", "NG=F"])
u = P["UNG"]["Close"].dropna()
ng = P["NG=F"]["Close"].dropna()
ng = ng[ng > 0]
idx = u.index
LAST = idx[-1]


def me_anchors(ix: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(ix, index=ix)
    m = pd.DatetimeIndex(s.groupby([ix.year, ix.month]).max().values)
    return m[m < LAST]  # the Sep-2026 group ends 09-29, not a real ME


ME = me_anchors(idx)


def ng_expiry(year: int, month: int) -> pd.Timestamp:
    """Last trade of the contract delivering in (year, month): 3 bd before the 1st."""
    first = pd.Timestamp(year=year, month=month, day=1)
    return first - pd.offsets.BDay(3)


# NG expiry dates (approx, business days ignoring holidays)
EXP = pd.DatetimeIndex([ng_expiry(y, m) for y in range(2000, 2028) for m in range(1, 13)])


def fwd_from(s: pd.Series, anchors: pd.DatetimeIndex, h: int) -> pd.Series:
    r = s.shift(-h) / s - 1.0
    a = anchors.intersection(s.index)
    return r.loc[a]


def to_next_me(s: pd.Series, anchors: pd.DatetimeIndex) -> pd.Series:
    out = {}
    a = list(anchors)
    for i in range(len(a) - 1):
        if a[i] in s.index and a[i + 1] in s.index:
            out[a[i]] = s.loc[a[i + 1]] / s.loc[a[i]] - 1.0
    return pd.Series(out)


def seam_in(anchor: pd.Timestamp, h: int, ix: pd.DatetimeIndex) -> bool:
    p = ix.get_loc(anchor)
    if p + h >= len(ix):
        return False
    lo, hi = ix[p], ix[p + h]
    return bool(((EXP >= lo) & (EXP <= hi)).any())


def ladder(ret: pd.Series, label: str) -> pd.DataFrame:
    """SHORT return by WINDOW month (anchor month + 1)."""
    r = -ret.dropna()
    wm = ((r.index.month % 12) + 1)
    rows = []
    for m in range(1, 13):
        v = r[wm == m].values
        s = summarize(v, f"{label} win-month {m:02d}")
        w = int((v > 0).sum())
        s["sign_p"] = sign_test(w, len(v)) if len(v) else np.nan
        rows.append(s)
    df = pd.DataFrame(rows)
    df["rank"] = df["mean_pct"].rank(ascending=False).astype(int)
    return df


print(f"UNG {idx[0].date()}..{LAST.date()}  ME anchors {len(ME)} ({ME[0].date()}..{ME[-1].date()})")
print(f"live: UNG 09-29 close {u.iloc[-1]:.2f}; NG=F {ng.iloc[-1]:.3f}")

# ---------------------------------------------------------------- 1. month ladder, UNG short
for h in (5, 10):
    r = fwd_from(u, ME, h)
    df = ladder(r, f"UNG h={h}")
    print(f"\n=== 1. SHORT UNG from ME close, h={h}, by WINDOW month ===")
    print(df[["label", "n", "mean_pct", "median_pct", "hit", "t", "worst_pct", "sign_p", "rank"]].round(3).to_string(index=False))
    allme = -r.dropna()
    oct_ = allme[allme.index.month == 9]
    oth = allme[allme.index.month != 9]
    drift = -(u.shift(-h) / u - 1.0).dropna()
    print(f"  CTRL all ME anchors: {100*allme.mean():+.3f}% (n={len(allme)}, hit {100*(allme>0).mean():.1f})")
    print(f"  CTRL own drift all days: {100*drift.mean():+.3f}% (n={len(drift)}, hit {100*(drift>0).mean():.1f})")
    print(f"  OCT window {100*oct_.mean():+.3f}% median {100*oct_.median():+.3f}% vs other-month ME {100*oth.mean():+.3f}% "
          f"-> excess {100*(oct_.mean()-oth.mean()):+.3f}pp ; median excess {100*(oct_.median()-oth.median()):+.3f}pp")
    # permutation: random 1-of-12 month label vs Oct excess
    rng = np.random.default_rng(7)
    vals = allme.values
    k = len(oct_)
    perm = np.array([rng.choice(vals, k, replace=False).mean() for _ in range(5000)])
    print(f"  perm P(random {k}-anchor draw >= Oct mean) = {(perm >= oct_.mean()).mean():.3f}")

# whole month (Sep ME -> Oct ME)
rw = to_next_me(u, ME)
df = ladder(rw, "UNG whole")
print("\n=== 1b. SHORT UNG whole month (ME -> next ME), by window month ===")
print(df[["label", "n", "mean_pct", "median_pct", "hit", "t", "worst_pct", "sign_p", "rank"]].round(3).to_string(index=False))
allw = -rw.dropna()
o = allw[allw.index.month == 9]
print(f"  Oct whole {100*o.mean():+.3f}% (median {100*o.median():+.3f}) vs other months {100*allw[allw.index.month!=9].mean():+.3f}%")

# ---------------------------------------------------------------- 2. per-horizon Oct vs rest
rows = []
for h in range(1, 11):
    r = -fwd_from(u, ME, h).dropna()
    o, x = r[r.index.month == 9], r[r.index.month != 9]
    w = int((o > 0).sum())
    rows.append({"h": h, "n_oct": len(o), "oct_short_pct": 100 * o.mean(), "oct_med": 100 * o.median(),
                 "rec": f"{w}-{len(o)-w}", "sign_p": sign_test(w, len(o)),
                 "other_me_pct": 100 * x.mean(), "excess_pp": 100 * (o.mean() - x.mean()),
                 "oct_worst": 100 * o.min()})
show(rows, "2. SHORT UNG from Sep ME close, h=1..10 vs other-month ME anchors")

# ---------------------------------------------------------------- 3. NG=F front leg
ME_ng = ME.intersection(ng.index)
rows = []
for h in (3, 5, 10):
    r = fwd_from(ng, ME_ng, h).dropna()
    seam = pd.Series([seam_in(a, h, ng.index) for a in r.index], index=r.index)
    print(f"\nNG=F h={h}: windows containing an NG expiry = {int(seam.sum())} of {len(r)}")
    r = r[~seam]
    df = ladder(r, f"NG=F h={h}")
    print(df[["label", "n", "mean_pct", "median_pct", "hit", "sign_p", "rank"]].round(3).to_string(index=False))
    s = -r
    o = s[s.index.month == 9]
    rows.append({"h": h, "oct_short_NG_pct": 100 * o.mean(), "n": len(o),
                 "other_short_NG_pct": 100 * s[s.index.month != 9].mean(),
                 "excess_pp": 100 * (o.mean() - s[s.index.month != 9].mean())})
show(rows, "3. SHORT NG=F front from Sep ME (seam-free windows): Oct vs other months")

# ---------------------------------------------------------------- 4. era / midterm on UNG h=10 and whole
for h in (5, 10):
    r = -fwd_from(u, ME, h).dropna()
    o = r[r.index.month == 9]
    x = r[r.index.month != 9]
    print(f"\n4. era / cycle, SHORT UNG Sep ME h={h}:")
    for lbl, m_o, m_x in (("pre-2018", o.index < "2018-01-01", x.index < "2018-01-01"),
                          ("2018+", o.index >= "2018-01-01", x.index >= "2018-01-01"),
                          ("midterm", (o.index.year % 4) == 2, (x.index.year % 4) == 2),
                          ("non-midterm", (o.index.year % 4) != 2, (x.index.year % 4) != 2)):
        ov, xv = o[m_o], x[m_x]
        w = int((ov > 0).sum())
        print(f"   {lbl:12s} Oct {100*ov.mean():+7.3f}% ({w}-{len(ov)-w})  other ME {100*xv.mean():+7.3f}%  "
              f"excess {100*(ov.mean()-xv.mean()):+7.3f}pp")
    print("   Oct episodes:", ", ".join(f"{d.year}:{100*v:+.1f}" for d, v in o.items()))

# ---------------------------------------------------------------- 5. tail + sizing
P2 = P["UNG"]
atr = pd.Series(wilder_atr(P2["High"], P2["Low"], P2["Close"]), index=P2.index)
atrp = (atr / P2["Close"])
print(f"\n5. UNG Wilder ATR% live (09-29): {100*atrp.iloc[-1]:.2f}%")
for h in (5, 10):
    r = -fwd_from(u, ME, h).dropna()
    o = r[r.index.month == 9]
    R = o / atrp.loc[o.index]
    print(f"   h={h}: Oct short in 1-ATR units: mean {R.mean():+.2f}R, worst {R.min():+.2f}R ({R.idxmin().date()}), "
          f"best {R.max():+.2f}R; NAV at 30bp/ATR = worst {0.30*R.min():+.2f}% NAV")
    allr = r / atrp.loc[r.index]
    print(f"        all-month short worst {allr.min():+.2f}R ({allr.idxmin().date()}), worst raw {100*r.min():+.2f}%")
# max adverse excursion inside the 10-day Oct window (short)
mae = []
for a in ME[ME.month == 9]:
    p = idx.get_loc(a)
    if p + 10 >= len(idx):
        continue
    path = u.iloc[p + 1:p + 11].values / u.iloc[p] - 1.0
    mae.append((a.year, 100 * path.max(), path.max() / atrp.loc[a]))
mae = pd.DataFrame(mae, columns=["year", "max_up_pct", "max_up_atr"])
print("   Oct h=10 short: worst intra-window rally (MAE):")
print(mae.sort_values("max_up_pct", ascending=False).head(5).round(2).to_string(index=False))
print(f"   stop at 1 ATR would be hit in {(mae.max_up_atr >= 1).sum()} of {len(mae)} Octobers; at 2 ATR {(mae.max_up_atr >= 2).sum()}")
