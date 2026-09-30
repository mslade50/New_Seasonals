"""d11b round 2: where the h=10 midterm cell lives, on the TRADEABLE objects.

Horizon walk h=1..12 from the Sept QE-2 entry (signal QE-3) for: ^VIX %, ^VIX3M % (closer to
the futures a vol ETP holds), SHORT SPY, short synthetic -0.5x SVXY residual (long vol,
beta 1.98 pre-2018-03 / 1.48 after). Midterm vs non-midterm vs all Septembers, drop-best-2,
cluster note, and the h=10 month-of-year ladder on SHORT SPY in midterm years.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

px = close_panel(["^VIX", "^VIX3M", "SPY", "SVXY"])
px = px[px["SPY"].notna()]
idx = px.index
BREAK = pd.Timestamp("2018-02-28")
r_sv = px["SVXY"].pct_change(fill_method=None)
S = (1 + r_sv.where(idx >= BREAK, 0.5 * r_sv).fillna(0)).cumprod()
S[idx <= px["SVXY"].first_valid_index()] = np.nan
px["S05"] = S
MID = {2002, 2006, 2010, 2014, 2018, 2022}


def seg(col, p0, p1):
    s = px[col].values
    if p1 >= len(s) or np.isnan(s[p0]) or np.isnan(s[p1]):
        return np.nan
    return s[p1] / s[p0] - 1.0


def cell(y, m, h, off=-2):
    md = idx[(idx.year == y) & (idx.month == m)]
    if len(md) == 0 or md[-1] == idx[-1]:
        return None
    ent = idx.get_loc(md[-1]) + off
    ex = ent + h
    if ex >= len(idx):
        return None
    spy = seg("SPY", ent, ex)
    s05 = seg("S05", ent, ex)
    b = 1.48 if idx[ent] > BREAK else 1.98
    return {"year": y, "month": m, "h": h, "vix": seg("^VIX", ent, ex), "vix3m": seg("^VIX3M", ent, ex),
            "short_spy": -spy, "long_vol_res": -(s05 - b * spy) if not np.isnan(s05) else np.nan}


D = pd.DataFrame([c for y in range(2000, 2026) for m in range(1, 13) for h in range(1, 13)
                  if (c := cell(y, m, h))])
D["mid"] = D.year.isin(MID)
sep = D[D.month == 9]

rows = []
for h in range(1, 13):
    s = sep[sep.h == h]
    mid, non = s[s.mid], s[~s.mid]
    row = {"h": h}
    for c in ("vix", "vix3m", "short_spy", "long_vol_res"):
        vm = mid[c].dropna()
        row[f"{c}_mid"] = round(100 * vm.mean(), 2)
        row[f"{c}_rec"] = f"{int((vm > 0).sum())}-{int((vm <= 0).sum())}"
        row[f"{c}_non"] = round(100 * non[c].dropna().mean(), 2)
    rows.append(row)
T = pd.DataFrame(rows)
print("=== Sept QE-2 horizon walk: midterm (mean, record) vs non-midterm mean ===")
print(T.to_string(index=False))

for h in (8, 10):
    s = sep[(sep.h == h) & sep.mid].set_index("year")
    print(f"\n=== h={h} midterm per year ===")
    print((100 * s[["vix", "vix3m", "short_spy", "long_vol_res"]]).round(2).to_string())
    for c in ("vix", "short_spy"):
        v = s[c].dropna()
        d = pd.DatetimeIndex([pd.Timestamp(f"{y}-09-28") for y in v.index])
        print(f"  {c}: {cluster_note(d, v.values)}")
        keep = v.sort_values(ascending=False).iloc[2:]
        non = sep[(sep.h == h) & ~sep.mid][c].dropna()
        print(f"  {c}: drop-best-2 midterm mean {100*keep.mean():+.2f}% on {len(keep)} "
              f"vs non-midterm {100*non.mean():+.2f}% vs all Sept {100*sep[sep.h == h][c].mean():+.2f}%")
    v = s["short_spy"].dropna()
    w = int((v > 0).sum())
    print(f"  short SPY h={h} midterm record {w}-{len(v)-w}, sign p {sign_test(w, len(v)):.4f}, "
          f"per-event {100*v.mean():+.2f}%")

# month-of-year ladder, SHORT SPY h=10, midterm only, and midterm-minus-non
lad = []
for m in range(1, 13):
    b = D[(D.h == 10) & (D.month == m)]
    lad.append({"month": m, "short_spy_mid": round(100 * b[b.mid].short_spy.mean(), 2),
                "short_spy_non": round(100 * b[~b.mid].short_spy.mean(), 2),
                "lvres_mid": round(100 * b[b.mid].long_vol_res.mean(), 2)})
L = pd.DataFrame(lad)
L["diff"] = L.short_spy_mid - L.short_spy_non
print("\n=== h=10 month-of-year ladder, SHORT SPY ===")
print(L.to_string(index=False))
print(f"  Sept rank: short_spy_mid {int(L.short_spy_mid.rank(ascending=False)[8])}/12, "
      f"diff {int(L['diff'].rank(ascending=False)[8])}/12, lvres_mid {int(L.lvres_mid.rank(ascending=False)[8])}/12")
print(f"  all-month midterm short SPY h=10 mean {100*D[(D.h == 10) & D.mid].short_spy.mean():+.2f}%")
