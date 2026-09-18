"""Post-Labor-Day volatility: is the "VIX rose 22 of 26" cell TRADEABLE?

Anchor = the last session BEFORE Labor Day, 2000..2026. Tonight (Friday
2026-09-04) IS an anchor; next session is Tuesday 2026-09-08.

Forms per instrument:
  A. lag0 h1  Friday close -> Tuesday close   (the brief's number; NOT
     tradeable from here, it requires having been long into the weekend)
  B. MOO Tuesday -> MOC Tuesday               (the tradeable auction form;
     the overnight gap is reported separately, it is what B forfeits)
  C. lag1     Tuesday close -> +1/+2/+3       (the pitch convention)

Instruments: ^VIX (sanity-reproduce the brief), ^VIX3M (2006+), SPY, and the
vol ETPs actually present in master_prices. CHECKED 2026-09-06: only UVXY and
SVXY exist in the cache; VXX, VIXY and VXZ are NOT there, so they cannot be
tested here at all. UVXY and SVXY both start 2011-10-04, so their post-Labor-
Day sample is 2012..2025 (n=14), not 26.

SVXY CAVEAT: SVXY was a -1.0x inverse VIX-futures ETP until Feb 2018, when
ProShares cut it to -0.5x (after the 2018-02-05 volmageddon, in which the old
-1x form lost ~90% in a session). The pre-2018 and post-2018 series are
DIFFERENT INSTRUMENTS by leverage; the era split below is therefore not just a
regime split, it is a product split, and pre-2018 SVXY magnitudes are ~2x the
post-2018 ones by construction. UVXY had its own leverage change (+2x to
+1.5x, Feb 2018) in the same direction.

Controls printed for every form: (i) any Friday before a 3-day weekend whose
next session is a Tuesday, EX Labor Day; (ii) all Tuesdays after any 3-day
weekend (Labor Day included, the superset); (iii) all-days same form, inline.
Splits: era 2018, midterm years (year %% 4 == 2), top-two concentration,
worst/best with dates, exact sign test.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    cluster_note, era_split, fwd_lag, load_events, load_prices, sign_test,
    summarize, wilder_atr,
)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
ASOF = pd.Timestamp("2026-09-04")

WANT = ["^VIX", "^VIX3M", "^VXV", "UVXY", "SVXY", "VXX", "VIXY", "VXZ",
        "SPY", "^GSPC"]
raw = load_prices(WANT)
present = sorted(raw)
print("present in master_prices:", present)
print("ABSENT (cannot be tested):", sorted(set(WANT) - set(raw)))

ref = raw["^GSPC"]["Close"].dropna().index

# ---------------------------------------------------------------- freezes
print("\n=== FREEZE: Friday 2026-09-04 close + Wilder-14 ATR ===")
for name in ("UVXY", "SVXY", "SPY", "^VIX", "^VIX3M"):
    if name not in raw:
        continue
    d = raw[name]
    c = d["Close"].dropna()
    a = pd.Series(wilder_atr(d["High"], d["Low"], d["Close"]),
                  index=d.index).reindex(c.index)
    print(f"  {name:<7} close {c.iloc[-1]:>9.4f}  bar {c.index[-1].date()}  "
          f"Wilder-14 ATR {a.iloc[-1]:>8.4f}  ({100*a.iloc[-1]/c.iloc[-1]:.2f}% of close)")

# ---------------------------------------------------------------- anchors
anchors = []
for y in range(2000, 2027):
    sept = pd.Timestamp(y, 9, 1)
    ld = sept + pd.Timedelta(days=(7 - sept.weekday()) % 7)   # first Monday
    before = ref[ref < ld]
    if len(before):
        anchors.append(before[-1])
anchors = pd.DatetimeIndex(anchors)
anchors = anchors[anchors <= ASOF]
print(f"\nanchors (n={len(anchors)}):", [a.date().isoformat() for a in anchors])
print("tonight is an anchor:", ASOF in set(anchors))
nfp = set(load_events(["nfp"])["date"])
print("Labor Day eves that were ALSO payrolls:",
      [a.date().isoformat() for a in anchors if a in nfp])

# control sets
gap3 = pd.DatetimeIndex([ref[i] for i in range(len(ref) - 1)
                         if (ref[i + 1] - ref[i]).days >= 4
                         and ref[i + 1].weekday() == 1])
gap3 = gap3[gap3 <= ASOF]
gap3_ex = gap3.difference(anchors)
print(f"controls: 3-day-weekend Fridays -> Tuesday n={len(gap3)} "
      f"(ex Labor Day n={len(gap3_ex)})")


# ---------------------------------------------------------------- helpers
def block(name, s, dates, h=1, lag=1, notes=False):
    """Close-to-close form. lag=0 -> anchor close as entry; lag=1 -> the
    session after the anchor (i.e. Tuesday's close) as entry."""
    f = fwd_lag(s, h, lag)
    v = f.reindex(pd.DatetimeIndex(dates)).dropna()
    if len(v) == 0:
        print(f"  {name:<52} n=0")
        return v
    st = summarize(v.values)
    nup = int((v > 0).sum())
    drift = 100 * f.dropna().mean()
    allhit = 100 * (f.dropna() > 0).mean()
    print(f"  {name:<52} n={st['n']:<4} mean={st['mean_pct']:+.3f}%  "
          f"med={st['median_pct']:+.3f}%  {nup}-{len(v)-nup} ({st['hit']:.1f}%)  "
          f"t={st['t']:+.2f}  sp={sign_test(nup, len(v)):.4f}  | ALL-DAYS "
          f"{drift:+.3f}% hit {allhit:.1f}%  | worst {st['worst_pct']:+.2f}% "
          f"({v.idxmin().date()})  best {st['best_pct']:+.2f}% ({v.idxmax().date()})")
    if notes:
        print("      era:", [(e["label"], e["n"], round(e.get("mean_pct", np.nan), 3),
                              round(e.get("hit", np.nan), 1))
                             for e in era_split(v.index, v.values)])
        print("      concentration:", cluster_note(v.index, v.values))
        mid = v[[d.year % 4 == 2 for d in v.index]]
        non = v[[d.year % 4 != 2 for d in v.index]]
        print(f"      midterm n={len(mid)} {int((mid>0).sum())}-{int((mid<=0).sum())} "
              f"mean={100*mid.mean():+.3f}%  | non-midterm n={len(non)} "
              f"{int((non>0).sum())}-{int((non<=0).sum())} mean={100*non.mean():+.3f}%")
        print(f"      midterm years: {[(d.year, round(100*x,2)) for d,x in mid.items()]}")
    return v


def open_to_close(d, dates, h_close=1):
    """Entry at the OPEN of the session after the anchor; exit at the close
    h_close sessions after the anchor. Also returns the overnight gap
    (anchor close -> next open) that this form forfeits."""
    c = d["Close"].dropna()
    o = d["Open"].reindex(c.index)
    # master_prices carries 6 ZERO opens on ^VIX3M (Aug 2026); an unguarded
    # divide there returns inf and poisons the all-days control mean.
    o = o.where(o > 0)
    p = {x: i for i, x in enumerate(c.index)}
    out, gap = {}, {}
    for a in pd.DatetimeIndex(dates):
        if a in p and p[a] + h_close < len(c):
            op = o.iloc[p[a] + 1]
            if not np.isfinite(op):
                continue
            out[a] = c.iloc[p[a] + h_close] / op - 1
            gap[a] = op / c.iloc[p[a]] - 1
    return pd.Series(out), pd.Series(gap)


def oc_block(name, d, dates, h_close=1, notes=False):
    v, g = open_to_close(d, dates, h_close)
    if len(v) == 0:
        print(f"  {name:<52} n=0")
        return v, g
    st = summarize(v.values)
    nup = int((v > 0).sum())
    allv, allg = open_to_close(d, d["Close"].dropna().index[252:-6], h_close)
    print(f"  {name:<52} n={st['n']:<4} mean={st['mean_pct']:+.3f}%  "
          f"med={st['median_pct']:+.3f}%  {nup}-{len(v)-nup} ({st['hit']:.1f}%)  "
          f"t={st['t']:+.2f}  sp={sign_test(nup, len(v)):.4f}  | ALL-DAYS same form "
          f"{100*allv.mean():+.3f}% hit {100*(allv>0).mean():.1f}%  | GAP forfeited "
          f"{100*g.mean():+.3f}% med {100*g.median():+.3f}% ({int((g>0).sum())}-"
          f"{int((g<=0).sum())})  | worst {st['worst_pct']:+.2f}% ({v.idxmin().date()})"
          f"  best {st['best_pct']:+.2f}%")
    if notes:
        print("      era:", [(e["label"], e["n"], round(e.get("mean_pct", np.nan), 3),
                              round(e.get("hit", np.nan), 1))
                             for e in era_split(v.index, v.values)])
        print("      concentration:", cluster_note(v.index, v.values))
        mid = v[[d_.year % 4 == 2 for d_ in v.index]]
        non = v[[d_.year % 4 != 2 for d_ in v.index]]
        print(f"      midterm n={len(mid)} {int((mid>0).sum())}-{int((mid<=0).sum())} "
              f"mean={100*mid.mean():+.3f}%  | non-midterm n={len(non)} "
              f"{int((non>0).sum())}-{int((non<=0).sum())} mean={100*non.mean():+.3f}%")
    return v, g


# ---------------------------------------------------------------- per instrument
ORDER = [t for t in ("^VIX", "^VIX3M", "UVXY", "SVXY", "SPY", "^GSPC") if t in raw]
lag0 = {}
ocs = {}
gaps = {}
for name in ORDER:
    d = raw[name]
    c = d["Close"].dropna()
    print(f"\n================ {name}  (bars {c.index[0].date()} .. "
          f"{c.index[-1].date()}) ================")
    lag0[name] = block("A. lag0 h1  Fri close -> Tue close", c, anchors, 1, 0,
                       notes=True)
    block("A. CTRL 3-day-wknd Fri -> Tue close, ex LD", c, gap3_ex, 1, 0)
    block("A. CTRL all 3-day-wknd Fri -> Tue (incl LD)", c, gap3, 1, 0)

    has_open = d["Open"].reindex(c.index).notna().any()
    if has_open:
        v, g = oc_block("B. MOO Tue -> MOC Tue", d, anchors, notes=True)
        ocs[name], gaps[name] = v, g
        oc_block("B. CTRL 3-day-wknd Tue MOO->MOC, ex LD", d, gap3_ex)
        oc_block("B. CTRL all post-3-day-wknd Tue MOO->MOC", d, gap3)
    else:
        print("  B. MOO->MOC: no Open series for this instrument (index), skipped")

    for h in (1, 2, 3):
        block(f"C. lag1 Tue close -> +{h}", c, anchors, h, 1, notes=(h == 3))
        block(f"C. CTRL 3-day-wknd Tue close -> +{h}, ex LD", c, gap3_ex, h, 1)

# ---------------------------------------------------------------- ETP year tables
print("\n\n================ FORM B PER-YEAR TABLE (ETPs + SPY) ================")
print("gap = Fri close -> Tue open (forfeited by MOO). oc = Tue open -> Tue close.")
for name in [t for t in ("UVXY", "SVXY", "SPY") if t in ocs]:
    v, g = ocs[name], gaps[name]
    a = lag0[name]
    rows = []
    for d_ in v.index:
        rows.append({"year": d_.year, "anchor": str(d_.date()),
                     "gap_pct": round(100 * g[d_], 2),
                     "open_to_close_pct": round(100 * v[d_], 2),
                     "fri_close_to_tue_close_pct": (round(100 * a[d_], 2)
                                                    if d_ in a.index else np.nan)})
    df = pd.DataFrame(rows)
    print(f"\n--- {name} ---")
    print(df.to_string(index=False))
    tot_full = df["fri_close_to_tue_close_pct"].sum()
    share = (f"{100*df['gap_pct'].sum()/tot_full:.0f}%"
             if abs(tot_full) > 1.0 else "n/a (full move sums to ~0)")
    print(f"  gap total {df['gap_pct'].sum():+.2f}pp | oc total "
          f"{df['open_to_close_pct'].sum():+.2f}pp | full-move total "
          f"{tot_full:+.2f}pp | gap share of full move: {share}")

# ------------------------------------------------- VIX-up years vs ETP oc leg
print("\n\n================ ON THE VIX-UP YEARS, DID THE ETP OPEN->CLOSE LEG "
      "ALSO WORK? ================")
vix_a = lag0["^VIX"]
up_years = vix_a[vix_a > 0]
dn_years = vix_a[vix_a <= 0]
print(f"^VIX form A record: {len(up_years)}-{len(dn_years)} "
      f"({[d.year for d in up_years.index]} up)")
for name in [t for t in ("UVXY", "SVXY") if t in ocs]:
    v, g = ocs[name], gaps[name]
    common = v.index.intersection(up_years.index)
    if len(common) == 0:
        continue
    sub = v.loc[common]
    subg = g.loc[common]
    right = (sub > 0) if name == "UVXY" else (sub < 0)
    rightg = (subg > 0) if name == "UVXY" else (subg < 0)
    want = "up" if name == "UVXY" else "down"
    print(f"\n  {name}: on the {len(common)} VIX-up anchors it also covers, the "
          f"OPEN->CLOSE leg went {want} {int(right.sum())}-{int((~right).sum())} "
          f"(sign p {sign_test(int(right.sum()), len(right)):.4f}), "
          f"mean {100*sub.mean():+.3f}%, median {100*sub.median():+.3f}%")
    print(f"    the GAP went {want} {int(rightg.sum())}-{int((~rightg).sum())}, "
          f"mean {100*subg.mean():+.3f}%")
    print(f"    per-anchor: "
          f"{[(d.year, round(100*subg[d],2), round(100*sub[d],2)) for d in common]}"
          "   (year, gap%, open->close%)")
    # how much of the ETP's full Fri->Tue move is in the gap on those years
    full = lag0[name].reindex(common).dropna()
    if len(full):
        print(f"    full Fri->Tue mean {100*full.mean():+.3f}% vs open->close "
              f"{100*sub.reindex(full.index).mean():+.3f}%  -> the auction form "
              f"keeps {100*sub.reindex(full.index).mean()/(100*full.mean()) * 100:.0f}%"
              " of it")

print("\nDONE.")
