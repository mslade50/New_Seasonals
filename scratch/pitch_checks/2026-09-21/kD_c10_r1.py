"""c10 round 1 (2026-09-21): long SPY from the late-September close into a
MIDTERM election (entry = election-31 sessions, which is today's 09-21 close
against the 2026-11-03 election; pre-election close = election-1 -> h=30).

Mechanism (pre-specified): political-uncertainty premium resolves into the vote
+ the midterm Q3/early-Q4 low. Falsifications run here:
  (a) same window in presidential years (also an election) and odd years (none)
  (b) SPY's own unconditional 30-session drift (full history, late-Sep all years)
  (c) offset ladder on the entry (E-40..E-20, fixed h=30 and exit-at-E-1) and
      h=5/10/21 from today's anchor
  (d) episode table with SPY vs its 200d at the signal close and the dial
  (e) post-anchor rows (entry at the pre-election close, h=5/10)
  (f) month x cycle grid: tdom-14 entry, h=30, 48 cells, rank of Sep x midterm
Returns: entry close -> exit close on adjusted SPY (total return), ^GSPC as a
price-only contrast row. Fractions in, percent out.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
BAR = pd.Timestamp("2026-09-18")

px = close_panel(["SPY", "^GSPC"])
px = px.loc[:BAR].dropna()
idx = px.index
spy = px["SPY"]
gspc = px["^GSPC"]
sma200 = spy.rolling(200).mean()
dist200 = spy / sma200 - 1.0
frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
dial = frag["63d"].rolling(10).mean()


def election_day(y: int) -> pd.Timestamp:
    d = pd.Timestamp(y, 11, 1)
    first_mon = d + pd.Timedelta(days=(0 - d.weekday()) % 7)
    return first_mon + pd.Timedelta(days=1)


# sanity: computed dates must match macro_events.csv on even years
ev = set(load_events(["election"])["date"])
for y in range(2000, 2027, 2):
    assert election_day(y) in ev, y

YEARS = list(range(2000, 2026))
EPOS = {}
for y in YEARS:
    e = election_day(y)
    p = int(idx.searchsorted(e))
    assert idx[p] == e, (y, e, idx[p])
    EPOS[y] = p


def cyc(y: int) -> str:
    return {2: "midterm", 0: "presidential", 1: "post-elec(odd)", 3: "pre-elec(odd)"}[y % 4]


def wret(s: pd.Series, p0: int, p1: int) -> float:
    if p0 < 0 or p1 >= len(s):
        return np.nan
    return s.iloc[p1] / s.iloc[p0] - 1.0


ENTRY_OFF = -31   # 2026: 09-21 close is election-31
# verify the live offset against a business-day calendar (no NYSE holidays 09-21..11-03)
fut = pd.bdate_range("2026-09-22", "2026-11-03")
print(f"live check: sessions after 09-21 through 11-03 = {len(fut)} (election at +{len(fut)}), "
      f"pre-election close 11-02 at +{len(fut)-1}")

# ---------------------------------------------------------------- core table
rows = []
for y in YEARS:
    E = EPOS[y]
    p0 = E + ENTRY_OFF
    sig = p0 - 1
    r = {
        "year": y, "cycle": cyc(y), "entry": idx[p0].date(),
        "to_E-1": wret(spy, p0, E - 1),
        "h5": wret(spy, p0, p0 + 5), "h10": wret(spy, p0, p0 + 10),
        "h21": wret(spy, p0, p0 + 21),
        "gspc_to_E-1": wret(gspc, p0, E - 1),
        "post_E-1_h5": wret(spy, E - 1, E + 4), "post_E-1_h10": wret(spy, E - 1, E + 9),
        "post_E_h10": wret(spy, E, E + 10),
        "vs200_sig": dist200.iloc[sig],
        "dial_sig": dial.get(idx[sig], np.nan),
        "maxdd_to_E-1": (spy.iloc[p0:E].min() / spy.iloc[p0] - 1.0),
    }
    rows.append(r)
T = pd.DataFrame(rows)
fmt = T.copy()
for c in ["to_E-1", "h5", "h10", "h21", "gspc_to_E-1", "post_E-1_h5", "post_E-1_h10",
          "post_E_h10", "vs200_sig", "maxdd_to_E-1"]:
    fmt[c] = (100 * fmt[c]).round(2)
fmt["dial_sig"] = fmt["dial_sig"].round(1)
print("\n=== every year, entry = election-31 close (SPY adj, %) ===")
print(fmt.to_string(index=False))


def block(col: str, title: str) -> None:
    out = []
    for c in ["midterm", "presidential", "post-elec(odd)", "pre-elec(odd)"]:
        v = T.loc[T.cycle == c, col].values
        s = summarize(v, c)
        w = int((v > 0).sum())
        s["rec"] = f"{w}-{len(v)-w}"
        s["sign_p"] = round(sign_test(w, len(v)), 4)
        out.append(s)
    v = T.loc[T.cycle.isin(["post-elec(odd)", "pre-elec(odd)"]), col].values
    s = summarize(v, "odd years (no election)")
    w = int((v > 0).sum()); s["rec"] = f"{w}-{len(v)-w}"; s["sign_p"] = round(sign_test(w, len(v)), 4)
    out.append(s)
    v = T.loc[T.cycle.isin(["midterm", "presidential"]), col].values
    s = summarize(v, "all election years")
    w = int((v > 0).sum()); s["rec"] = f"{w}-{len(v)-w}"; s["sign_p"] = round(sign_test(w, len(v)), 4)
    out.append(s)
    s = summarize(T[col].values, "all years")
    w = int((T[col] > 0).sum()); s["rec"] = f"{w}-{len(T)-w}"
    out.append(s)
    show(out, title)


block("to_E-1", "(a) entry E-31 -> pre-election close (h=30), by cycle")
block("h5", "h=5 from the anchor, by cycle")
block("h10", "h=10 from the anchor, by cycle")
block("h21", "h=21 from the anchor, by cycle")
block("post_E-1_h5", "(e) POST: pre-election close -> +5, by cycle")
block("post_E-1_h10", "(e) POST: pre-election close -> +10, by cycle")

# ---------------------------------------------------------------- (b) own drift
r30 = spy.shift(-30) / spy - 1.0
base_all = r30.dropna()
print("\n=== (b) own drift, SPY 30-session windows ===")
show([summarize(base_all.values, "all days 2000-2026"),
      summarize(base_all[base_all.index >= "2002-01-01"].values, "all days 2002+"),
      summarize(base_all[(base_all.index.month == 9) & (base_all.index.day >= 15)
                         & (base_all.index.day <= 25)].values,
                "entries Sep 15-25, all years (day-level)")])
mid = T.loc[T.cycle == "midterm", "to_E-1"].values
print(f"  midterm per-event edge vs all-days drift: {100*(mid.mean()-base_all.mean()):+.3f}pp; "
      f"base up-rate {100*(base_all>0).mean():.1f}% -> sign p vs base "
      f"{sign_test(int((mid>0).sum()), len(mid), float((base_all>0).mean())):.4f}")
print(f"  concentration: {cluster_note(pd.DatetimeIndex([idx[EPOS[y]+ENTRY_OFF] for y in T.loc[T.cycle=='midterm','year']]), mid)}")
m = T.cycle == "midterm"
print(f"  drop-best midterm mean: {100*np.sort(mid)[:-1].mean():+.3f}% ; drop-best-two {100*np.sort(mid)[:-2].mean():+.3f}%")
print(f"  era: pre-2018 midterms {100*T.loc[m & (T.year<2018),'to_E-1'].mean():+.3f}% "
      f"(N={int((m & (T.year<2018)).sum())}); 2018+ {100*T.loc[m & (T.year>=2018),'to_E-1'].mean():+.3f}% "
      f"(N={int((m & (T.year>=2018)).sum())})")
ab = m & (T.vs200_sig > 0)
be = m & (T.vs200_sig <= 0)
print(f"  midterm entered ABOVE 200d: {100*T.loc[ab,'to_E-1'].mean():+.3f}% on "
      f"{int((T.loc[ab,'to_E-1']>0).sum())}-{int((T.loc[ab,'to_E-1']<=0).sum())}; "
      f"BELOW 200d: {100*T.loc[be,'to_E-1'].mean():+.3f}% on "
      f"{int((T.loc[be,'to_E-1']>0).sum())}-{int((T.loc[be,'to_E-1']<=0).sum())}")
allab = T.vs200_sig > 0
print(f"  ALL years entered above 200d: {100*T.loc[allab,'to_E-1'].mean():+.3f}% "
      f"(N={int(allab.sum())}, hit {100*(T.loc[allab,'to_E-1']>0).mean():.0f}%); below: "
      f"{100*T.loc[~allab,'to_E-1'].mean():+.3f}% (N={int((~allab).sum())})")

# ---------------------------------------------------------------- (c) ladder
print("\n=== (c) offset ladder, MIDTERM years: entry at E+k ===")
lad = []
for k in range(-45, -14):
    fixed = [wret(spy, EPOS[y] + k, EPOS[y] + k + 30) for y in YEARS if y % 4 == 2]
    toE = [wret(spy, EPOS[y] + k, EPOS[y] - 1) for y in YEARS if y % 4 == 2]
    pres = [wret(spy, EPOS[y] + k, EPOS[y] + k + 30) for y in YEARS if y % 4 == 0]
    odd = [wret(spy, EPOS[y] + k, EPOS[y] + k + 30) for y in YEARS if y % 2 == 1]
    lad.append({"k": k, "mid_h30": 100*np.mean(fixed),
                "mid_h30_rec": f"{sum(v>0 for v in fixed)}-{sum(v<=0 for v in fixed)}",
                "mid_toE-1": 100*np.mean(toE),
                "pres_h30": 100*np.mean(pres), "odd_h30": 100*np.mean(odd)})
L = pd.DataFrame(lad).round(3)
print(L.to_string(index=False))
sub = L[(L.k >= -40) & (L.k <= -20)]
live = L.loc[L.k == ENTRY_OFF].iloc[0]
rk1 = int((sub.mid_h30 > live.mid_h30).sum()) + 1
rk2 = int((sub["mid_toE-1"] > live["mid_toE-1"]).sum()) + 1
print(f"  live k={ENTRY_OFF}: fixed-h30 ranks {rk1} of {len(sub)} (k in -40..-20); "
      f"exit-at-E-1 ranks {rk2} of {len(sub)} (shorter holds for later k)")
print(f"  midterm minus odd-years at the live rung: {live.mid_h30 - live.odd_h30:+.3f}pp; "
      f"minus presidential: {live.mid_h30 - live.pres_h30:+.3f}pp")

# ---------------------------------------------------------------- (f) month x cycle grid
print("\n=== (f) month x cycle grid: entry at the 14th session of month m, h=30 ===")
grid = []
ser = pd.Series(range(len(idx)), index=idx)
for mth in range(1, 13):
    for c in range(4):
        vals = []
        for y in range(2000, 2026):
            if y % 4 != c:
                continue
            sess = idx[(idx.year == y) & (idx.month == mth)]
            if len(sess) < 14:
                continue
            p0 = int(ser[sess[13]])
            vals.append(wret(spy, p0, p0 + 30))
        vals = [v for v in vals if not np.isnan(v)]
        grid.append({"month": mth, "cyc": c, "n": len(vals), "mean": 100*np.mean(vals),
                     "hit": 100*np.mean([v > 0 for v in vals])})
G = pd.DataFrame(grid)
cell = G[(G.month == 9) & (G.cyc == 2)].iloc[0]
rk = int((G["mean"] > cell["mean"]).sum()) + 1
print(f"  Sep x midterm (tdom-14, h=30): {cell['mean']:+.3f}% hit {cell['hit']:.0f}% n={int(cell['n'])}; "
      f"ranks {rk} of 48 from the top")
print(f"  grid median {G['mean'].median():+.3f}%, 48-cell mean {G['mean'].mean():+.3f}%")
top = G.sort_values("mean", ascending=False).head(8)
print(top.round(3).to_string(index=False))
# permutation: shuffle cycle labels within month across years -> P(Sep-midterm >= observed)
rng = np.random.default_rng(42)
sep_vals = []
for y in range(2000, 2026):
    sess = idx[(idx.year == y) & (idx.month == 9)]
    p0 = int(ser[sess[13]])
    sep_vals.append((y, wret(spy, p0, p0 + 30)))
sv = np.array([v for _, v in sep_vals])
lab = np.array([y % 4 == 2 for y, _ in sep_vals])
obs = sv[lab].mean()
perm = np.array([sv[rng.permutation(lab)].mean() for _ in range(20000)])
print(f"  within-September label permutation: P(random 6-7 Septembers >= midterm mean) = "
      f"{(perm >= obs).mean():.3f}")
# how many of the 48 cells would a max-of-48 search find at this level
print(f"  cells >= Sep-midterm: {int((G['mean'] >= cell['mean']).sum())} of 48")

# ---------------------------------------------------------------- live state
sig = idx[-1]
print(f"\nlive: signal close {sig.date()} SPY {spy.iloc[-1]:.2f}, vs 200d {100*dist200.iloc[-1]:+.2f}%, "
      f"dial ma10(63d) {dial.iloc[-1]:.1f} (append-only PIT vintage since 2026-07-02; "
      f"2016-2024 rows are the recompute vintage)")
