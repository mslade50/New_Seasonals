"""C10 round 1 + the alphabetical placebo (run EARLY, per the brief).

State: a megacap at a 21-day return rank <= 5 (trailing 252d) while SPY is
within 2% of its own 52-week high. META is today's instance (21d rank 4.0,
SPY -1.34% off its high).

Both directions are priced. The long thesis is idiosyncratic damage that
mean-reverts; the short thesis is early information about a leadership
change. The market-relative column (name minus SPY) is reported beside the
outright because a one-name pitch is an outright and the two answer
different questions.

THE PLACEBO THAT DECIDES IT: on the SAME trigger dates, does selecting the
most-washed names beat selecting the four ALPHABETICALLY-FIRST names of the
universe? If not, the date carries everything and the rank gate is decoration.
(The 2026-08-18 cross-sectional short died on exactly this test.)

SURVIVORSHIP: the universe is today's megacaps, so every name in it survived.
That biases the LONG direction UP. Stated, not corrected -- it makes the long
an upper bound and the short a lower bound.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

ASOF = pd.Timestamp("2026-08-18")
UNIV = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "TSLA", "JPM", "V",
        "MA", "UNH", "JNJ", "XOM", "LLY", "AVGO", "WMT", "PG", "HD", "COST",
        "ORCL", "CVX", "MRK", "PEP", "KO", "ADBE", "CRM", "BAC", "NFLX",
        "TMO", "AMD", "CSCO", "ACN", "MCD", "ABT", "LIN", "DIS", "PFE",
        "CMCSA", "INTC", "WFC", "VZ", "TXN", "QCOM", "NKE", "PM", "UPS",
        "MS", "GS", "CAT", "HON", "IBM", "BA", "SBUX", "T", "AMGN", "LOW",
        "UNP", "RTX", "BLK", "DE", "SPGI", "NOW", "GE", "MDT", "SCHW",
        "AXP", "C", "TJX", "MU", "LMT", "SYK", "BMY", "GILD", "MDLZ", "ADP",
        "CVS", "TGT", "MMM", "SO", "DUK", "CI", "MO", "REGN", "PANW"]

px = close_panel(sorted(set(UNIV + ["SPY"])))
idx = px.index
UNIV = [t for t in UNIV if t in px.columns]
print(f"universe {len(UNIV)} names; panel {px.shape}")

# ---- date-level gate: SPY within 2% of its own 52w high ----
spy = px["SPY"].dropna()
spy_hi = spy.rolling(252).max().reindex(idx)
spy_near = (px["SPY"] >= spy_hi * 0.98).fillna(False)
print(f"SPY off its 52w high on {ASOF.date()}: "
      f"{100*(px['SPY'].loc[ASOF]/spy_hi.loc[ASOF]-1):+.2f}%  "
      f"gate(<=2%) = {bool(spy_near.loc[ASOF])}")
print(f"SPY-near-high days: {int(spy_near.sum())} of {len(idx)} "
      f"({100*spy_near.mean():.1f}%)")

# ---- per-name 21d rank ----
r21 = {}
for t in UNIV:
    s = px[t].dropna()
    r21[t] = pct_rank(s, 21).reindex(idx)
r21 = pd.DataFrame(r21)
print(f"\nMETA 21d rank on {ASOF.date()}: {r21['META'].loc[ASOF]:.1f}   "
      f"names with rank<=5 today: "
      f"{sorted(r21.columns[(r21.loc[ASOF] <= 5).fillna(False)])}")

H, GAP, THRESH = 10, 21, 5


def fwd(t, h):
    return fwd_lag(px[t], h, 1)


FWD = {h: pd.DataFrame({t: fwd(t, h) for t in UNIV}) for h in (1, 2, 3, 5, 7, 10)}
SPYF = {h: fwd_lag(px["SPY"], h, 1) for h in FWD}


def triggers(thresh=THRESH, date_gate=True, h=H):
    """(name, date) pairs, declustered per name."""
    m = (r21 <= thresh).fillna(False)
    if date_gate:
        m = m & np.repeat(spy_near.values[:, None], m.shape[1], axis=1)
    out = []
    for t in UNIV:
        d = idx[m[t].values & FWD[h][t].notna().values]
        for dd in declusters(d, GAP, idx):
            out.append((t, dd))
    return out


trig = triggers()
print(f"\ntriggers (declustered {GAP}td per name): N={len(trig)}, "
      f"{len(set(t for t, _ in trig))} distinct names, "
      f"{len(set(d for _, d in trig))} distinct dates")


def stats(pairs, h=H, label=""):
    v = np.array([FWD[h][t].get(d, np.nan) for t, d in pairs], float)
    rel = np.array([FWD[h][t].get(d, np.nan) - SPYF[h].get(d, np.nan)
                    for t, d in pairs], float)
    v, rel = v[~np.isnan(v)], rel[~np.isnan(rel)]
    s = summarize(v, label)
    s["rel_spy_pct"] = round(100 * rel.mean(), 3) if len(rel) else np.nan
    s["rel_hit"] = round(100 * (rel > 0).mean(), 1) if len(rel) else np.nan
    return s


rows = [stats(trig, H, f"COND rank<={THRESH} & SPY near high (N={len(trig)})")]

# CTRL-a: same names, unconditional, over the trigger span
span_lo = min(d for _, d in trig)
names_hit = sorted(set(t for t, _ in trig))
allpairs = [(t, d) for t in names_hit for d in idx[(idx >= span_lo)
            & FWD[H][t].notna().values]]
rows.append(stats(allpairs, H, f"CTRL-a same names, all days in span (N={len(allpairs)})"))

# CTRL-b: whole universe, full history
allu = [(t, d) for t in UNIV for d in idx[FWD[H][t].notna().values]]
rows.append(stats(allu, H, f"CTRL-b whole universe, all days (N={len(allu)})"))

# CTRL-c: SAME DATES, all universe names (removes the market factor entirely)
tdates = sorted(set(d for _, d in trig))
same_day = [(t, d) for d in tdates for t in UNIV if not np.isnan(FWD[H][t].get(d, np.nan))]
rows.append(stats(same_day, H, f"CTRL-c SAME DATES, every universe name (N={len(same_day)})"))

# CTRL-d: SPY-near-high days only, whole universe (the date gate alone)
nh = idx[spy_near.values]
nh_pairs = [(t, d) for d in nh for t in UNIV if not np.isnan(FWD[H][t].get(d, np.nan))]
rows.append(stats(nh_pairs, H, f"CTRL-d SPY-near-high days, every name (N={len(nh_pairs)})"))
show(rows, f"1. conditional vs controls, h={H}, LONG the washed name")

print("\n  (short direction = negate mean_pct and rel_spy_pct above)")

# ------------------------------------------------- THE ALPHABETICAL PLACEBO
print("\n" + "=" * 78)
print("ALPHABETICAL PLACEBO -- run early, it decides the candidate")
print("On each trigger DATE take the 4 alphabetically-first universe names")
print("(no rank condition at all) and compare to the 4 most-washed qualifiers.")
print("=" * 78)
alpha4 = sorted(UNIV)[:4]
print("alphabetically-first four:", alpha4)

sel_washed, sel_alpha, sel_rand = [], [], []
rng = np.random.default_rng(42)
for d in tdates:
    q = [t for t, dd in trig if dd == d]
    q = sorted(q, key=lambda t: r21[t].get(d, 999))[:4]
    sel_washed += [(t, d) for t in q]
    sel_alpha += [(t, d) for t in alpha4 if not np.isnan(FWD[H][t].get(d, np.nan))]
    pool = [t for t in UNIV if not np.isnan(FWD[H][t].get(d, np.nan))]
    sel_rand += [(t, d) for t in rng.choice(pool, size=min(4, len(pool)),
                                            replace=False)]
show([stats(sel_washed, H, f"4 most-washed qualifiers (N={len(sel_washed)})"),
      stats(sel_alpha, H, f"4 alphabetically-first, ANY rank (N={len(sel_alpha)})"),
      stats(sel_rand, H, f"4 random names, ANY rank (N={len(sel_rand)})")],
     f"placebo: selection rule on the same {len(tdates)} dates, h={H}")

# ------------------------------------------------------ threshold gradient
print("\n" + "=" * 78)
print("MAGNITUDE GRADIENT at today's reading (META 21d rank 4.0)")
print("=" * 78)
rows = []
for lo, hi in [(0, 2), (0, 5), (2, 5), (5, 10), (10, 20), (20, 35), (35, 65),
               (65, 101)]:
    m = ((r21 >= lo) & (r21 < hi)).fillna(False)
    m = m & np.repeat(spy_near.values[:, None], m.shape[1], axis=1)
    pairs = []
    for t in UNIV:
        d = idx[m[t].values & FWD[H][t].notna().values]
        pairs += [(t, dd) for dd in declusters(d, GAP, idx)]
    if not pairs:
        rows.append({"bucket": f"[{lo},{hi})", "n": 0})
        continue
    s = stats(pairs, H, f"21d rank [{lo},{hi})")
    rows.append({"bucket": f"[{lo},{hi})", "n": s["n"],
                 "mean_pct": round(s["mean_pct"], 3), "hit": round(s["hit"], 1),
                 "t": round(s["t"], 2), "rel_spy_pct": s["rel_spy_pct"],
                 "rel_hit": s["rel_hit"]})
show(rows, "21d-rank buckets, SPY-near-high days, h=10, LONG")

# ---------------------------------------------------------- gate attribution
print("\n" + "=" * 78)
print("GATE ATTRIBUTION: does 'SPY within 2% of its 52w high' do anything?")
print("=" * 78)
rows = []
for lbl, dg in [("SPY-near-high ON (the candidate)", True),
                ("SPY gate OFF (rank<=5 any tape)", False)]:
    p = triggers(THRESH, dg)
    rows.append(stats(p, H, f"{lbl} (N={len(p)})"))
# and the complement: rank<=5 while SPY is NOT near its high
m = (r21 <= THRESH).fillna(False) & np.repeat(~spy_near.values[:, None],
                                              len(UNIV), axis=1)
p = []
for t in UNIV:
    d = idx[m[t].values & FWD[H][t].notna().values]
    p += [(t, dd) for dd in declusters(d, GAP, idx)]
rows.append(stats(p, H, f"rank<=5 while SPY NOT near high (N={len(p)})"))
show(rows, "date-gate attribution")

# --------------------------------------------------------------- era split
print("\n" + "=" * 78)
print("ERA / REGIME")
print("=" * 78)
for cut, labs in [("2018-01-01", ("pre-2018", "2018+")),
                  ("2013-01-01", ("pre-2013", "2013+"))]:
    a = [(t, d) for t, d in trig if d < pd.Timestamp(cut)]
    b = [(t, d) for t, d in trig if d >= pd.Timestamp(cut)]
    show([stats(a, H, f"{labs[0]} (N={len(a)})"), stats(b, H, f"{labs[1]} (N={len(b)})")],
         f"era split at {cut}")
mid = [(t, d) for t, d in trig if d.year % 4 == 2]
non = [(t, d) for t, d in trig if d.year % 4 != 2]
show([stats(mid, H, f"midterm years (N={len(mid)})"),
      stats(non, H, f"non-midterm (N={len(non)})")], "midterm split")

# ------------------------------------------------------------ horizon scan
print("\n" + "=" * 78)
print("HORIZON SCAN (episode-level, long the washed name)")
print("=" * 78)
rows = []
for h in (1, 2, 3, 5, 7, 10):
    p = triggers(THRESH, True, h)
    s = stats(p, h, f"h={h}")
    base = np.nanmean(FWD[h][UNIV].values)
    s["ctl_all_pct"] = round(100 * base, 3)
    s["edge_pct"] = round(s["mean_pct"] - 100 * base, 3)
    rows.append(s)
show(rows, "h=1..10")

# --------------------------------------------------- concentration + names
print("\nname concentration of the trigger set:")
cnt = pd.Series([t for t, _ in trig]).value_counts()
print(cnt.head(12).to_string())
v = np.array([FWD[H][t].get(d, np.nan) for t, d in trig], float)
print(f"\ntop-5 |contribution| episodes:")
o = np.argsort(-np.abs(v))[:5]
for i in o:
    print(f"   {trig[i][0]:6s} {trig[i][1].date()}  {100*v[i]:+7.2f}%")
yr = pd.Series(v).groupby([d.year for _, d in trig]).mean()
print("\nmean by year (%):")
print((100 * yr).round(2).to_string())
