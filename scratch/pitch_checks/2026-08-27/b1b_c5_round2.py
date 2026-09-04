"""C5 round 2 — the four pitched pairs against the sector-pair reference class,
the state-matched cell (rank AND magnitude, which is what today actually is),
the midterm split, regime over-selection and dial support.

The map chose XLK/XLV, SMH/XLV, SMH/IBB, QQQ/XLV out of a sector grid. That is
a SEARCH, so the honest multiplicity object is every sector pair under the
identical rule (max-of-K), not the four that were noticed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

ASOF = pd.Timestamp("2026-08-26")
RNG = np.random.default_rng(11)
H = 10
THR = 2.5

CLASS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",
         "SMH", "IBB", "QQQ"]
PITCHED = [("XLK", "XLV"), ("SMH", "XLV"), ("SMH", "IBB"), ("QQQ", "XLV")]

px = close_panel(sorted(set(CLASS + ["SPY"])))
px = px[px.index <= ASOF]


def vret(s, n):
    v = s.dropna()
    return (v / v.shift(n) - 1.0).reindex(s.index)


def pit_pctile(s, lookback=252):
    v = s.dropna()
    return v.rolling(lookback).apply(lambda w: (w[:-1] < w[-1]).mean() * 100.0,
                                     raw=True).reindex(s.index)


R63 = {t: vret(px[t], 63) for t in CLASS}
FWD = {t: fwd_lag(px[t], H, 1) for t in CLASS}


def pair_cell(a, b, thr=THR, extra=None, h=H):
    sp = (R63[a] - R63[b]).dropna()
    pit = pit_pctile(sp)
    m = (pit <= thr)
    if extra is not None:
        m = m & extra(sp)
    ret = (fwd_lag(px[a], h, 1) - fwd_lag(px[b], h, 1))
    valid = ret.dropna().index
    trig = pd.DatetimeIndex(px.index[m.reindex(px.index, fill_value=False)
                                     .fillna(False).values]).intersection(valid)
    if len(trig) < 3:
        return None
    epi = declusters(trig, h, valid)
    v = ret.loc[epi].values
    base = ret.loc[valid]
    return {"pair": f"{a}-{b}", "n": len(v), "mean_pct": 100 * v.mean(),
            "drift_pct": 100 * base.mean(),
            "excess_pct": 100 * (v.mean() - base.mean()),
            "se_pct": 100 * v.std(ddof=1) / np.sqrt(len(v)),
            "hit": 100 * (v > 0).mean(), "_v": v, "_e": epi, "_ret": ret,
            "_valid": valid}


# ---------------------------------------------------------------------------
print("=" * 78)
print("1. STATE-MATCHED CELL: today is BOTH a rank floor AND a big magnitude.")
print("   (registry: 'a rank is not a magnitude'. Today's spreads are -17.9pp")
print("    to -33.4pp, so the honest cell is rank-floor AND spread <= -15pp.)")
print("=" * 78)
rows = []
for a, b in PITCHED:
    sp_now = 100 * (R63[a] - R63[b]).iloc[-1]
    r_rank = pair_cell(a, b)
    r_both = pair_cell(a, b, extra=lambda sp: sp <= -0.15)
    r_mag = None
    spx = (R63[a] - R63[b]).dropna()
    mm = (spx <= -0.15)
    ret = FWD[a] - FWD[b]
    valid = ret.dropna().index
    trig = pd.DatetimeIndex(px.index[mm.reindex(px.index, fill_value=False).fillna(False).values]).intersection(valid)
    if len(trig) >= 3:
        epi = declusters(trig, H, valid)
        v = ret.loc[epi].values
        r_mag = {"n": len(v), "mean_pct": 100 * v.mean(),
                 "excess_pct": 100 * (v.mean() - ret.loc[valid].mean()),
                 "hit": 100 * (v > 0).mean()}
    print(f"\n  {a}-{b}  today {sp_now:+.2f}pp")
    for lbl, r in [("rank floor only (PIT<=2.5)", r_rank),
                   ("magnitude only (<=-15pp)", r_mag),
                   ("BOTH = today's state", r_both)]:
        if r is None:
            print(f"    {lbl:<28} n<3")
            continue
        print(f"    {lbl:<28} n={r['n']:>3}  mean {r['mean_pct']:+7.3f}%  "
              f"excess {r['excess_pct']:+7.3f}pp  hit {r['hit']:5.1f}%")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print(f"2. SECTOR-PAIR REFERENCE CLASS — identical rule, every ordered pair "
      f"of {len(CLASS)} names, h={H}")
print("=" * 78)
cells = []
for a in CLASS:
    for b in CLASS:
        if a == b:
            continue
        c = pair_cell(a, b)
        if c and c["n"] >= 5:
            cells.append(c)
RC = pd.DataFrame([{k: v for k, v in c.items() if not k.startswith("_")}
                   for c in cells]).set_index("pair").sort_values("excess_pct", ascending=False)
print(f"  K = {len(RC)} ordered pairs with n>=5 episodes")
print(RC.round(3).head(10).to_string())
print("  ...")
print(RC.round(3).tail(5).to_string())
w = 1.0 / RC.se_pct.values ** 2
th = RC.excess_pct.values
fe = float((w * th).sum() / w.sum())
Q = float((w * (th - fe) ** 2).sum())
dfree = len(RC) - 1
I2 = max(0.0, (Q - dfree) / Q) * 100 if Q > 0 else 0.0
try:
    from scipy import stats as _st
    pQ = float(_st.chi2.sf(Q, dfree))
except Exception:
    pQ = float("nan")
print(f"\n  fixed-effect common excess {fe:+.3f}pp  se {1/np.sqrt(w.sum()):.3f}pp  "
      f"z {fe*np.sqrt(w.sum()):+.2f}")
print(f"  Cochran Q {Q:.2f} on {dfree} df  p {pQ:.3f}   I^2 {I2:.1f}%")
for a, b in PITCHED:
    k = f"{a}-{b}"
    if k in RC.index:
        print(f"  {k:<9} excess {RC.loc[k,'excess_pct']:+7.3f}pp -> rank "
              f"{int((RC.excess_pct >= RC.loc[k,'excess_pct']).sum())} of {len(RC)}")

store = {c["pair"]: c for c in cells if c["pair"] in RC.index}
B = 1500
maxes = np.empty(B)
for i in range(B):
    best = -1e9
    for k in RC.index:
        c = store[k]
        f = c["_ret"].loc[c["_valid"]].values
        pos = RNG.integers(0, len(f), size=c["n"])
        best = max(best, 100 * (f[pos].mean() - f.mean()))
    maxes[i] = best
print(f"\n  null max-of-{len(RC)}: median {np.median(maxes):+.3f}pp  "
      f"p95 {np.percentile(maxes,95):+.3f}pp")
for a, b in PITCHED:
    k = f"{a}-{b}"
    if k in RC.index:
        o = RC.loc[k, "excess_pct"]
        print(f"  FAMILY-WISE p for {k:<9} ({o:+.3f}pp) = {(maxes >= o).mean():.4f}")
print(f"  best observed pair {RC.excess_pct.idxmax()} {RC.excess_pct.max():+.3f}pp"
      f" -> family-wise p {(maxes >= RC.excess_pct.max()).mean():.4f}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("3. MIDTERM SPLIT, REGIME OVER-SELECTION, DIAL SUPPORT (pitched pairs)")
print("=" * 78)
spy = px["SPY"].dropna()
sma200 = rolling_on_valid(spy, lambda x: x.rolling(200).mean())
below = (spy < sma200)
frag = pd.read_parquet(ROOT / "data/rd2_fragility.parquet")
ma10 = frag["63d"].rolling(10).mean()
for a, b in PITCHED:
    c = pair_cell(a, b)
    epi, v = c["_e"], c["_v"]
    mid = np.array([d.year % 4 == 2 for d in epi])
    print(f"\n  --- {a}-{b} (n={c['n']} episodes, h={H}) ---")
    show([summarize(v[mid], f"MIDTERM (N={int(mid.sum())})"),
          summarize(v[~mid], f"non-midterm (N={int((~mid).sum())})")], "")
    bl = below.reindex(epi).fillna(False).values
    print(f"    SPY below its 200d on {100*bl.mean():.1f}% of trigger episodes "
          f"vs {100*below.mean():.1f}% base rate  -> over-selection "
          f"{100*(bl.mean()-below.mean()):+.1f}pp")
    show([summarize(v[bl], f"SPY<200d (N={int(bl.sum())})"),
          summarize(v[~bl], f"SPY>=200d (N={int((~bl).sum())})")], "")
    dl = ma10.reindex(epi).dropna()
    print(f"    dial ma10(63d) on trigger episodes: n_with_reading {len(dl)}  "
          f"max {dl.max():.1f}  >=85: {(dl>=85).sum()}   [today 88.6]")
    # drop-best-episode
    o = np.argsort(-v)
    print(f"    drop best episode -> mean {100*np.delete(v,o[0]).mean():+.3f}%; "
          f"drop best 2 -> {100*np.delete(v,o[:2]).mean():+.3f}%  "
          f"(full {100*v.mean():+.3f}%, drift {c['drift_pct']:+.3f}%)")
