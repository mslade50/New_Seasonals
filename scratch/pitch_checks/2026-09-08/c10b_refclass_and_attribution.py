"""C10 round 2 -- is the FOOD group special, or is this generic short-term
reversal wearing a staples label?

Round 1 left exactly one survivor: V1, the equal-weight basket of the FLUSHED
members, held outright (+0.931% at h=5 over 106 gated episodes, 69-37,
edge +0.671pp, bootP 0.007). The pair against XLP was 56-50 and is already
dead. This script attacks the survivor on the four things that can kill it:

  A  LIVE BAND. Today is n=5 flushed and XLP r5 27.4. Quote those bands on
     their own, not inside "n>=3" and "XLP r5>=25".
  B  CONCENTRATION. drop-best-3, ex-2020, ex-2002, ex-GFC.
  C  REFERENCE CLASS. The identical rule on nine sector subgroups built by a
     deterministic rule (10 longest-history names per sector, gated by that
     sector's own SPDR), plus 500 RANDOM 10-name subsets of the same
     Consumer Defensive pool. NULL-3 max-of-K permutation, the 2026-08-21
     estimator.
  D  MECHANISM. On the SAME trigger days, the equal-weight forward return of
     the k worst-5d-rank names in a broad NON-staples universe. If that pays
     the same, there is no staples content and the cell is the short-term
     reversal factor with a sector-intact gate bolted on.
  E  GATE SWAP. XLP-intact vs SPY-intact.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

warnings.filterwarnings("ignore")

FOOD = ["CPB", "GIS", "TSN", "HRL", "SYY", "CAG", "SJM", "MKC", "KR", "HSY"]
RANK_MAX, N_MIN, XLP_FLOOR, H = 10.0, 3, 25.0, 5
SPDR = {"Industrials": "XLI", "Technology": "XLK", "Financial Services": "XLF",
        "Consumer Cyclical": "XLY", "Basic Materials": "XLB",
        "Consumer Defensive": "XLP", "Energy": "XLE", "Healthcare": "XLV",
        "Utilities": "XLU"}

meta = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                       columns=["ticker", "date"])
meta["date"] = pd.to_datetime(meta["date"])
agg = meta.groupby("ticker")["date"].agg(["min", "max", "count"])
sm = pd.read_parquet(ROOT / "data" / "sector_map.parquet").set_index("ticker")
agg = agg.join(sm)
pool = agg[(agg["min"] <= "2000-06-01") & (agg["max"] >= "2026-09-04")]

sector_groups, cd_pool = {}, []
for sec, etf in SPDR.items():
    names = pool[pool["sector"] == sec].sort_values(
        ["count"], ascending=False).index.tolist()
    names = sorted(names)                       # deterministic, not cap-picked
    if sec == "Consumer Defensive":
        cd_pool = [n for n in names]
    if len(names) >= 10:
        sector_groups[sec] = (names[:10], etf)

ALL = sorted(set(FOOD + [t for g, _ in sector_groups.values() for t in g]
                 + cd_pool + list(SPDR.values()) + ["SPY"]))
BROAD = sorted(set(t for sec, (g, _) in sector_groups.items()
                   if sec != "Consumer Defensive" for t in g))
print(f"loading {len(ALL)} tickers ...")
PXD = load_prices(ALL)
IDX = PXD["SPY"].index
C = pd.DataFrame({t: PXD[t]["Close"] for t in PXD}).reindex(IDX)
R5 = pd.DataFrame({t: pct_rank(C[t], 5) for t in C.columns})
F5 = pd.DataFrame({t: fwd_lag(C[t], H, 1) for t in C.columns})

xlp_gate = (R5["XLP"] >= XLP_FLOOR).fillna(False)
spy_gate = (R5["SPY"] >= XLP_FLOOR).fillna(False)


def flush(group, gate, rank_max=RANK_MAX, n_min=N_MIN, fwd=None, ret_mask=False):
    fwd = F5 if fwd is None else fwd
    M = (R5[group] <= rank_max).fillna(False) & C[group].notna()
    v = fwd[group].where(M).mean(axis=1)
    trig = IDX[((M.sum(axis=1) >= n_min) & gate).values]
    if ret_mask:
        return M, v, trig
    return v, trig


def stats(ret, trig, h=H, label="", quiet=False):
    valid = ret.dropna().index
    t = pd.DatetimeIndex(trig).intersection(valid)
    if len(t) < 3:
        if not quiet:
            print(f"  {label}: {len(t)} days -- too few")
        return None
    epi = declusters(t, h, valid)
    ep = ret.loc[epi].values
    base = float(ret.loc[valid].mean())
    w = int((ep > 0).sum())
    d = {"label": label, "n_days": len(t), "n": len(epi),
         "mean_pct": 100 * ep.mean(), "ctl": 100 * base,
         "excess_pp": 100 * (ep.mean() - base), "hit": 100 * (ep > 0).mean(),
         "rec": f"{w}-{len(epi)-w}", "sign_p": sign_test(w, len(epi)),
         "t": summarize(ep)["t"], "worst": 100 * ep.min(),
         "epi": epi, "ep": ep}
    if not quiet:
        print(f"  {label:<44s} Nd {d['n_days']:4d} Ne {d['n']:3d} mean "
              f"{d['mean_pct']:+7.3f}%  ctl {d['ctl']:+6.3f}%  excess "
              f"{d['excess_pp']:+6.3f}pp  t {d['t']:+5.2f}  rec {d['rec']:>7s}  "
              f"p {d['sign_p']:.4f}  worst {d['worst']:+7.2f}%")
    return d


M_food, v_food, trig_food = flush(FOOD, xlp_gate, ret_mask=True)
cnt_food = M_food.sum(axis=1)

print("=" * 78)
print("A. THE LIVE BANDS, QUOTED ON THEIR OWN (today: n=5 flushed, XLP r5 27.4)")
print("=" * 78)
for lo, hi, lbl in ((3, 3, "n == 3"), (4, 4, "n == 4"), (5, 10, "n >= 5"),
                    (3, 10, "n >= 3 (pitched)")):
    m = (cnt_food >= lo) & (cnt_food <= hi) & xlp_gate
    stats(v_food, IDX[m.values], label=f"count band {lbl}")
print()
for lo, hi in ((0, 15), (15, 25), (25, 35), (35, 50), (50, 101)):
    m = (cnt_food >= N_MIN) & (R5["XLP"] >= lo).fillna(False) & (R5["XLP"] < hi)
    stats(v_food, IDX[m.values], label=f"XLP r5 in [{lo},{hi})")
print("\n  the joint live cell (n>=5 AND XLP r5 in [25,35)):")
m = (cnt_food >= 5) & (R5["XLP"] >= 25).fillna(False) & (R5["XLP"] < 35)
d_live = stats(v_food, IDX[m.values], label="LIVE CELL n>=5 & XLP r5 25-35")
if d_live:
    print("     episodes: " + ", ".join(str(x.date()) for x in d_live["epi"]))

print("\n" + "=" * 78)
print("B. CONCENTRATION (pitched cell n>=3, XLP r5>=25, h=5)")
print("=" * 78)
d = stats(v_food, trig_food, label="pitched cell")
ep, epi = d["ep"], d["epi"]
order = np.argsort(-ep)
for k in (1, 3, 5):
    keep = np.ones(len(ep), bool)
    keep[order[:k]] = False
    w = int((ep[keep] > 0).sum())
    print(f"  drop-best-{k}: mean {100*ep[keep].mean():+.3f}%  excess "
          f"{100*ep[keep].mean()-d['ctl']:+.3f}pp  rec {w}-{keep.sum()-w}  "
          f"cost mult {100*100*ep[keep].mean()/5.0:.1f}x (5 bps weighted)")
for drop, lbl in (({2020}, "ex-2020"), ({2002}, "ex-2002"),
                  ({2008, 2009}, "ex-GFC"), ({2020, 2002, 2008, 2009}, "ex-all-crises")):
    keep = ~np.isin(epi.year, list(drop))
    w = int((ep[keep] > 0).sum())
    print(f"  {lbl:14s}: n {keep.sum():3d}  mean {100*ep[keep].mean():+.3f}%  "
          f"rec {w}-{keep.sum()-w}  sign p {sign_test(w, int(keep.sum())):.4f}")
print(f"  {cluster_note(epi, ep, k=3)}")

print("\n" + "=" * 78)
print("C. REFERENCE CLASS -- the identical rule on nine sector subgroups")
print("=" * 78)
book, rows = {}, []
for sec, (g, etf) in sector_groups.items():
    gate = (R5[etf] >= XLP_FLOOR).fillna(False)
    v, tr = flush(g, gate)
    s = stats(v, tr, label=f"{sec[:22]:22s} [{etf}]", quiet=True)
    if s is None or s["n"] < 10:
        continue
    book[sec] = 100 * s["ep"] - s["ctl"]
    rows.append({"sector": sec, "etf": etf, "n": s["n"],
                 "mean_pct": round(s["mean_pct"], 3), "ctl": round(s["ctl"], 3),
                 "excess_pp": round(s["excess_pp"], 3), "rec": s["rec"],
                 "sign_p": round(s["sign_p"], 4),
                 "members": ",".join(g[:4]) + "..."})
book["FOOD"] = 100 * ep - d["ctl"]
rows.append({"sector": "FOOD (focus)", "etf": "XLP", "n": d["n"],
             "mean_pct": round(d["mean_pct"], 3), "ctl": round(d["ctl"], 3),
             "excess_pp": round(d["excess_pp"], 3), "rec": d["rec"],
             "sign_p": round(d["sign_p"], 4), "members": ",".join(FOOD[:4]) + "..."})
print(pd.DataFrame(rows).to_string(index=False))

rng = np.random.default_rng(42)
NB = 20000


def null3(bk, focus):
    names = list(bk)
    obs = float(bk[focus].mean())
    means = {c: float(bk[c].mean()) for c in names}
    cm = float(np.mean(list(means.values())))
    cen = {c: bk[c] - means[c] + cm for c in names}
    ns = {c: len(bk[c]) for c in names}
    mx = np.array([max(rng.choice(cen[c], size=ns[c], replace=True).mean()
                       for c in names) for _ in range(NB)])
    p = float((mx >= obs).mean())
    rank = 1 + sum(1 for c in names if means[c] > obs)
    other = np.concatenate([bk[c] for c in names if c != focus])
    a = bk[focus]
    se = np.sqrt(a.var(ddof=1) / len(a) + other.var(ddof=1) / len(other))
    ses = {c: bk[c].std(ddof=1) / np.sqrt(len(bk[c])) for c in names}
    wts = {c: 1 / ses[c] ** 2 for c in names}
    fe = sum(wts[c] * means[c] for c in names) / sum(wts.values())
    Q = sum(wts[c] * (means[c] - fe) ** 2 for c in names)
    dfree = len(names) - 1
    I2 = max(0.0, 100 * (Q - dfree) / Q) if Q > 0 else 0.0
    print(f"\n  class K={len(names)}; positive {sum(1 for c in names if means[c]>0)}; "
          f"equal-weight mean {cm:+.3f}pp; median {np.median(list(means.values())):+.3f}pp")
    print(f"  FOOD observed {obs:+.3f}pp -> rank {rank} of {len(names)}")
    print(f"  null max-of-{len(names)}: median {np.median(mx):+.3f}  "
          f"95th {np.percentile(mx, 95):+.3f}")
    print(f"  **P(max-of-{len(names)} >= FOOD) = {p:.4f}**")
    print(f"  Welch FOOD vs pooled others {a.mean()-other.mean():+.3f}pp "
          f"t {(a.mean()-other.mean())/se:+.2f}   fixed-effect common excess "
          f"{fe:+.3f}pp   Cochran Q {Q:.2f} on {dfree} df   I-squared {I2:.1f}%")
    return p


p_sector = null3(book, "FOOD")

print("\n" + "=" * 78)
print(f"C2. 500 RANDOM 10-NAME SUBSETS of the {len(cd_pool)}-name Consumer "
      f"Defensive pool, same XLP gate")
print("=" * 78)
print(f"  pool: {', '.join(cd_pool)}")
draws = []
for i in range(500):
    g = list(rng.choice(cd_pool, size=10, replace=False))
    v, tr = flush(g, xlp_gate)
    s = stats(v, tr, label="", quiet=True)
    if s and s["n"] >= 10:
        draws.append(s["excess_pp"])
draws = np.array(draws)
print(f"  {len(draws)} valid draws; excess distribution: mean {draws.mean():+.3f}pp "
      f"median {np.median(draws):+.3f}pp  sd {draws.std(ddof=1):.3f}")
print(f"  percentiles 5/25/50/75/95: "
      + " / ".join(f"{np.percentile(draws, q):+.3f}" for q in (5, 25, 50, 75, 95)))
print(f"  FOOD excess {d['excess_pp']:+.3f}pp -> "
      f"**P(random staples 10-subset >= FOOD) = {(draws >= d['excess_pp']).mean():.4f}**")

print("\n" + "=" * 78)
print("D. MECHANISM -- generic short-term reversal on the SAME trigger days")
print(f"   (broad NON-staples universe, {len(BROAD)} names)")
print("=" * 78)
Mb = (R5[BROAD] <= RANK_MAX).fillna(False) & C[BROAD].notna()
vb = F5[BROAD].where(Mb).mean(axis=1)
print("  the same trigger dates, but the basket is the market's own r5<=10 names:")
sb = stats(vb, trig_food, label="broad r5<=10 basket on FOOD trigger days")
print(f"  FOOD basket on the same days: {d['mean_pct']:+.3f}%  "
      f"broad basket: {sb['mean_pct']:+.3f}%  "
      f"difference {d['mean_pct']-sb['mean_pct']:+.3f}pp")
# paired, episode by episode
common = d["epi"].intersection(sb["epi"])
pa = v_food.loc[common].values
pb = vb.loc[common].values
diff = pa - pb
w = int((diff > 0).sum())
print(f"  PAIRED on {len(common)} common episodes: FOOD {100*pa.mean():+.3f}% "
      f"vs broad {100*pb.mean():+.3f}%  diff {100*diff.mean():+.3f}pp "
      f"t {summarize(diff)['t']:+.2f}  rec {w}-{len(diff)-w} "
      f"sign p {sign_test(w, len(diff)):.4f}")
print("\n  and the broad universe's OWN cell (its own >=3-flushed trigger, XLP gate off):")
vb2, tb2 = flush(BROAD, pd.Series(True, index=IDX), n_min=3)
stats(vb2, tb2, label="broad basket, own trigger, no gate")

print("\n" + "=" * 78)
print("E. GATE SWAP -- is it XLP that must be intact, or just SPY?")
print("=" * 78)
stats(v_food, IDX[((cnt_food >= N_MIN) & xlp_gate).values], label="XLP r5>=25 (pitched)")
stats(v_food, IDX[((cnt_food >= N_MIN) & spy_gate).values], label="SPY r5>=25 (swap)")
stats(v_food, IDX[((cnt_food >= N_MIN) & xlp_gate & spy_gate).values],
      label="BOTH intact")
stats(v_food, IDX[((cnt_food >= N_MIN) & xlp_gate & ~spy_gate).values],
      label="XLP intact, SPY NOT")
stats(v_food, IDX[((cnt_food >= N_MIN) & ~xlp_gate & spy_gate).values],
      label="SPY intact, XLP NOT")
