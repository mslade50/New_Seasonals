"""C12 -- discharge watchlist entry 40's arm.

The arm, verbatim: "TURNS ON when the identical rule run across a 13-name
subsector reference class (ITA IHI IBB XBI ITB XHB XRT XME XOP OIH KRE SMH
IYR) puts ITA's excess outside a max-of-13 permutation at P <= 0.10."

Part A reproduces the 2026-09-07 ITA numbers exactly (same rule, same
declustering, same controls) so the arm is run against a verified object.
Part B runs the identical rule on all 13 names and prices ITA against the
class with the NULL-3 estimator the repo settled on 2026-08-21
(a4c_c11_class_null_ownvol.py): impose a COMMON class mean, keep each name's
OWN episode dispersion, resample each name at its own N, take the max over
K=13, and ask how often that max reaches ITA's observed excess.

Two bases, because the watchlist entry itself names both:
  EXCESS   = episode mean minus that name's own all-days lag-1 h-day drift
  RESIDUAL = beta-neutral (name - beta*SPY) episode mean minus its own drift
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change  # noqa

warnings.filterwarnings("ignore")

ASOF = pd.Timestamp("2026-09-04")
CLASS = ["ITA", "IHI", "IBB", "XBI", "ITB", "XHB", "XRT",
         "XME", "XOP", "OIH", "KRE", "SMH", "IYR"]
RANK_MAX = 10.0
SPY_GATE = 0.98            # within 2% of the trailing-252 high
NB = 20000
rng = np.random.default_rng(42)


def roll_max(s, n=252):
    return rolling_on_valid(s, lambda x: x.rolling(n).max())


PX = load_prices(CLASS + ["SPY"])
IDX = PX["SPY"].index
IDX = IDX[IDX <= ASOF]
C = {t: PX[t]["Close"].reindex(IDX) for t in PX}

spy_hi = roll_max(C["SPY"], 252)
near_hi = (C["SPY"] >= SPY_GATE * spy_hi).fillna(False)
RANK = {t: pct_rank(C[t], 21) for t in CLASS}

print("=" * 78)
print(f"C12  ITA WASHOUT UNDER AN INDEX AT ITS HIGH -- reference class of 13"
      f"   (asof {IDX[-1].date()})")
print("=" * 78)
print(f"  SPY {C['SPY'].iloc[-1]:.2f}   off 252d high "
      f"{100 * (C['SPY'].iloc[-1] / spy_hi.iloc[-1] - 1):+.2f}%   gate live: "
      f"{bool(near_hi.iloc[-1])}")
print("  live class state (r21, 21d ret, off-52wh):")
for t in CLASS:
    s = C[t].dropna()
    print(f"    {t:5s} r21 {RANK[t].iloc[-1]:5.1f}   21d "
          f"{100 * (s.iloc[-1] / s.iloc[-22] - 1):+6.2f}%   off52wh "
          f"{100 * (s.iloc[-1] / roll_max(s, 252).iloc[-1] - 1):+7.2f}%"
          f"   {'IN CELL' if RANK[t].iloc[-1] <= RANK_MAX else ''}")


def cellstats(t: str, h: int, rank_max: float = RANK_MAX, gate: bool = True,
              min_gap: int | None = None):
    """Episode-level stats for one name under the rule."""
    f = fwd_lag(C[t], h, 1)
    valid = f.dropna().index
    m = (RANK[t] <= rank_max).fillna(False)
    if gate:
        m = m & near_hi
    trig = pd.DatetimeIndex(IDX[m.values]).intersection(valid)
    if len(trig) == 0:
        return None
    epi = declusters(trig, min_gap or h, valid)
    ep = f.loc[epi].values
    drift = float(f.loc[valid].mean())
    loc = local_control(valid, trig)
    span = valid[(valid >= trig[0]) & (valid <= trig[-1])]
    w = int((ep > 0).sum())
    return {"tkr": t, "h": h, "n_days": len(trig), "n": len(epi),
            "mean_pct": 100 * ep.mean(), "drift_pct": 100 * drift,
            "excess_pp": 100 * (ep.mean() - drift),
            "span_drift_pct": 100 * float(f.loc[span].mean()),
            "local_pct": 100 * float(f.loc[loc].mean()) if len(loc) else np.nan,
            "hit": 100 * (ep > 0).mean(), "worst_pct": 100 * ep.min(),
            "rec": f"{w}-{len(ep) - w}", "sign_p": sign_test(w, len(ep)),
            "bootP": bootstrap_p_le0(ep), "epi": epi, "ep": ep,
            "yrs": len(set(epi.year))}


# ---------------------------------------------------------------- Part A
print("\n" + "=" * 78)
print("PART A -- reproduce the 2026-09-07 ITA cell (claims: h=10 +1.223% N=29 "
      "bootP 0.010 edge +0.659pp; h=5 +0.543% N=43 26-17 bootP 0.071 edge +0.261pp)")
print("=" * 78)
repro = []
for h in (5, 10):
    r = cellstats("ITA", h)
    repro.append({k: v for k, v in r.items() if k not in ("epi", "ep")})
    print(f"\n  h={h}: N_epi {r['n']} (days {r['n_days']}), mean {r['mean_pct']:+.3f}%, "
          f"drift {r['drift_pct']:+.3f}%, edge {r['excess_pp']:+.3f}pp, "
          f"rec {r['rec']}, sign p {r['sign_p']:.4f}, bootP<=0 {r['bootP']:.3f}, "
          f"worst {r['worst_pct']:.2f}%, {r['yrs']} distinct yrs")
    print(f"     local +/-126td control {r['local_pct']:+.3f}%   same-span drift "
          f"{r['span_drift_pct']:+.3f}%")
    print("     " + cluster_note(r["epi"], r["ep"]))

# ---------------------------------------------------------------- Part B
print("\n" + "=" * 78)
print("PART B -- the identical rule on all 13 names")
print("=" * 78)

beta = {}
for t in CLASS:
    dd = pd.DataFrame({"a": C[t], "s": C["SPY"]}).pct_change().dropna()
    beta[t] = float(np.polyfit(dd["s"], dd["a"], 1)[0])


def residual_stats(t: str, h: int):
    b = beta[t]
    rr = vehicle_ret(pd.DataFrame({t: C[t], "SPY": C["SPY"]}),
                     [(t, 1.0), ("SPY", -b)], h, 1)
    valid = rr.dropna().index
    m = ((RANK[t] <= RANK_MAX) & near_hi).fillna(False)
    trig = pd.DatetimeIndex(IDX[m.values]).intersection(valid)
    if len(trig) == 0:
        return None
    epi = declusters(trig, h, valid)
    ep = rr.loc[epi].values
    drift = float(rr.loc[valid].mean())
    w = int((ep > 0).sum())
    return {"n": len(epi), "mean_pct": 100 * ep.mean(),
            "drift_pct": 100 * drift, "excess_pp": 100 * (ep.mean() - drift),
            "rec": f"{w}-{len(ep) - w}", "sign_p": sign_test(w, len(ep)),
            "ep": ep, "beta": b}


books = {}
for h in (5, 10):
    rows, ex_book, res_book = [], {}, {}
    for t in CLASS:
        a = cellstats(t, h)
        if a is None or a["n"] < 5:
            rows.append({"tkr": t, "n": 0 if a is None else a["n"],
                         "note": "too few episodes"})
            continue
        b = residual_stats(t, h)
        rows.append({"tkr": t, "n_days": a["n_days"], "n": a["n"],
                     "yrs": a["yrs"],
                     "mean_pct": round(a["mean_pct"], 3),
                     "drift_pct": round(a["drift_pct"], 3),
                     "excess_pp": round(a["excess_pp"], 3),
                     "hit": round(a["hit"], 1), "rec": a["rec"],
                     "sign_p": round(a["sign_p"], 4),
                     "resid_excess_pp": round(b["excess_pp"], 3),
                     "resid_rec": b["rec"], "beta": round(b["beta"], 2),
                     "worst_pct": round(a["worst_pct"], 2)})
        # centred episode books, in PERCENT units on both sides (unit assert below)
        ex_book[t] = 100 * a["ep"] - a["drift_pct"]
        res_book[t] = 100 * b["ep"] - b["drift_pct"]
    print(f"\n=== h={h} per-name ===")
    print(pd.DataFrame(rows).to_string(index=False))
    books[h] = (ex_book, res_book)


def null3(book: dict, focus: str, label: str):
    names = list(book)
    obs = float(book[focus].mean())
    means = {c: float(book[c].mean()) for c in names}
    cm = float(np.mean(list(means.values())))
    cen = {c: book[c] - means[c] + cm for c in names}
    ns = {c: len(book[c]) for c in names}
    mx = np.empty(NB)
    for i in range(NB):
        mx[i] = max(rng.choice(cen[c], size=ns[c], replace=True).mean()
                    for c in names)
    p = float((mx >= obs).mean())
    rank = 1 + sum(1 for c in names if means[c] > obs)
    other = np.concatenate([book[c] for c in names if c != focus])
    a = book[focus]
    se = np.sqrt(a.var(ddof=1) / len(a) + other.var(ddof=1) / len(other))
    print(f"\n--- NULL 3 on {label} (K={len(names)}) ---")
    print(f"  class members positive: "
          f"{sum(1 for c in names if means[c] > 0)}/{len(names)};  "
          f"class equal-weight mean {cm:+.3f}pp;  class median "
          f"{np.median(list(means.values())):+.3f}pp")
    print(f"  {focus} observed {obs:+.3f}pp -> rank {rank} of {len(names)}")
    print(f"  null max-of-{len(names)}: median {np.median(mx):+.3f}  "
          f"95th {np.percentile(mx, 95):+.3f}")
    print(f"  **P(max-of-{len(names)} >= {focus}) = {p:.4f}**")
    print(f"  Welch {focus} vs pooled others: "
          f"{a.mean() - other.mean():+.3f}pp  t {(a.mean() - other.mean()) / se:+.2f}"
          f"  (others mean {other.mean():+.3f}pp, N={len(other)})")
    # Cochran Q on the per-name excesses (homogeneity of the class)
    ses = {c: book[c].std(ddof=1) / np.sqrt(len(book[c])) for c in names}
    wts = {c: 1 / ses[c] ** 2 for c in names}
    fe = sum(wts[c] * means[c] for c in names) / sum(wts.values())
    Q = sum(wts[c] * (means[c] - fe) ** 2 for c in names)
    df = len(names) - 1
    I2 = max(0.0, 100 * (Q - df) / Q) if Q > 0 else 0.0
    print(f"  fixed-effect common excess {fe:+.3f}pp;  Cochran Q {Q:.2f} on "
          f"{df} df;  I-squared {I2:.1f}%")
    return p


print("\n" + "=" * 78)
print("PART B2 -- the max-of-13 permutation (20,000 draws, seed 42)")
print("=" * 78)
results = {}
for h in (5, 10):
    ex_book, res_book = books[h]
    assert abs(np.mean(ex_book["ITA"])) < 50, "unit check: percent, not fraction"
    results[(h, "excess")] = null3(ex_book, "ITA", f"EXCESS h={h}")
    results[(h, "residual")] = null3(res_book, "ITA", f"BETA-NEUTRAL RESIDUAL h={h}")

print("\n" + "=" * 78)
print("ARM VERDICT")
print("=" * 78)
for k, v in results.items():
    print(f"  h={k[0]:2d} {k[1]:9s}  P(max-of-13 >= ITA) = {v:.4f}   "
          f"{'ARMS (<=0.10)' if v <= 0.10 else 'STAYS PARKED (>0.10)'}")
