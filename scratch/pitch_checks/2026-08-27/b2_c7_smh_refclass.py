"""C7 round 1 — the leader's deep correction, with the CROSS-SECTIONAL
reference class built FIRST (the 2026-08-13 IHI protocol).

Trigger: 63d return PIT trailing-252 rank <= 5 AND trailing-252d return in the
top decile (two forms: an absolute >= +40% gate, and a cross-sectional
top-decile gate against the reference class itself). Long, h = 1..10.

Order of work, deliberately: reference class -> fixed-effect meta + Cochran Q
-> permutation max-of-K -> only then SMH's own cell.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

ASOF = pd.Timestamp("2026-08-26")
RNG = np.random.default_rng(42)

REF = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV",
       "XLY", "SMH", "IBB", "IHI", "ITA", "ITB", "XBI", "XHB", "XME", "XOP",
       "XRT", "KRE", "OIH", "VNQ", "IYR", "GDX", "QQQ", "IWM", "SPY", "DIA",
       "EEM", "EFA", "FXI"]
H = 10
R63_MAX = 5.0
R252_MIN = 0.40          # absolute top-decile proxy; SMH today is +0.8968
MIN_HIST = 1500          # sessions

px = close_panel(REF)
px = px[px.index <= ASOF]


def vret(s, n):
    v = s.dropna()
    return (v / v.shift(n) - 1.0).reindex(s.index)


# ---------------------------------------------------------------------------
print("=" * 78)
print("0. TODAY'S STATE across the reference class")
print("=" * 78)
today = []
for t in REF:
    s = px[t].dropna()
    if len(s) < MIN_HIST:
        continue
    today.append({"t": t, "r63": pct_rank(s, 63).iloc[-1],
                  "ret252_pct": 100 * vret(s, 252).iloc[-1],
                  "ret63_pct": 100 * vret(s, 63).iloc[-1],
                  "r5": pct_rank(s, 5).iloc[-1]})
T = pd.DataFrame(today).set_index("t").sort_values("r63")
print(T.round(2).to_string())
fires = T[(T.r63 <= R63_MAX) & (T.ret252_pct >= 100 * R252_MIN)]
print(f"\n  FIRES TODAY (r63<={R63_MAX} & ret252>={100*R252_MIN}%): {list(fires.index)}")
print(f"  ret252 cross-sectional decile cut today = {T.ret252_pct.quantile(0.90):.2f}%; "
      f"SMH {T.loc['SMH','ret252_pct']:.2f}% -> rank "
      f"{int((T.ret252_pct >= T.loc['SMH','ret252_pct']).sum())} of {len(T)}")

# ---------------------------------------------------------------------------
# 1. reference class: identical rule on every member
# ---------------------------------------------------------------------------
def cell(t, h=H, r63_max=R63_MAX, r252_min=R252_MIN, gate=True, min_gap=None):
    s = px[t].dropna()
    if len(s) < MIN_HIST:
        return None
    r63 = pct_rank(s, 63)
    r252 = vret(s, 252)
    m = (r63 <= r63_max)
    if gate:
        m = m & (r252 >= r252_min)
    fwd = fwd_lag(s, h, 1)
    valid = fwd.dropna().index
    trig = s.index[m.reindex(s.index, fill_value=False).fillna(False).values]
    trig = pd.DatetimeIndex(trig).intersection(valid)
    if len(trig) == 0:
        return {"t": t, "n_days": 0, "n": 0}
    epi = declusters(trig, min_gap or h, valid)
    v = fwd.loc[epi].values
    base = fwd.loc[valid]
    n = len(v)
    sd = v.std(ddof=1) if n > 1 else np.nan
    return {"t": t, "n_days": len(trig), "n": n,
            "mean_pct": 100 * v.mean(),
            "drift_pct": 100 * base.mean(),
            "excess_pct": 100 * (v.mean() - base.mean()),
            "se_pct": 100 * sd / np.sqrt(n) if n > 1 else np.nan,
            "hit": 100 * (v > 0).mean(),
            "worst_pct": 100 * v.min(),
            "_vals": v, "_dates": epi}


rows = [cell(t) for t in REF]
rows = [r for r in rows if r]
RC = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                   for r in rows]).set_index("t")
RC = RC[RC.n >= 3].sort_values("excess_pct", ascending=False)
print("\n" + "=" * 78)
print(f"1. REFERENCE CLASS, identical rule, h={H}  (K={len(RC)} members with n>=3)")
print("=" * 78)
print(RC.round(3).to_string())

# fixed-effect meta + Cochran Q
w = 1.0 / RC.se_pct.values ** 2
th = RC.excess_pct.values
fe = float((w * th).sum() / w.sum())
Q = float((w * (th - fe) ** 2).sum())
df = len(RC) - 1
I2 = max(0.0, (Q - df) / Q) * 100 if Q > 0 else 0.0
try:
    from scipy import stats as _st
    pQ = float(_st.chi2.sf(Q, df))
except Exception:
    pQ = float("nan")
print(f"\n  fixed-effect common excess = {fe:+.3f}pp   se = {1/np.sqrt(w.sum()):.3f}pp"
      f"   z = {fe*np.sqrt(w.sum()):+.2f}")
print(f"  Cochran Q = {Q:.2f} on {df} df   p = {pQ:.3f}   I^2 = {I2:.1f}%")
print(f"  SMH excess = {RC.loc['SMH','excess_pct']:+.3f}pp  ->  rank "
      f"{int((RC.excess_pct >= RC.loc['SMH','excess_pct']).sum())} of {len(RC)} by excess; "
      f"by |t|: rank {int(((RC.excess_pct/RC.se_pct).abs() >= abs(RC.loc['SMH','excess_pct']/RC.loc['SMH','se_pct'])).sum())}")

# ---------------------------------------------------------------------------
# 2. permutation max-of-K (circular shift of the whole trigger set)
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("2. PERMUTATION max-of-K  (common circular date shift, preserves clustering)")
print("=" * 78)
store = {r["t"]: r for r in rows if r["t"] in RC.index}
fwds, bases, idxs = {}, {}, {}
for t in RC.index:
    s = px[t].dropna()
    f = fwd_lag(s, H, 1).dropna()
    fwds[t], bases[t], idxs[t] = f.values, f.mean(), f.index

B = 2000
maxes = np.empty(B)
smh_null = np.empty(B)
for b in range(B):
    best = -1e9
    for t in RC.index:
        f, base, idx = fwds[t], bases[t], idxs[t]
        n = store[t]["n"]
        pos = RNG.integers(0, len(f), size=n)
        ex = 100 * (f[pos].mean() - base)
        if t == "SMH":
            smh_null[b] = ex
        best = max(best, ex)
    maxes[b] = best
obs = RC.loc["SMH", "excess_pct"]
print(f"  observed SMH excess {obs:+.3f}pp")
print(f"  null max-of-{len(RC)} : mean {maxes.mean():+.3f}pp  median {np.median(maxes):+.3f}pp  "
      f"p95 {np.percentile(maxes,95):+.3f}pp  max {maxes.max():+.3f}pp")
print(f"  FAMILY-WISE p = P(max_null >= obs) = {(maxes >= obs).mean():.4f}")
print(f"  per-instrument (SMH only) null p = {(smh_null >= obs).mean():.4f}")

# ---------------------------------------------------------------------------
# 3. SMH's own cell — full battery, both gate forms, and the gate attribution
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("3. SMH's own cell")
print("=" * 78)
s = px["SMH"].dropna()
r63 = pct_rank(s, 63)
r252 = vret(s, 252)
mask = ((r63 <= R63_MAX) & (r252 >= R252_MIN)).reindex(px.index, fill_value=False).fillna(False)
print(f"  trigger days: {int(mask.sum())}")
variants = {
    "r63<=5 NO 252 gate": (r63 <= 5).reindex(px.index, fill_value=False).fillna(False),
    "r63<=2 & 252>=40%": ((r63 <= 2) & (r252 >= 0.40)).reindex(px.index, fill_value=False).fillna(False),
    "r63<=10 & 252>=40%": ((r63 <= 10) & (r252 >= 0.40)).reindex(px.index, fill_value=False).fillna(False),
    "r63<=5 & 252>=25%": ((r63 <= 5) & (r252 >= 0.25)).reindex(px.index, fill_value=False).fillna(False),
    "r63<=5 & 252>=70%": ((r63 <= 5) & (r252 >= 0.70)).reindex(px.index, fill_value=False).fillna(False),
    "r63<=5 & 252 gate & r5>=25": ((r63 <= 5) & (r252 >= 0.40)
                                   & (pct_rank(s, 5) >= 25)).reindex(px.index, fill_value=False).fillna(False),
}
for h in (5, 10):
    battery(px, mask, [("SMH", 1.0)], h, "C7 LONG SMH: r63<=5 & 252d>=+40%",
            cost_bps=5.0, variants=variants if h == 10 else None,
            event_kinds=("cpi", "ppi", "nfp"))

trig = px.index[mask.values]
show(horizon_scan(px, trig, [("SMH", 1.0)], hs=(1, 2, 3, 5, 10)),
     "HORIZON SCAN long SMH")

# gate attribution: what does the 252d gate actually remove?
m_nogate = (r63 <= R63_MAX)
print(f"\n  GATE ATTRIBUTION: r63<=5 alone fires {int(m_nogate.sum())} days; "
      f"with the 252d>=+40% gate {int(mask.sum())} days "
      f"-> the gate removes {100*(1-mask.sum()/max(1,m_nogate.sum())):.1f}%")

# midterm split (mandatory today)
epi = declusters(trig, H, fwd_lag(s, H, 1).dropna().index)
f10 = fwd_lag(s, H, 1)
mid = np.array([d.year % 4 == 2 for d in epi])
show([summarize(f10.loc[epi[mid]].values, f"MIDTERM years (N={int(mid.sum())})"),
      summarize(f10.loc[epi[~mid]].values, f"non-midterm (N={int((~mid).sum())})")],
     "MIDTERM CYCLE SPLIT, h=10 episodes")
print("  midterm episode dates:", [str(d.date()) for d in epi[mid]])
