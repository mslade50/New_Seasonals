"""C10 round 1 — the TJX-class washout, judged against the 218-name tape as a
reference class built FIRST (2026-08-13 IHI protocol).

Trigger on a large-cap single name: z10 <= -2.0 AND 21d PIT return rank <= 2
AND within 2% of a 52-week low. Long, h = 1..10.

Order: pooled cross-sectional cell -> heterogeneity + permutation max-of-K ->
today's firing basket (enumerated) -> earnings proximity -> TJX alone -> book
overlap in the 23y ledger.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

ASOF = pd.Timestamp("2026-08-26")
RNG = np.random.default_rng(7)
Z_MAX, R21_MAX, LOW_PCT = -2.0, 2.0, 2.0
MIN_HIST = 1500

tape = sorted(json.load(open(ROOT / "data/pitch_tape.json"))["tickers"])
frames = load_prices(tape)
print(f"loaded {len(frames)} of {len(tape)} tape names")

# ---------------------------------------------------------------------------
# build per-name masks + forward returns once
# ---------------------------------------------------------------------------
def build(t):
    s = frames[t]["Close"].dropna()
    s = s[s.index <= ASOF]
    if len(s) < MIN_HIST:
        return None
    z = zscore(s, 10)
    r21 = pct_rank(s, 21)
    lo = s.rolling(252).min()
    dl = 100 * (s / lo - 1.0)
    m = (z <= Z_MAX) & (r21 <= R21_MAX) & (dl <= LOW_PCT)
    return {"s": s, "z": z, "r21": r21, "dl": dl, "mask": m.fillna(False)}


built = {}
for t in tape:
    if t not in frames:
        continue
    b = build(t)
    if b is not None:
        built[t] = b
print(f"usable names (>= {MIN_HIST} sessions): {len(built)}")

# ---------------------------------------------------------------------------
# 0. today's firing basket
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("0. WHO FIRES TODAY (state as of 2026-08-26 close)")
print("=" * 78)
today = []
for t, b in built.items():
    today.append({"t": t, "z10": b["z"].iloc[-1], "r21": b["r21"].iloc[-1],
                  "dist_low_pct": b["dl"].iloc[-1], "fires": bool(b["mask"].iloc[-1])})
TD = pd.DataFrame(today).set_index("t")
fire = TD[TD.fires].sort_values("z10")
print(fire.round(2).to_string() if len(fire) else "  NOBODY FIRES")
print(f"\n  basket size today = {len(fire)} of {len(built)}")
near = TD[(TD.z10 <= -1.5) & (TD.r21 <= 10) & (TD.dist_low_pct <= 5)].sort_values("z10")
print(f"  loosened (z<=-1.5, r21<=10, <=5% off low): {len(near)} -> {list(near.index)}")
for t in ["TJX", "WMT", "NKE", "ROST", "TGT", "XRT"]:
    if t in TD.index:
        r = TD.loc[t]
        print(f"  {t:<6} z10 {r.z10:+6.2f}  r21 {r.r21:5.1f}  off-low {r.dist_low_pct:6.2f}%  fires={r.fires}")

# ---------------------------------------------------------------------------
# 1. POOLED cross-sectional cell (the reference class IS the pool)
# ---------------------------------------------------------------------------
H_LIST = (1, 2, 3, 5, 10)
print("\n" + "=" * 78)
print("1. POOLED cell across all names, all history (episodes declustered per name)")
print("=" * 78)
pooled = {h: [] for h in H_LIST}
pooled_dates = {h: [] for h in H_LIST}
pooled_name = {h: [] for h in H_LIST}
drift = {h: [] for h in H_LIST}
per_name = {h: {} for h in H_LIST}
for t, b in built.items():
    s = b["s"]
    for h in H_LIST:
        f = fwd_lag(s, h, 1)
        valid = f.dropna().index
        trig = pd.DatetimeIndex(s.index[b["mask"].reindex(s.index, fill_value=False).values]).intersection(valid)
        if len(trig) == 0:
            continue
        epi = declusters(trig, h, valid)
        v = f.loc[epi].values
        pooled[h].extend(v.tolist())
        pooled_dates[h].extend(list(epi))
        pooled_name[h].extend([t] * len(epi))
        drift[h].append(f.loc[valid].mean())
        per_name[h][t] = v

rows = []
for h in H_LIST:
    v = np.array(pooled[h])
    d = float(np.mean(drift[h]))
    r = summarize(v, f"POOLED h={h}")
    r["n_names"] = len(per_name[h])
    r["drift_pct"] = round(100 * d, 3)
    r["excess_pct"] = round(r["mean_pct"] - 100 * d, 3)
    r["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
    rows.append(r)
show(rows, "pooled conditional vs the average name's unconditional drift")

# name-level control: each name's OWN drift, matched
print("\n  name-matched excess (each episode minus its own name's drift):")
for h in H_LIST:
    ex = []
    for t, v in per_name[h].items():
        f = fwd_lag(built[t]["s"], h, 1)
        ex.extend((v - f.dropna().mean()).tolist())
    ex = np.array(ex)
    print(f"    h={h:>2}  N={len(ex):>4}  mean excess {100*ex.mean():+7.3f}pp  "
          f"t {ex.mean()/(ex.std(ddof=1)/np.sqrt(len(ex))):+6.2f}  "
          f"hit {100*(ex>0).mean():5.1f}%  median {100*np.median(ex):+6.3f}pp")

# era + midterm split on the pool
H = 10
dts = pd.DatetimeIndex(pooled_dates[H])
vals = np.array(pooled[H])
show(era_split(dts, vals), f"pooled era split h={H}")
mid = np.array([d.year % 4 == 2 for d in dts])
show([summarize(vals[mid], f"MIDTERM (N={int(mid.sum())})"),
      summarize(vals[~mid], f"non-midterm (N={int((~mid).sum())})")],
     f"pooled MIDTERM split h={H}")
by_year = pd.Series(vals).groupby(dts.year.values).agg(["size", "mean"])
print("\n  pooled by year (h=10):")
print((by_year.assign(mean=lambda d: (100 * d["mean"]).round(2))).to_string())

# ---------------------------------------------------------------------------
# 2. heterogeneity + permutation max-of-K over names with n>=5 episodes
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print(f"2. HETEROGENEITY + permutation max-of-K, h={H}, names with n>=5 episodes")
print("=" * 78)
mem = []
for t, v in per_name[H].items():
    if len(v) < 5:
        continue
    f = fwd_lag(built[t]["s"], H, 1)
    d = f.dropna().mean()
    mem.append({"t": t, "n": len(v), "mean_pct": 100 * v.mean(),
                "drift_pct": 100 * d, "excess_pct": 100 * (v.mean() - d),
                "se_pct": 100 * v.std(ddof=1) / np.sqrt(len(v)),
                "hit": 100 * (v > 0).mean()})
M = pd.DataFrame(mem).set_index("t").sort_values("excess_pct", ascending=False)
print(f"  K = {len(M)} names")
print(M.round(3).head(12).to_string())
print("  ...")
print(M.round(3).tail(6).to_string())
w = 1.0 / M.se_pct.values ** 2
th = M.excess_pct.values
fe = float((w * th).sum() / w.sum())
Q = float((w * (th - fe) ** 2).sum())
df = len(M) - 1
I2 = max(0.0, (Q - df) / Q) * 100 if Q > 0 else 0.0
try:
    from scipy import stats as _st
    pQ = float(_st.chi2.sf(Q, df))
except Exception:
    pQ = float("nan")
print(f"\n  fixed-effect common excess {fe:+.3f}pp  se {1/np.sqrt(w.sum()):.3f}pp  "
      f"z {fe*np.sqrt(w.sum()):+.2f}")
print(f"  Cochran Q {Q:.2f} on {df} df   p {pQ:.3f}   I^2 {I2:.1f}%")
if "TJX" in M.index:
    print(f"  TJX excess {M.loc['TJX','excess_pct']:+.3f}pp (n={int(M.loc['TJX','n'])}) -> rank "
          f"{int((M.excess_pct >= M.loc['TJX','excess_pct']).sum())} of {len(M)}")

# permutation: random anchors per name, same episode counts, max over K
fwd_cache = {t: fwd_lag(built[t]["s"], H, 1).dropna() for t in M.index}
B = 2000
maxes = np.empty(B)
for b in range(B):
    best = -1e9
    for t in M.index:
        f = fwd_cache[t].values
        n = int(M.loc[t, "n"])
        pos = RNG.integers(0, len(f), size=n)
        ex = 100 * (f[pos].mean() - f.mean())
        best = max(best, ex)
    maxes[b] = best
tjx_obs = M.loc["TJX", "excess_pct"] if "TJX" in M.index else np.nan
best_obs = M.excess_pct.max()
print(f"\n  null max-of-{len(M)}: median {np.median(maxes):+.3f}pp  p95 {np.percentile(maxes,95):+.3f}pp")
print(f"  observed BEST member {M.excess_pct.idxmax()} {best_obs:+.3f}pp -> family-wise p "
      f"{(maxes >= best_obs).mean():.4f}")
if not np.isnan(tjx_obs):
    print(f"  observed TJX {tjx_obs:+.3f}pp -> family-wise p {(maxes >= tjx_obs).mean():.4f}")

# ---------------------------------------------------------------------------
# 3. TJX alone
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("3. TJX's own cell")
print("=" * 78)
px = pd.DataFrame({t: built[t]["s"] for t in ["TJX", "WMT", "NKE", "XRT", "SPY"]
                   if t in built}).dropna(how="all")
b = built["TJX"]
mask = b["mask"].reindex(px.index, fill_value=False).fillna(False)
print(f"  TJX trigger days: {int(mask.sum())}")
variants = {
    "z<=-2 & r21<=2 (no low gate)": ((b["z"] <= -2) & (b["r21"] <= 2)).reindex(px.index, fill_value=False).fillna(False),
    "z<=-2 only": (b["z"] <= -2).reindex(px.index, fill_value=False).fillna(False),
    "r21<=2 only": (b["r21"] <= 2).reindex(px.index, fill_value=False).fillna(False),
    "within 2% of low only": (b["dl"] <= 2).reindex(px.index, fill_value=False).fillna(False),
    "z<=-2.5 & r21<=2 & low2": ((b["z"] <= -2.5) & (b["r21"] <= 2) & (b["dl"] <= 2)).reindex(px.index, fill_value=False).fillna(False),
}
for h in (5, 10):
    battery(px, mask, [("TJX", 1.0)], h, "C10 LONG TJX washout", cost_bps=5.0,
            variants=variants if h == 10 else None, event_kinds=("cpi", "ppi", "nfp"))
trig = px.index[mask.values]
show(horizon_scan(px, trig, [("TJX", 1.0)], hs=(1, 2, 3, 5, 10)), "HORIZON SCAN long TJX")

# ---------------------------------------------------------------------------
# 4. earnings proximity for the firing basket
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("4. EARNINGS PROXIMITY for every name firing today (hold = 10 td from 08-28)")
print("=" * 78)
ec = pd.read_parquet(ROOT / "data/earnings_calendar.parquet")
col = "date" if "date" in ec.columns else ec.columns[0]
ec[col] = pd.to_datetime(ec[col])
sym = "symbol" if "symbol" in ec.columns else ("ticker" if "ticker" in ec.columns else ec.columns[1])
for t in list(fire.index) + ["TJX", "WMT", "NKE"]:
    e = ec[(ec[sym] == t) & (ec[col] >= pd.Timestamp("2026-08-01")) & (ec[col] <= pd.Timestamp("2026-10-15"))]
    print(f"  {t:<8} next prints: {[str(d.date()) for d in sorted(e[col].unique())][:4]}")

# ---------------------------------------------------------------------------
# 5. book overlap
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("5. BOOK OVERLAP — ledger signals fired in this exact state")
print("=" * 78)
led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
led["Signal Date"] = pd.to_datetime(led["Signal Date"])
hits = []
for t, bb in built.items():
    sub = led[led.Ticker == t]
    if sub.empty:
        continue
    m = bb["mask"]
    for _, r in sub.iterrows():
        d = r["Signal Date"]
        if d in m.index and bool(m.loc[d]):
            hits.append({"Ticker": t, "Strategy": r["Strategy"], "date": d,
                         "R": r["R_Multiple"], "Dir": r["Direction"]})
Hb = pd.DataFrame(hits)
if len(Hb):
    print(f"  {len(Hb)} ledger trades signalled on a C10 trigger day")
    print(Hb.groupby("Strategy").agg(n=("R", "size"), avgR=("R", "mean")).round(3).to_string())
    print(f"  overall avgR {Hb.R.mean():+.3f} on N={len(Hb)}")
else:
    print("  0 ledger trades on a C10 trigger day")
