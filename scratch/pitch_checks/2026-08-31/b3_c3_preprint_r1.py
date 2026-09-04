"""C3 round 1: pre-print drift in a deeply LAGGING large cap (AVGO, print
2026-09-02, r63 rank 2.4).

Trade under test, matched to the live tape exactly:
  conditioner observed on the close of p-3   (today that is Fri 2026-08-28)
  entry  MOC on the close of p-2             (Mon 2026-08-31)
  exit   MOC on the close of p-1             (Tue 2026-09-01)
  print on p                                 (Wed 2026-09-02)
so the object is a ONE-SESSION hold, close[p-2] -> close[p-1], and the pitch
lag=1 convention is satisfied (signal p-3, entry p-2).

REFERENCE CLASS IS RUN FIRST, per the 2026-08-25/27 registry entries: the
identical rule across every name with earnings data, Cochran Q / I-squared /
common fixed-effect excess / permutation max-of-N. The registry records this
as the modal kill and the common excess as frequently negative.

Data hygiene (b3_c3_recon.py): pre-1993 earnings rows are quarter ENDS, not
announcement dates (82% land exactly on a quarter end), so the calendar is
restricted to 1996+ where that share is under 8%, and prices only start ~2000
anyway.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
CACHE = OUT / "_c3_panel.pkl"

# ---------------------------------------------------------------------------
# panel
# ---------------------------------------------------------------------------
E = pd.read_parquet(ROOT / "data" / "earnings_calendar.parquet",
                    columns=["ticker", "date"])
E["date"] = pd.to_datetime(E["date"])
E = E[(E["date"] >= "1996-01-01") & (E["date"] <= "2026-08-28")]
E = E.drop_duplicates(["ticker", "date"]).sort_values(["ticker", "date"])

if CACHE.exists():
    close = pd.read_pickle(CACHE)
else:
    mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                         columns=["ticker", "date", "Close"])
    mp["date"] = pd.to_datetime(mp["date"])
    keep = set(E["ticker"].unique()) | {"SPY", "SMH", "QQQ"}
    mp = mp[mp["ticker"].isin(keep)]
    mp = mp.drop_duplicates(["ticker", "date"], keep="last")
    close = mp.pivot(index="date", columns="ticker", values="Close").sort_index()
    close.to_pickle(CACHE)
print("panel", close.shape, close.index[0].date(), "->", close.index[-1].date())

SPY = close["SPY"].dropna()
spy_r = SPY.pct_change()

TICKERS = [t for t in E["ticker"].unique() if t in close.columns]
print("tickers with both earnings and prices:", len(TICKERS))

# ---------------------------------------------------------------------------
# per-ticker anchor build.  Uses each ticker's OWN valid-bar index (close_panel
# unions calendars; a foreign-calendar hole would shift every offset).
# ---------------------------------------------------------------------------
R63_GATE = 5.0          # "deeply lagging": trailing-252 pctile of the 63d return
K_ENTRY = 2             # entry at p - K_ENTRY
recs = []
for t in TICKERS:
    s = close[t].dropna()
    if len(s) < 400:
        continue
    idx = s.index
    r63 = pct_rank(s, 63, 252)          # computed on valid sessions only
    r21 = pct_rank(s, 21, 252)
    v = s.values
    dates = E.loc[E["ticker"] == t, "date"]
    pos, kept = anchor_positions(idx, dates, offset=0)
    for p, d in zip(pos, kept):
        # require the anchor to BE a real session for this ticker (searchsorted
        # snaps a non-session report date forward; that is the right anchor)
        if p - 10 < 0 or p + 1 >= len(v):
            continue
        rec = {"ticker": t, "report": d, "p": p,
               "r63": r63.iloc[p - 3], "r21": r21.iloc[p - 3]}
        for k in range(2, 11):
            rec[f"k{k}"] = v[p - 1] / v[p - k] - 1.0
        rec["post"] = v[p + 1] / v[p] - 1.0           # print reaction (context)
        rec["entry_date"] = idx[p - K_ENTRY]
        rec["exit_date"] = idx[p - 1]
        recs.append(rec)
D = pd.DataFrame(recs)
print("anchors", len(D), "tickers", D['ticker'].nunique(),
      "span", D['report'].min().date(), "->", D['report'].max().date())

# market return over the identical calendar span (for the beta residual)
spos = pd.Series(range(len(SPY)), index=SPY)
sp_idx = SPY.index
D["spy"] = [
    (SPY.reindex([ed, xd]).values[1] / SPY.reindex([ed, xd]).values[0] - 1.0)
    if (ed in SPY.index and xd in SPY.index) else np.nan
    for ed, xd in zip(D["entry_date"], D["exit_date"])]

D = D.dropna(subset=["k2"])
D["ret"] = D["k2"]

# ---------------------------------------------------------------------------
# 0. unconditional own-drift benchmark: the SAME ticker's average 1-session
#    return over the whole sample (all days), so "excess" is like-for-like.
# ---------------------------------------------------------------------------
drift = {}
for t in D["ticker"].unique():
    s = close[t].dropna()
    drift[t] = s.pct_change().mean()
D["drift"] = D["ticker"].map(drift)
D["excess"] = D["ret"] - D["drift"]

print("\n" + "=" * 78)
print("1. POOLED pre-print 1-session cell, ALL names (no lagging gate)")
print("=" * 78)
show([summarize(D["ret"].values, f"all prints k=2 (N={len(D)})"),
      summarize(D["drift"].values, "own-drift benchmark"),
      summarize(D["excess"].values, "excess over own drift"),
      summarize(D["spy"].values, "SPY same span")],
     "pooled, ungated")

gate = D["r63"] <= R63_GATE
G = D[gate]
print("\n" + "=" * 78)
print(f"2. GATED cell: r63 rank <= {R63_GATE:.0f} at p-3 (today AVGO = 2.4)")
print("=" * 78)
show([summarize(G["ret"].values, f"GATED (N={len(G)})"),
      summarize(G["excess"].values, "GATED excess over own drift"),
      summarize(D.loc[~gate, "excess"].values, "UNGATED complement excess"),
      summarize(G["spy"].values, "SPY same span, gated")],
     "gate attribution, pooled")
print(f"  gate value = {100*(G['excess'].mean() - D.loc[~gate,'excess'].mean()):+.4f}pp "
      f"(gated minus complement)")

# beta-neutral residual, pooled
beta_pool = np.polyfit(G["spy"].values, G["ret"].values, 1)[0]
G_res = G["ret"] - beta_pool * G["spy"]
print(f"  pooled beta of cell on SPY = {beta_pool:.3f}; "
      f"beta-neutral residual mean = {100*G_res.mean():+.4f}% "
      f"(t {G_res.mean()/(G_res.std(ddof=1)/np.sqrt(len(G_res))):+.2f})")

# ---------------------------------------------------------------------------
# 3. REFERENCE CLASS -- run BEFORE anything else is believed
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("3. REFERENCE CLASS: identical gated rule on every name (min 8 anchors)")
print("=" * 78)
rows = []
for t, g in G.groupby("ticker"):
    if len(g) < 8:
        continue
    e = g["excess"].values
    sd = e.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        continue
    rows.append({"ticker": t, "n": len(e), "excess_pp": 100 * e.mean(),
                 "se_pp": 100 * sd / np.sqrt(len(e))})
RC = pd.DataFrame(rows)
w = 1.0 / RC["se_pp"] ** 2
common = float((w * RC["excess_pp"]).sum() / w.sum())
se_common = float(np.sqrt(1.0 / w.sum()))
Q = float((w * (RC["excess_pp"] - common) ** 2).sum())
dfq = len(RC) - 1
I2 = max(0.0, 100 * (Q - dfq) / Q) if Q > 0 else 0.0
from scipy import stats as _st
pQ = float(_st.chi2.sf(Q, dfq))
print(f"  class size {len(RC)} names, {int(RC['n'].sum())} gated anchors")
print(f"  common fixed-effect excess = {common:+.4f}pp (se {se_common:.4f}, "
      f"z {common/se_common:+.2f})")
print(f"  Cochran Q = {Q:.2f} on {dfq} df, p = {pQ:.4f}, I-squared = {I2:.1f}%")
RC["t"] = RC["excess_pp"] / RC["se_pp"]
RC = RC.sort_values("excess_pp", ascending=False).reset_index(drop=True)
if "AVGO" in set(RC["ticker"]):
    rk = int(RC.index[RC["ticker"] == "AVGO"][0]) + 1
    a = RC[RC["ticker"] == "AVGO"].iloc[0]
    print(f"  AVGO excess {a['excess_pp']:+.4f}pp on n={int(a['n'])}, "
          f"t {a['t']:+.2f}, RANK {rk} of {len(RC)}  "
          f"-> family-wise p (max-of-N) = {1-(1-rk/len(RC))**1:.4f} "
          f"[empirical rank share {rk/len(RC):.4f}]")
else:
    print("  AVGO has too few gated anchors to enter the class")
print("\n  top 8 / bottom 8 of the class:")
print(RC.head(8).round(4).to_string(index=False))
print(RC.tail(8).round(4).to_string(index=False))

# permutation: max |t| under a sign-flip null on the per-name excesses
rng = np.random.default_rng(7)
obs_max = RC["t"].max()
nulls = []
for _ in range(2000):
    sgn = rng.choice([-1.0, 1.0], size=len(RC))
    nulls.append((RC["t"].values * sgn).max())
nulls = np.array(nulls)
print(f"\n  permutation (sign-flip) P(max t >= observed max {obs_max:.2f}) = "
      f"{(nulls >= obs_max).mean():.4f}")
if "AVGO" in set(RC["ticker"]):
    at = float(RC.loc[RC['ticker'] == 'AVGO', 't'].iloc[0])
    print(f"  permutation P(max t >= AVGO's t {at:.2f}) = {(nulls >= at).mean():.4f}")

# ---------------------------------------------------------------------------
# 4. AVGO ALONE (the pitched object)
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("4. AVGO alone")
print("=" * 78)
A = D[D["ticker"] == "AVGO"].sort_values("report")
print(f"  AVGO prints with prices: {len(A)}  "
      f"{A['report'].min().date()} .. {A['report'].max().date()}")
show([summarize(A["ret"].values, f"AVGO all prints (N={len(A)})"),
      summarize(A["excess"].values, "AVGO excess over own drift"),
      summarize(A["spy"].values, "SPY same span")], "AVGO ungated")
AG = A[A["r63"] <= R63_GATE]
print(f"\n  AVGO with r63<={R63_GATE:.0f} at p-3: N={len(AG)}")
if len(AG):
    print(AG[["report", "r63", "r21", "k2", "post"]].assign(
        k2=lambda x: (100 * x["k2"]).round(3),
        post=lambda x: (100 * x["post"]).round(3)).to_string(index=False))
    wins = int((AG["ret"] > 0).sum())
    up = float((close["AVGO"].dropna().pct_change() > 0).mean())
    print(f"  record {wins}-{len(AG)-wins}; AVGO's own unconditional up-rate "
          f"{100*up:.1f}%; sign p (vs own up-rate) = "
          f"{sign_test(wins, len(AG), p=up):.4f}")
# looser gates for AVGO
for g in [10, 20, 33, 50]:
    sub = A[A["r63"] <= g]
    if len(sub) >= 3:
        wins = int((sub["ret"] > 0).sum())
        print(f"    r63<={g:3d}: N={len(sub):3d} mean {100*sub['ret'].mean():+.3f}% "
              f"excess {100*sub['excess'].mean():+.3f}pp record {wins}-{len(sub)-wins}")

# ---------------------------------------------------------------------------
# 5. k-LADDER (how far before the print) -- pooled gated
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("5. k-LADDER: entry at p-k, exit at p-1, gated cell (pooled)")
print("=" * 78)
lad = []
for k in range(2, 11):
    e = (G[f"k{k}"] - (k - 1) * G["drift"]).dropna()
    lad.append({"k": k, "hold_td": k - 1, "n": len(e),
                "mean_pct": 100 * G[f"k{k}"].mean(),
                "excess_pp": 100 * e.mean(),
                "per_td_pp": 100 * e.mean() / (k - 1),
                "hit": 100 * (G[f"k{k}"] > 0).mean()})
print(pd.DataFrame(lad).round(4).to_string(index=False))

# ---------------------------------------------------------------------------
# 6. OFFSET PLACEBO LADDER (the repo's 11-for-11 kill)
#    shift the anchor by m and run the identical (p'-2 -> p'-1) trade
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("6. OFFSET PLACEBO LADDER on the anchor, m = -10..+5 (gated)")
print("=" * 78)
plac = []
for m in range(-10, 6):
    vals, exs = [], []
    for t, g in G.groupby("ticker"):
        s = close[t].dropna()
        v = s.values
        dr = drift[t]
        for p in g["p"].values:
            q = p + m
            if q - 2 < 0 or q >= len(v):
                continue
            r = v[q - 1] / v[q - 2] - 1.0
            vals.append(r)
            exs.append(r - dr)
    if vals:
        plac.append({"m": m, "n": len(vals), "mean_pct": 100 * np.mean(vals),
                     "excess_pp": 100 * np.mean(exs),
                     "t": np.mean(exs) / (np.std(exs, ddof=1) / np.sqrt(len(exs)))})
P = pd.DataFrame(plac)
P["rank"] = P["excess_pp"].rank(ascending=False).astype(int)
print(P.round(4).to_string(index=False))
true_rank = int(P.loc[P["m"] == 0, "rank"].iloc[0])
print(f"  TRUE anchor (m=0) ranks {true_rank} of {len(P)}; "
      f"ladder mean excess {P['excess_pp'].mean():+.4f}pp, "
      f"true minus ladder mean = "
      f"{float(P.loc[P['m']==0,'excess_pp'].iloc[0]) - P['excess_pp'].mean():+.4f}pp")

# ---------------------------------------------------------------------------
# 7. era + cost + tail
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("7. era split / cost / tail (pooled gated)")
print("=" * 78)
pre = G[G["report"] < "2018-01-01"]
post = G[G["report"] >= "2018-01-01"]
show([summarize(pre["excess"].values, f"pre-2018 excess (N={len(pre)})"),
      summarize(post["excess"].values, f"2018+ excess (N={len(post)})")], "era")
print(f"  worst single anchor: {100*G['ret'].min():.2f}% on "
      f"{G.loc[G['ret'].idxmin(),'ticker']} {G.loc[G['ret'].idxmin(),'report'].date()}")
print(f"  p01 / p05 of the gated 1-session return: "
      f"{100*G['ret'].quantile(0.01):.2f}% / {100*G['ret'].quantile(0.05):.2f}%")
edge_bps = 100 * G["excess"].mean() * 100
print(f"  cost: single-name round trip ~8 bps; gated excess "
      f"{edge_bps:.2f} bps -> {edge_bps/8:.2f}x cost (bar is 5x)")

D.to_pickle(OUT / "_c3_anchors.pkl")
print("\nsaved anchors ->", OUT / "_c3_anchors.pkl")
