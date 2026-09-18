"""K2 c3 round 1b/2: the SHORT side of new-low breadth under a near-high index.

Round 1 (k2_c3_newlow_breadth.py) killed the long (cell below near-high-without-
breadth in every universe) and showed negative cell means on the full cache and
the ETF universe. This script: (1) rebuilds the full-cache universe STOCKS ONLY
(round 1's 'full' included leveraged/inverse ETFs, which sit at 52w lows exactly
when the index is near a high); (2) effective N of today's reading; (3) SHORT SPY
gate attribution h=1..10 on every universe; (4) count ladder on the ETF universe;
(5) era / midterm / dial splits; (6) concentration.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import json

OUT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
exec(open(OUT / "k2_c3_newlow_breadth.py").read().split("tape = json.load")[0].split("OUT = Path")[1].split("\n", 1)[1])  # NON_STOCK, ETF_CAND
tape = json.load(open(ROOT / "data" / "pitch_tape.json"))
TAPE_STOCKS = sorted(t for t in tape["tickers"] if t not in NON_STOCK)

mp = pd.read_parquet(PRICES_PATH, columns=["ticker", "date", "Close"])
mp["date"] = pd.to_datetime(mp["date"])
mp = mp.drop_duplicates(["ticker", "date"], keep="last")
W = mp.pivot(index="date", columns="ticker", values="Close").sort_index()
del mp
W = W.loc[W.index >= W["SPY"].dropna().index[0]]

sm = pd.read_parquet(ROOT / "data" / "sector_map.parquet")
print("sector values:", sm["sector"].value_counts().to_dict())
ETF_WORDS = {"", "ETF", "Etf", "Fund", "N/A", "None", "Unknown"}
stock_names = set(sm.loc[~sm["sector"].fillna("").isin(ETF_WORDS), "ticker"])
try:
    import strategy_config as sc
    excl = set(getattr(sc, "OLV_CAP_EXEMPT_ETFS", [])) | set(getattr(sc, "LEV3X_ALL", []))
except Exception as e:  # noqa
    print("strategy_config import failed:", e)
    excl = set()
excl |= NON_STOCK | set(ETF_CAND)
STOCKS = sorted(c for c in W.columns if c in stock_names and c not in excl)
leaks = [c for c in STOCKS if c in {"SQQQ", "SPXU", "SH", "PSQ", "SDS", "TZA", "UVXY", "VXX", "TLT", "XLU"}]
print(f"stock-only full universe {len(STOCKS)} (leak check {leaks}); tape stocks {len(TAPE_STOCKS)}")

ETFS = [t for t in ETF_CAND if t in W.columns]


def low_dist(cols):
    lo = {}
    for c in cols:
        s = W[c].dropna()
        if len(s) < 253:
            continue
        lo[c] = (s / s.rolling(252).min() - 1.0).reindex(W.index)
    return pd.DataFrame(lo)


D = {"tape": low_dist(TAPE_STOCKS), "stocks": low_dist(STOCKS), "etf": low_dist(ETFS)}
last = W.index[-1]
print(f"\n=== effective N of today's reading ({last.date()}) ===")
for k, d in D.items():
    names = d.columns[(d.loc[last] <= 0.01).values].tolist()
    print(f"{k}: {len(names)} of {int(d.loc[last].notna().sum())} within 1% of 252 low: {names[:40]}")

spy = W["SPY"]
spy_dd = rolling_on_valid(spy, lambda x: x / x.rolling(252).max() - 1.0)
px = W[["SPY"]].dropna().copy()
fr = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
fr.index = pd.to_datetime(fr.index)
dial = fr["63d"].rolling(10).mean().reindex(px.index).ffill(limit=3)
print(f"dial ma10(63d) last {dial.iloc[-1]:.1f}")


def pit(s, lb=252):
    return rolling_on_valid(s, lambda x: x.rolling(lb).rank(pct=True) * 100.0)


def share(d, x=0.01, min_n=30):
    n = d.notna().sum(axis=1)
    s = (d <= x).sum(axis=1) / n.replace(0, np.nan)
    return s[n >= min_n]


sh = {"tape": share(D["tape"]), "stocks": share(D["stocks"]),
      "etf": share(D["etf"], min_n=13)}
for k, s in sh.items():
    print(f"{k}: today share {100*s.iloc[-1]:.2f}% pit252 {pit(s).iloc[-1]:.1f} pit504 {pit(s,504).iloc[-1]:.1f}")

dd = spy_dd.reindex(px.index)
H = (1, 2, 3, 5, 10)


def short_row(mask, h, lab):
    r = -vehicle_ret(px, [("SPY", 1.0)], h, 1)
    v = r.notna()
    m = mask.reindex(px.index, fill_value=False).fillna(False).astype(bool) & v
    ep = declusters(px.index[m], h, px.index)
    s = summarize(r.loc[ep].values, lab)
    if s["n"]:
        w = int((r.loc[ep] > 0).sum())
        s["rec"] = f"{w}-{s['n']-w}"
        s["sign_p"] = round(sign_test(w, s["n"]), 4)
        s["h"] = h
    return s, ep, r


print("\n=== SHORT SPY gate attribution (episode means, SHORT P&L %) ===")
att = []
for k, s in sh.items():
    p = pit(s).reindex(px.index)
    near = dd >= -0.05
    cells = {"CELL near5 & pit>=90": near & (p >= 90),
             "near5 & pit<90 (decisive ctrl)": near & (p < 90),
             "pit>=90 alone": p >= 90,
             "near5 alone": near,
             "all days": dd.notna()}
    for h in (1, 5, 10):
        for lab, m in cells.items():
            row, _, _ = short_row(m, h, f"{k} {lab}")
            att.append(row)
show(att)

print("\n=== neighbours (SHORT P&L, h=5, episodes) ===")
nb = []
for k, d in D.items():
    for x in (0.005, 0.01, 0.02):
        s = share(d, x, 13 if k == "etf" else 30)
        for lb in (252, 504):
            p = pit(s, lb).reindex(px.index)
            for thr in (90, 95, 98):
                for idd in (-0.03, -0.05, -0.07):
                    row, _, _ = short_row((dd >= idd) & (p >= thr), 5, f"{k} x{x} lb{lb} thr{thr} idx{idd}")
                    nb.append(row)
NB = pd.DataFrame(nb)
for k in D:
    sub = NB[NB["label"].str.startswith(k + " ")]
    print(f"{k}: {len(sub)} variants, positive-short share {(sub['mean_pct']>0).mean():.2f}, "
          f"median {sub['mean_pct'].median():+.3f}%, min {sub['mean_pct'].min():+.3f}, max {sub['mean_pct'].max():+.3f}")
show(NB[NB["label"].str.contains("x0.01 lb252")].to_dict("records"), "1% / lb252 rows")

print("\n=== ETF universe COUNT ladder with SPY within 5% of high (SHORT P&L h=5) ===")
cnt = (D["etf"] <= 0.01).sum(axis=1).reindex(px.index)
lad = []
for c in (0, 1, 2, 3):
    m = (dd >= -0.05) & ((cnt == c) if c < 3 else (cnt >= 3))
    row, _, _ = short_row(m, 5, f"count {'>=3' if c == 3 else c}")
    lad.append(row)
show(lad)
print("stock-universe share-decile ladder near5 (SHORT h=5):")
p = pit(sh["stocks"]).reindex(px.index)
lad = []
for lo_, hi_ in ((0, 50), (50, 80), (80, 90), (90, 95), (95, 101)):
    row, _, _ = short_row((dd >= -0.05) & (p >= lo_) & (p < hi_), 5, f"pit [{lo_},{hi_})")
    lad.append(row)
show(lad)

print("\n=== splits of the canonical cells (SHORT h=5, episodes) ===")
for k in ("tape", "stocks", "etf"):
    p = pit(sh[k]).reindex(px.index)
    m = (dd >= -0.05) & (p >= 90)
    row, ep, r = short_row(m, 5, k)
    v = r.loc[ep]
    yrs = ep.year
    out = [row,
           summarize(v[yrs < 2018].values, f"{k} pre-2018"), summarize(v[yrs >= 2018].values, f"{k} 2018+"),
           summarize(v[(yrs % 4) == 2].values, f"{k} midterm"), summarize(v[(yrs % 4) != 2].values, f"{k} non-midterm"),
           summarize(v[(dial.loc[ep] >= 70).values].values, f"{k} dial>=70"),
           summarize(v[(dial.loc[ep] < 70).values].values, f"{k} dial<70")]
    show(out, f"{k}")
    print("  ", cluster_note(ep, v.values))
    fl = event_in_window(ep, px.index, 5, 1, ("fomc_decision",))
    print(f"   fomc in hold: {summarize(v[fl].values)['mean_pct'] if fl.any() else float('nan'):+.3f}% n={int(fl.sum())} | "
          f"out {summarize(v[~fl].values)['mean_pct']:+.3f}% n={int((~fl).sum())}")
