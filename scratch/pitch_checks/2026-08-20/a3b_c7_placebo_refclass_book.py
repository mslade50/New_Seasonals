"""C7 round 2: the alphabetical placebo, the industry reference class, vehicle
cost, and what the systematic book does when it meets this state.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

pd.set_option("display.width", 240)

GROUPS = {
    "banks": ["JPM", "BAC", "C", "WFC", "GS", "MS", "USB", "KEY", "RF", "STT", "SCHW"],
    "insurance": ["AIG", "ALL", "HIG", "MET", "PGR", "TRV", "AON", "MRSH"],
    "utilities": ["AEP", "CMS", "CNP", "D", "DTE", "DUK", "ED", "EIX", "ETR", "EXC",
                  "FE", "NEE", "PCG", "PEG", "PNW", "PPL", "SO", "SRE"],
    "staples": ["ADM", "CAG", "CL", "CPB", "GIS", "HRL", "HSY", "KMB", "KO", "KR",
                "MO", "PEP", "PG", "SYY", "TAP", "TSN", "WMT"],
    "semis": ["ADI", "AMAT", "AMD", "AVGO", "INTC", "MU", "NVDA", "QCOM", "TXN"],
    "pharma": ["ABT", "AMGN", "BMY", "GILD", "JNJ", "LLY", "MRK", "PFE", "REGN"],
    "medtech": ["BAX", "BDX", "MDT", "SYK", "TMO", "CVS", "HUM", "UNH"],
    "machinery": ["CAT", "DE", "DOV", "EMR", "GD", "HON", "ITW", "MMM", "PH", "ROK", "SNA", "SWK"],
    "retail": ["COST", "HD", "LOW", "ROST", "TGT", "TJX", "SBUX", "MCD", "NKE"],
    "energy": ["COP", "CVX", "EOG", "HAL", "OXY", "SLB", "VLO", "WMB", "XOM"],
    "transports": ["CSX", "FDX", "LUV", "NSC", "UNP"],
    "software": ["ADBE", "ADSK", "CRM", "CSCO", "GOOG", "IBM", "META", "MSFT", "ORCL"],
}
ALL = sorted({t for g in GROUPS.values() for t in g} | {"SPY", "XLF", "KRE"})
raw = load_prices(ALL)
d = raw["SPY"]["Close"].dropna().index
close = pd.DataFrame({t: raw[t]["Close"].reindex(d) for t in raw if t in raw})
R5 = pd.DataFrame({t: pct_rank(raw[t]["Close"].dropna(), 5).reindex(d) for t in ALL if t in raw})
R63 = pd.DataFrame({t: pct_rank(raw[t]["Close"].dropna(), 63).reindex(d) for t in ALL if t in raw})
FWD = {h: pd.DataFrame({t: fwd_lag(close[t], h) for t in close}) for h in (3, 5, 10)}
SPY_F = {h: fwd_lag(close["SPY"], h) for h in (3, 5, 10)}


def group_mask(names, brd=0.70, med=70.0):
    r5 = R5[names]
    r63 = R63[names]
    nv = r5.notna().sum(axis=1)
    b = (r5 <= 20).sum(axis=1) / nv.replace(0, np.nan)
    m63 = r63.median(axis=1)
    return ((b >= brd) & (m63 >= med) & (nv >= max(4, int(0.7 * len(names))))).fillna(False).astype(bool), b, m63


BANKS = GROUPS["banks"]
m_int, brd, med = group_mask(BANKS)
epi = declusters(d[m_int], 10, d)
print(f"banks INTACT episodes: {len(epi)}  years {sorted(set(epi.year))}")

# ------------------------------------------------------------- 1. alphabetical placebo
print("\n===== 1. ALPHABETICAL PLACEBO (registry: 6-for-6 as a killer) =====")
alpha4 = sorted(BANKS)[:4]
print(f" alphabetically first four: {alpha4}")
for h in (3, 5, 10):
    sig_rows, alp_rows, all_rows = [], [], []
    for dt in epi:
        if dt not in FWD[h].index:
            continue
        ranks = R5.loc[dt, BANKS].dropna()
        if len(ranks) < 8:
            continue
        washed4 = ranks.nsmallest(4).index.tolist()
        f = FWD[h].loc[dt]
        s = SPY_F[h].loc[dt]
        if np.isnan(s):
            continue
        sig_rows.append(np.nanmean([-(f[t] - s) for t in washed4]))
        alp_rows.append(np.nanmean([-(f[t] - s) for t in alpha4]))
        all_rows.append(np.nanmean([-(f[t] - s) for t in BANKS if not np.isnan(f.get(t, np.nan))]))
    sg, al, aw = np.array(sig_rows), np.array(alp_rows), np.array(all_rows)
    print(f" h={h:>2} N={len(sg)}  SHORT market-relative:  signal-picked 4 {100*np.nanmean(sg):+.3f}% "
          f"(hit {100*np.nanmean(sg>0):.1f}%)   ALPHABETICAL 4 {100*np.nanmean(al):+.3f}% "
          f"(hit {100*np.nanmean(al>0):.1f}%)   all 11 {100*np.nanmean(aw):+.3f}%")
    print(f"       selection premium (signal - alphabetical) = {100*np.nanmean(sg-al):+.3f}pp")

# ------------------------------------------------------------- 2. reference class
print("\n\n===== 2. REFERENCE CLASS: the identical rule on 12 industry groups =====")
for h in (3, 5):
    rows = []
    for gname, names in GROUPS.items():
        mm, _, _ = group_mask(names)
        f = FWD[h]
        ew = pd.Series({dt: -np.nanmean([f.loc[dt, t] for t in names if t in f.columns])
                        for dt in d if dt in f.index})
        ew = ew.dropna()
        e = declusters(pd.DatetimeIndex(d[mm]).intersection(ew.index), 10, ew.index)
        if len(e) < 5:
            rows.append({"group": gname, "n": len(e)})
            continue
        v = ew.loc[e].values
        base = ew.values
        se = v.std(ddof=1) / np.sqrt(len(v))
        rows.append({"group": gname, "n": len(v), "short_mean_pct": round(100 * v.mean(), 3),
                     "own_drift": round(100 * base.mean(), 3),
                     "excess_pp": round(100 * (v.mean() - base.mean()), 3),
                     "se_pp": round(100 * se, 3),
                     "hit": round(100 * (v > 0).mean(), 1)})
    df = pd.DataFrame(rows)
    print(f"\n h={h}:")
    print(df.to_string(index=False))
    sub = df.dropna(subset=["excess_pp"])
    x = sub["excess_pp"].values
    s = sub["se_pp"].values
    w = 1 / s ** 2
    common = (w * x).sum() / w.sum()
    Q = (w * (x - common) ** 2).sum()
    dfree = len(x) - 1
    print(f"  fixed-effect common excess = {common:+.3f}pp;  Cochran Q = {Q:.2f} on {dfree} df")
    print(f"  observed cross-group sd of excess = {x.std(ddof=1):.3f}pp against a mean sampling SE of "
          f"{s.mean():.3f}pp  (dispersion ratio {x.std(ddof=1)/s.mean():.2f})")
    bk = float(sub.loc[sub["group"] == "banks", "excess_pp"].iloc[0])
    print(f"  banks excess {bk:+.3f}pp, rank {int((x > bk).sum()) + 1} of {len(x)}")
    rng = np.random.default_rng(7)
    draws = rng.normal(loc=common, scale=s[None, :], size=(20000, len(s)))
    print(f"  permutation under the common effect: P(max group excess >= banks) = "
          f"{(draws.max(axis=1) >= bk).mean():.3f}")

# ------------------------------------------------------------- 3. vehicle cost
print("\n\n===== 3. VEHICLE COST =====")
print(" XLF ~2 bps round trip, KRE ~5 bps, a 4-name single-stock basket ~10 bps"
      " (4 legs, 2.5 bps each), the KRE/XLF pair ~7 bps.")
for h in (3, 5, 10):
    for vn, legs, cost in [("XLF short", [("XLF", -1.0)], 2.0),
                           ("KRE short", [("KRE", -1.0)], 5.0),
                           ("KRE short vs XLF long", [("KRE", -1.0), ("XLF", 1.0)], 7.0)]:
        ret = vehicle_ret(close, legs, h)
        e = pd.DatetimeIndex(epi).intersection(ret.dropna().index)
        v = ret.loc[e].values
        bps = 100 * v.mean() * 100
        print(f" h={h:>2} {vn:<24} N={len(v):>2} mean {bps:+7.1f} bps  -> {bps/cost:+.1f}x its "
              f"{cost} bps round trip  hit {100*(v>0).mean():.0f}%")

# ------------------------------------------------------------- 4. book overlap
print("\n\n===== 4. BOOK OVERLAP =====")
led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
led["Signal Date"] = pd.to_datetime(led["Signal Date"])
trig = set(pd.DatetimeIndex(d[m_int]))
FIN = set(BANKS) | {"XLF", "KRE"}
on = led[led["Signal Date"].isin(trig)]
fin = led[led["Ticker"].isin(FIN)]
fin_on = fin[fin["Signal Date"].isin(trig)]
print(f" trigger days = {len(trig)} of {len(d)} sessions ({100*len(trig)/len(d):.2f}%)")
print(f" ALL book trades signalled on trigger days: {len(on)} ({100*len(on)/len(led):.2f}% of the ledger)")
if len(on):
    print(on.groupby("Direction")[["PnL_flat_750k", "R_Multiple"]].agg(["count", "mean"]).to_string())
print(f"\n book trades in the bank complex ever: {len(fin)};  on trigger days: {len(fin_on)}")
if len(fin_on):
    print(fin_on.groupby(["Strategy", "Direction"])["R_Multiple"].agg(["count", "mean"]).to_string())
    print(f" enrichment: {100*len(fin_on)/max(1,len(fin)):.2f}% of bank-complex trades against a "
          f"{100*len(trig)/len(d):.2f}% day share")
