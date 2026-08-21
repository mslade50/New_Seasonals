"""C7 round 1: LONG the four most-washed bank names on a 5d breadth washout
with the 63-day trend BROKEN (watchlist W10's untested long leg).

Order of attack, per the arming note:
  1. alphabetical placebo on the LONG side (the explicitly untested leg)
  2. reference class over 12 industry groups (already 2-for-2 as a killer)
  3. the 70 threshold walk (today's median 63d rank is 69.8, a knife edge)
  4. basket vs simply buying KRE / XLF
  5. cost
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

pd.set_option("display.width", 250)

# today's stated complex (13 incl. the two ETFs) and yesterday's 11 singles
BANKS13 = ["JPM", "BAC", "C", "WFC", "GS", "MS", "USB", "PNC", "TFC", "SCHW",
           "STT", "KRE", "XLF"]
BANKS_SINGLE = [t for t in BANKS13 if t not in ("KRE", "XLF")]
BANKS_Y = ["JPM", "BAC", "C", "WFC", "GS", "MS", "USB", "KEY", "RF", "STT", "SCHW"]

GROUPS = {
    "banks": BANKS_SINGLE,
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
ALL = sorted({t for g in GROUPS.values() for t in g} | set(BANKS13) | set(BANKS_Y) | {"SPY"})
raw = load_prices(ALL)
d = raw["SPY"]["Close"].dropna().index
have = [t for t in ALL if t in raw]
close = pd.DataFrame({t: raw[t]["Close"].reindex(d) for t in have})
R5 = pd.DataFrame({t: pct_rank(raw[t]["Close"].dropna(), 5).reindex(d) for t in have})
R63 = pd.DataFrame({t: pct_rank(raw[t]["Close"].dropna(), 63).reindex(d) for t in have})
HS = (1, 2, 3, 5, 10)
FWD = {h: pd.DataFrame({t: fwd_lag(close[t], h) for t in close}) for h in HS}
SPY_F = {h: fwd_lag(close["SPY"], h) for h in HS}


def group_mask(names, brd=0.70, med_lo=None, med_hi=70.0):
    names = [t for t in names if t in R5.columns]
    r5, r63 = R5[names], R63[names]
    nv = r5.notna().sum(axis=1)
    b = (r5 <= 20).sum(axis=1) / nv.replace(0, np.nan)
    m63 = r63.median(axis=1)
    m = (b >= brd) & (nv >= max(4, int(0.7 * len(names))))
    if med_hi is not None:
        m = m & (m63 < med_hi)
    if med_lo is not None:
        m = m & (m63 >= med_lo)
    return m.fillna(False).astype(bool), b, m63


# ------------------------------------------------------------------ 0. today
m_brk, brd, med = group_mask(BANKS_SINGLE)
m_brk13, brd13, med13 = group_mask(BANKS13)
print("===== 0. live state =====")
print(f" bar {d[-1].date()}")
print(f" 11 singles : breadth {100*brd.iloc[-1]:.1f}%  median63 {med.iloc[-1]:.1f}  armed={bool(m_brk.iloc[-1])}")
print(f" 13 w/ ETFs : breadth {100*brd13.iloc[-1]:.1f}%  median63 {med13.iloc[-1]:.1f}  armed={bool(m_brk13.iloc[-1])}")
epi = declusters(d[m_brk], 10, d)
print(f" BROKEN episodes (gap10): {len(epi)}  years {sorted(set(epi.year))}")
print(f" day count {int(m_brk.sum())}")


def basket_fwd(dt, names, h, k=4, pick="washed"):
    ranks = R5.loc[dt, names].dropna()
    if len(ranks) < 8:
        return None
    if pick == "washed":
        sel = ranks.nsmallest(k).index.tolist()
    elif pick == "alpha":
        sel = sorted(names)[:k]
    elif pick == "all":
        sel = ranks.index.tolist()
    elif pick == "strong":
        sel = ranks.nlargest(k).index.tolist()
    f = FWD[h].loc[dt, sel]
    if f.isna().any():
        return None
    return float(f.mean())


# ------------------------------------------------------------ 1. alphabetical placebo, LONG
print("\n===== 1. ALPHABETICAL PLACEBO, LONG SIDE (the untested leg) =====")
alpha4 = sorted(BANKS_SINGLE)[:4]
print(f" alphabetically-first four: {alpha4}")
rows = []
for h in HS:
    w, a, al, st, sp = [], [], [], [], []
    for dt in epi:
        if dt not in FWD[h].index or np.isnan(SPY_F[h].loc[dt]):
            continue
        rw = basket_fwd(dt, BANKS_SINGLE, h, 4, "washed")
        ra = basket_fwd(dt, BANKS_SINGLE, h, 4, "alpha")
        rl = basket_fwd(dt, BANKS_SINGLE, h, 4, "all")
        rs = basket_fwd(dt, BANKS_SINGLE, h, 4, "strong")
        if None in (rw, ra, rl, rs):
            continue
        w.append(rw); a.append(ra); al.append(rl); st.append(rs)
        sp.append(float(SPY_F[h].loc[dt]))
    w, a, al, st, sp = map(np.array, (w, a, al, st, sp))
    if not len(w):
        continue
    rows.append({"h": h, "n": len(w),
                 "washed4_pct": round(100 * w.mean(), 3),
                 "alpha4_pct": round(100 * a.mean(), 3),
                 "allnames_pct": round(100 * al.mean(), 3),
                 "strongest4_pct": round(100 * st.mean(), 3),
                 "SPY_pct": round(100 * sp.mean(), 3),
                 "washed-alpha_pp": round(100 * (w - a).mean(), 3),
                 "washed-SPY_pp": round(100 * (w - sp).mean(), 3),
                 "alpha-SPY_pp": round(100 * (a - sp).mean(), 3),
                 "washed hit": round(100 * (w > 0).mean(), 1),
                 "signp_vs_alpha": round(sign_test(int((w > a).sum()), len(w)), 4)})
show(rows, "washed-4 vs alphabetical-4 vs all-11 vs strongest-4, LONG, episodes")

# ------------------------------------------------------------ 2. reference class
print("\n===== 2. REFERENCE CLASS: identical BROKEN-trend long rule, 12 groups =====")
for h in (3, 5, 10):
    rows = []
    for g, names in GROUPS.items():
        gm, _, _ = group_mask(names)
        ge = declusters(d[gm], 10, d)
        ex, ba = [], []
        for dt in ge:
            if dt not in FWD[h].index or np.isnan(SPY_F[h].loc[dt]):
                continue
            rw = basket_fwd(dt, [t for t in names if t in R5.columns], h, 4, "washed")
            if rw is None:
                continue
            ex.append(rw - float(SPY_F[h].loc[dt]))
            ba.append(rw)
        if len(ex) < 3:
            rows.append({"group": g, "n": len(ex)})
            continue
        ex = np.array(ex)
        rows.append({"group": g, "n": len(ex),
                     "excess_vs_SPY_pp": round(100 * ex.mean(), 3),
                     "se_pp": round(100 * ex.std(ddof=1) / np.sqrt(len(ex)), 3),
                     "raw_pct": round(100 * np.mean(ba), 3),
                     "hit": round(100 * (ex > 0).mean(), 1)})
    df = pd.DataFrame(rows).dropna(subset=["excess_vs_SPY_pp"])
    ex = df["excess_vs_SPY_pp"].values / 100
    se = df["se_pp"].values / 100
    wgt = 1 / se**2
    fe = float((wgt * ex).sum() / wgt.sum())
    Q = float((wgt * (ex - fe) ** 2).sum())
    dfree = len(ex) - 1
    I2 = max(0.0, 100 * (Q - dfree) / Q) if Q > 0 else 0.0
    disp = float(ex.std(ddof=1) / se.mean())
    bank_ex = float(df.loc[df["group"] == "banks", "excess_vs_SPY_pp"].iloc[0]) / 100
    rank = int((df["excess_vs_SPY_pp"] >= 100 * bank_ex).sum())
    # P(max group excess >= banks) under the common-effect null, parametric
    rng = np.random.default_rng(42)
    sim = rng.normal(fe, se[None, :], size=(20000, len(ex)))
    pmax = float((sim.max(axis=1) >= bank_ex).mean())
    show(rows, f"h={h} groups")
    print(f"  h={h}: fixed-effect common excess {100*fe:+.3f}pp | Cochran Q {Q:.2f} on {dfree} df"
          f" | I^2 {I2:.1f}% | cross-group sd {100*ex.std(ddof=1):.3f}pp vs mean SE {100*se.mean():.3f}pp"
          f" (ratio {disp:.2f}) | banks rank {rank} of {len(ex)} | P(max >= banks) = {pmax:.3f}")

# ------------------------------------------------------------ 3. the 70 knife edge
print("\n===== 3. MEDIAN-63d THRESHOLD WALK (today = 69.8) =====")
rows = []
for thr in (50, 55, 60, 65, 70, 75, 80, None):
    gm, _, _ = group_mask(BANKS_SINGLE, med_hi=thr)
    ge = declusters(d[gm], 10, d)
    for h in (3, 5, 10):
        v, sp = [], []
        for dt in ge:
            if dt not in FWD[h].index or np.isnan(SPY_F[h].loc[dt]):
                continue
            rw = basket_fwd(dt, BANKS_SINGLE, h, 4, "washed")
            if rw is None:
                continue
            v.append(rw); sp.append(float(SPY_F[h].loc[dt]))
        if len(v) < 3:
            continue
        v, sp = np.array(v), np.array(sp)
        rows.append({"med63 <": thr if thr else "no gate", "h": h, "n": len(v),
                     "washed4_pct": round(100 * v.mean(), 3),
                     "excess_vs_SPY_pp": round(100 * (v - sp).mean(), 3),
                     "hit": round(100 * (v > 0).mean(), 1)})
# also the INTACT complement for attribution
for h in (3, 5, 10):
    gm, _, _ = group_mask(BANKS_SINGLE, med_lo=70.0, med_hi=None)
    ge = declusters(d[gm], 10, d)
    v, sp = [], []
    for dt in ge:
        if dt not in FWD[h].index or np.isnan(SPY_F[h].loc[dt]):
            continue
        rw = basket_fwd(dt, BANKS_SINGLE, h, 4, "washed")
        if rw is None:
            continue
        v.append(rw); sp.append(float(SPY_F[h].loc[dt]))
    if len(v) >= 3:
        v, sp = np.array(v), np.array(sp)
        rows.append({"med63 <": "INTACT >=70", "h": h, "n": len(v),
                     "washed4_pct": round(100 * v.mean(), 3),
                     "excess_vs_SPY_pp": round(100 * (v - sp).mean(), 3),
                     "hit": round(100 * (v > 0).mean(), 1)})
show(rows, "gate walk, washed-4 long")

# ------------------------------------------------------------ 4. basket vs ETF
print("\n===== 4. IS THE BASKET JUST KRE/XLF? =====")
rows = []
for h in HS:
    v, kre, xlf, sp = [], [], [], []
    for dt in epi:
        if dt not in FWD[h].index or np.isnan(SPY_F[h].loc[dt]):
            continue
        rw = basket_fwd(dt, BANKS_SINGLE, h, 4, "washed")
        if rw is None or np.isnan(FWD[h].loc[dt, "KRE"]) or np.isnan(FWD[h].loc[dt, "XLF"]):
            continue
        v.append(rw); kre.append(float(FWD[h].loc[dt, "KRE"]))
        xlf.append(float(FWD[h].loc[dt, "XLF"])); sp.append(float(SPY_F[h].loc[dt]))
    if len(v) < 3:
        continue
    v, kre, xlf, sp = map(np.array, (v, kre, xlf, sp))
    rows.append({"h": h, "n": len(v),
                 "washed4_pct": round(100 * v.mean(), 3),
                 "KRE_pct": round(100 * kre.mean(), 3),
                 "XLF_pct": round(100 * xlf.mean(), 3),
                 "SPY_pct": round(100 * sp.mean(), 3),
                 "corr(w,KRE)": round(float(np.corrcoef(v, kre)[0, 1]), 3),
                 "corr(w,XLF)": round(float(np.corrcoef(v, xlf)[0, 1]), 3),
                 "w-KRE_pp": round(100 * (v - kre).mean(), 3),
                 "w-XLF_pp": round(100 * (v - xlf).mean(), 3)})
show(rows, "four-name basket vs the two ETFs")

# ------------------------------------------------------------ 5. era + concentration + cost
print("\n===== 5. ERA / CONCENTRATION on the pitched h=5 washed-4 long =====")
h = 5
dts, vals, sps = [], [], []
for dt in epi:
    if dt not in FWD[h].index or np.isnan(SPY_F[h].loc[dt]):
        continue
    rw = basket_fwd(dt, BANKS_SINGLE, h, 4, "washed")
    if rw is None:
        continue
    dts.append(dt); vals.append(rw); sps.append(float(SPY_F[h].loc[dt]))
dts = pd.DatetimeIndex(dts); vals = np.array(vals); sps = np.array(sps)
show([summarize(vals, "washed4 raw"), summarize(vals - sps, "washed4 excess vs SPY")],
     "h=5 episodes")
show(era_split(dts, vals), "era split, raw")
show(era_split(dts, vals - sps), "era split, excess")
print(" concentration raw   :", cluster_note(dts, vals))
print(" concentration excess:", cluster_note(dts, vals - sps))
print(f" bootstrap P(mean<=0) raw {bootstrap_p_le0(vals):.3f}  excess {bootstrap_p_le0(vals-sps):.3f}")
print(f" record raw {int((vals>0).sum())}-{int((vals<=0).sum())} sign p {sign_test(int((vals>0).sum()), len(vals)):.4f}")
print(f" cost: 4 single-name legs ~10 bps rt = 40 bps vs mean {100*vals.mean()*100:.1f} bps"
      f" -> {vals.mean()*10000/40:.1f}x ; one ETF at 2 bps -> KRE/XLF above")
print(" episode dates:", ", ".join(str(x.date()) for x in dts))

# ------------------------------------------------------------ 6. tape over-selection
sma200 = rolling_on_valid(close["SPY"], lambda x: x.rolling(200).mean())
abv = (close["SPY"] > sma200)
print(f"\n===== 6. tape over-selection: trigger days below SPY 200d "
      f"{100*(~abv.loc[d[m_brk]]).mean():.1f}% vs base {100*(~abv.loc[d]).mean():.1f}% =====")

# ------------------------------------------------------------ 7. universe definition fragility
print("\n===== 7. universe definition (11 singles vs 13 with ETFs vs yesterday's 11) =====")
for nm, uni in [("today 11 singles", BANKS_SINGLE), ("today 13 incl KRE/XLF", BANKS13),
                ("2026-08-20's 11", BANKS_Y)]:
    gm, b_, m_ = group_mask(uni)
    ge = declusters(d[gm], 10, d)
    sel_pool = [t for t in uni if t in R5.columns]
    v, sp = [], []
    for dt in ge:
        if dt not in FWD[5].index or np.isnan(SPY_F[5].loc[dt]):
            continue
        rw = basket_fwd(dt, sel_pool, 5, 4, "washed")
        if rw is None:
            continue
        v.append(rw); sp.append(float(SPY_F[5].loc[dt]))
    v, sp = np.array(v), np.array(sp)
    print(f" {nm:<24} armed_today={bool(gm.iloc[-1])} breadth {100*b_.iloc[-1]:.1f}% "
          f"med63 {m_.iloc[-1]:.1f} | h=5 n={len(v)} raw {100*v.mean():+.3f}% "
          f"excess {100*(v-sp).mean():+.3f}pp")
