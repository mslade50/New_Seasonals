"""C1x rounds 1-2: CONTINUATION short after a non-earnings, intraday-led drop of
>= k Wilder ATR (prior-day ATR) with SPY 1d > -1%. Short the name hedged at
trailing-252 beta against its sector SPDR (banks + other financials vs XLF),
lag=1, h=1..5. Returns reported as the SHORT's P&L.

Family: same rule on every single name in the tape grouped by sector (banks
split out). Fixed-effect common excess, Cochran Q, P(max group >= banks).
Neighbours: ATR 1.5/2/2.5, intraday- vs gap-led, earnings vs not, sector vs
SPY hedge. Era, concentration, year count. GLW failed-thrust (r5@t-1>=80,
>=2 ATR) decomposition against the same continuation family.
Direction was named before data (counter-story) but is one of two signs:
sign tests are reported TWO-SIDED."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch_lab import *  # noqa
from k2_common import *  # noqa

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 260)
BANKS = ["JPM", "BAC", "C", "WFC", "GS", "MS", "BNY", "STT"]
SN = single_names()
for _b in BANKS:
    SN[_b] = "Financial Services"
NAMES = sorted(t for t, s in SN.items() if s in SECTOR_ETF)
ETFS_USED = sorted(set(SECTOR_ETF.values()))
px, cal, P = load_panels(NAMES + ETFS_USED + ["SPY"])
NAMES = [t for t in NAMES if t in px]
D = derive(px, cal)
C = P["Close"]
ret1, sh, gap = D["ret1"], D["shock"], D["gap"]
spy1 = ret1["SPY"]


def group_of(t: str) -> str:
    if t in BANKS:
        return "Banks"
    s = SN[t]
    return "Fin ex-banks" if s == "Financial Services" else s


GROUP = {t: group_of(t) for t in NAMES}
HEDGE = {t: SECTOR_ETF[SN[t]] for t in NAMES}
print("names:", len(NAMES), " groups:", pd.Series(GROUP).value_counts().to_dict())

# betas vs each name's own sector ETF (fallback SPY where the ETF is missing, e.g. XLC pre-2018)
BETA_SEC, BETA_SPY = {}, {}
bspy = rolling_beta(ret1[NAMES + ["SPY"]], "SPY")
for etf in ETFS_USED:
    cols = [t for t in NAMES if HEDGE[t] == etf]
    b = rolling_beta(ret1[cols + [etf]], etf)
    for t in cols:
        BETA_SEC[t] = b[t]
for t in NAMES:
    BETA_SPY[t] = bspy[t]


def short_pair(h: int, hedge: str) -> pd.DataFrame:
    F = fwd_panel(C, h)
    out = {}
    for t in NAMES:
        if hedge == "sector":
            etf = HEDGE[t]
            leg = BETA_SEC[t] * F[etf]
            fb = BETA_SPY[t] * F["SPY"]
            leg = leg.where(leg.notna(), fb)
        else:
            leg = BETA_SPY[t] * F["SPY"]
        out[t] = -(F[t] - leg)
    return pd.DataFrame(out)


PAIRS = {(h, hd): short_pair(h, hd) for h in (1, 2, 3, 5) for hd in ("sector", "spy")}

ern = pd.read_parquet(ROOT / "data" / "earnings_calendar.parquet", columns=["ticker", "date"])
ern["date"] = pd.to_datetime(ern["date"])
EARN = pd.DataFrame(False, index=cal, columns=NAMES)
for j, t in enumerate(NAMES):
    pos = cal.searchsorted(pd.DatetimeIndex(ern.loc[ern["ticker"] == t, "date"]))
    for p in pos:
        if p < len(cal):
            EARN.iat[p, j] = True
            if p + 1 < len(cal):
                EARN.iat[p + 1, j] = True
GS = (gap[NAMES] / ret1[NAMES])
SPYOK = pd.DataFrame(np.broadcast_to((spy1 > -0.01).to_numpy()[:, None], (len(cal), len(NAMES))),
                     index=cal, columns=NAMES)
R5P = D["r5"][NAMES].shift(1)


def mask(k=1.5, earn="non", lead="intraday", spy=True, names=None) -> pd.DataFrame:
    m = sh[NAMES].le(-k)
    if spy:
        m &= SPYOK
    if earn == "non":
        m &= ~EARN
    elif earn == "only":
        m &= EARN
    if lead == "intraday":
        m &= GS < 0.5
    elif lead == "gap":
        m &= GS >= 0.5
    if names is not None:
        m &= pd.DataFrame(np.broadcast_to(np.isin(NAMES, names), m.shape), index=cal, columns=NAMES)
    return m.fillna(False)


def two_sided(w: int, n: int) -> float:
    return min(1.0, 2 * min(sign_test(w, n), sign_test(n - w, n)))


def stats(M: pd.DataFrame, h: int, hedge: str, label: str, names=None) -> dict:
    A = PAIRS[(h, hedge)]
    cols = names if names is not None else NAMES
    Mv = M[cols] & A[cols].notna()
    s = A[cols].where(Mv).mean(axis=1).dropna()
    ctrl_series = A[cols].mean(axis=1)
    xs = A.sub(A.mean(axis=1), axis=0)[cols].where(Mv).mean(axis=1).dropna()
    if len(s) == 0:
        return {"label": label, "n": 0}
    keep = declusters(s.index, max(h, 5), cal)
    v = s.loc[keep].values
    ctrl = float(ctrl_series.mean())
    r = summarize(v, label)
    w = int((v > 0).sum())
    r.update({"name_days": int(Mv.to_numpy().sum()), "dates": len(s), "rec": f"{w}-{len(v)-w}",
              "sign_p2": round(two_sided(w, len(v)), 4), "ctrl_pct": round(100 * ctrl, 3),
              "excess_pp": round(r["mean_pct"] - 100 * ctrl, 3),
              "xs_pp": round(100 * xs.loc[xs.index.intersection(keep)].mean(), 3)})
    yrs = s.loc[keep].groupby(keep.year).sum()
    r["pos_yrs"] = f"{int((yrs > 0).sum())}/{len(yrs)}"
    return r


def episodes(M, h, hedge, names):
    A = PAIRS[(h, hedge)]
    Mv = M[names] & A[names].notna()
    s = A[names].where(Mv).mean(axis=1).dropna()
    ctrl = float(A[names].mean(axis=1).mean())
    keep = declusters(s.index, max(h, 5), cal)
    return keep, s.loc[keep].values, ctrl


live = mask(1.5).loc[cal[-1]]
print("\nlive 09-14 (1.5 ATR, non-earn, intraday-led, SPY ok):", list(live[live].index))
for t in ["BAC", "GS", "MS", "BNY", "GLW", "ADI", "AMAT", "AMD"]:
    print(f"  {t}: shock {sh[t].iloc[-1]:+.2f}  gap share {GS[t].iloc[-1]:+.2f}  earn {bool(EARN[t].iloc[-1])}"
          f"  r5@t-1 {R5P[t].iloc[-1]:.1f}  hedge {HEDGE[t]} beta {BETA_SEC[t].iloc[-1]:.2f}")

BANKS_IN = [b for b in BANKS if b in NAMES]
NONFIN = [t for t in NAMES if GROUP[t] not in ("Banks", "Fin ex-banks")]

print("\n##### 1. THE CELL (short, sector-beta hedge) #####")
rows = []
for h in (1, 2, 3, 5):
    rows.append(stats(mask(1.5), h, "sector", f"h={h} BANKS 1.5 non-earn intraday", BANKS_IN))
    rows.append(stats(mask(1.5), h, "sector", f"h={h} BAC only", ["BAC"]))
    rows.append(stats(mask(1.5), h, "sector", f"h={h} NON-FIN family", NONFIN))
    rows.append(stats(mask(1.5), h, "sector", f"h={h} ALL single names", NAMES))
show(rows, "cell vs control (short P&L; ctrl = all-days short pair of the same names)")

print("\n##### 2. FAMILY heterogeneity (h=1,3,5; excess over own-group all-days short pair) #####")
groups = sorted(set(GROUP.values()))
for h in (1, 3, 5):
    rows, ms, ses = [], [], []
    for g in groups:
        cols = [t for t in NAMES if GROUP[t] == g]
        keep, v, ctrl = episodes(mask(1.5), h, "sector", cols)
        if len(v) < 5:
            continue
        ex = v - ctrl
        se = ex.std(ddof=1) / np.sqrt(len(ex))
        w = int((v > 0).sum())
        rows.append({"group": g, "names": len(cols), "n_ep": len(v), "mean_pct": 100 * v.mean(),
                     "excess_pp": 100 * ex.mean(), "se_pp": 100 * se, "t": ex.mean() / se,
                     "rec": f"{w}-{len(v)-w}", "sign_p2": round(two_sided(w, len(v)), 4)})
        ms.append(ex.mean()); ses.append(se)
    ms, ses = np.array(ms), np.array(ses)
    wts = 1 / ses ** 2
    fe = (wts * ms).sum() / wts.sum()
    fe_se = 1 / np.sqrt(wts.sum())
    Q = (wts * (ms - fe) ** 2).sum()
    df = len(ms) - 1
    I2 = max(0.0, (Q - df) / Q) if Q > 0 else 0.0
    df_rows = pd.DataFrame(rows)
    bank_obs = df_rows.loc[df_rows["group"] == "Banks", "excess_pp"].iloc[0] / 100
    rng = np.random.default_rng(7)
    sims = rng.normal(fe, ses, size=(20000, len(ms))).max(axis=1)
    pmax = float((sims >= bank_obs).mean())
    rank = int((df_rows["excess_pp"] > 100 * bank_obs).sum()) + 1
    show(rows, f"h={h} groups")
    print(f"  FE common excess {100*fe:+.3f}pp (se {100*fe_se:.3f}, z {fe/fe_se:+.2f}); Cochran Q {Q:.2f} on {df} df;"
          f" I2 {100*I2:.1f}%; banks rank {rank} of {len(ms)}; P(max group >= banks | homogeneous) {pmax:.3f}")
    ex_b = ms[df_rows['group'].values != 'Banks']
    wb = wts[df_rows['group'].values != 'Banks']
    print(f"  FE ex-banks {100*(wb*ex_b).sum()/wb.sum():+.3f}pp")

print("\n##### 3. neighbours (banks and non-fin family; short P&L) #####")
for h in (1, 5):
    rows = []
    for nm, cols in [("BANKS", BANKS_IN), ("NON-FIN", NONFIN)]:
        for k in (1.5, 2.0, 2.5):
            rows.append(stats(mask(k), h, "sector", f"{nm} k={k} non-earn intraday", cols))
        rows.append(stats(mask(1.5, lead="gap"), h, "sector", f"{nm} 1.5 non-earn GAP-led", cols))
        rows.append(stats(mask(1.5, lead="any"), h, "sector", f"{nm} 1.5 non-earn any-lead", cols))
        rows.append(stats(mask(1.5, earn="only", lead="any"), h, "sector", f"{nm} 1.5 EARNINGS any-lead", cols))
        rows.append(stats(mask(1.5), h, "spy", f"{nm} 1.5 non-earn intraday, SPY-beta hedge", cols))
        rows.append(stats(mask(1.5, spy=False), h, "sector", f"{nm} 1.5 non-earn intraday, no SPY gate", cols))
    show(rows, f"neighbours h={h}")

print("\n##### 4. era + concentration (h=5, sector hedge) #####")
for nm, cols in [("BANKS", BANKS_IN), ("NON-FIN", NONFIN), ("ALL", NAMES)]:
    keep, v, ctrl = episodes(mask(1.5), 5, "sector", cols)
    show(era_split(keep, v), f"{nm} era split (short P&L, episodes)")
    print(f"  {nm}: {cluster_note(keep, v)}")
    yrs = pd.Series(v, index=keep).groupby(keep.year).sum()
    print(f"  {nm}: positive years {int((yrs > 0).sum())}/{len(yrs)}; bootstrap P(mean<=0) {bootstrap_p_le0(v):.3f}")

print("\n##### 5. GLW: failed thrust >= 2 ATR short vs the continuation family (sector hedge) #####")
for h in (1, 3, 5):
    rows = []
    base2 = mask(2.0, earn="any", lead="any")
    thrust = (R5P >= 80).fillna(False)
    rows.append(stats(base2 & thrust, h, "sector", "ALL >=2ATR r5>=80 (any earn/lead)"))
    rows.append(stats(base2 & ~thrust, h, "sector", "ALL >=2ATR r5<80 (any earn/lead)"))
    rows.append(stats(mask(2.0) & thrust, h, "sector", "ALL >=2ATR r5>=80 non-earn intraday"))
    rows.append(stats(mask(2.0) & ~thrust, h, "sector", "ALL >=2ATR r5<80 non-earn intraday"))
    rows.append(stats(mask(2.0, lead="gap") & thrust, h, "sector", "ALL >=2ATR r5>=80 non-earn GAP-led"))
    rows.append(stats(mask(2.0, lead="gap") & ~thrust, h, "sector", "ALL >=2ATR r5<80 non-earn GAP-led"))
    rows.append(stats(base2 & thrust, h, "spy", "ALL >=2ATR r5>=80, SPY hedge"))
    rows.append(stats(base2 & thrust, h, "sector", "GLW own record >=2ATR r5>=80", ["GLW"]))
    rows.append(stats(mask(2.0, earn="any", lead="any"), h, "sector", "GLW own >=2ATR any", ["GLW"]))
    show(rows, f"GLW decomposition h={h}")

print("\ncost: BAC ~ $59.5, 1-2 bp spread per side -> ~2-4 bp RT + XLF ~1-2 bp RT + commissions ~0.5 bp = ~5 bp")
