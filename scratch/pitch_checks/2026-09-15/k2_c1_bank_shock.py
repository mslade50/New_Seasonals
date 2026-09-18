"""C1 round 1: money-centre bank one-day shock (<= -2 Wilder ATR) with KRE >= 0
and SPY > -1%; long the name hedged against XLF, h=1..5, lag=1.
Parent (no KRE gate), complement (KRE < 0), gap-vs-intraday split, earnings
split, and the non-bank large-cap reference class."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch_lab import *  # noqa
from k2_common import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
BANKS = ["JPM", "BAC", "C", "WFC", "GS", "MS", "BNY", "STT"]
SN = single_names()
NONFIN = [t for t, s in SN.items() if s in SECTOR_ETF and s != "Financial Services"]
ETF_NEED = ["SPY", "XLF", "KRE", "^VIX"] + sorted(set(SECTOR_ETF.values()))
px, cal, P = load_panels(BANKS + NONFIN + ETF_NEED)
D = derive(px, cal)
C = P["Close"]
ret1, sh = D["ret1"], D["shock"]
bS = rolling_beta(ret1, "SPY")
bX = rolling_beta(ret1, "XLF")
spy1, kre1 = ret1["SPY"], ret1["KRE"]
spy_ok = spy1 > -0.01
sma200 = C["SPY"].rolling(200).mean()
vix = C["^VIX"]
vq = vix.quantile([1 / 3, 2 / 3]).values

ern = pd.read_parquet(ROOT / "data" / "earnings_calendar.parquet", columns=["ticker", "date"])
ern["date"] = pd.to_datetime(ern["date"])


def earn_flag(ev: pd.DataFrame) -> np.ndarray:
    out = []
    for d, t in zip(ev["date"], ev["name"]):
        e = ern.loc[ern["ticker"] == t, "date"]
        p = cal.get_loc(d)
        lo = cal[max(0, p - 1)]
        out.append(bool(((e >= lo) & (e <= d)).any()))
    return np.asarray(out)


def colmask(frame: pd.DataFrame, names, cond: pd.Series) -> pd.DataFrame:
    return frame[names].apply(lambda c: c & cond.fillna(False))


parent = colmask(sh.le(-2.0), BANKS, spy_ok)
cell = colmask(parent, BANKS, kre1 >= 0)
comp = colmask(parent, BANKS, kre1 < 0)
parent_kre_era = colmask(parent, BANKS, kre1.notna())

EV = {"CELL KRE>=0": events_from_mask(cell, BANKS),
      "PARENT (no KRE gate, 2000+)": events_from_mask(parent, BANKS),
      "PARENT KRE-era (2006-06+)": events_from_mask(parent_kre_era, BANKS),
      "COMPLEMENT KRE<0": events_from_mask(comp, BANKS)}
for k, v in EV.items():
    print(f"{k}: name-days {len(v)}, dates {v['date'].nunique()}")

live = cell.loc[cal[-1]]
print("\nlive 2026-09-14 cell members:", list(live[live].index),
      " BAC shock", round(sh.loc[cal[-1], "BAC"], 2), " KRE 1d", round(100 * kre1.iloc[-1], 2))

kre_era = cal[cal >= pd.Timestamp("2006-06-22")]


def pair(h: int, hedge: str, lag: int = 1) -> pd.DataFrame:
    F = fwd_panel(C, h, lag)
    if hedge == "XLFb":
        return F[BANKS].sub(bX[BANKS].mul(F["XLF"], axis=0))
    if hedge == "XLF1":
        return F[BANKS].sub(F["XLF"], axis=0)
    if hedge == "SPYb":
        return F.sub(bS.mul(F["SPY"], axis=0))
    if hedge == "raw":
        return F
    raise ValueError(hedge)


def local_ctrl(panel: pd.DataFrame, ev: pd.DataFrame, win: int = 126) -> float:
    vals = []
    for t, g in ev.groupby("name"):
        pos = cal.get_indexer(pd.DatetimeIndex(g["date"]))
        keep = np.zeros(len(cal), bool)
        for p in pos:
            keep[max(0, p - win):p + win + 1] = True
        keep[pos] = False
        x = panel[t].to_numpy(float)[keep]
        vals.append(x[~np.isnan(x)])
    v = np.concatenate(vals)
    return float(v.mean())


print("\n##### 1. Pattern vs controls (date-averaged across names, declustered gap=max(h,5)) #####")
for hedge in ["XLFb", "XLF1", "SPYb", "raw"]:
    rows = []
    for h in (1, 2, 3, 5):
        pan = pair(h, hedge)
        ctrl_all = pan[BANKS].loc[kre_era].mean(axis=1).mean()
        for lbl, ev in EV.items():
            e = ev.copy()
            e["v"] = lookup(pan, e["date"], e["name"])
            r = date_stats(e, "v", cal, max(h, 5), f"h={h} {lbl}", ctrl_all)
            if lbl.startswith("CELL"):
                r["local_pp"] = round(r["mean_pct"] - 100 * local_ctrl(pan, e), 3)
                e["n0"] = lookup(pair(h, hedge, lag=0), e["date"], e["name"])
                r["lag0_pct"] = round(100 * date_series(e, "n0").mean(), 3)
            rows.append(r)
    show(rows, f"hedge={hedge}  (ctrl = all bank-days KRE era; local_pp = excess over +/-126td ex-trigger)")

# ---- splits on the CELL, XLF-beta hedge
print("\n##### 2. CELL splits (hedge XLF beta) #####")
ev = EV["CELL KRE>=0"].copy()
ev["shock"] = lookup(sh, ev["date"], ev["name"])
ev["ret1"] = lookup(ret1, ev["date"], ev["name"])
ev["gap"] = lookup(D["gap"], ev["date"], ev["name"])
ev["gap_share"] = ev["gap"] / ev["ret1"]
ev["earn"] = earn_flag(ev)
ev["above200"] = (C["SPY"] > sma200).reindex(ev["date"]).values
ev["vix"] = vix.reindex(ev["date"]).values
for h in (1, 3, 5):
    pan = pair(h, "XLFb")
    ev[f"h{h}"] = lookup(pan, ev["date"], ev["name"])
for h in (1, 3, 5):
    col = f"h{h}"
    g = max(h, 5)
    rows = []
    d = pd.DatetimeIndex(ev["date"])
    for lbl, m in [("pre-2018", d < "2018-01-01"), ("2018+", d >= "2018-01-01"),
                   ("gap-led (gap>=50% of move)", ev["gap_share"] >= 0.5),
                   ("intraday-led (gap<50%)", ev["gap_share"] < 0.5),
                   ("earnings day/after", ev["earn"]), ("not earnings", ~ev["earn"]),
                   ("SPY above 200d", ev["above200"] == True),
                   ("SPY below 200d", ev["above200"] == False),
                   ("VIX low tercile", ev["vix"] <= vq[0]),
                   ("VIX mid", (ev["vix"] > vq[0]) & (ev["vix"] <= vq[1])),
                   ("VIX high tercile", ev["vix"] > vq[1]),
                   ("shock -2..-3 ATR", ev["shock"] > -3), ("shock <= -3 ATR", ev["shock"] <= -3),
                   ("ex 2008-2009", ~d.year.isin([2008, 2009]))]:
        rows.append(date_stats(ev[np.asarray(m, dtype=bool)], col, cal, g, f"h={h} {lbl}"))
    show(rows, f"CELL splits h={h}")

print("\nper-bank, CELL h=5 (name-days, not declustered):")
print(ev.groupby("name")["h5"].agg(["count", "mean", lambda x: (x > 0).mean()]).round(4))
e5 = ev.dropna(subset=["h5"])
s5 = date_series(e5, "h5")
k5 = declusters(s5.index, 5, cal)
print("\nconcentration (h=5 declustered):", cluster_note(k5, s5.loc[k5].values))
print("bootstrap P(mean<=0):", round(bootstrap_p_le0(s5.loc[k5].values), 4))
print("\nlast 25 CELL events:")
print(ev.tail(25)[["date", "name", "shock", "ret1", "gap_share", "earn", "h1", "h3", "h5"]]
      .round(4).to_string(index=False))

# ---- reference class: same rule on non-bank large caps, hedge SPY beta
print("\n##### 3. Reference class: non-financial single names (SPY-beta hedge) #####")
sec_ok = pd.DataFrame({t: (ret1[SECTOR_ETF[SN[t]]] >= 0) for t in NONFIN})
ref_parent = colmask(sh.le(-2.0), NONFIN, spy_ok)
ref_cell = ref_parent & sec_ok.reindex(columns=NONFIN).fillna(False)
bank_parent_spy = parent
rows = []
for h in (1, 3, 5):
    pan = pair(h, "SPYb")
    ctrl_nf = pan[NONFIN].mean(axis=1).mean()
    ctrl_b = pan[BANKS].mean(axis=1).mean()
    for lbl, m, names, ctl in [
        ("banks parent", parent, BANKS, ctrl_b),
        ("banks CELL KRE>=0", cell, BANKS, ctrl_b),
        ("non-fin parent", ref_parent, NONFIN, ctrl_nf),
        ("non-fin + own sector ETF>=0", ref_cell, NONFIN, ctrl_nf),
        ("non-fin + own sector ETF<0", ref_parent & ~sec_ok.reindex(columns=NONFIN).fillna(True), NONFIN, ctrl_nf)]:
        e = events_from_mask(m, names)
        e["v"] = lookup(pan, e["date"], e["name"])
        rows.append(date_stats(e, "v", cal, max(h, 5), f"h={h} {lbl}", ctl))
show(rows, "reference class (date-averaged, declustered gap=max(h,5))")

# gap-led vs intraday-led in the non-fin reference class (news vs de-grossing)
rows = []
e = events_from_mask(ref_parent, NONFIN)
e["gs"] = lookup(D["gap"], e["date"], e["name"]) / lookup(ret1, e["date"], e["name"])
for h in (1, 5):
    e["v"] = lookup(pair(h, "SPYb"), e["date"], e["name"])
    rows.append(date_stats(e[e["gs"] >= 0.5], "v", cal, 5, f"h={h} non-fin parent gap-led"))
    rows.append(date_stats(e[e["gs"] < 0.5], "v", cal, 5, f"h={h} non-fin parent intraday-led"))
show(rows, "reference class gap split")

# live BAC session anatomy
b = px["BAC"].loc["2026-09-10":]
print("\nBAC daily bars:\n", b[["Open", "High", "Low", "Close"]].round(3))
print("BAC 09-14 gap share of move:", round(D["gap"].loc[cal[-1], "BAC"] / ret1.loc[cal[-1], "BAC"], 3))
try:
    from intraday_data import get_intraday
    ib = get_intraday("BAC", start="2026-09-11", end="2026-09-14")
    print(ib.tail(30).to_string(index=False))
except Exception as exc:  # noqa
    print("intraday unavailable:", exc)
