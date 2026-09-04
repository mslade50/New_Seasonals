"""C2 / C2b / C9 round 2 — gate attribution, the short side, cycle/era, tape
over-selection, and the cross-sectional REFERENCE-CLASS placebo.

Round 1 (b2_insurance_breadth.py) showed the long is negative at every horizon
and below its own drift. Round 2 asks the four questions that decide whether
anything at all is in here:

  (1) GATE ATTRIBUTION — drop each gate in turn. If the "intact 63d uptrend"
      gate is not doing work, nothing may be attributed to it. (The round-1
      sensitivity table already hinted it INVERTS the sign.)
  (2) THE SHORT SIDE — a negative long is only an idea if the sign is stable.
      Era + midterm-cycle split on the short.
  (3) TAPE OVER-SELECTION — what fraction of trigger days sit below SPY's 200d
      against the unconditional base rate.
  (4) REFERENCE-CLASS PLACEBO — run the IDENTICAL breadth rule on 12 other
      industry groups. If insurance is not distinguishable from its peer
      groups, the "insurance complex" label is decoration.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pitch_lab import (bootstrap_p_le0, cluster_note, declusters, fwd_lag,  # noqa: E402
                       load_prices, pct_rank, show, sign_test, summarize)

ASOF = pd.Timestamp("2026-08-13")

GROUPS = {
    "insurance": ["HIG", "ALL", "TRV", "AIG", "MET", "PGR", "CB", "AFL",
                  "PRU", "LNC", "WRB", "CINF", "L", "GL"],
    "banks": ["JPM", "BAC", "WFC", "C", "USB", "PNC", "TFC", "MTB", "FITB",
              "KEY", "RF", "CFG", "HBAN", "ZION"],
    "semis": ["NVDA", "AMD", "INTC", "MU", "TXN", "ADI", "AMAT", "LRCX",
              "KLAC", "MCHP", "NXPI", "SWKS", "QCOM", "AVGO"],
    "retail": ["WMT", "TGT", "COST", "HD", "LOW", "TJX", "ROST", "DG",
               "DLTR", "BBY", "KSS", "M"],
    "pharma": ["JNJ", "PFE", "MRK", "LLY", "ABBV", "BMY", "AMGN", "GILD",
               "BIIB", "VRTX", "ZTS"],
    "energy": ["XOM", "CVX", "COP", "EOG", "SLB", "HAL", "OXY", "PSX",
               "VLO", "MPC", "DVN", "HES"],
    "utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL", "ED", "WEC",
                  "ES", "PEG", "PNW", "ETR"],
    "reits": ["AMT", "PLD", "CCI", "EQIX", "PSA", "SPG", "O", "AVB", "EQR",
              "VTR", "WELL"],
    "transports": ["UNP", "CSX", "NSC", "UPS", "FDX", "DAL", "UAL", "LUV",
                   "ODFL", "CHRW", "EXPD"],
    "staples": ["KO", "PEP", "PG", "MO", "PM", "MDLZ", "GIS", "K", "CL",
                "KMB", "SYY", "HSY"],
    "machinery": ["CAT", "DE", "HON", "GE", "MMM", "EMR", "ITW", "PH",
                  "ROK", "DOV", "ETN"],
    "media_telecom": ["DIS", "CMCSA", "T", "VZ", "NFLX", "CHTR", "FOXA",
                      "PARA", "WBD"],
    "homebuild_disc": ["DHI", "LEN", "NVR", "PHM", "WHR", "F", "GM", "APTV",
                       "BWA", "LKQ", "GPC"],
}

ALL = sorted({t for v in GROUPS.values() for t in v} | {"SPY", "XLF"})
raw = load_prices(ALL)
spy = raw["SPY"]["Close"]
spy = spy[spy.index <= ASOF]
BASE_IDX = spy.index


def group_state(names: list[str]):
    have = [t for t in names if t in raw]
    panel = pd.DataFrame({t: raw[t]["Close"] for t in have}).reindex(BASE_IDX)
    r5 = pd.DataFrame({t: pct_rank(panel[t].dropna(), 5).reindex(BASE_IDX)
                       for t in panel})
    r63 = pd.DataFrame({t: pct_rank(panel[t].dropna(), 63).reindex(BASE_IDX)
                        for t in panel})
    navail = r5.notna().sum(axis=1)
    ew = (1.0 + panel.pct_change().mean(axis=1, skipna=True).fillna(0.0)).cumprod()
    return panel, r5, r63, navail, ew, have


def breadth_mask(r5, r63, navail, rank5_cut=20, frac=0.70, med63=70):
    m = ((r5 <= rank5_cut).sum(axis=1) / navail >= frac) & (navail >= 8)
    if med63 is not None:
        m = m & (r63.median(axis=1) >= med63)
    return m.fillna(False)


panel, r5, r63, navail, insew, have = group_state(GROUPS["insurance"])
print(f"insurance names in cache: {have}")

H = 5
ret = fwd_lag(insew, H)
valid = ret.notna()

print("\n" + "=" * 78)
print("1. GATE ATTRIBUTION — drop each gate in turn (h=5, LONG the basket)")
print("=" * 78)
CFGS = [
    ("BOTH gates (as pitched)", dict(rank5_cut=20, frac=0.70, med63=70)),
    ("drop INTACT gate (breadth only)", dict(rank5_cut=20, frac=0.70, med63=None)),
    ("drop BREADTH gate (intact only)", dict(rank5_cut=20, frac=0.0, med63=70)),
    ("neither gate = all days", dict(rank5_cut=100, frac=0.0, med63=None)),
]
rows = []
for lbl, kw in CFGS:
    m = breadth_mask(r5, r63, navail, **kw)
    t = BASE_IDX[m.values & valid.values]
    if len(t) == 0:
        continue
    epi = declusters(t, H, BASE_IDX)
    s = summarize(ret.loc[epi].values, f"{lbl} (N_epi={len(epi)}, N_days={len(t)})")
    rows.append(s)
show(rows, "gate attribution")
print("  READ: if 'drop INTACT gate' is materially BETTER than 'BOTH gates',")
print("  the intact-uptrend gate is not a filter, it is the damage.")

# 1b. the honest attribution: inside the breadth-only cell, split by whether
# the complex is 'intact'. TODAY sits in the intact half (median rank63 82.9).
mb = breadth_mask(r5, r63, navail, med63=None)
intact = (r63.median(axis=1) >= 70).fillna(False)
rows = []
for lbl, sub in (("breadth + INTACT (today's cell)", mb & intact),
                 ("breadth + NOT intact", mb & ~intact)):
    t = BASE_IDX[sub.values & valid.values]
    epi = declusters(t, H, BASE_IDX)
    rows.append(summarize(ret.loc[epi].values, f"{lbl} (N_epi={len(epi)})"))
show(rows, "1b. inside the breadth-only cell, where does the +0.917% live?")

print("\n" + "=" * 78)
print("2. THE SHORT SIDE — era + midterm-cycle stability (h=3/5/10)")
print("=" * 78)
m0 = breadth_mask(r5, r63, navail)
for h in (3, 5, 10):
    r = fwd_lag(insew, h)
    t = BASE_IDX[m0.values & r.notna().values]
    epi = declusters(t, h, BASE_IDX)
    v = -r.loc[epi].values  # SHORT the basket
    d = pd.DatetimeIndex(epi)
    wins = int((v > 0).sum())
    s = summarize(v, f"SHORT basket h={h} (N={len(v)})")
    s["sign_p"] = round(sign_test(wins, len(v)), 4)
    rows = [s]
    for lbl, m in (("  pre-2018", d < pd.Timestamp("2018-01-01")),
                   ("  2018+", d >= pd.Timestamp("2018-01-01")),
                   ("  midterm yrs", np.asarray(d.year % 4 == 2)),
                   ("  non-midterm", np.asarray(d.year % 4 != 2))):
        if m.sum():
            rows.append(summarize(v[m], f"{lbl} (N={int(m.sum())})"))
    show(rows, f"h={h}")
    print(f"  {cluster_note(d, v)}")

print("\n" + "=" * 78)
print("3. TAPE OVER-SELECTION — SPY below its 200d on trigger days")
print("=" * 78)
below200 = (spy < spy.rolling(200).mean())
m0v = m0.reindex(BASE_IDX, fill_value=False).values
base_rate = 100 * below200[below200.notna()].mean()
trig_rate = 100 * below200[m0v & below200.notna().values].mean()
print(f"  base rate  SPY<200d = {base_rate:.1f}%")
print(f"  trigger days         = {trig_rate:.1f}%  (N={int(m0v.sum())})")
print(f"  over-selection = {trig_rate - base_rate:+.1f}pp")

print("\n" + "=" * 78)
print("4. REFERENCE-CLASS PLACEBO — the identical breadth rule on 13 groups")
print("=" * 78)
res = []
for g, names in GROUPS.items():
    _, gr5, gr63, gnav, gew, ghave = group_state(names)
    if len(ghave) < 8:
        print(f"  {g}: only {len(ghave)} names in cache, skipped")
        continue
    m = breadth_mask(gr5, gr63, gnav)
    r = fwd_lag(gew, H)
    t = BASE_IDX[m.values & r.notna().values]
    if len(t) < 3:
        res.append({"group": g, "n_epi": 0, "mean_pct": np.nan})
        continue
    epi = declusters(t, H, BASE_IDX)
    v = r.loc[epi].values
    drift = r.dropna()
    res.append({
        "group": g, "n_names": len(ghave), "n_days": len(t), "n_epi": len(epi),
        "mean_pct": round(100 * v.mean(), 3),
        "hit": round(100 * (v > 0).mean(), 1),
        "own_drift_pct": round(100 * drift.mean(), 3),
        "excess_pp": round(100 * (v.mean() - drift.mean()), 3),
        "sign_p": round(sign_test(int((v > 0).sum()), len(v)), 3),
    })
df = pd.DataFrame(res).sort_values("excess_pp", ascending=False)
print(df.to_string(index=False))
ex = df["excess_pp"].dropna().values
print(f"\n  cross-group excess: mean {ex.mean():+.3f}pp, sd {ex.std(ddof=1):.3f}pp, "
      f"min {ex.min():+.3f}, max {ex.max():+.3f}")
ins = df[df["group"] == "insurance"]["excess_pp"].iloc[0]
rank = int((df["excess_pp"].dropna() >= ins).sum())
print(f"  INSURANCE excess {ins:+.3f}pp ranks {rank} of {len(ex)} groups "
      f"(family-wise p = {rank/len(ex):.3f} for 'best group under no effect')")
print("  READ: if insurance is mid-pack or worse, the industry label carries")
print("  no information and the cell is a generic shape, not an insurance one.")
