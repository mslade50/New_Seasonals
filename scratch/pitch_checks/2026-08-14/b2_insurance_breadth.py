"""C2 / C2b / C9 round 1 — the insurance complex breaking together inside an
intact uptrend while XLF prints a 63d rank of 100.

The claimed difference from the registry-dead "single market breaking inside an
intact thrust" family is that this is a synchronized BREADTH event across ~10
names, not one index decoupling. This script decides whether that difference is
real or cosmetic.

Trigger (base): >= 70% of the available insurer universe with a 5-day return
rank <= 20 on the same day, while the universe's MEDIAN 63d rank >= 70
("intact"). Today (2026-08-13): 13 of 14 at rank5 <= 20 (93%), median rank63
82.9, XLF rank63 100.0 -> live.

Vehicles measured, in the order the registry demands (price the legs BEFORE the
spread):
  L1  equal-weight insurance basket (the outright, C2)
  L2  XLF alone on the same days (the leg the spread is subtracted from)
  L3  SPY alone on the same days (tape control)
  S1  basket - XLF (C2b)
  T4  the 4 most-washed names at each trigger (the only TRADEABLE form, <= 4 legs)
  C9  the single strongest name at each trigger
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pitch_lab import (battery, bootstrap_p_le0, cluster_note, declusters,  # noqa: E402
                       load_prices, local_control, pct_rank, show, sign_test,
                       summarize)

ASOF = pd.Timestamp("2026-08-13")
INSURERS = ["HIG", "ALL", "TRV", "AIG", "MET", "PGR", "CB", "AFL", "PRU",
            "LNC", "WRB", "CINF", "L", "GL"]
OTHER = ["XLF", "SPY"]

raw = load_prices(INSURERS + OTHER)
panel = pd.DataFrame({t: raw[t]["Close"] for t in INSURERS}).sort_index()
panel = panel[panel.index <= ASOF]
spy_c = raw["SPY"]["Close"].reindex(panel.index)
xlf_c = raw["XLF"]["Close"].reindex(panel.index)

# common index: days where XLF and SPY both trade (XLF starts 1998-12)
idx = panel.index[xlf_c.notna() & spy_c.notna()]
panel = panel.loc[idx]
spy_c, xlf_c = spy_c.loc[idx], xlf_c.loc[idx]

r5 = pd.DataFrame({t: pct_rank(panel[t].dropna(), 5).reindex(idx) for t in panel})
r21 = pd.DataFrame({t: pct_rank(panel[t].dropna(), 21).reindex(idx) for t in panel})
r63 = pd.DataFrame({t: pct_rank(panel[t].dropna(), 63).reindex(idx) for t in panel})
navail = r5.notna().sum(axis=1)

# equal-weight insurance basket as a synthetic price (cross-sectional mean of
# daily simple returns, rebalanced daily). FRACTIONS throughout.
rets = panel.pct_change()
ew_ret = rets.mean(axis=1, skipna=True).fillna(0.0)
assert abs(ew_ret).max() < 0.5, "EW basket daily return out of fraction range"
insew = (1.0 + ew_ret).cumprod()

px = pd.DataFrame({"INSEW": insew, "XLF": xlf_c, "SPY": spy_c}).dropna()


def mask_for(rank5_cut: float, frac: float, med63: float | None,
             use21: float | None = None) -> pd.Series:
    n_wash = (r5 <= rank5_cut).sum(axis=1)
    m = (n_wash / navail >= frac) & (navail >= 8)
    if med63 is not None:
        m = m & (r63.median(axis=1) >= med63)
    if use21 is not None:
        m = m & (r21.median(axis=1) >= use21)
    return m.reindex(px.index, fill_value=False).fillna(False)


BASE = dict(rank5_cut=20, frac=0.70, med63=70)
m0 = mask_for(**BASE)
print("=" * 78)
print("TRIGGER: >=70% of insurers with rank5<=20, median rank63>=70")
print(f"  fires on {int(m0.sum())} days, {len(set(m0[m0].index.year))} years, "
      f"live today={bool(m0.reindex([ASOF]).fillna(False).iloc[0])}")
print(f"  trigger years: {sorted(set(m0[m0].index.year))}")
print("=" * 78)

VARIANTS = {
    "rank5<=10": mask_for(10, 0.70, 70),
    "rank5<=25": mask_for(25, 0.70, 70),
    "rank5<=30": mask_for(30, 0.70, 70),
    "frac>=0.5": mask_for(20, 0.50, 70),
    "frac>=0.6": mask_for(20, 0.60, 70),
    "frac>=0.8": mask_for(20, 0.80, 70),
    "med63>=60": mask_for(20, 0.70, 60),
    "med63>=80": mask_for(20, 0.70, 80),
    "NO intact gate": mask_for(20, 0.70, None),
    "med21>=60 too": mask_for(20, 0.70, 70, use21=60),
}

for h in (3, 5, 10):
    battery(px, m0, [("INSEW", 1.0)], h,
            f"C2 L1 equal-weight insurance basket, LONG", cost_bps=10,
            variants=VARIANTS if h == 5 else None)
    battery(px, m0, [("XLF", 1.0)], h, "C2 L2 XLF alone (the leg)", cost_bps=5)
    battery(px, m0, [("SPY", 1.0)], h, "C2 L3 SPY alone (tape)", cost_bps=3)
    battery(px, m0, [("INSEW", 1.0), ("XLF", -1.0)], h,
            "C2b S1 basket - XLF (the spread)", cost_bps=10)

# ---------------------------------------------------------------------------
# T4 / C9: name-selection forms. Path-dependent, so computed by hand.
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("T4 (4 most-washed names) and C9 (single strongest name) at each trigger")
print("=" * 78)


def selection_ret(trig: pd.DatetimeIndex, h: int, k: int, mode: str,
                  lag: int = 1) -> tuple[pd.DatetimeIndex, np.ndarray, list]:
    """Equal-weight forward return of the k names picked at each trigger.
    mode='washed' -> lowest rank5; mode='strong' -> highest rank63."""
    pos = pd.Series(range(len(panel.index)), index=panel.index)
    dates, vals, picks = [], [], []
    for d in trig:
        p = pos.get(d)
        if p is None or p + lag + h >= len(panel.index):
            continue
        key = r5.loc[d] if mode == "washed" else -r63.loc[d]
        key = key.dropna()
        if len(key) < k:
            continue
        sel = list(key.sort_values().index[:k])
        e = panel.iloc[p + lag][sel]
        x = panel.iloc[p + lag + h][sel]
        if e.isna().any() or x.isna().any():
            continue
        dates.append(d)
        vals.append(float((x / e - 1.0).mean()))
        picks.append(sel)
    return pd.DatetimeIndex(dates), np.asarray(vals), picks


trig0 = px.index[m0.values]
for h in (3, 5, 10):
    epi = declusters(trig0, h, px.index)
    for mode, k, lbl in (("washed", 4, "T4 4 most-washed"),
                         ("washed", 1, "T1 single most-washed"),
                         ("strong", 1, "C9 single strongest (rank63)")):
        d, v, picks = selection_ret(epi, h, k, mode)
        if len(v) == 0:
            continue
        wins = int((v > 0).sum())
        # control: same names, all days in span (own drift)
        r = summarize(v, f"{lbl} h={h} (episodes N={len(v)})")
        r["sign_p"] = round(sign_test(wins, len(v)), 4)
        r["boot_p_le0"] = round(bootstrap_p_le0(v), 4)
        show([r])
        print(f"    picks (last 3): {picks[-3:]}")
        print(f"    {cluster_note(d, v)}")

# unconditional drift of the same selection rule, for T4's control:
# pick the 4 lowest-rank5 insurers on EVERY day, same horizon.
print("\n  T4 control: the same 'pick 4 most-washed insurers' rule on ALL days")
for h in (3, 5, 10):
    alld = declusters(px.index[:-h - 2], h, px.index)
    d, v, _ = selection_ret(alld, h, 4, "washed")
    show([summarize(v, f"ALL-days 4-most-washed h={h} (N={len(v)})")])
