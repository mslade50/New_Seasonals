"""a1 — ADVERSARIAL round 1 on CANDIDATE A1.

"The duration rotation at both extremes": LONG XLRE / SHORT XLF, equal
dollar weight, entry lag=1 MOC, exit MOC, on the session where ^TNX closes
at a trailing-252d HIGH *and* XLRE's 63d return rank (trailing-252 PIT) is
<= 10.

Order convention (rule 7, fixed and stated): FILTER first, then DECLUSTER
(min_gap = h). Every episode number below uses that order.

Calendar: the whole panel is reindexed to SPY's session index before any
rolling window, because ^TNX (like ^VIX) carries bars on NYSE closures.

Cost: 2 bp per leg per side => 4 bp per leg round trip, 8 bp for the pair.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change  # noqa

H = 5
COST_LEG_BPS = 4.0          # per leg, round trip
SPDR = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU",
        "XLV", "XLY"]
TICKERS = SPDR + ["^TNX", "IYR", "VNQ", "SPY"]

raw = close_panel(TICKERS)
CAL = raw["SPY"].dropna().index
px = raw.reindex(CAL)
POS = pd.Series(range(len(CAL)), index=CAL)

print("panel calendar:", CAL[0].date(), "..", CAL[-1].date(), len(CAL), "sessions")


# ---------------------------------------------------------------- primitives
def tnx_at_252_high() -> pd.Series:
    s = px["^TNX"]
    hi = rolling_on_valid(s, lambda x: x.rolling(252).max())
    return (s >= hi - 1e-12) & s.notna() & hi.notna()


def r63_floor(tkr: str, thr: float = 10.0) -> pd.Series:
    return pct_rank(px[tkr], 63) <= thr


def pair_ret(long_t: str, short_t: str, h: int = H, lag: int = 1) -> pd.Series:
    return vehicle_ret(px, [(long_t, 1.0), (short_t, -1.0)], h, lag)


def leg_ret(tkr: str, h: int = H, lag: int = 1) -> pd.Series:
    return fwd_lag(px[tkr], h, lag)


def episodes(mask: pd.Series, ret: pd.Series, h: int = H) -> pd.DatetimeIndex:
    """FILTER then DECLUSTER."""
    days = CAL[mask.reindex(CAL, fill_value=False).values & ret.notna().values]
    return declusters(days, h, CAL)


TNX_HI = tnx_at_252_high()
print(f"^TNX 252d-high sessions: {int(TNX_HI.sum())} "
      f"({TNX_HI[TNX_HI].index[0].date()} .. {TNX_HI[TNX_HI].index[-1].date()})")
print("  today's reading is one of them:", bool(TNX_HI.iloc[-1]))


# ==========================================================================
# 1. the defended cell — battery
# ==========================================================================
mask_join = TNX_HI & r63_floor("XLRE")
ret_pair = pair_ret("XLRE", "XLF")
print(f"\nDEFENDED trigger days (XLRE series, 2015-10+): "
      f"{int((mask_join & ret_pair.notna()).sum())}")

variants = {
    "r63<=5": TNX_HI & r63_floor("XLRE", 5),
    "r63<=10 (defended)": mask_join,
    "r63<=15": TNX_HI & r63_floor("XLRE", 15),
    "r63<=20": TNX_HI & r63_floor("XLRE", 20),
    "r63<=25": TNX_HI & r63_floor("XLRE", 25),
}
battery(px, mask_join, [("XLRE", 1.0), ("XLF", -1.0)], H,
        "A1 DEFENDED  LONG XLRE / SHORT XLF", COST_LEG_BPS,
        variants=variants, min_gap=H, event_kinds=("cpi",))

# same cell, extended proxies
for proxy in ("IYR", "VNQ"):
    m = TNX_HI & r63_floor(proxy)
    battery(px, m, [(proxy, 1.0), ("XLF", -1.0)], H,
            f"A1 EXTENDED  LONG {proxy} / SHORT XLF (proxy series)",
            COST_LEG_BPS, min_gap=H, event_kinds=("cpi",))


# ==========================================================================
# 2. leg attribution on the SAME episodes
# ==========================================================================
def leg_table(long_t: str, mask: pd.Series, label: str) -> None:
    rp = pair_ret(long_t, "XLF")
    epi = episodes(mask, rp, H)
    if len(epi) == 0:
        print(f"\n--- {label}: no episodes")
        return
    rl = leg_ret(long_t)
    rs = leg_ret("XLF")
    rspy = leg_ret("SPY")
    span = (CAL >= epi[0]) & (CAL <= epi[-1])
    rows = [
        summarize(rp.loc[epi].values, f"PAIR {long_t}-XLF (N={len(epi)})"),
        summarize(rl.loc[epi].values, f"  LONG leg {long_t} on same episodes"),
        summarize(rl[span & rl.notna().values].values,
                  f"  CTRL {long_t} own drift, same span"),
        summarize(rs.loc[epi].values, "  SHORT leg XLF raw (sign as held: -1x below)"),
        summarize(-rs.loc[epi].values, "  SHORT leg XLF contribution (-1x)"),
        summarize(-rs[span & rs.notna().values].values,
                  "  CTRL short-XLF contribution, same span"),
        summarize((rl - rspy).loc[epi].values,
                  f"  {long_t} minus SPY on same episodes"),
        summarize((rl - rspy)[span & (rl - rspy).notna().values].values,
                  f"  CTRL {long_t}-SPY own drift, same span"),
    ]
    show(rows, f"LEG ATTRIBUTION — {label}")


print("\n" + "=" * 78)
leg_table("XLRE", mask_join, "XLRE cell (2015-10+)")
leg_table("IYR", TNX_HI & r63_floor("IYR"), "IYR cell (2000-06+)")
leg_table("VNQ", TNX_HI & r63_floor("VNQ"), "VNQ cell (2004-09+)")


# ==========================================================================
# 3. gate attribution + discarded complements
# ==========================================================================
def gate_attribution(long_t: str) -> None:
    rp = pair_ret(long_t, "XLF")
    ok = rp.notna().values
    a = TNX_HI.reindex(CAL, fill_value=False).values & ok      # yield gate only
    b = r63_floor(long_t).reindex(CAL, fill_value=False).values & ok  # floor only
    j = a & b
    rows = [
        summarize(rp.values[j], f"JOIN (both gates)  N={int(j.sum())}"),
        summarize(rp.values[a], f"(a) ^TNX 252d high ALONE  N={int(a.sum())}"),
        summarize(rp.values[a & ~b],
                  f"    DISCARDED by the {long_t}-floor gate  N={int((a & ~b).sum())}"),
        summarize(rp.values[b], f"(b) {long_t} r63<=10 ALONE  N={int(b.sum())}"),
        summarize(rp.values[b & ~a],
                  f"    DISCARDED by the yield gate  N={int((b & ~a).sum())}"),
        summarize(rp.values[~a & ~b & ok], f"neither gate  N={int((~a & ~b & ok).sum())}"),
        summarize(rp.values[ok], f"ALL DAYS  N={int(ok.sum())}"),
    ]
    show(rows, f"GATE ATTRIBUTION (day level) — LONG {long_t} / SHORT XLF, h={H}")


for t in ("XLRE", "IYR", "VNQ"):
    gate_attribution(t)


# ==========================================================================
# 4. reference class + max-of-K permutation against the DEFENDED statistic
# ==========================================================================
def epi_positions(mask: pd.Series, ret: pd.Series, h: int = H) -> np.ndarray:
    epi = episodes(mask, ret, h)
    return np.asarray([POS[d] for d in epi], dtype=int)


def perm_max_of_k(defended_stat: float, ret_arrays: dict[str, np.ndarray],
                  base_pos: np.ndarray, n_perm: int = 4000,
                  seed: int = 42) -> tuple[float, float, dict]:
    """Circular-shift permutation. Shifts the DECLUSTERED trigger positions as
    a rigid block, preserving episode count and spacing, and rescores every
    sibling. Returns (p_uncharged_defended, p_charged_max_of_K, observed)."""
    rng = np.random.default_rng(seed)
    n = len(CAL)
    obs = {k: np.nanmean(v[base_pos]) for k, v in ret_arrays.items()}
    ge_def = 0
    ge_max = 0
    used = 0
    for _ in range(n_perm):
        off = int(rng.integers(1, n))
        sh = (base_pos + off) % n
        stats = []
        for k, v in ret_arrays.items():
            m = np.nanmean(v[sh])
            stats.append(m)
            if k == "DEFENDED" and not np.isnan(m) and m >= defended_stat:
                ge_def += 1
        stats = [s for s in stats if not np.isnan(s)]
        if not stats:
            continue
        used += 1
        if max(stats) >= defended_stat:
            ge_max += 1
    return ((ge_def + 1) / (n_perm + 1), (ge_max + 1) / (used + 1), obs)


rp_def = pair_ret("XLRE", "XLF")
base_pos = epi_positions(mask_join, rp_def, H)
DEF_STAT = float(np.nanmean(rp_def.values[base_pos]))
print(f"\n\nDEFENDED statistic: episode mean of LONG XLRE/SHORT XLF, "
      f"h={H}, N={len(base_pos)} episodes = {100*DEF_STAT:+.3f}%")

# class 1: LONG(S) / SHORT XLF, identical construction (S's own r63 floor)
print("\n--- REFERENCE CLASS 1: LONG(sector S) / SHORT XLF, identical construction")
cls1 = []
for s in SPDR:
    if s == "XLF":
        continue
    m = TNX_HI & r63_floor(s)
    r = pair_ret(s, "XLF")
    e = episodes(m, r, H)
    d = summarize(r.loc[e].values, f"LONG {s} / SHORT XLF")
    cls1.append(d)
show(cls1, "class 1 (each sibling on its OWN r63 floor)")

# class 2: LONG XLRE / SHORT(S), trigger held fixed at the defended trigger
print("\n--- REFERENCE CLASS 2: LONG XLRE / SHORT(sector S), defended trigger")
cls2 = []
for s in SPDR:
    if s == "XLRE":
        continue
    r = pair_ret("XLRE", s)
    e = episodes(mask_join, r, H)
    cls2.append(summarize(r.loc[e].values, f"LONG XLRE / SHORT {s}"))
show(cls2, "class 2 (short leg varies)")

# permutation, class 2 (same trigger for every member -> same base positions)
arrays2 = {"DEFENDED": rp_def.values}
for s in SPDR:
    if s == "XLRE":
        continue
    arrays2[f"XLRE-{s}"] = pair_ret("XLRE", s).values
p_unch, p_ch, obs2 = perm_max_of_k(DEF_STAT, arrays2, base_pos)
rank2 = 1 + sum(1 for k, v in obs2.items()
                if k != "DEFENDED" and not np.isnan(v) and v > DEF_STAT)
print(f"\nclass 2 observed rank of XLRE-vs-XLF: {rank2} of {len(obs2)-1} short legs")
print(f"  permutation, statistic = EPISODE MEAN of LONG XLRE/SHORT XLF (h={H})")
print(f"  UNCHARGED p (defended cell vs its own shifted null) = {p_unch:.4f}")
print(f"  CHARGED   p (max over K={len(arrays2)-1} short legs >= defended stat) "
      f"= {p_ch:.4f}")

# permutation, class 1: each sibling has its OWN trigger, so shift each mask
print("\nclass 1 permutation (each sibling shifted on its own trigger positions)")
rng = np.random.default_rng(7)
sib = {}
for s in SPDR:
    if s == "XLF":
        continue
    r = pair_ret(s, "XLF")
    sib[s] = (r.values, epi_positions(TNX_HI & r63_floor(s), r, H))
obs1 = {s: float(np.nanmean(v[p])) if len(p) else np.nan for s, (v, p) in sib.items()}
n = len(CAL)
ge_max1 = 0
NP = 4000
for _ in range(NP):
    off = int(rng.integers(1, n))
    best = -1e9
    for s, (v, p) in sib.items():
        if len(p) == 0:
            continue
        m = np.nanmean(v[(p + off) % n])
        if not np.isnan(m):
            best = max(best, m)
    if best >= DEF_STAT:
        ge_max1 += 1
rank1 = 1 + sum(1 for s, v in obs1.items()
                if s != "XLRE" and not np.isnan(v) and v > DEF_STAT)
print(f"  observed rank of XLRE: {rank1} of {len([v for v in obs1.values() if not np.isnan(v)])}")
print(f"  CHARGED max-of-K p (vs DEFENDED stat {100*DEF_STAT:+.3f}%) = "
      f"{(ge_max1+1)/(NP+1):.4f}")


# ==========================================================================
# 5. era split, 2022 shock, concentration, drop-best
# ==========================================================================
def concentration(long_t: str, mask: pd.Series, label: str) -> None:
    r = pair_ret(long_t, "XLF")
    e = episodes(mask, r, H)
    v = r.loc[e].values
    if len(v) == 0:
        return
    print(f"\n--- CONCENTRATION / ERA — {label} (N={len(v)} episodes)")
    print("  ", cluster_note(e, v))
    order = np.argsort(-v)
    keep = np.ones(len(v), bool)
    keep[order[:2]] = False
    show([summarize(v, "all episodes"),
          summarize(v[keep], "drop best 2")], "drop-best-2")
    yrs = pd.DatetimeIndex(e).year
    by = pd.Series(v).groupby(yrs.values).agg(["count", "mean", "sum"])
    by["mean"] = (100 * by["mean"]).round(3)
    by["sum"] = (100 * by["sum"]).round(3)
    print("  by year (mean_pct, sum_pct):")
    print(by.to_string())
    best_yr = by["sum"].idxmax()
    kb = yrs != best_yr
    show([summarize(v[kb], f"drop best year {best_yr}")], "drop-best-year")
    is22 = yrs == 2022
    if is22.any():
        show([summarize(v[is22], "2022 only"),
              summarize(v[~is22], "ex-2022")], "2022 rate shock")


concentration("XLRE", mask_join, "XLRE cell")
concentration("IYR", TNX_HI & r63_floor("IYR"), "IYR cell")
concentration("VNQ", TNX_HI & r63_floor("VNQ"), "VNQ cell")


# ==========================================================================
# 6. direction honesty + event-in-window + cost
# ==========================================================================
print("\n\n--- DIRECTION HONESTY: the momentum side (SHORT XLRE / LONG XLF)")
for t, m in (("XLRE", mask_join), ("IYR", TNX_HI & r63_floor("IYR")),
             ("VNQ", TNX_HI & r63_floor("VNQ"))):
    r = pair_ret(t, "XLF")
    e = episodes(m, r, H)
    if len(e) == 0:
        continue
    v = r.loc[e].values
    w = int((v > 0).sum())
    print(f"  {t}: fade  mean {100*v.mean():+.3f}%  record {w}-{len(v)-w}  "
          f"sign p {sign_test(w, len(v)):.4f}   |   momentum (negate) "
          f"mean {-100*v.mean():+.3f}%  record {len(v)-w}-{w}  "
          f"sign p {sign_test(len(v)-w, len(v)):.4f}")

print("\n--- EVENT IN WINDOW (episodes), CPI and FOMC separately")
for t, m in (("XLRE", mask_join), ("IYR", TNX_HI & r63_floor("IYR"))):
    r = pair_ret(t, "XLF")
    e = episodes(m, r, H)
    if len(e) == 0:
        continue
    v = r.loc[e].values
    for kind in (("cpi",), ("fomc_decision",), ("ppi",)):
        fl = event_in_window(e, CAL, H, 1, kind)
        show([summarize(v[fl], f"{kind[0]} IN (N={int(fl.sum())})"),
              summarize(v[~fl], f"{kind[0]} OUT (N={int((~fl).sum())})")],
             f"{t} — {kind[0]} in hold")

print(f"\n--- COST: 8 bps for the pair (2 bp/leg/side). "
      f"Defended episode mean = {100*DEF_STAT*100:.1f} bps "
      f"= {100*DEF_STAT*100/8:.1f}x cost (need >=5x).")
