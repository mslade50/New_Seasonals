"""Test 5: hysteresis band vs plain threshold on the live multiplier signal.
Counts regime flips/year and short-lived episodes; checks trade-level impact.
Also: LOYO on the 21d/21MA runner-up and its overlap with 63d>=50."""
import numpy as np
import pandas as pd

from dial_setup import DIALS, clustered_t, load_frag, load_trades_joined, smooth

frag = load_frag()
sm10 = smooth(frag, 10).dropna()
s63 = sm10["63d"]
years = (s63.index[-1] - s63.index[0]).days / 365.25


def regime_series(s: pd.Series, enter: float, exit_: float) -> pd.Series:
    """True while in high-frag regime; hysteresis when exit_ < enter."""
    state, out = False, []
    for v in s.values:
        if not state and v >= enter:
            state = True
        elif state and v < exit_:
            state = False
        out.append(state)
    return pd.Series(out, index=s.index)


def episode_stats(reg: pd.Series):
    flips = reg.astype(int).diff().abs().sum()
    grp = (reg != reg.shift()).cumsum()
    eps = reg.groupby(grp).agg(["first", "size"])
    on_eps = eps[eps["first"]]
    return {
        "flips/yr": flips / years,
        "episodes": len(on_eps),
        "days_on_pct": reg.mean() * 100,
        "median_ep_len": on_eps["size"].median() if len(on_eps) else np.nan,
        "eps<=5d": int((on_eps["size"] <= 5).sum()),
        "eps<=10d": int((on_eps["size"] <= 10).sum()),
    }


t = load_trades_joined(10)
nb = t[~t.is_ovs].copy()
# map each trade's signal date to regime state
print("=== TEST 5: hysteresis vs plain threshold, 63d 10d-MA ===")
print(f"(daily series {s63.index[0].date()}..{s63.index[-1].date()}, {years:.1f} yrs)\n")
configs = [
    ("plain 50", 50, 50), ("hyst 50/40", 50, 40), ("hyst 50/45", 50, 45),
    ("plain 55", 55, 55), ("hyst 55/45", 55, 45),
    ("plain 47.5", 47.5, 47.5), ("hyst 55/40", 55, 40),
]
rows = []
for name, en, ex in configs:
    reg = regime_series(s63, en, ex)
    st = episode_stats(reg)
    m = nb["Signal Date"].map(reg).fillna(False).astype(bool)
    hi, lo = nb[m], nb[~m]
    ts, p, nmh, _ = clustered_t(hi, lo)
    rows.append({"config": name, **{k: round(v, 2) for k, v in st.items()},
                 "N_tr_hi": len(hi), "avgR_hi": round(hi.R_Multiple.mean(), 3),
                 "avgR_lo": round(lo.R_Multiple.mean(), 3),
                 "t": round(ts, 2), "p": round(p, 3)})
print(pd.DataFrame(rows).to_string(index=False))

# LOYO on hysteresis 50/40
print("\nLOYO, hyst 50/40:")
reg = regime_series(s63, 50, 40)
m_all = nb["Signal Date"].map(reg).fillna(False).astype(bool)
ts_all = []
for yr in sorted(nb.yr.unique()):
    sub = nb[nb.yr != yr]
    m = m_all.loc[sub.index]
    ts, p, _, _ = clustered_t(sub[m], sub[~m])
    ts_all.append((yr, ts))
print("  t range [{:+.2f}, {:+.2f}], weakest dropping {} ({:+.2f})".format(
    min(x[1] for x in ts_all), max(x[1] for x in ts_all),
    max(ts_all, key=lambda x: x[1])[0], max(x[1] for x in ts_all)))

# ---- runner-up scrutiny: 21d dial with 21d MA at p90 ----
print("\n=== 21d dial, 21-day MA, thr=p90 (~36) — LOYO + overlap with 63d>=50 ===")
t21 = load_trades_joined(21)
nb21 = t21[~t21.is_ovs].copy()
sm21 = smooth(frag, 21).dropna()
thr21 = sm21["21d"].quantile(0.90)
m21 = nb21["frag_21d"] >= thr21
ts_all = []
for yr in sorted(nb21.yr.unique()):
    sub = nb21[nb21.yr != yr]
    m = m21.loc[sub.index]
    ts, p, _, _ = clustered_t(sub[m], sub[~m])
    ts_all.append((yr, ts))
print(f"thr={thr21:.1f}, N_hi={m21.sum()}, LOYO t range "
      f"[{min(x[1] for x in ts_all):+.2f}, {max(x[1] for x in ts_all):+.2f}], "
      f"weakest dropping {max(ts_all, key=lambda x: x[1])[0]}")
# overlap with 63d>=50 (10d MA) at trade level — align on shared trades
shared = nb.index.intersection(nb21.index)
a = (nb.loc[shared, "frag_63d"] >= 50)
b = m21.loc[shared]
print(f"trade overlap: 63d-flag {a.sum()}, 21d/21-flag {b.sum()}, "
      f"both {(a & b).sum()}, jaccard {(a & b).sum() / (a | b).sum():.2f}")
# does 21d/21 add anything on top of 63d>=50?
extra = shared[b & ~a]
print(f"trades flagged ONLY by 21d/21: N={len(extra)}, "
      f"avgR={nb.loc[extra, 'R_Multiple'].mean():+.3f}")

# ---- 2026 YTD inversion check for chosen rule ----
print("\n=== 2026 YTD check (established caveat) ===")
for name, mask in [("63d>=50 plain", nb.frag_63d >= 50),
                   ("hyst 50/40", m_all)]:
    y26 = nb[nb.yr == 2026]
    m = mask.loc[y26.index]
    print(f"{name}: 2026 flagged avgR {y26[m].R_Multiple.mean():+.2f} (N={m.sum()}), "
          f"unflagged {y26[~m].R_Multiple.mean():+.2f} (N={(~m).sum()})")
