"""B1 round 1 -- the pre-FOMC midterm cross-asset family, done honestly.

Geometry (today's, exactly): signal close D = decision - 11 td, entry MOC at
D+1 = decision - 10 td, hold h=10 -> exit ON the decision close.
2026-08-31 signal, 2026-09-01 entry, 2026-09-16 decision.

What the map (00b) did NOT do and this does:
  - lag ladder 0/1/2 (registry: an effect one session wide starting a session
    LATE has no shape -- killed the closest candidate on 2026-08-31)
  - declustering VERIFIED rather than assumed
  - the three controls (own drift same span / all days / local +-126td)
  - a family multiplicity charge over the grid actually walked
    (15 classes x 2 signs), by permutation max-of-k with the cross-sectional
    correlation preserved (same random dates across classes)
  - homogeneity: Cochran Q + I-squared (five families closed this way here)
  - the placebo offset ladder k=-14..-6 (currently 12-for-12 as a killer)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd
from scipy import stats

ASOF = pd.Timestamp("2026-08-31")
H = 10           # hold from the entry close to the decision close
K_ENTRY = -10    # entry session sits 10 td before the decision
K_SIG = -11      # signal session (entry is lag=1 off this)

CLASSES = {
    "us_large": "SPY", "us_small": "IWM", "rates": "TLT", "rates_belly": "IEF",
    "credit": "HYG", "gold": "GLD", "miners": "GDX", "metals": "SLV",
    "energy": "USO", "energy_eq": "XLE", "dollar": "UUP", "intl_dev": "EFA",
    "intl_em": "EEM", "vol_inv": "SVXY", "tech": "XLK",
}

px = load_prices(sorted(set(CLASSES.values())))
S = {k: px[v]["Close"].dropna()[lambda s: s.index <= ASOF] for k, v in CLASSES.items()}

ev = load_events(["fomc_decision", "vix_expiry"])
FOM = pd.DatetimeIndex(sorted(ev.loc[ev["event"] == "fomc_decision", "date"].unique()))
FOM = FOM[FOM <= ASOF]
VIXEXP = set(pd.DatetimeIndex(sorted(ev.loc[ev["event"] == "vix_expiry", "date"].unique())))
MID = pd.DatetimeIndex([d for d in FOM if d.year % 4 == 2])

print("FOMC decisions on/before %s: %d (%s .. %s) | midterm %d"
      % (ASOF.date(), len(FOM), FOM[0].date(), FOM[-1].date(), len(MID)))


# ---------------------------------------------------------------- geometry
def window_ret(s: pd.Series, anchors: pd.DatetimeIndex, k_entry: int,
               h: int) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Return (decision dates kept, return from close[dec+k_entry] to
    close[dec+k_entry+h]).  Uses anchor_positions so pre-inception anchors
    cannot collapse onto the opening bars."""
    pos, kept = anchor_positions(s.index, anchors, offset=k_entry)
    out_d, out_v = [], []
    v = s.values
    for p, d in zip(pos, kept):
        if p < 0 or p + h >= len(v):
            continue
        out_d.append(d)
        out_v.append(v[p + h] / v[p] - 1.0)
    return pd.DatetimeIndex(out_d), np.asarray(out_v, float)


def drift(s: pd.Series, h: int, span: tuple | None = None) -> np.ndarray:
    r = (s.shift(-h) / s - 1.0).dropna()
    if span:
        r = r[(r.index >= span[0]) & (r.index <= span[1])]
    return r.values


# ------------------------------------------------- 0. decluster VERIFICATION
print("\n== 0. declustering: do consecutive [dec-10, dec] windows ever overlap?")
spy = S["us_large"]
pos, kept = anchor_positions(spy.index, FOM, offset=K_ENTRY)
gaps = np.diff(pos)
print("   consecutive entry-session gaps (td): min %d  p05 %d  median %d  max %d"
      % (gaps.min(), np.percentile(gaps, 5), np.median(gaps), gaps.max()))
print("   windows are %d td long; overlapping pairs = %d of %d"
      % (H, int((gaps <= H).sum()), len(gaps)))
posm, keptm = anchor_positions(spy.index, MID, offset=K_ENTRY)
gm = np.diff(posm)
print("   midterm-only gaps: min %d median %d -> overlaps %d"
      % (gm.min(), np.median(gm), int((gm <= H).sum())))
print("   VERDICT: declustering is a NO-OP on this anchor (each decision is its"
      " own episode); every N below is already an episode count.")


# --------------------------------------------- 1. lag ladder + the 3 controls
def block(name: str, anchors: pd.DatetimeIndex, label: str) -> dict:
    s = S[name]
    d, v = window_ret(s, anchors, K_ENTRY, H)
    if len(v) < 5:
        return {"class": name, "n": len(v)}
    span = (d[0], d[-1])
    ctrl_a = drift(s, H, span)
    ctrl_b = drift(s, H)
    loc = local_control(s.index, pd.DatetimeIndex(
        [s.index[p] for p in anchor_positions(s.index, anchors, K_ENTRY)[0]]), 126)
    r_loc = (s.shift(-H) / s - 1.0).reindex(loc).dropna().values
    wins = int((v > 0).sum())
    return {
        "class": name, "tic": CLASSES[name], "n": len(v),
        "cond_pct": 100 * v.mean(),
        "ctrlA_same_span": 100 * ctrl_a.mean(),
        "ctrlB_all_days": 100 * ctrl_b.mean(),
        "ctrlC_local126": 100 * r_loc.mean() if len(r_loc) else np.nan,
        "edge_vs_A_pp": 100 * (v.mean() - ctrl_a.mean()),
        "edge_vs_C_pp": 100 * (v.mean() - r_loc.mean()) if len(r_loc) else np.nan,
        "hit": 100 * wins / len(v),
        "t_vs_A": (v.mean() - ctrl_a.mean()) / np.sqrt(
            v.var(ddof=1) / len(v) + ctrl_a.var(ddof=1) / len(ctrl_a)),
        "sign_p": sign_test(max(wins, len(v) - wins), len(v)),
    }


for lbl, anchors in (("FULL SAMPLE", FOM), ("MIDTERM ONLY", MID)):
    rows = [block(k, anchors, lbl) for k in CLASSES]
    show(rows, f"1. {lbl}: entry dec-10 MOC -> decision close, vs three controls")

print("\n== 1b. LAG LADDER (signal D = dec-11; lag 0/1/2 -> entry dec-11/-10/-9,"
      " hold 10)")
for lbl, anchors in (("full", FOM), ("midterm", MID)):
    rows = []
    for lag in (0, 1, 2):
        r = {"sample": lbl, "lag": lag, "entry_at": f"dec{K_SIG + lag:+d}"}
        for k in ("us_large", "energy", "energy_eq", "tech", "vol_inv"):
            d, v = window_ret(S[k], anchors, K_SIG + lag, H)
            base = drift(S[k], H, (d[0], d[-1]))
            r[k] = round(100 * (v.mean() - base.mean()), 3)
        rows.append(r)
    show(rows, f"lag ladder, edge vs own drift (pp), {lbl}")


# ------------------------------------- 2. family charge on the MIDTERM slice
print("\n== 2. FAMILY MULTIPLICITY CHARGE (the grid walked: 15 classes x 2 signs)")
obs = {}
for k in CLASSES:
    d, v = window_ret(S[k], MID, K_ENTRY, H)
    if len(v) < 5:
        continue
    base = drift(S[k], H, (d[0], d[-1]))
    e = v.mean() - base.mean()
    se = np.sqrt(v.var(ddof=1) / len(v) + base.var(ddof=1) / len(base))
    obs[k] = (e, se, len(v))

e_arr = np.array([obs[k][0] for k in obs])
se_arr = np.array([obs[k][1] for k in obs])
w = 1.0 / se_arr ** 2
ebar = float((w * e_arr).sum() / w.sum())
Q = float((w * (e_arr - ebar) ** 2).sum())
df = len(e_arr) - 1
I2 = max(0.0, (Q - df) / Q) if Q > 0 else 0.0
print("   fixed-effect common excess = %+.3fpp (SE %.3fpp, z %+.2f)"
      % (100 * ebar, 100 / np.sqrt(w.sum()), ebar * np.sqrt(w.sum())))
print("   Cochran Q = %.2f on %d df, p = %.4f | I-squared = %.1f%%"
      % (Q, df, 1 - stats.chi2.cdf(Q, df), 100 * I2))
print("   cross-sectional sd of edges %.3fpp vs mean sampling SE %.3fpp -> ratio %.2f"
      % (100 * e_arr.std(ddof=1), 100 * se_arr.mean(),
         e_arr.std(ddof=1) / se_arr.mean()))

# permutation max-of-k with cross-sectional correlation preserved
rng = np.random.default_rng(42)
n_mid = len(MID)
cal = spy.index[(spy.index >= pd.Timestamp("2002-08-01"))]  # all classes alive-ish
cal = cal[:-(H + 15)]
NB = 4000
maxabs = np.zeros(NB)
pre = {}
for k in obs:
    s = S[k]
    pre[k] = (s, (s.shift(-H) / s - 1.0))
for b in range(NB):
    dates = pd.DatetimeIndex(rng.choice(cal, size=n_mid, replace=False))
    best = 0.0
    for k in obs:
        s, fr = pre[k]
        pos, kept = anchor_positions(s.index, dates, offset=K_ENTRY)
        if len(pos) < 10:
            continue
        vv = fr.iloc[pos].dropna().values
        if len(vv) < 10:
            continue
        base = fr.dropna()
        base = base[(base.index >= kept[0]) & (base.index <= kept[-1])]
        best = max(best, abs(vv.mean() - base.mean()))
    maxabs[b] = best
obs_max = float(np.abs(e_arr).max())
obs_max_k = list(obs)[int(np.argmax(np.abs(e_arr)))]
print("   observed max |edge| = %.3fpp on '%s'" % (100 * obs_max, obs_max_k))
print("   permutation max-of-15 (%d random date sets of size %d, correlation preserved):"
      % (NB, n_mid))
print("     P(max |edge| >= observed) = %.4f   [null median %.3fpp, p95 %.3fpp]"
      % ((maxabs >= obs_max).mean(), 100 * np.median(maxabs),
         100 * np.percentile(maxabs, 95)))
for k in sorted(obs, key=lambda x: -abs(obs[x][0])):
    e, se, n = obs[k]
    print("     %-11s edge %+.3fpp  n=%d  P(perm max >= |this|) = %.4f"
          % (k, 100 * e, n, (maxabs >= abs(e)).mean()))


# -------------------------------------------- 3. the placebo offset ladder
print("\n== 3. PLACEBO OFFSET LADDER (entry at dec+k, hold 10; k=-10 is the true"
      " anchor, the only k whose exit lands ON the decision)")
for lbl, anchors in (("full", FOM), ("midterm", MID)):
    rows = []
    for k in range(-16, -4):
        r = {"sample": lbl, "k": k, "exit_at": f"dec{k + H:+d}"}
        for cls in ("us_large", "energy", "energy_eq", "tech", "rates"):
            d, v = window_ret(S[cls], anchors, k, H)
            if len(v) < 5:
                r[cls] = np.nan
                continue
            base = drift(S[cls], H, (d[0], d[-1]))
            r[cls] = round(100 * (v.mean() - base.mean()), 3)
        rows.append(r)
    show(rows, f"placebo ladder, edge vs own drift (pp), {lbl}")
    df_l = pd.DataFrame(rows).set_index("k")
    for cls in ("us_large", "energy", "energy_eq", "tech", "rates"):
        col = df_l[cls].astype(float)
        rank = int((col.abs() >= abs(col.loc[-10])).sum())
        print("   %-10s true anchor |edge| ranks %d of %d in the ladder"
              % (cls, rank, col.notna().sum()))
