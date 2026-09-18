import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

# C8 round 1b: the pitched state has ONE FOMC anchor, so test the flip on the definitions that
# produced the 09-14 kill number (thrust-union, r5>=95 alone, looser proximity), at k=-2 (today)
# and k=-3 (the kill's offset), with drop-best, mechanism split, residual, midterm/era, and a
# calendar-permutation charge over sign x horizon x offset.
pd.set_option("future.no_silent_downcasting", True)
raw = close_panel(["GLD", "GC=F", "^TNX", "DX-Y.NYB"])
tx = raw["^TNX"]
r5 = pct_rank(tx, 5)
hi252 = rolling_on_valid(tx, lambda x: x.rolling(252).max())
hi63 = rolling_on_valid(tx, lambda x: x.rolling(63).max())
DEFS = {
    "UNION (09-14 kill def)": ((r5 >= 90) & (tx >= hi63 - 1e-9)) | (r5 >= 95) | ((r5 >= 90) & (tx >= 0.98 * hi252)),
    "r5>=95 alone": r5 >= 95,
    "r5>=95 & within 1% max (pitched)": (r5 >= 95) & (tx >= 0.99 * hi252),
    "r5>=95 & within 3% max": (r5 >= 95) & (tx >= 0.97 * hi252),
    "r5>=90 & within 3% max": (r5 >= 90) & (tx >= 0.97 * hi252),
}
live = pd.Timestamp("2026-09-14")
print("LIVE membership:", {k: bool(v.loc[live]) for k, v in DEFS.items()})
fomc = pd.DatetimeIndex(load_events(["fomc_decision"])["date"])


def rec(v, label):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    r = summarize(v, label)
    if r["n"]:
        w = int((v > 0).sum())
        r["rec"] = f"{w}-{len(v)-w}"
        r["sign_p"] = round(sign_test(w, len(v)), 4)
        r["drop_best1"] = round(100 * np.sort(v)[:-1].mean(), 3) if len(v) > 1 else np.nan
        r["drop_best2"] = round(100 * np.sort(v)[:-2].mean(), 3) if len(v) > 2 else np.nan
    return r


for V in ["GLD", "GC=F"]:
    cal = raw[V].dropna().index
    g = raw[V].reindex(cal)
    TC = tx.reindex(cal).ffill(limit=1)
    DX = raw["DX-Y.NYB"].reindex(cal).ffill(limit=1)
    gdd = g / rolling_on_valid(g, lambda x: x.rolling(252).max()) - 1
    N = len(cal)
    fpos, fk = anchor_positions(cal, fomc, 0)
    fpos = np.array(fpos)
    fk = pd.DatetimeIndex(fk)
    print(f"\n==================== {V} ====================")
    for h in (1, 3, 5):
        ret = -fwd_lag(g, h, 1)
        dxr = fwd_lag(DX, h, 1)
        dy = (TC.shift(-(1 + h)) - TC.shift(-1)) * 100
        ok = ret.notna() & dxr.notna() & dy.notna()
        X = np.column_stack([np.ones(ok.sum()), dxr[ok].values, dy[ok].values])
        coef = np.linalg.lstsq(X, ret[ok].values, rcond=None)[0]
        resid = ret - (coef[0] + coef[1] * dxr + coef[2] * dy)
        for off in (-2, -3):
            sp = fpos + off
            okp = (sp >= 0) & (sp < N)
            d = cal[sp[okp]]
            fd = fk[okp]
            vals = ret.reindex(d).values
            mid = np.array([x.year % 4 == 2 for x in fd])
            post = d >= pd.Timestamp("2018-01-01")
            dyv = dy.reindex(d).values
            deep = (gdd.reindex(d) <= -0.10).values
            rows = [rec(vals, "PARENT all FOMC")]
            for name, s in DEFS.items():
                m = s.reindex(d).fillna(False).values.astype(bool)
                r = rec(vals[m], name)
                if r["n"]:
                    r["resid"] = round(100 * np.nanmean(resid.reindex(d).values[m]), 3)
                    r["dTNX_bp"] = round(float(np.nanmean(dyv[m])), 1)
                    r["TNXup_n"] = int((dyv[m] > 0).sum())
                rows.append(r)
                if name.startswith("UNION"):
                    for lbl, mm in [("  UNION & TNX rose", m & (dyv > 0)), ("  UNION & TNX fell", m & (dyv <= 0)),
                                    ("  UNION midterm", m & mid), ("  UNION non-midterm", m & ~mid),
                                    ("  UNION pre-2018", m & ~post), ("  UNION 2018+", m & post),
                                    ("  UNION & GLD>=10% off high (LIVE)", m & deep),
                                    ("  complement (no union)", ~m)]:
                        rows.append(rec(vals[mm], lbl))
            show(rows, f"{V} SHORT h={h} signal k={off} lag1 (resid = ret~DX+dTNX residual)")
            if off == -2 and h in (1, 5):
                m = DEFS["UNION (09-14 kill def)"].reindex(d).fillna(False).values.astype(bool)
                print("  UNION episodes:", [(str(x.date()), round(100 * ret.loc[x], 2), round(float(dy.loc[x]), 1)) for x in d[m]])

    # placebo ladder on UNION, short, h=1/3/5
    rows = []
    for off in range(-7, 4):
        row = {"k_signal": off}
        sp = fpos + off
        sp = sp[(sp >= 0) & (sp < N)]
        dd = cal[sp]
        m = DEFS["UNION (09-14 kill def)"].reindex(dd).fillna(False).values.astype(bool)
        for h in (1, 3, 5):
            x = (-fwd_lag(g, h, 1)).reindex(dd).values[m]
            x = x[~np.isnan(x)]
            row[f"n{h}"] = len(x)
            row[f"short_h{h}"] = round(100 * x.mean(), 3) if len(x) else np.nan
        rows.append(row)
    lad = pd.DataFrame(rows)
    print(f"\n{V} PLACEBO LADDER (UNION, short):")
    print(lad.to_string(index=False))
    for h in (1, 3, 5):
        c = f"short_h{h}"
        t0 = lad.loc[lad.k_signal == -2, c].values[0]
        print(f"  {c}: true k=-2 ranks {int((lad[c] > t0).sum()) + 1} of {lad[c].notna().sum()}")

    # calendar-permutation charge: circularly shift the FOMC position list; for each draw take the
    # best t over sign {+,-} x h 1..5 x offset k=-7..+3 on the UNION state (n>=4); compare to the
    # observed traded cell (short, k=-2) at each h.
    U = DEFS["UNION (09-14 kill def)"].reindex(cal).fillna(False).values.astype(bool)
    RET = {h: fwd_lag(g, h, 1).values for h in range(1, 6)}

    def grid_best(pos):
        best = -np.inf
        for off in range(-7, 4):
            sp = pos + off
            sp = sp[(sp >= 0) & (sp < N)]
            sp = sp[U[sp]]
            for h in range(1, 6):
                x = RET[h][sp]
                x = x[~np.isnan(x)]
                if len(x) < 4 or x.std(ddof=1) == 0:
                    continue
                t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
                best = max(best, abs(t))
        return best

    obs = {}
    for h in range(1, 6):
        sp = fpos - 2
        sp = sp[(sp >= 0) & (sp < N)]
        sp = sp[U[sp]]
        x = -RET[h][sp]
        x = x[~np.isnan(x)]
        obs[h] = x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 3 else np.nan
    rng = np.random.default_rng(7)
    null = []
    for _ in range(400):
        sh = int(rng.integers(30, N - 30))
        null.append(grid_best(np.sort((fpos + sh) % N)))
    null = np.array(null)
    gb = grid_best(fpos)
    print(f"\n{V} CHARGE (UNION, 2 signs x 5 h x 11 offsets, 400 circular calendar shifts): observed-grid best |t| {gb:.2f}, "
          f"charged P(null best >= it) = {(null >= gb).mean():.3f}; null median {np.median(null):.2f}")
    for h in range(1, 6):
        print(f"  traded short k=-2 h={h}: t {obs[h]:+.2f}  charged P = {((null >= obs[h]).mean() if obs[h] > 0 else float('nan')):.3f} (nan = SHORT wrong-signed, nothing to charge)")
print("\ncost: GLD 5 bp round trip; 5x bar = +0.25%")
