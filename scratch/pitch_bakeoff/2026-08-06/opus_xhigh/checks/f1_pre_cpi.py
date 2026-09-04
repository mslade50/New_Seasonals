"""F -- PRE-CPI DRIFT (axis: event_fingerprint).

Adversarial check for the 2026-08-06 Daily Pitch.

Geometry under test (matches today exactly):
    print bar p = the CPI release session (2026-08-12)
    entry       = MOO at bar p-k, k in {3,4,5}   (today: k=4 -> 2026-08-06)
    exit 'eve'  = MOC at bar p-1                 (today: 2026-08-11)
    exit 'print'= MOC at bar p                   (today: 2026-08-12)

Grid = 4 assets x 3 entry offsets x 2 exits = 24 cells, all reported.
Controls: unconditional same-length MOO->MOC hold, two multiplicity nulls
(random pseudo-events + circular trading-day shift of the real calendar).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

ROOT = C.ROOT
ASSETS = ["SPY", "TLT", "GLD", "DX-Y.NYB"]
OFFSETS = [3, 4, 5]
EXITS = ["eve", "print"]
RNG = np.random.default_rng(20260806)
NSIM = 3000

pd.set_option("display.width", 200)


def hr(title: str) -> None:
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)


# ----------------------------------------------------------------------------
# 0. events
# ----------------------------------------------------------------------------
def load_events() -> dict[str, pd.DatetimeIndex]:
    raw = pd.read_csv(ROOT / "data" / "macro_events.csv", parse_dates=["date"])
    raw["date"] = raw["date"].dt.normalize()
    out = {}
    for ev in ("cpi", "nfp", "ppi"):
        d = raw.loc[raw["event"] == ev, "date"].drop_duplicates().sort_values()
        out[ev] = pd.DatetimeIndex(d.values)
    return out


def dedupe_monthly(dates: pd.DatetimeIndex) -> tuple[pd.DatetimeIndex, list]:
    """One event per calendar month (keep first). Returns (kept, dropped)."""
    s = pd.Series(dates, index=dates)
    key = dates.to_period("M")
    keep_mask = ~pd.Series(key).duplicated(keep="first").to_numpy()
    return dates[keep_mask], list(dates[~keep_mask])


# ----------------------------------------------------------------------------
# 1. asset container
# ----------------------------------------------------------------------------
@dataclass
class Asset:
    name: str
    idx: pd.DatetimeIndex
    op: np.ndarray
    cl: np.ndarray
    hi: np.ndarray
    lo: np.ndarray

    def pos_of(self, dates: pd.DatetimeIndex) -> np.ndarray:
        p = self.idx.searchsorted(dates, side="left")
        return p[p < len(self.idx)]


def build_assets(px: dict[str, pd.DataFrame]) -> dict[str, Asset]:
    out = {}
    for t, d in px.items():
        out[t] = Asset(t, d.index, d["Open"].to_numpy(), d["Close"].to_numpy(),
                       d["High"].to_numpy(), d["Low"].to_numpy())
    return out


def cell(a: Asset, ppos: np.ndarray, k: int, exit_kind: str):
    """Returns (entry_positions, exit_positions, pct returns)."""
    e = ppos - k
    x = ppos - 1 if exit_kind == "eve" else ppos
    ok = (e >= 0) & (x < len(a.cl))
    e, x = e[ok], x[ok]
    r = (a.cl[x] / a.op[e] - 1.0) * 100.0
    good = np.isfinite(r)
    return e[good], x[good], r[good]


def uncond(a: Asset, h: int, lo_pos: int, hi_pos: int) -> np.ndarray:
    """All-bar MOO->MOC hold of h sessions, restricted to [lo_pos, hi_pos]."""
    i = np.arange(lo_pos, min(hi_pos, len(a.cl) - h - 1) + 1)
    r = (a.cl[i + h] / a.op[i] - 1.0) * 100.0
    return r[np.isfinite(r)]


def welch_t(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x), np.asarray(y)
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    return float((x.mean() - y.mean()) / np.sqrt(vx + vy))


# ----------------------------------------------------------------------------
def main() -> None:
    ev = load_events()
    cpi_all = ev["cpi"]
    cpi, dropped = dedupe_monthly(cpi_all)

    hr("0. EVENT SOURCE VERIFICATION (data/macro_events.csv)")
    print(f"raw CPI rows          : {len(cpi_all)}")
    print(f"date range            : {cpi_all.min().date()} .. {cpi_all.max().date()}")
    print(f"dropped as duplicate-month scrape artifacts: {[str(d.date()) for d in dropped]}")
    print(f"CPI events used       : {len(cpi)}")
    yr = pd.Series(1, index=cpi).groupby(cpi.year).sum()
    print("per-year counts:")
    print("  " + "  ".join(f"{y}:{n}" for y, n in yr.items()))
    print(f"\nnext scheduled CPI after 2026-08-05: "
          f"{cpi[cpi > pd.Timestamp('2026-08-05')][0].date()}")
    print(f"NFP dates near today : "
          f"{[str(d.date()) for d in ev['nfp'] if pd.Timestamp('2026-08-01') <= d <= pd.Timestamp('2026-08-31')]}")
    print(f"PPI dates near today : "
          f"{[str(d.date()) for d in ev['ppi'] if pd.Timestamp('2026-08-01') <= d <= pd.Timestamp('2026-08-31')]}")

    px = C.load(ASSETS)
    A = build_assets(px)
    hr("0b. PRICE DATA")
    for t, a in A.items():
        syn = float(np.mean(a.op[1:] == a.cl[:-1])) * 100
        print(f"{t:10s} bars={len(a.idx):5d}  {a.idx[0].date()} .. {a.idx[-1].date()}  "
              f"Open==prior Close on {syn:5.1f}% of bars")

    # ------------------------------------------------------------------
    # 1. FULL GRID, every cell reported
    # ------------------------------------------------------------------
    hr("1. FULL PRE-SPECIFIED GRID -- 4 assets x 3 entry offsets x 2 exits = 24 cells")
    print("entry = MOO at bar p-k ; exit eve = MOC at p-1 ; exit print = MOC at p")
    print("hold  = sessions from entry OPEN to exit CLOSE (eve: k-1, print: k)")
    print("uncond = same-length MOO->MOC hold over ALL bars in the treated span")
    print("lift_t = Welch t of (cell - uncond)\n")

    rows = []
    store = {}
    for t in ASSETS:
        a = A[t]
        ppos = a.pos_of(cpi)
        for k in OFFSETS:
            for xk in EXITS:
                e, x, r = cell(a, ppos, k, xk)
                h = k - 1 if xk == "eve" else k
                u = uncond(a, h, int(e.min()), int(e.max()))
                store[(t, k, xk)] = dict(e=e, x=x, r=r, h=h, u=u)
                rows.append(dict(
                    asset=t, k=k, exit=xk, hold=h, n=len(r),
                    avg=round(r.mean(), 4), med=round(float(np.median(r)), 4),
                    hit=round((r > 0).mean() * 100, 1), t=round(C.tstat(r), 2),
                    uncond=round(u.mean(), 4), lift=round(r.mean() - u.mean(), 4),
                    lift_t=round(welch_t(r, u), 2),
                    worst=round(r.min(), 2), best=round(r.max(), 2)))
    grid = pd.DataFrame(rows)
    print(grid.to_string(index=False))

    print(f"\nCELLS EXAMINED: {len(grid)}  (4 assets x 3 offsets x 2 exits)")
    imax = grid["t"].abs().idxmax()
    obs_max_t = float(grid.loc[imax, "t"])
    print(f"max |t| in grid  : {abs(obs_max_t):.2f}  -> "
          f"{grid.loc[imax,'asset']} k={grid.loc[imax,'k']} exit={grid.loc[imax,'exit']}")
    imax2 = grid["lift_t"].abs().idxmax()
    print(f"max |lift_t|     : {abs(float(grid.loc[imax2,'lift_t'])):.2f} -> "
          f"{grid.loc[imax2,'asset']} k={grid.loc[imax2,'k']} exit={grid.loc[imax2,'exit']}")

    # ------------------------------------------------------------------
    # 2. MULTIPLICITY NULLS on max|t| across the whole grid
    # ------------------------------------------------------------------
    hr("2. MULTIPLICITY NULL on max |t| across all 24 cells")
    cal = A["SPY"].idx           # master trading calendar
    real_pos_spy = A["SPY"].pos_of(cpi)
    m = len(real_pos_spy)

    def grid_max_t(pseudo_dates: pd.DatetimeIndex) -> float:
        best = 0.0
        for t in ASSETS:
            a = A[t]
            pp = a.pos_of(pseudo_dates)
            for k in OFFSETS:
                for xk in EXITS:
                    _, _, r = cell(a, pp, k, xk)
                    if len(r) < 20:
                        continue
                    tv = abs(C.tstat(r))
                    if np.isfinite(tv) and tv > best:
                        best = tv
        return best

    # Null A: random pseudo-event dates on the master calendar (same count)
    nullA = np.empty(NSIM)
    lo, hi = 6, len(cal) - 2
    for s in range(NSIM):
        p = RNG.choice(np.arange(lo, hi), size=m, replace=False)
        nullA[s] = grid_max_t(pd.DatetimeIndex(cal[np.sort(p)]))

    # Null B: circular shift of the REAL CPI calendar by a random td offset
    nullB = np.empty(NSIM)
    shifts = RNG.integers(6, 250, size=NSIM) * RNG.choice([-1, 1], size=NSIM)
    for s in range(NSIM):
        p = (real_pos_spy + shifts[s]) % len(cal)
        nullB[s] = grid_max_t(pd.DatetimeIndex(cal[np.sort(p)]))

    for nm, nl in (("A random-position", nullA), ("B circular-shift", nullB)):
        pv = float((nl >= abs(obs_max_t)).mean())
        print(f"null {nm}: sims={NSIM}  median max|t|={np.median(nl):.2f}  "
              f"p90={np.percentile(nl,90):.2f}  p95={np.percentile(nl,95):.2f}  "
              f"p99={np.percentile(nl,99):.2f}   "
              f"P(null max|t| >= {abs(obs_max_t):.2f}) = {pv:.3f}")

    print("\nPer-cell family-wise p (fraction of null grids whose MAX |t| beat that cell):")
    fam = []
    for _, rw in grid.iterrows():
        tv = abs(rw["t"])
        fam.append(dict(asset=rw["asset"], k=rw["k"], exit=rw["exit"], t=rw["t"],
                        fwer_pA=round(float((nullA >= tv).mean()), 3),
                        fwer_pB=round(float((nullB >= tv).mean()), 3)))
    print(pd.DataFrame(fam).to_string(index=False))

    survivors = [(r["asset"], r["k"], r["exit"]) for r in fam
                 if r["fwer_pB"] <= 0.10 and r["fwer_pA"] <= 0.10]
    print(f"\nCells clearing BOTH multiplicity nulls at p<=0.10: {survivors or 'NONE'}")

    # ------------------------------------------------------------------
    # 3. FULL ATTACK on the primary geometry (k=4) for every asset
    #    (run for all four regardless, so the reader sees the whole picture)
    # ------------------------------------------------------------------
    hr("3. FULL ATTACK -- today's geometry k=4, both exits, ALL FOUR assets")

    nfp_pos = {t: set(A[t].pos_of(ev["nfp"]).tolist()) for t in ASSETS}
    cpi_pos = {t: set(A[t].pos_of(cpi).tolist()) for t in ASSETS}

    # trading-day-of-month for each asset
    tdom = {}
    for t in ASSETS:
        idx = A[t].idx
        tdom[t] = pd.Series(1, index=idx).groupby([idx.year, idx.month]).cumsum().to_numpy()

    for t in ASSETS:
        a = A[t]
        for xk in EXITS:
            d = store[(t, 4, xk)]
            e, x, r, h = d["e"], d["x"], d["r"], d["h"]
            dates = a.idx[e]
            print("\n" + "-" * 92)
            print(f"### {t}  k=4  exit={xk}  hold={h} sessions  "
                  f"({dates[0].date()} .. {dates[-1].date()})")
            print("-" * 92)
            print(pd.DataFrame([C.describe("all signals", r, d["u"])]).to_string(index=False))

            # (a) eras
            print("\n(a) ERA SPLIT")
            er = []
            for lab, msk in (("pre-2013", dates < "2013-01-01"),
                             ("2013-2018", (dates >= "2013-01-01") & (dates < "2019-01-01")),
                             ("2019+", dates >= "2019-01-01"),
                             ("--pre-2018", dates < "2018-01-01"),
                             ("--2018+", dates >= "2018-01-01")):
                if msk.sum() >= 2:
                    er.append(C.describe(lab, r[msk]))
            C.show(er)

            # (b) episodes + LOYO
            print("\n(b) EPISODES + LOYO")
            eprows = []
            for gap in (10, 21):
                km = C.declusterize(dates, gap_td=gap)
                rr, dd = r[km], dates[km]
                eprows.append(dict(gap=gap, n_ep=int(km.sum()),
                                   avg=round(rr.mean(), 4), t=round(C.tstat(rr), 2)))
                yrs = pd.Series(dd.year)
                loyo = []
                for y in sorted(yrs.unique()):
                    m2 = (yrs != y).to_numpy()
                    if m2.sum() > 5:
                        loyo.append((y, C.tstat(rr[m2])))
                if loyo:
                    fl = min(loyo, key=lambda z: z[1])
                    eprows[-1]["loyo_floor_t"] = round(fl[1], 2)
                    eprows[-1]["loyo_worst_yr"] = fl[0]
            C.show(eprows)

            # (c) worst window / worst year / year table
            print("\n(c) WORST WINDOW / YEAR TABLE")
            wi = int(np.argmin(r))
            print(f"worst single window: {dates[wi].date()} -> {a.idx[x[wi]].date()}  "
                  f"{r[wi]:+.2f}%   |  best: {dates[int(np.argmax(r))].date()} "
                  f"{r.max():+.2f}%")
            yt = pd.DataFrame({"year": dates.year, "r": r}).groupby("year")["r"].agg(
                n="size", avg="mean", tot="sum", hit=lambda z: (z > 0).mean() * 100)
            yt = yt.round(3)
            print(yt.to_string())
            print(f"worst year: {yt['tot'].idxmin()} tot={yt['tot'].min():+.2f}%  |  "
                  f"years negative: {(yt['tot'] < 0).sum()}/{len(yt)}")

            # (d) trading-day-of-month matched control
            print("\n(d) TRADING-DAY-OF-MONTH MATCHED CONTROL")
            td_t = tdom[t][e]
            n = len(a.cl)
            all_i = np.arange(0, n - h - 1)
            td_all = tdom[t][all_i]
            # window [i, i+h] must contain NO cpi print position
            cps = np.zeros(n, dtype=bool)
            cps[list(cpi_pos[t])] = True
            cum = np.concatenate([[0], np.cumsum(cps)])
            has_cpi = (cum[all_i + h + 1] - cum[all_i]) > 0
            in_span = (all_i >= e.min()) & (all_i <= e.max())
            r_all = (a.cl[all_i + h] / a.op[all_i] - 1.0) * 100.0
            ctrl_ok = (~has_cpi) & in_span & np.isfinite(r_all)
            strat, wts = [], []
            for v in sorted(set(td_t.tolist())):
                tv_ = r[td_t == v]
                cv = r_all[ctrl_ok & (td_all == v)]
                if len(cv) >= 5 and len(tv_) >= 3:
                    strat.append(dict(tdom=int(v), n_treat=len(tv_),
                                      avg_treat=round(tv_.mean(), 4),
                                      n_ctrl=len(cv), avg_ctrl=round(cv.mean(), 4),
                                      diff=round(tv_.mean() - cv.mean(), 4)))
                    wts.append(len(tv_))
            if strat:
                sd = pd.DataFrame(strat)
                print(sd.to_string(index=False))
                w = np.array(wts, dtype=float)
                print(f"WEIGHTED stratified diff (treat - same-tdom no-CPI control): "
                      f"{np.average(sd['diff'], weights=w):+.4f}%  "
                      f"[raw cell avg {r.mean():+.4f}%, "
                      f"pooled tdom-matched control avg "
                      f"{r_all[ctrl_ok & np.isin(td_all, list(set(td_t.tolist())))].mean():+.4f}%]")
            else:
                print("  no usable tdom-matched control strata (CPI saturates these tdom slots)")

            # (e) NFP contamination
            print("\n(e) NFP CONTAMINATION OF THE WINDOW")
            nf = np.zeros(n, dtype=bool)
            nf[list(nfp_pos[t])] = True
            cn = np.concatenate([[0], np.cumsum(nf)])
            has_nfp = (cn[x + 1] - cn[e]) > 0
            print(f"windows containing an NFP print: {has_nfp.sum()}/{len(r)} "
                  f"({has_nfp.mean()*100:.0f}%)   [TODAY'S WINDOW DOES]")
            C.show([C.describe("NFP inside (like today)", r[has_nfp]),
                    C.describe("no NFP inside", r[~has_nfp])])

            # (f) today's state sub-cell
            print("\n(f) TODAY'S-STATE SUB-CELL")
            close = pd.Series(a.cl, index=a.idx)
            r5 = C.pct_rank(close.pct_change(5), 252).to_numpy()
            r21 = C.pct_rank(close.pct_change(21), 252).to_numpy()
            r63 = C.pct_rank(close.pct_change(63), 252).to_numpy()
            sma200 = close.rolling(200).mean().to_numpy()
            hi52 = close.rolling(252).max().to_numpy()
            prior = e - 1                     # state known at the pre-entry close
            ok = prior >= 0
            if t == "SPY":
                st = (close.to_numpy()[prior] >= hi52[prior] * 0.999) & (r5[prior] >= 80)
                lab = "SPY at ~52w high AND 5d rank>=80 (today)"
            elif t == "DX-Y.NYB":
                st = (r5[prior] <= 25) & (r63[prior] >= 70)
                lab = "DX 5d rank<=25 AND 63d rank>=70 (today)"
            elif t == "TLT":
                st = (r21[prior] <= 33) & (close.to_numpy()[prior] < sma200[prior])
                lab = "TLT 21d rank<=33 AND below 200d SMA (today)"
            else:
                st = (close.to_numpy()[prior] < sma200[prior]) & \
                     (close.to_numpy()[prior] <= hi52[prior] * 0.85)
                lab = "GLD below 200d SMA AND >=15% off 52w high (today)"
            st = st & ok & np.isfinite(st.astype(float))
            C.show([C.describe(lab, r[st]), C.describe("rest of the cell", r[~st])])
            print(f"today's state sub-cell is the "
                  f"{'STRONG' if (st.sum() and r[st].mean() > r[~st].mean()) else 'WEAK'} half")

    # ------------------------------------------------------------------
    hr("4. COST BENCHMARK")
    print("round-trip cost assumptions: SPY/TLT/GLD 1-3 bps, DX futures ~1.5 bps")
    print("5x-cost hurdle: SPY/TLT/GLD ~0.15% (at 3 bps), DX ~0.075%")
    ck = grid[grid["k"] == 4][["asset", "exit", "n", "avg", "t", "lift"]].copy()
    ck["cost_bps"] = np.where(ck["asset"] == "DX-Y.NYB", 1.5, 3.0)
    ck["edge_x_cost"] = (ck["avg"].abs() * 100 / ck["cost_bps"]).round(1)
    ck["lift_x_cost"] = (ck["lift"].abs() * 100 / ck["cost_bps"]).round(1)
    print(ck.to_string(index=False))

    # ------------------------------------------------------------------
    hr("5. TODAY'S EXACT SPY CELL (best cell in the grid, intersected with today)")
    a = A["SPY"]
    close = pd.Series(a.cl, index=a.idx)
    r5 = C.pct_rank(close.pct_change(5), 252).to_numpy()
    hi52 = close.rolling(252).max().to_numpy()
    nf = np.zeros(len(a.cl), dtype=bool)
    nf[list(nfp_pos["SPY"])] = True
    cn = np.concatenate([[0], np.cumsum(nf)])
    for xk in EXITS:
        d = store[("SPY", 4, xk)]
        e, x, r = d["e"], d["x"], d["r"]
        prior = e - 1
        st = (prior >= 0) & (a.cl[prior] >= hi52[prior] * 0.999) & (r5[prior] >= 80)
        hn = (cn[x + 1] - cn[e]) > 0
        print(f"\n-- exit={xk}")
        C.show([C.describe("52w-high state (any NFP)", r[st]),
                C.describe("52w-high AND NFP-in-window = TODAY", r[st & hn]),
                C.describe("52w-high, NO NFP in window", r[st & ~hn])])
        print("  52w-high state entry dates: " +
              ", ".join(str(dt.date()) for dt in a.idx[e[st]]))
        print("  of those, NFP-contaminated (today's exact case): " +
              (", ".join(str(dt.date()) for dt in a.idx[e[st & hn]]) or "NONE"))

    # ------------------------------------------------------------------
    hr("6. CROSS-CHECK vs the LIVE DX IDEA (entry MOO p-4 = 08-06, exit MOC p-2 = 08-10)")
    a = A["DX-Y.NYB"]
    ppos = a.pos_of(cpi)
    e = ppos - 4
    x = ppos - 2
    ok = (e >= 0) & (x < len(a.cl))
    e, x = e[ok], x[ok]
    r = (a.cl[x] / a.op[e] - 1.0) * 100.0
    dates = a.idx[e]
    close = pd.Series(a.cl, index=a.idx)
    r5 = C.pct_rank(close.pct_change(5), 252).to_numpy()
    r63 = C.pct_rank(close.pct_change(63), 252).to_numpy()
    prior = e - 1
    st = (prior >= 0) & (r5[prior] <= 25) & (r63[prior] >= 70)
    u = uncond(a, 2, int(e.min()), int(e.max()))
    print("hold = 2 sessions (MOO p-4 -> MOC p-2), i.e. the live DX idea's exact horizon")
    C.show([C.describe("all pre-CPI windows", r, u),
            C.describe("live-idea state INSIDE a pre-CPI window", r[st], u),
            C.describe("not in live-idea state", r[~st], u)])
    print("  live-idea-state pre-CPI entry dates: " +
          ", ".join(str(dt.date()) for dt in dates[st]))


if __name__ == "__main__":
    main()
