"""C6 round 1 — cross-sectional new-high breadth while the index is off its high.

Live state: 20/218 tape names within 0.25% of a 52w high, SPY -1.56% off its own.
Claim: broad participation without index confirmation -> long SPY (or IWM) 1-10 td.

Round-1 obligations discharged here:
  0. does the trigger even FIRE on a point-in-time definition, and on a
     survivorship-free universe? (2026-08-19 / -08-20 registry)
  1. gate-OFF FIRST: the plain index-off-its-high state, and the plain
     breadth state, BEFORE the interaction (2026-08-19 registry)
  2. battery() vs three controls incl. CTRL-c local (2026-08-14 registry)
  3. survivorship-contaminated tape vs 9 sector SPDRs (2026-08-20 registry:
     "swapping the tape for 11 sector ETFs flips the sign outright")
  4. tape-over-selection: share of trigger days above SPY's 200d vs base rate
     (2026-08-21 registry)
  5. cost, era, concentration, registry collision measured
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BAR = pd.Timestamp("2026-08-21")
TOL = 0.0025          # "at a 52-week high" = within 0.25%, the live definition
COST_SPY = 1.5        # bps round trip

tape_json = json.load(open(ROOT / "data" / "pitch_tape.json"))["tickers"]
TAPE = sorted(tape_json)
SECT9 = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
SECT11 = SECT9 + ["XLRE", "XLC"]
VEH = ["SPY", "QQQ", "IWM", "DIA"]

px_all = load_prices(sorted(set(TAPE + SECT11 + VEH)))
spy = px_all["SPY"]["Close"].dropna()
CAL = spy.index[spy.index <= BAR]


def dist_52wh(c: pd.Series, look: int = 252) -> pd.Series:
    hi = c.rolling(look).max()
    return c / hi - 1.0


def breadth(univ, tol=TOL, label=""):
    """Share of the universe within `tol` of its own 52-week high, on SPY's
    calendar. Names without a defined 252-day max leave the DENOMINATOR."""
    num = pd.Series(0.0, index=CAL)
    den = pd.Series(0.0, index=CAL)
    used = []
    for t in univ:
        d = px_all.get(t)
        if d is None:
            continue
        c = d["Close"].dropna()
        c = c[c.index <= BAR]
        if len(c) < 300:
            continue
        used.append(t)
        dd = dist_52wh(c)
        flag = (dd >= -tol).astype(float)
        flag[dd.isna()] = np.nan
        f = flag.reindex(CAL)
        ok = f.notna()
        num[ok] += f[ok].values
        den[ok] += 1.0
    b = (num / den).where(den >= 5)
    print(f"  breadth[{label}] built on {len(used)} names, "
          f"denominator today = {den.iloc[-1]:.0f}")
    return b, den


print("=" * 100)
print("0. DOES THE TRIGGER FIRE?  live values + PIT trailing-252 percentile")
print("=" * 100)

b_tape, den_tape = breadth(TAPE, label="tape218")
b_s9, den_s9 = breadth(SECT9, label="sect9")
b_s11, den_s11 = breadth(SECT11, label="sect11")

spy_d = dist_52wh(spy).reindex(CAL)
qqq_d = dist_52wh(px_all["QQQ"]["Close"].dropna()).reindex(CAL)
iwm_d = dist_52wh(px_all["IWM"]["Close"].dropna()).reindex(CAL)


def pit_pct(s, look=252):
    return rolling_on_valid(s, lambda x: x.rolling(look).rank(pct=True) * 100.0)


for lbl, b in (("tape218", b_tape), ("sect9", b_s9), ("sect11", b_s11)):
    pit = pit_pct(b)
    full = 100.0 * (b <= b.iloc[-1]).mean()
    print(f"\n  {lbl:8s} today = {100*b.iloc[-1]:6.2f}%  "
          f"(count {b.iloc[-1]*den_tape.iloc[-1] if lbl=='tape218' else b.iloc[-1]*(den_s9.iloc[-1] if lbl=='sect9' else den_s11.iloc[-1]):.0f})"
          f"   PIT trailing-252 pctile = {pit.iloc[-1]:5.1f}"
          f"   FULL-SAMPLE pctile = {full:5.1f}   (lookahead gap "
          f"{full - pit.iloc[-1]:+.1f} pts)")
    print(f"           trailing-252 mean {100*b.tail(252).mean():.2f}%  "
          f"max {100*b.tail(252).max():.2f}%   full-sample mean "
          f"{100*b.mean():.2f}%")

print(f"\n  SPY dist 52wh today = {100*spy_d.iloc[-1]:+.2f}%   "
      f"QQQ {100*qqq_d.iloc[-1]:+.2f}%   IWM {100*iwm_d.iloc[-1]:+.2f}%")
print(f"  corr(tape breadth, sect9 breadth) = "
      f"{b_tape.corr(b_s9):.3f}   sect11 {b_tape.corr(b_s11):.3f}")
print("  which sector ETFs are within 0.25% of a 52w high today: "
      + ", ".join(t for t in SECT11
                  if px_all.get(t) is not None
                  and len(px_all[t]['Close'].dropna()) > 300
                  and dist_52wh(px_all[t]['Close'].dropna()[
                      px_all[t]['Close'].dropna().index <= BAR]).iloc[-1] >= -TOL))

# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("1. GATE-OFF FIRST: each half of the trigger alone, long SPY")
print("=" * 100)

px = pd.DataFrame({t: px_all[t]["Close"] for t in VEH}).reindex(CAL)

IDX_LO, IDX_HI = -0.05, -0.005      # "off its high but not broken": -5% .. -0.5%
idx_gate = (spy_d > IDX_LO) & (spy_d <= IDX_HI)
print(f"  index gate (SPY {100*IDX_LO:.1f}% < dist <= {100*IDX_HI:.1f}%): "
      f"{int(idx_gate.sum())} days of {int(spy_d.notna().sum())} "
      f"({100*idx_gate.mean():.1f}%)  -- live today: {bool(idx_gate.iloc[-1])}")

pit_tape = pit_pct(b_tape)
pit_s9 = pit_pct(b_s9)

for H in (1, 3, 5, 10):
    ret = fwd_lag(px["SPY"], H, 1)
    valid = ret.notna()
    rows = [summarize(ret[valid].values, f"CTRL-b all days h={H}")]
    rows.append(summarize(ret[valid & idx_gate].values,
                          f"GATE-OFF: index off-high alone (N={int((valid&idx_gate).sum())})"))
    for lbl, pit in (("tape218", pit_tape), ("sect9", pit_s9)):
        m = pit >= 80
        rows.append(summarize(ret[valid & m].values,
                              f"GATE-OFF: breadth>=80pct {lbl} alone (N={int((valid&m).sum())})"))
        rows.append(summarize(ret[valid & m & idx_gate].values,
                              f"BOTH: {lbl} breadth>=80 & index off-high (N={int((valid&m&idx_gate).sum())})"))
    show(rows, f"h={H} td, long SPY, day level")

# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("2. BATTERY, contaminated tape universe, long SPY h=5")
print("=" * 100)
H = 5
mask_tape = (pit_tape >= 80) & idx_gate
variants_t = {
    "tape pit>=70 & idx": (pit_tape >= 70) & idx_gate,
    "tape pit>=90 & idx": (pit_tape >= 90) & idx_gate,
    "tape raw>=9.17% & idx": (b_tape >= b_tape.iloc[-1] - 1e-9) & idx_gate,
    "tape pit>=80, idx -3..-0.5%": (pit_tape >= 80) & (spy_d > -0.03) & (spy_d <= -0.005),
    "tape pit>=80, idx -10..-0.5%": (pit_tape >= 80) & (spy_d > -0.10) & (spy_d <= -0.005),
    "tape pit>=80, NO idx gate": (pit_tape >= 80),
}
battery(px, mask_tape, [("SPY", 1.0)], H, "C6-tape long SPY", COST_SPY,
        variants=variants_t, min_gap=10)

print("\n" + "=" * 100)
print("3. BATTERY, SURVIVORSHIP-FREE 9 sector SPDRs, long SPY h=5")
print("=" * 100)
mask_s9 = (pit_s9 >= 80) & idx_gate
variants_s = {
    "sect9 pit>=70 & idx": (pit_s9 >= 70) & idx_gate,
    "sect9 pit>=90 & idx": (pit_s9 >= 90) & idx_gate,
    "sect9 raw>=2/9 & idx": (b_s9 >= 2 / 9 - 1e-9) & idx_gate,
    "sect9 raw>=3/9 & idx": (b_s9 >= 3 / 9 - 1e-9) & idx_gate,
    "sect9 pit>=80, NO idx gate": (pit_s9 >= 80),
}
battery(px, mask_s9, [("SPY", 1.0)], H, "C6-sect9 long SPY", COST_SPY,
        variants=variants_s, min_gap=10)

# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("4. VEHICLE: does IWM or DIA do better than SPY on the same trigger?")
print("=" * 100)
for lbl, m in (("tape218", mask_tape), ("sect9", mask_s9)):
    rows = []
    for v in VEH:
        ret = fwd_lag(px[v], H, 1)
        valid = ret.dropna().index
        t = CAL[m.reindex(CAL, fill_value=False).values].intersection(valid)
        epi = declusters(t, 10, valid)
        r = summarize(ret.loc[epi].values, f"{v} (N_epi={len(epi)})")
        r["ctl_all_pct"] = round(100 * ret.loc[valid].mean(), 3)
        r["edge_pct"] = round(r["mean_pct"] - 100 * ret.loc[valid].mean(), 3)
        rows.append(r)
    show(rows, f"h=5, trigger={lbl}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("5. TAPE-OVER-SELECTION: is the trigger just a bull-tape selector?")
print("=" * 100)
sma200 = rolling_on_valid(spy, lambda x: x.rolling(200).mean()).reindex(CAL)
above = (spy > sma200).reindex(CAL)
base = 100 * above[above.notna()].mean()
for lbl, m in (("tape218 pit>=80 & idx", mask_tape), ("sect9 pit>=80 & idx", mask_s9),
               ("idx gate alone", idx_gate)):
    mm = m.reindex(CAL, fill_value=False) & above.notna()
    print(f"  {lbl:26s}: {100*above[mm].mean():5.1f}% of trigger days above SPY's 200d "
          f"(base rate {base:.1f}%)  N={int(mm.sum())}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("6. REGISTRY COLLISION, measured: fragility dial coverage of trigger days")
print("=" * 100)
frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
frag.index = pd.to_datetime(frag.index)
for lbl, m in (("tape218", mask_tape), ("sect9", mask_s9)):
    t = CAL[m.reindex(CAL, fill_value=False).values]
    have = frag.index.intersection(t)
    pit_vintage = have[have >= pd.Timestamp("2026-07-02")]
    print(f"  {lbl}: {len(t)} trigger days, {len(have)} have a dial reading "
          f"({100*len(have)/max(1,len(t)):.0f}%), {len(pit_vintage)} on the "
          f"post-2026-07-02 PIT vintage")
print(f"  today's ma10(63d) dial = "
      f"{frag['63d'].rolling(10).mean().iloc[-1]:.1f}  (extreme; NOT used as a signal here)")
