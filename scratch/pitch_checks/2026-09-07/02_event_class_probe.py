"""Stage B1: adversarial probe of the cells the 192-cell grid screen flagged.

Nothing here promotes a cell. This exists so the survey map can record WHY a
flagged cell is or is not worth a stage-C hour, with the numbers an attacker
would reach for first:

  1. ANCHOR-SHIFT PLACEBO. Slide the event anchor -10..+10 trading days and
     re-measure the identical window. A real event effect peaks at offset 0.
     A cell that is really "mid-month drift wearing an event's name" is flat
     across the whole scan. This is the single test that separates the two,
     and it is cheap.
  2. SVXY IS TWO INSTRUMENTS. ProShares cut SVXY from -1.0x to -0.5x on
     2018-02-28 after the 2018-02-05 vol blowup took it down ~90% in a
     session. Every SVXY mean spanning that date is a mixture of two
     different vehicles, and the pre-2018 half is the one with the fat left
     tail. Split it.
  3. THE SEPTEMBER FOMC *IS* THE SEPTEMBER VIX EXPIRY, most years. Both land
     on the third Wednesday. So the vix_expiry x SVXY cell and the pre-FOMC
     SVXY cell are largely the SAME observations, and any midterm inversion
     that applies to one applies to the other. Count the coincidence.
  4. Per-year record and worst instance for anything that survives 1-3.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _eventgrid import event_dates, midterm_mask, runway  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    anchor_positions, cluster_note, load_prices, sign_test, summarize,
)

pd.set_option("display.width", 200)

# (label, event kind, proxy, k, mode, month filter or None)
PULSES = [
    ("A  pre-FOMC long SVXY",        "fomc_decision", "SVXY", 6, "pre",  None),
    ("B  pre-VIXexp long SVXY, SEP", "vix_expiry",    "SVXY", 6, "pre",  9),
    ("C  run-up into Sep quad, IWM", "quad_witching", "IWM",  8, "pre",  9),
    ("D  day before PPI, long USO",  "ppi",           "USO",  2, "pre",  None),
    ("E  into VIXexp short UUP, SEP", "vix_expiry",   "UUP",  6, "thru", 9),
    ("F  into VIXexp long SLV, SEP", "vix_expiry",    "SLV",  6, "thru", 9),
]


def shifted(close: pd.Series, evd, k: int, h: int, off: int) -> np.ndarray:
    """The identical entry/exit window with every anchor slid `off` td."""
    idx = close.index
    pos, _ = anchor_positions(idx, evd, offset=off)
    out = []
    for p in pos:
        a, b = p - k, p - k + h
        if a < 0 or b >= len(idx):
            continue
        out.append(float(close.iloc[b] / close.iloc[a] - 1.0))
    return np.asarray(out)


def placebo_scan(label, kind, tkr, k, mode, month, close) -> None:
    h = k - 1 if mode == "pre" else k
    evd = event_dates(kind)
    if month:
        evd = evd[evd.month == month]
    rows = []
    for off in range(-10, 11):
        v = shifted(close, evd, k, h, off)
        if len(v) == 0:
            continue
        w = int((v > 0).sum())
        rows.append({"offset_td": off, "n": len(v),
                     "mean_pct": round(100 * v.mean(), 3),
                     "med_pct": round(100 * float(np.median(v)), 3),
                     "rec": f"{w}-{len(v)-w}",
                     "p_coin": round(sign_test(w, len(v)), 4),
                     "TRUE": "<== true anchor" if off == 0 else ""})
    df = pd.DataFrame(rows)
    print(f"\n--- PLACEBO anchor-shift scan: {label} "
          f"({kind}{' SEP only' if month else ''} x {tkr}, k={k} {mode}, h={h}) ---")
    print(df.to_string(index=False))
    true = df[df["offset_td"] == 0].iloc[0]
    others = df[df["offset_td"] != 0]
    better = int((others["mean_pct"].abs() >= abs(true["mean_pct"])).sum())
    print(f"    RANK OF THE TRUE ANCHOR: {better} of {len(others)} placebo "
          f"offsets have |mean| >= the true anchor's {true['mean_pct']:+.3f}%."
          f"  (0-2 = the event date is doing work; 8+ = this is drift with an "
          f"event's name on it.)")


def era_and_years(label, kind, tkr, k, mode, month, px) -> None:
    h = k - 1 if mode == "pre" else k
    close = px[tkr]["Close"]
    evd = event_dates(kind)
    if month:
        evd = evd[evd.month == month]
    r = runway(close, evd, k, mode)
    v, ent = r["ret"].values, pd.DatetimeIndex(r.index)
    w = int((v > 0).sum())
    s = summarize(v, label)
    print(f"\n--- {label}: {kind}{' SEP' if month else ''} x {tkr} "
          f"k={k} {mode} h={h} ---")
    print(f"    HEADLINE n={len(v)} mean {s['mean_pct']:+.3f}% med "
          f"{s['median_pct']:+.3f}% rec {w}-{len(v)-w} hit {s['hit']:.1f}% "
          f"p_coin={sign_test(w, len(v)):.4f} worst {s['worst_pct']:+.2f}% "
          f"best {s['best_pct']:+.2f}%")
    print(f"    edge/cost: mean {100*s['mean_pct']:.0f} bps vs ~5 bps "
          f"round trip = {abs(100*s['mean_pct'])/5:.1f}x")
    print(f"    concentration: {cluster_note(ent, v)}")
    cuts = [("2018-02-28", "SVXY -1.0x era / -0.5x era")] if tkr == "SVXY" \
        else [("2018-01-01", "pre-2018 / 2018+")]
    for cut, nm in cuts:
        m = ent < pd.Timestamp(cut)
        for sub, lbl in ((m, f"< {cut}"), (~m, f">= {cut}")):
            vv = v[sub]
            if len(vv) == 0:
                continue
            ww = int((vv > 0).sum())
            print(f"    {nm} | {lbl:12s} n={len(vv)} mean {100*vv.mean():+.3f}%"
                  f" med {100*np.median(vv):+.3f}% rec {ww}-{len(vv)-ww} "
                  f"p_coin={sign_test(ww, len(vv)):.4f}")
    mm = midterm_mask(ent)
    for sub, lbl in ((mm, "MIDTERM"), (~mm, "non-midterm")):
        vv = v[sub]
        if len(vv) == 0:
            continue
        ww = int((vv > 0).sum())
        print(f"    cycle | {lbl:12s} n={len(vv)} mean {100*vv.mean():+.3f}% "
              f"med {100*np.median(vv):+.3f}% rec {ww}-{len(vv)-ww} "
              f"p_coin={sign_test(ww, len(vv)):.4f}")
    if len(v) <= 30:
        print("    every instance: " + ", ".join(
            f"{d.date()}:{100*x:+.2f}%" for d, x in zip(ent, v)))


def fomc_vixexp_coincidence() -> None:
    print("\n" + "=" * 100)
    print("PROBE 3 -- how much of 'pre-VIX-expiry' is just 'pre-FOMC'?")
    print("=" * 100)
    ve = event_dates("vix_expiry")
    fo = event_dates("fomc_decision")
    fos = set(fo)
    sep = ve[ve.month == 9]
    same = sum(1 for d in sep if d in fos)
    near = sum(1 for d in sep
               if min((abs((d - f).days) for f in fo), default=999) <= 3)
    print(f"  September VIX expiries 2000-2027: {len(sep)}")
    print(f"    landing EXACTLY on an FOMC decision day: {same} "
          f"({100*same/len(sep):.0f}%)")
    print(f"    within 3 CALENDAR days of one:           {near} "
          f"({100*near/len(sep):.0f}%)")
    allsame = sum(1 for d in ve if d in fos)
    print(f"  All months: {allsame} of {len(ve)} VIX expiries ({100*allsame/len(ve):.0f}%) "
          f"are also FOMC decision days.")
    print("  2026-09-16 is BOTH. So the vix_expiry x SVXY September cell and "
          "the pre-FOMC SVXY cell are largely the SAME observations, and the "
          "midterm inversion measured on one transfers to the other.")


def decompose_sep_vixexp() -> None:
    """PROBE 4 -- the decisive attack on the one surviving cell.

    SVXY is short vol. If SPY simply rises in the week before the September
    VIX expiry, then a long-SVXY cell is a beta trade wearing a vol-crush
    story, and the honest vehicle is SPY. Put the same 5-session window on
    ^VIX, SPY and SVXY side by side and see which one is carrying it.
    """
    print("\n" + "=" * 100)
    print("PROBE 4 -- pre-Sep-VIX-expiry window decomposed: ^VIX vs SPY vs "
          "SVXY on the IDENTICAL entry/exit sessions")
    print("=" * 100)
    px = load_prices(["^VIX", "SPY", "SVXY"])
    ve = event_dates("vix_expiry")
    ve = ve[ve.month == 9]
    frames = {t: runway(px[t]["Close"], ve, 6, "pre") for t in
              ("^VIX", "SPY", "SVXY")}
    j = pd.DataFrame({t: f["ret"] for t, f in frames.items()}).dropna()
    j = (100 * j).round(2)
    j["year"] = j.index.year
    print(j.to_string())
    print()
    for t in ("^VIX", "SPY", "SVXY"):
        v = j[t].values / 100.0
        w = int((v > 0).sum())
        print(f"  {t:5s} n={len(v)} mean {100*v.mean():+.3f}% med "
              f"{100*np.median(v):+.3f}% rec {w}-{len(v)-w} "
              f"p_coin_up={sign_test(w, len(v)):.4f} "
              f"p_coin_down={sign_test(len(v)-w, len(v)):.4f}")
    # how many of SVXY's wins came with SPY DOWN? that is the vol-crush half
    both = j[(j["SVXY"] > 0)]
    print(f"\n  SVXY up on {len(both)} of {len(j)} instances; of those, SPY "
          f"was DOWN on {(both['SPY'] < 0).sum()} and ^VIX was down on "
          f"{(both['^VIX'] < 0).sum()}.")
    print("  Read: SVXY up WITH SPY down is unambiguous vol crush. SVXY up "
          "only when SPY is up would make this a levered beta trade and the "
          "cell should be pitched on SPY instead.")
    # magnitude decay
    last5 = j.tail(5)
    print(f"\n  DECAY CHECK, last 5 instances ({list(last5['year'])}): SVXY "
          f"mean {last5['SVXY'].mean():+.3f}%, rec "
          f"{(last5['SVXY']>0).sum()}-{(last5['SVXY']<=0).sum()}; "
          f"first 5 ({list(j.head(5)['year'])}) mean "
          f"{j.head(5)['SVXY'].mean():+.3f}%.")
    print("  SVXY was -1.0x until 2018-02-28 and -0.5x after, so roughly half "
          "the decay is the vehicle, not the market.")


def main() -> None:
    tks = sorted({p[2] for p in PULSES})
    px = load_prices(tks)
    print("=" * 100)
    print("PROBE of the 9 screen flags from 02_event_class_grid.py "
          "(192-cell grid). STILL UNCHARGED.")
    print("=" * 100)
    print("\n" + "=" * 100)
    print("PROBE 1 -- anchor-shift placebo scans (21 offsets x 6 cells = 126 "
          "additional measurements, all of them CONTROLS, not candidates)")
    print("=" * 100)
    for label, kind, tkr, k, mode, month in PULSES:
        placebo_scan(label, kind, tkr, k, mode, month, px[tkr]["Close"])

    print("\n" + "=" * 100)
    print("PROBE 2 -- era / instrument-regime / cycle splits")
    print("=" * 100)
    for label, kind, tkr, k, mode, month in PULSES:
        era_and_years(label, kind, tkr, k, mode, month, px)

    fomc_vixexp_coincidence()
    decompose_sep_vixexp()


if __name__ == "__main__":
    main()
