"""Stage B1 probe 5: is the pre-Sep-VIX-expiry SVXY cell just levered beta?

The one cell that survived the 192-cell grid's screen, the anchor-shift
placebo, the SVXY leverage break and the cycle split is:

    long SVXY, MOC at (Sep VIX expiry - 6 td), exit MOC the session before
    the expiry  -- 14 instances 2012..2025, 13-1, mean +4.05%, median +1.82%

SPY was up 11 of those 14 windows. So the honest question an adversarial
checker asks is whether SVXY simply delivered its beta to a market that
happened to be up, in which case the trade is SPY and the vol story is
decoration. This regresses SVXY's 5-session returns on SPY's over the SAME
5-session, non-overlapping grid, era by era (SVXY was -1.0x before
2018-02-28 and -0.5x after), and reports the cell's BETA-ADJUSTED alpha.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _eventgrid import event_dates, runway  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import fwd_ret, load_prices, sign_test  # noqa: E402

pd.set_option("display.width", 200)
SPLIT = pd.Timestamp("2018-02-28")   # ProShares cut SVXY to -0.5x


def fit_beta(spy: pd.Series, svxy: pd.Series, h: int,
             lo=None, hi=None) -> tuple[float, float, int]:
    """OLS of SVXY h-session return on SPY's, sampled every h sessions so the
    windows do not overlap (overlapping windows fake precision)."""
    a, b = fwd_ret(spy, h), fwd_ret(svxy, h)
    df = pd.DataFrame({"spy": a, "svxy": b}).dropna()
    if lo is not None:
        df = df[df.index >= lo]
    if hi is not None:
        df = df[df.index < hi]
    df = df.iloc[::h]
    x, y = df["spy"].values, df["svxy"].values
    beta, alpha = np.polyfit(x, y, 1)
    return float(beta), float(alpha), len(df)


def main() -> None:
    px = load_prices(["SPY", "SVXY"])
    spy, svxy = px["SPY"]["Close"], px["SVXY"]["Close"]
    h = 5
    print("=" * 100)
    print("PROBE 5 -- beta decomposition of the pre-Sep-VIX-expiry SVXY cell")
    print("=" * 100)

    eras = [("full 2011-10+", None, None),
            ("-1.0x era (< 2018-02-28)", None, SPLIT),
            ("-0.5x era (>= 2018-02-28)", SPLIT, None)]
    betas = {}
    for lbl, lo, hi in eras:
        beta, alpha, n = fit_beta(spy, svxy, h, lo, hi)
        betas[lbl] = beta
        print(f"  {lbl:28s} beta(SVXY,SPY) = {beta:+.2f}  intercept "
              f"{100*alpha:+.3f}%/{h}d   n={n} non-overlapping windows")

    ve = event_dates("vix_expiry")
    ve = ve[ve.month == 9]
    j = pd.DataFrame({
        "spy": runway(spy, ve, 6, "pre")["ret"],
        "svxy": runway(svxy, ve, 6, "pre")["ret"],
    }).dropna()
    j["era"] = np.where(j.index < SPLIT, "-1.0x era (< 2018-02-28)",
                        "-0.5x era (>= 2018-02-28)")
    j["beta"] = j["era"].map(betas)
    j["expected_from_spy"] = j["beta"] * j["spy"]
    j["alpha"] = j["svxy"] - j["expected_from_spy"]

    out = (100 * j[["spy", "svxy", "expected_from_spy", "alpha"]]).round(2)
    out["era"] = j["era"].str[:5]
    print("\n  per-instance, SVXY minus what its own beta to SPY explains:")
    print(out.to_string())

    for lbl, sub in (("ALL 14", j), ("-1.0x era", j[j.index < SPLIT]),
                     ("-0.5x era", j[j.index >= SPLIT])):
        v = sub["alpha"].values
        if len(v) == 0:
            continue
        w = int((v > 0).sum())
        print(f"\n  ALPHA {lbl:10s} n={len(v)} mean {100*v.mean():+.3f}% med "
              f"{100*np.median(v):+.3f}% rec {w}-{len(v)-w} "
              f"p_coin={sign_test(w, len(v)):.4f} worst {100*v.min():+.2f}%")

    print("\n  READ: if the alpha column is mostly positive with a clean "
          "record, the cell is a vol-crush trade and SVXY is the right "
          "vehicle. If alpha collapses to noise, the cell is SPY beta and "
          "should be pitched as SPY (cheaper, no term-structure risk).")

    print("\n  LIVE-SIZING NOTE: at the -0.5x era beta, a 2026 instance is "
          f"expected around {100*j[j.index >= SPLIT]['svxy'].mean():.2f}% "
          "(the -0.5x era mean), NOT the +4.05% full-sample headline. The "
          "full-sample mean is a -1.0x vehicle's number and cannot be "
          "traded today.")


if __name__ == "__main__":
    main()
