"""Idea check: long IWM into the Jackson Hole symposium session.

Flagged 08-21 as the idea-shaped candidate for this queue. The symposium's
main session is Friday 2026-08-28. The trade shape is buy Thursday's close
(MOC 08-27), sell Friday's close (MOC 08-28): the session's own
close-to-close move. Anchoring on the session BEFORE the symposium with
lag-1 entry would be a Wednesday-close entry, so this script measures both
the symposium session itself (lag 0 from the prior close, which IS the
Thursday-MOC order) and the two-session run from Wednesday's close.

Kill attempts: full-history + local control, era split, midterm years
(2026 is one), concentration, ^RUT as the index cross-check, and SPY as
the comparison vehicle.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    cluster_note, era_split, load_events, load_prices, local_control,
    sign_test, summarize, wilder_atr,
)

px = load_prices(["IWM", "^RUT", "SPY"])
iwm_raw = px["IWM"]
iwm, rut, spy = iwm_raw["Close"], px["^RUT"]["Close"], px["SPY"]["Close"]
idx = iwm.index
atr = pd.Series(wilder_atr(iwm_raw["High"], iwm_raw["Low"], iwm_raw["Close"]),
                index=idx)
print("IWM panel", idx[0].date(), "->", idx[-1].date(), "n", len(idx))
print("tonight: IWM close %.2f | Wilder-14 ATR %.4f (%.2f%%)"
      % (iwm.iloc[-1], atr.iloc[-1], atr.iloc[-1] / iwm.iloc[-1] * 100))

jh = load_events(["jackson_hole"])["date"]
jh = pd.DatetimeIndex([d for d in jh if d <= idx[-1]])
print("\nsymposium sessions on file:", len(jh), jh[0].date(), "..", jh[-1].date())


def session_ret(s: pd.Series, dates: pd.DatetimeIndex, back: int = 1) -> pd.Series:
    """close(D) / close(D-back) - 1 for each symposium date D present in s."""
    pos = pd.Series(range(len(s)), index=s.index)
    out = {}
    for d in dates:
        p = pos.get(d)
        if p is None or p - back < 0:
            continue
        out[d] = s.iloc[p] / s.iloc[p - back] - 1.0
    return pd.Series(out)


def report(name: str, r: pd.Series, s: pd.Series, back: int) -> None:
    v = r.values
    st = summarize(v)
    nup = int((r > 0).sum())
    allr = (s / s.shift(back) - 1.0).dropna()
    loc = allr.reindex(local_control(s.index, r.index, 126)).dropna()
    print(f"  {name} back={back} n={st['n']:<3} mean={st['mean_pct']:+.3f}%  med={st['median_pct']:+.2f}%  "
          f"{nup}-{len(r)-nup} up  t={st['t']:+.2f}  sign_p={sign_test(nup, len(r)):.4f}  "
          f"| all {100*allr.mean():+.3f}%  local {100*loc.mean():+.3f}%  "
          f"| worst {st['worst_pct']:+.2f}% ({r.idxmin().date()}) best {st['best_pct']:+.2f}% ({r.idxmax().date()})")
    print("    era:", [(e["label"], e["n"], round(e.get("mean_pct", np.nan), 3),
                        round(e.get("hit", np.nan), 1)) for e in era_split(r.index, v)])
    print("    concentration:", cluster_note(r.index, v))
    mid = r[[d.year % 4 == 2 for d in r.index]]
    non = r[[d.year % 4 != 2 for d in r.index]]
    for lab, sub in (("midterm", mid), ("non-midterm", non)):
        if len(sub):
            su = summarize(sub.values)
            nu = int((sub > 0).sum())
            print(f"    {lab:<12} n={su['n']:<3} mean={su['mean_pct']:+.3f}%  {nu}-{len(sub)-nu} up  "
                  f"sign_p={sign_test(nu, len(sub)):.4f}  worst={su['worst_pct']:+.2f}%")
    print("    midterm years:", [(d.year, round(100 * x, 2)) for d, x in mid.items()])


print("\n-- the symposium session itself (Thursday MOC -> Friday MOC)")
for name, s in (("IWM", iwm), ("^RUT", rut), ("SPY", spy)):
    report(name, session_ret(s, jh, 1), s, 1)

print("\n-- two-session run: Wednesday MOC -> Friday MOC")
for name, s in (("IWM", iwm), ("SPY", spy)):
    report(name, session_ret(s, jh, 2), s, 2)

print("\n-- three-session run: Tuesday MOC (tonight) -> Friday MOC")
for name, s in (("IWM", iwm), ("SPY", spy)):
    report(name, session_ret(s, jh, 3), s, 3)

# the session AFTER the symposium (Monday), in case the move is the reaction
print("\n-- the session after the symposium (Friday MOC -> Monday MOC)")
pos = pd.Series(range(len(idx)), index=idx)
after = pd.DatetimeIndex([idx[pos[d] + 1] for d in jh if d in pos and pos[d] + 1 < len(idx)])
report("IWM", session_ret(iwm, after, 1), iwm, 1)
