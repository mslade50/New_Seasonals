"""The run-up into a VIX expiry vs the expiry session itself.

Last night published the k2 cell: the session two td before an expiry gained
0.190% at t 2.87 (SPY). Tonight the anchor is k1 and h1 IS the expiry session,
where the engine has SPY at -0.104%. Same 319 events, opposite sign. The
question is whether that is one round trip (the drift gets given back on
settlement day) or two unrelated numbers that happen to differ.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, load_events, summarize, show, sign_test, era_split,
    cluster_note,
)

TKRS = ["SPY", "^GSPC", "SI=F"]
px = close_panel(TKRS)
idx = px.index
pos = pd.Series(range(len(idx)), index=idx)

ev = load_events(["vix_expiry"])
exp_dates = pd.DatetimeIndex(ev["date"])

# map each expiry to its own df position, then walk back k sessions
rows = []
for d in exp_dates:
    p = idx.searchsorted(d)
    if p >= len(idx) or idx[p] != d:
        continue  # expiry not a session in the panel
    rows.append({"expiry": d, "p": p})
E = pd.DataFrame(rows)
print(f"expiries matched to sessions: {len(E)} of {len(exp_dates)}")

ret = {t: px[t].pct_change() for t in TKRS}


def leg(t, p_from, p_to):
    """close-to-close return from session p_from to session p_to."""
    if p_from < 0 or p_to >= len(idx):
        return np.nan
    return px[t].values[p_to] / px[t].values[p_from] - 1.0


print("\n" + "=" * 74)
print("A. the three sessions around a VIX expiry, per ticker")
print("=" * 74)
for t in TKRS:
    out = []
    for name, a, b in [
        ("k2 -> k1  (two-out to one-out, last night's h1)", -2, -1),
        ("k1 -> k0  (one-out to the EXPIRY session, tonight)", -1, 0),
        ("k0 -> k+1 (expiry to the session after)", 0, 1),
        ("k2 -> k0  (the pair, two sessions)", -2, 0),
    ]:
        v = np.array([leg(t, p + a, p + b) for p in E["p"]], float)
        v = v[~np.isnan(v)]
        r = summarize(v, name)
        r["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
        out.append(r)
    base = ret[t].dropna()
    out.append(summarize(base.values, "CTRL all days"))
    show(out, f"{t}: sessions around the expiry")

print("\n" + "=" * 74)
print("B. is it a round trip? condition the expiry session on the run-up")
print("=" * 74)
for t in TKRS:
    runup = np.array([leg(t, p - 2, p - 1) for p in E["p"]], float)
    expday = np.array([leg(t, p - 1, p) for p in E["p"]], float)
    ok = ~np.isnan(runup) & ~np.isnan(expday)
    runup, expday = runup[ok], expday[ok]
    up = runup > 0
    rows2 = [
        summarize(expday[up], f"expiry session | run-up POSITIVE (n={up.sum()})"),
        summarize(expday[~up], f"expiry session | run-up negative (n={(~up).sum()})"),
    ]
    for r, m in zip(rows2, [up, ~up]):
        r["sign_p"] = round(sign_test(int((expday[m] > 0).sum()), int(m.sum())), 4)
    corr = np.corrcoef(runup, expday)[0, 1]
    show(rows2, f"{t}: expiry session split by the prior session's sign")
    print(f"  corr(run-up, expiry session) = {corr:+.3f}")

print("\n" + "=" * 74)
print("C. August expiries only, and the midterm subset")
print("=" * 74)
E["month"] = pd.DatetimeIndex(E["expiry"]).month
E["year"] = pd.DatetimeIndex(E["expiry"]).year
E["midterm"] = E["year"] % 4 == 2
for t in TKRS:
    out = []
    for name, m in [
        ("all expiries", np.ones(len(E), bool)),
        ("AUGUST expiries", (E["month"] == 8).values),
        ("August + midterm", ((E["month"] == 8) & E["midterm"]).values),
        ("non-August", (E["month"] != 8).values),
    ]:
        v = np.array([leg(t, p - 1, p) for p in E["p"][m]], float)
        v = v[~np.isnan(v)]
        if len(v) == 0:
            continue
        r = summarize(v, name)
        r["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
        r["n_up"] = int((v > 0).sum())
        out.append(r)
    show(out, f"{t}: the EXPIRY SESSION by month bucket")

print("\n" + "=" * 74)
print("D. era stability and concentration, expiry session, SPY and SI=F")
print("=" * 74)
for t in ["SPY", "SI=F"]:
    v, dts = [], []
    for p, d in zip(E["p"], E["expiry"]):
        x = leg(t, p - 1, p)
        if not np.isnan(x):
            v.append(x)
            dts.append(idx[p])
    v = np.array(v)
    dts = pd.DatetimeIndex(dts)
    show(era_split(dts, v), f"{t}: expiry session, era split at 2018")
    print(" ", cluster_note(dts, v, k=2))
