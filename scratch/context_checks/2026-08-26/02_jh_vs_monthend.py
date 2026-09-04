"""Is the Jackson Hole bond bid its own effect, or is it month end?

Jackson Hole always lands in the last week of August, so the k2 anchor sits
inside or beside the final-3-sessions window every single year. Tomorrow is
BOTH. This separates them.

Convention: context lane, lag=0 close-to-close from the anchor close, so h=1
is the next session.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

from pitch_lab import close_panel, load_events, summarize, sign_test, era_split, cluster_note

SUBJECTS = ["TLT", "IEF", "^TNX", "HYG", "SPY"]
px = close_panel(SUBJECTS)
px = px.sort_index()
all_dates = px.index

ev = load_events(["jackson_hole"])
jh = pd.to_datetime(sorted(ev["date"].unique()))
print(f"Jackson Hole events loaded: {len(jh)}  {jh.min().date()} .. {jh.max().date()}")


def anchor_k_before(event_dates, k):
    """Session k trading days before each event (the context event-lane anchor)."""
    out = []
    for d in event_dates:
        pos = all_dates.searchsorted(pd.Timestamp(d))
        if pos >= len(all_dates):
            continue
        a = pos - k
        if a < 0:
            continue
        out.append(all_dates[a])
    return pd.DatetimeIndex(sorted(set(out)))


def td_from_month_end(dates):
    """How many sessions after `date` is the month's last session. 0 = is the last."""
    ser = pd.Series(all_dates, index=all_dates)
    ym = ser.index.to_period("M")
    last_of_month = ser.groupby(ym).transform("max")
    pos = pd.Series(np.arange(len(all_dates)), index=all_dates)
    last_pos = pos.reindex(last_of_month.values).values
    dist = pd.Series(last_pos - pos.values, index=all_dates)
    return dist.reindex(dates)


def fwd(sym, dates, h=1):
    s = px[sym].dropna()
    pos = s.index.searchsorted(dates)
    ok = (pos >= 0) & (pos + h < len(s))
    pos = pos[ok]
    used = s.index[pos]
    vals = (s.values[pos + h] / s.values[pos]) - 1.0
    return used, vals


jh_k2 = anchor_k_before(jh, 2)
dist = td_from_month_end(jh_k2)
print("\nJackson Hole k2 anchors and their distance to month end (sessions):")
print(f"  n anchors {len(jh_k2)}")
vc = dist.value_counts().sort_index()
for k, v in vc.items():
    print(f"    {int(k)} sessions from month end : {v}")
print(f"  share inside the final-3 window (dist <= 2): "
      f"{(dist <= 2).mean():.1%}")

# The month-end window universe, as the engine defines the trigger.
me_all = all_dates[(td_from_month_end(all_dates) <= 2).values]
print(f"\nfinal-3-session universe: {len(me_all)} sessions")

print("\n" + "=" * 78)
print("h1 forward returns, three nested cells")
print("=" * 78)
for sym in SUBJECTS:
    print(f"\n--- {sym} ---")

    d_jh, v_jh = fwd(sym, jh_k2, 1)
    s = summarize(v_jh, "JH k2")
    up = int((v_jh > 0).sum())
    print(f"  JH k2 (all)              n={len(v_jh):4d}  mean={s['mean_pct']:+7.3f}%  "
          f"{up}-{len(v_jh)-up} up  signp={sign_test(up, len(v_jh)):.4f}")

    # JH anchors that are NOT in the final-3 window
    dj = td_from_month_end(d_jh)
    out_mask = (dj > 2).values
    if out_mask.sum() >= 3:
        v_out = v_jh[out_mask]
        up_o = int((v_out > 0).sum())
        s_o = summarize(v_out, "")
        print(f"  JH k2, OUTSIDE month end n={len(v_out):4d}  mean={s_o['mean_pct']:+7.3f}%  "
              f"{up_o}-{len(v_out)-up_o} up  signp={sign_test(up_o, len(v_out)):.4f}")
    in_mask = (dj <= 2).values
    if in_mask.sum() >= 3:
        v_in = v_jh[in_mask]
        up_i = int((v_in > 0).sum())
        s_i = summarize(v_in, "")
        print(f"  JH k2, INSIDE month end  n={len(v_in):4d}  mean={s_i['mean_pct']:+7.3f}%  "
              f"{up_i}-{len(v_in)-up_i} up  signp={sign_test(up_i, len(v_in)):.4f}")

    # Month end WITHOUT Jackson Hole anywhere near
    d_me, v_me = fwd(sym, me_all, 1)
    near_jh = np.zeros(len(d_me), dtype=bool)
    for j in jh_k2:
        near_jh |= (np.abs((d_me - j).days) <= 7)
    v_me_nojh = v_me[~near_jh]
    up_m = int((v_me_nojh > 0).sum())
    s_m = summarize(v_me_nojh, "")
    print(f"  month end, NO JH nearby  n={len(v_me_nojh):4d}  mean={s_m['mean_pct']:+7.3f}%  "
          f"{up_m}-{len(v_me_nojh)-up_m} up  signp={sign_test(up_m, len(v_me_nojh)):.4f}  "
          f"t={s_m['t']:.2f}")

    # August-only month end, excluding JH-adjacent, to isolate the calendar slot
    aug = d_me.month == 8
    v_aug = v_me[aug & ~near_jh]
    if len(v_aug) >= 5:
        up_a = int((v_aug > 0).sum())
        s_a = summarize(v_aug, "")
        print(f"  AUGUST month end, no JH  n={len(v_aug):4d}  mean={s_a['mean_pct']:+7.3f}%  "
              f"{up_a}-{len(v_aug)-up_a} up  signp={sign_test(up_a, len(v_aug)):.4f}")

    # all-days control
    s_all = px[sym].dropna().pct_change().shift(-1).dropna()
    print(f"  all-days control         n={len(s_all):4d}  mean={s_all.mean()*100:+7.3f}%  "
          f"hit={(s_all > 0).mean():.1%}")
