"""Guards for the Market Context sweep (scripts/build_context_state.py).

Three things are pinned here, because all three are silent when wrong:

1. THE ANCHOR CONVENTION. Every cell anchors on "today's analogue" and h=1 is
   the next session. An event cell that anchored on the event day instead
   would report the day AFTER the event as the event's own move, and nothing
   downstream could tell.
2. UNITS. summarize() takes fractions and returns percent. The daily-pitch
   product shipped a 100x scaling bug on this exact boundary (pitch_lab
   docstring, 2026-08-07), so the cell builder is checked against a synthetic
   panel with a known answer.
3. TAG COHERENCE. The confidence tag is code-computed from N, t and era
   stability; the prose stage may downgrade it but may never invent one.

Synthetic panels, not the live cache: a test that reads master_prices fails
on the day the price pull is stale, which is exactly when the engine most
needs its guards working.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_context_state as ctx  # noqa: E402
from pitch_lab import sign_test as pitch_sign_test  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _panel(returns: list[float], start: str = "2010-01-04") -> pd.DataFrame:
    """OHLCV frame whose close-to-close returns are exactly `returns`."""
    idx = pd.date_range(start, periods=len(returns) + 1, freq="B")
    close = pd.Series(100.0, index=idx)
    for i, r in enumerate(returns, start=1):
        close.iloc[i] = close.iloc[i - 1] * (1 + r)
    frame = pd.DataFrame({"date": idx, "Open": close, "High": close * 1.01,
                          "Low": close * 0.99, "Close": close,
                          "Volume": 1_000_000}, index=idx)
    return frame


def _subject(frame: pd.DataFrame) -> ctx.Subject:
    return ctx.make_subjects({"X": frame})["X"]


# ---------------------------------------------------------------------------
# 1. anchor convention
# ---------------------------------------------------------------------------
def test_anchors_before_selects_the_prior_session():
    ref = pd.DatetimeIndex(pd.date_range("2020-01-01", periods=6, freq="D"))
    prop = np.array([False, False, True, False, False, True])
    got = ctx.anchors_before(ref, prop)
    assert list(got) == [ref[1], ref[4]]


def test_event_anchor_is_one_session_before_the_event():
    """h=1 on an event cell must BE the event session's own return."""
    returns = [0.0] * 40
    returns[10] = 0.05          # the "event day" move
    returns[20] = 0.05
    frame = _panel(returns)
    subject = _subject(frame)
    idx = frame.index
    # Anchor on the sessions immediately before each spike.
    anchors = pd.DatetimeIndex([idx[10], idx[20]])
    cell = ctx.build_cell(subject, anchors)
    h1 = cell["h"]["1"]
    assert h1["n"] == 2
    assert h1["mean_pct"] == pytest.approx(5.0, abs=1e-6)
    assert h1["hit"] == pytest.approx(100.0)


def test_month_window_anchors_preview_the_month_end():
    ref = pd.DatetimeIndex(pd.bdate_range("2021-01-01", "2021-03-31"))
    anchors = ctx.month_window_anchors(ref, final_n=1, first_n=0)
    # Every anchor's NEXT session must be the last session of its month.
    pos = {d: i for i, d in enumerate(ref)}
    for a in anchors:
        nxt = ref[pos[a] + 1]
        assert nxt.month != ref[pos[a] + 2].month if pos[a] + 2 < len(ref) else True
    assert len(anchors) >= 2


def test_holiday_flags_come_from_the_calendar_not_the_price_index():
    """The panel is truncated at asof, so the bar revealing a holiday gap has
    not printed yet. Reading adjacency off the index silently never fires."""
    # 2018-11-21 (Wed) -> 11-23 (Fri): Thanksgiving sits in between.
    assert ctx.holiday_flags(pd.Timestamp("2018-11-21"),
                             pd.Timestamp("2018-11-23"))["post"] is True
    # The evening before that: tomorrow is the last session before the break.
    assert ctx.holiday_flags(pd.Timestamp("2018-11-20"),
                             pd.Timestamp("2018-11-21"))["pre"] is True
    # An ordinary Friday -> Monday is not holiday adjacency.
    flags = ctx.holiday_flags(pd.Timestamp("2026-08-07"),
                              pd.Timestamp("2026-08-10"))
    assert flags == {"pre": False, "post": False}


def test_holiday_anchor_needs_a_weekday_gap_not_a_weekend():
    # A clean run of business days has no holiday adjacency at all.
    ref = pd.DatetimeIndex(pd.bdate_range("2021-02-01", "2021-02-26"))
    assert len(ctx.holiday_adjacent_anchors(ref, "pre")) == 0
    # Drop a Wednesday to simulate a mid-week closure.
    holiday = pd.Timestamp("2021-02-17")
    gapped = ref[ref != holiday]
    pre = ctx.holiday_adjacent_anchors(gapped, "pre")
    assert pd.Timestamp("2021-02-15") in pre   # anchor -> 02-16 is pre-holiday


# ---------------------------------------------------------------------------
# 2. units and statistics
# ---------------------------------------------------------------------------
def test_cell_reports_percent_not_fractions():
    """A +1% flat drift must read 1.0, not 0.01 and not 100."""
    frame = _panel([0.01] * 60)
    subject = _subject(frame)
    anchors = pd.DatetimeIndex(frame.index[5:40])
    cell = ctx.build_cell(subject, anchors)
    assert cell["h"]["1"]["mean_pct"] == pytest.approx(1.0, abs=1e-6)
    assert cell["h"]["5"]["mean_pct"] == pytest.approx(
        100 * (1.01 ** 5 - 1), abs=1e-4)


def test_edge_is_measured_against_the_all_days_control():
    up, flat = 0.02, 0.0
    returns = [up if i % 2 else flat for i in range(200)]
    frame = _panel(returns)
    subject = _subject(frame)
    # _panel puts returns[i] on the move from bar i to bar i+1, so anchoring
    # on bar i previews returns[i]. Anchor where the NEXT bar is an up bar.
    anchors = pd.DatetimeIndex([d for i, d in enumerate(frame.index)
                                if i < len(returns) and returns[i] == up])
    cell = ctx.build_cell(subject, anchors)
    h1 = cell["h"]["1"]
    assert h1["mean_pct"] == pytest.approx(2.0, abs=1e-6)
    assert h1["control_all_days"]["mean_pct"] == pytest.approx(1.0, abs=0.02)
    assert h1["edge_pct"] == pytest.approx(h1["mean_pct"]
                                           - h1["control_all_days"]["mean_pct"],
                                           abs=1e-6)


def test_record_and_sign_test_direction():
    vals = np.array([-0.01] * 6)
    rec = ctx._record(vals)
    assert (rec["up"], rec["down"], rec["sign_dir"]) == (0, 6, "down")
    assert rec["sign_p"] == pytest.approx(0.0156, abs=1e-4)   # rounded to 4dp


def test_sign_test_matches_pitch_lab_at_nugget_sample_sizes():
    """The engine swaps in binom_p_greater because sign_test overflows on the
    n-in-the-thousands cells the sweep also scores. At the sizes a nugget
    actually lives at, the two must be the same number."""
    for wins, n in [(6, 6), (8, 9), (10, 12), (14, 20), (30, 50)]:
        assert ctx.sign_test(wins, n) == pytest.approx(
            pitch_sign_test(wins, n), rel=1e-9)


def test_sign_test_survives_a_large_cell():
    assert 0.0 <= ctx.sign_test(800, 1500) <= 1.0


# ---------------------------------------------------------------------------
# 3. tags
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n,t,stable,expected", [
    (3, 5.0, True, "dead"),
    (9, 5.0, True, "anecdote"),
    (30, 5.0, True, "suggestive"),
    (80, 1.0, True, "suggestive"),     # big N, weak t
    (80, 3.0, False, "suggestive"),    # big N, strong t, era flip
    (80, 3.0, True, "solid"),
])
def test_tag_hint(n, t, stable, expected):
    assert ctx._tag_hint(n, t, stable) == expected


def test_tag_hint_tracks_the_cell_it_describes():
    frame = _panel([0.0] * 30)
    subject = _subject(frame)
    cell = ctx.build_cell(subject, pd.DatetimeIndex(frame.index[:3]))
    assert cell["h"]["1"]["tag_hint"] in ("dead", "anecdote")
    assert cell["h"]["1"]["n"] <= 3


# ---------------------------------------------------------------------------
# 4. price-lane trigger masks
# ---------------------------------------------------------------------------
def test_a_relentless_grind_higher_is_one_piece_of_news():
    """Not one per bar and not one per month: the novelty filter fires on the
    first high and stays quiet while the state persists."""
    frame = _panel([0.002] * 400)
    assert int(ctx._t_new_high(frame).sum()) == 1


def test_a_second_high_after_a_quiet_stretch_fires_again():
    # Up, then a drawdown long enough to clear the 30-day window, then a push
    # back through the old high.
    frame = _panel([0.004] * 300 + [-0.004] * 60 + [0.006] * 60)
    hits = frame.index[ctx._t_new_high(frame).to_numpy()]
    assert len(hits) == 2
    assert (hits[1] - hits[0]).days > 30


def test_drop_after_high_needs_both_legs():
    returns = [0.002] * 300 + [-0.008] + [0.0] * 5
    frame = _panel(returns)
    mask = ctx._drop_after_high(frame, 50)
    fired = frame.index[mask.to_numpy()]
    assert len(fired) == 1
    # The fired bar is the DROP, and the bar before it was the 52w high.
    close = frame["Close"]
    drop_pos = frame.index.get_loc(fired[0])
    assert close.iloc[drop_pos] < close.iloc[drop_pos - 1]
    assert close.iloc[drop_pos - 1] == close.iloc[:drop_pos].max()


def test_drop_after_high_respects_the_threshold():
    returns = [0.002] * 300 + [-0.003] + [0.0] * 5    # 30bp, under the 50bp bar
    frame = _panel(returns)
    assert not ctx._drop_after_high(frame, 50).any()


def test_streak_counts_consecutive_closes():
    returns = [0.01] * 5 + [-0.01] * 6 + [0.0] * 5
    frame = _panel(returns)
    up = ctx._t_streak(frame, 1)
    down = ctx._t_streak(frame, -1)
    assert up.sum() == 1              # the 5th up close only
    assert down.sum() == 2            # the 5th and 6th down closes


def test_two_atr_uses_the_prior_bar_atr():
    frame = _panel([0.0] * 60)
    # A quiet tape, then one big session.
    frame.loc[frame.index[-1], "Close"] = frame["Close"].iloc[-2] * 1.10
    mask = ctx._t_two_atr(frame)
    assert bool(mask.iloc[-1])


def test_sma200_cross_deduplicates_whipsaw():
    # Long uptrend, a crash through the mean, a whipsaw back and forth, then a
    # durable recovery. Four raw crosses, but the whipsaw pair is one event.
    frame = _panel([0.004] * 260 + [-0.02] * 30 + [0.03] * 6 + [-0.03] * 6
                   + [0.01] * 120)
    raw = (frame["Close"] > _sma(frame["Close"], 200))
    raw_crosses = int(((raw != raw.shift(1)) & raw.notna()
                       & raw.shift(1).notna()).sum())
    hits = np.flatnonzero(ctx._t_sma200_cross(frame).to_numpy())
    assert raw_crosses > len(hits) >= 1
    assert all(np.diff(hits) >= 63)


def _sma(close, window):
    return close.rolling(window).mean()


def test_pct_change_does_not_pad_across_a_missing_bar():
    """The cross-asset masks run on series reindexed to the NYSE grid. A padded
    pct_change turns a missing bar into a 0.0% day and a real move into a
    two-day move, which would fire co-movement triggers on sessions that never
    traded for that instrument."""
    idx = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=5))
    s = pd.Series([100.0, np.nan, 110.0, 110.0, 110.0], index=idx)
    got = ctx._pct_change(s)
    assert np.isnan(got.iloc[1])
    assert np.isnan(got.iloc[2])
    assert got.iloc[3] == pytest.approx(0.0)


def test_co_movement_needs_the_magnitude_floor():
    idx = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=4))
    a = pd.Series([100.0, 100.1, 101.0, 101.0], index=idx)   # +10bp, +90bp
    b = pd.Series([50.0, 50.05, 50.5, 50.5], index=idx)
    got = ctx._both_direction(a, b, 1)
    assert not bool(got.iloc[1])       # 10bp is co-movement, not an event
    assert bool(got.iloc[2])


# ---------------------------------------------------------------------------
# 5. session resolution and the freshness gate
# ---------------------------------------------------------------------------
def test_sunday_run_resolves_to_friday_and_previews_monday():
    asof, nxt = ctx.resolve_sessions(pd.Timestamp("2026-08-09"))   # a Sunday
    assert str(asof.date()) == "2026-08-07"
    assert str(nxt.date()) == "2026-08-10"


def test_session_evening_resolves_to_itself():
    asof, nxt = ctx.resolve_sessions(pd.Timestamp("2026-08-06"))   # a Thursday
    assert str(asof.date()) == "2026-08-06"
    assert str(nxt.date()) == "2026-08-07"


def test_month_end_position_counts_sessions():
    pos = ctx.month_end_position(pd.Timestamp("2026-01-30"))
    assert pos["td_from_month_end"] == 0
    assert pos["sessions_in_month"] == pos["td_of_month"]


# ---------------------------------------------------------------------------
# 6. novelty
# ---------------------------------------------------------------------------
def test_novelty_suppresses_new_claims_without_a_state_file(tmp_path):
    flags, suppressed = ctx.load_flag_state(tmp_path / "nope.json", [], True)
    assert flags == {} and suppressed is True
    cells = [{"fingerprint": "P1:new_52w_high|SPY",
              "h": {"1": {"mean_pct": 1.0}}}]
    out = ctx.novelty_for(cells, flags, pd.Timestamp("2026-08-07"), suppressed)
    assert out["delta_suppressed"] is True
    assert out["flags"]["P1:new_52w_high|SPY"]["is_new"] is False


def test_novelty_blocks_a_repeat_inside_five_sessions():
    flags = {"P1:new_52w_high|SPY": {"last_published": "2026-08-05",
                                     "last_headline_number": 1.0, "count": 1}}
    cells = [{"fingerprint": "P1:new_52w_high|SPY",
              "h": {"1": {"mean_pct": 1.02}}}]
    out = ctx.novelty_for(cells, flags, pd.Timestamp("2026-08-07"), False)
    flag = out["flags"]["P1:new_52w_high|SPY"]
    assert flag["is_new"] is False
    assert flag["materially_moved"] is False
    assert flag["repeat_blocked"] is True


def test_novelty_reopens_when_the_number_moves():
    flags = {"P1:new_52w_high|SPY": {"last_published": "2026-08-05",
                                     "last_headline_number": 1.0, "count": 1}}
    cells = [{"fingerprint": "P1:new_52w_high|SPY",
              "h": {"1": {"mean_pct": 1.9}}}]
    out = ctx.novelty_for(cells, flags, pd.Timestamp("2026-08-07"), False)
    flag = out["flags"]["P1:new_52w_high|SPY"]
    assert flag["materially_moved"] is True
    assert flag["repeat_blocked"] is False


# ---------------------------------------------------------------------------
# 7. registry / index integrity
# ---------------------------------------------------------------------------
def test_trigger_ids_are_unique():
    ids = [t.id for t in ctx.PRICE_TRIGGERS]
    assert len(ids) == len(set(ids))


def test_two_sided_triggers_declare_a_side():
    """A trigger that can fire from either tail MUST split, or its pooled
    number describes neither state. Found live on 2026-08-07: pooled ^NDX
    scored +0.24% at t=2.53 and earned `solid` while the live top side was
    -0.02%."""
    two_sided = {"P4:z10_extreme", "P5:rank5_extreme", "P5b:rank21_extreme",
                 "P6:two_atr_day", "P8:sma200_cross"}
    for trig in ctx.PRICE_TRIGGERS:
        if trig.id in two_sided:
            assert trig.side_fn is not None, f"{trig.id} pools its two sides"
            assert "{side}" in trig.cell, f"{trig.id} does not name its side"


def test_side_label_formats_into_the_cell_name():
    for trig in ctx.PRICE_TRIGGERS:
        if trig.side_fn is None:
            continue
        for label in trig.side_labels:
            assert "{side}" not in trig.cell.format(side=label)


def test_rank_side_marks_only_the_tails():
    frame = _panel([0.001] * 400)
    side = ctx._side_rank(frame, 5)
    rank = ctx._pct_rank(frame["Close"].astype(float), 5)
    for pos in range(300, len(frame)):
        r, s = rank.iloc[pos], side.iloc[pos]
        if r >= 95:
            assert s == 1.0
        elif r <= 5:
            assert s == -1.0
        else:
            assert np.isnan(s)


def test_event_lane_excludes_foreign_cash_indices():
    """They close before a US release prints, so their 'event day' bar does
    not contain the event."""
    foreign = {"^N225", "^FTSE", "^GDAXI", "^HSI", "^AXJO", "^KS11", "^FCHI",
               "^MXX", "^BVSP"}
    assert not (set(ctx.EVENT_LANE_SUBJECTS) & foreign)


def test_universe_has_no_duplicates_and_no_sector_subjects():
    flat = [t for names in ctx.CONTEXT_UNIVERSE.values() for t in names]
    assert len(flat) == len(set(flat))
    assert not (set(flat) & set(ctx.BREADTH_ONLY)), \
        "sector ETFs are breadth context only, never a nugget subject"


def test_index_line_resolves_to_a_cell():
    frame = _panel([0.001] * 300)
    subject = _subject(frame)
    cell = {"trigger_id": "P1:new_52w_high", "lane": "price",
            "cell": "first 52w high in 30+ days", "anchor_rule": "test",
            "n_anchors": 3, "fingerprint": "P1:new_52w_high|X"}
    cell.update(ctx.build_cell(subject, pd.DatetimeIndex(frame.index[10:13])))
    index = ctx.build_index([cell])
    assert len(index) == 1
    assert index[0]["subjects"][0]["fp"] == "P1:new_52w_high|X"
    assert index[0]["subjects"][0]["n"] == 3
