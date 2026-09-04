

# ---------------------------------------------------------------------------
# anchor_positions — both searchsorted guards (promoted 2026-08-28)
# ---------------------------------------------------------------------------
@pytest.fixture()
def late_index():
    """An instrument that starts in 2020, against events spanning 2015-2030."""
    return pd.bdate_range("2020-01-01", periods=400)


def test_anchor_positions_drops_pre_inception_anchors(late_index):
    """searchsorted returns 0 for every date before the first bar, which
    collapses all pre-inception anchors onto the opening sessions. On
    2026-08-28 that counted one early SVXY value twelve times and reported
    n=26 against a real history of 14."""
    events = pd.DatetimeIndex(["2015-08-28", "2016-08-26", "2017-08-25",
                               "2021-08-27", "2022-08-26"])
    pos, kept = pl.anchor_positions(late_index, events)
    assert list(kept) == list(pd.DatetimeIndex(["2021-08-27", "2022-08-26"]))
    assert len(pos) == 2
    # the collapsed form would have put three anchors at position 0
    assert 0 not in pos
    assert len(set(pos)) == len(pos)


def test_anchor_positions_drops_future_anchors(late_index):
    """The documented guard: an unrealised event must not resolve to the end
    of the index and mint a spurious recent anchor."""
    events = pd.DatetimeIndex(["2021-08-27", "2030-08-30"])
    pos, kept = pl.anchor_positions(late_index, events)
    assert list(kept) == [pd.Timestamp("2021-08-27")]
    assert pos[0] < len(late_index) - 1


def test_anchor_positions_offset_stays_in_range(late_index):
    """An offset that walks off either end drops the anchor rather than
    clipping it onto a neighbouring session."""
    first, last = late_index[0], late_index[-1]
    pos, kept = pl.anchor_positions(late_index, [first], offset=-1)
    assert pos == [] and len(kept) == 0
    pos, kept = pl.anchor_positions(late_index, [last], offset=1)
    assert pos == [] and len(kept) == 0
    pos, kept = pl.anchor_positions(late_index, [late_index[10]], offset=3)
    assert pos == [13] and list(kept) == [late_index[10]]


def test_anchor_positions_lands_on_or_after_a_non_session(late_index):
    """A holiday anchor resolves forward to the next real session, and the
    returned anchor date is the EVENT date so a caller can report it."""
    holiday = late_index[10] + pd.Timedelta(days=1)   # between two sessions
    assert holiday not in late_index
    pos, kept = pl.anchor_positions(late_index, [holiday])
    assert late_index[pos[0]] > holiday
    assert list(kept) == [holiday]


def test_anchor_positions_handles_an_empty_index():
    pos, kept = pl.anchor_positions(pd.DatetimeIndex([]), ["2021-08-27"])
    assert pos == [] and len(kept) == 0
