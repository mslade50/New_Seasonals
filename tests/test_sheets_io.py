from copy import deepcopy
from types import SimpleNamespace

import pytest

from sheets_io import replace_worksheet_values


class Worksheet:
    id, row_count, col_count = 123, 5, 5

    def __init__(self, values, failure=None):
        self.values = deepcopy(values)
        self.failure = failure
        self.requests = []
        self.spreadsheet = SimpleNamespace(batch_update=self.batch_update)

    def get_all_values(self):
        return deepcopy(self.values)

    def batch_update(self, payload):
        self.requests.append(payload)
        if self.failure == "before":
            raise OSError("offline")
        request = payload["requests"][-1]["updateCells"]
        self.values = [[c["userEnteredValue"]["stringValue"] for c in row["values"]] for row in request["rows"]]
        if self.failure == "corrupt":
            self.values[-1][-1] = "wrong value"
        if self.failure == "after":
            raise OSError("response lost")


def test_replacement_failure_keeps_the_previous_table():
    ws = Worksheet([["symbol", "state"], ["SPY", "PENDING"]], "before")
    before = deepcopy(ws.values)
    with pytest.raises(OSError):
        replace_worksheet_values(ws, [["symbol", "state"], ["QQQ", "FILLED"]])
    assert ws.values == before
    assert len(ws.requests) == 1


def test_one_atomic_batch_clears_trailing_cells_and_expands_grid():
    ws = Worksheet([["old", "trailing"], ["remove"]])
    replace_worksheet_values(ws, [["=literal", "1", "2", "3", "4", "5"]])
    requests = ws.requests[0]["requests"]
    assert len(requests) == 2
    assert requests[0]["updateSheetProperties"]["properties"]["gridProperties"]["columnCount"] == 6
    assert requests[-1]["updateCells"]["range"]["endRowIndex"] == 5
    assert ws.values == [["=literal", "1", "2", "3", "4", "5"]]


def test_ambiguous_response_accepts_only_full_matching_readback():
    ws = Worksheet([["old"]], "after")
    assert replace_worksheet_values(ws, [["new"]]) == [["new"]]
    ws = Worksheet([["old"]], "corrupt")
    with pytest.raises(RuntimeError, match="readback differs"):
        replace_worksheet_values(ws, [["new"]])


def test_changed_snapshot_prevents_replacement():
    ws = Worksheet([["manual edit"]])
    with pytest.raises(RuntimeError, match="changed since read"):
        replace_worksheet_values(ws, [["new"]], expected_values=[["old"]])
    assert ws.requests == []


def test_empty_and_ragged_rows_match_sheets_readback():
    ws = Worksheet([["old"]])
    assert replace_worksheet_values(ws, [["A", ""], ["", ""], []]) == [["A"]]
    assert replace_worksheet_values(ws, []) == []
