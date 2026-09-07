"""Whole-table Sheets writes without a destructive clear/write gap.

The Sheets batch is atomic. The optional read precondition catches intervening
edits, but is not a server-side compare-and-swap: callers still need one writer
or a lease when the table has concurrent owners.
"""
import math


def _cell_text(value):
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return str(value)


def _table(values):
    rows = [[_cell_text(value) for value in row] for row in values]
    for row in rows:
        while row and row[-1] == "":
            row.pop()
    while rows and not rows[-1]:
        rows.pop()
    return rows


def replace_worksheet_values(worksheet, rows, *, expected_values=None):
    """Replace values atomically, verify every cell, return observed values.

    All cells are literal strings (including strings beginning with '='); no
    formula evaluation is introduced. Trailing old cells are cleared in the
    same updateCells request. An ambiguous transport error is accepted only
    after a complete matching readback.
    """
    desired = _table(rows)
    if expected_values is not None:
        if _table(worksheet.get_all_values()) != _table(expected_values):
            raise RuntimeError("Worksheet changed since read; replacement was not sent")
    row_count = max(int(worksheet.row_count), len(desired), 1)
    col_count = max(int(worksheet.col_count), max(map(len, desired), default=0), 1)
    requests = []
    if row_count > worksheet.row_count or col_count > worksheet.col_count:
        requests.append({"updateSheetProperties": {
            "properties": {"sheetId": worksheet.id, "gridProperties": {
                "rowCount": row_count, "columnCount": col_count}},
            "fields": "gridProperties.rowCount,gridProperties.columnCount"}})
    requests.append({"updateCells": {
        "range": {"sheetId": worksheet.id, "startRowIndex": 0,
                  "startColumnIndex": 0, "endRowIndex": row_count,
                  "endColumnIndex": col_count},
        "rows": [{"values": [{"userEnteredValue": {"stringValue": cell}}
                              for cell in row]} for row in desired],
        "fields": "userEnteredValue"}})
    try:
        worksheet.spreadsheet.batch_update({"requests": requests})
    except Exception:
        # A response can be lost after the server has committed the batch.
        observed = worksheet.get_all_values()
        if _table(observed) == desired:
            return observed
        raise
    observed = worksheet.get_all_values()
    if _table(observed) != desired:
        raise RuntimeError("Worksheet replacement readback differs from intended values")
    return observed
