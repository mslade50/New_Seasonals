"""Exercise the scanner's actual date-validation loop without any staging I/O."""
import ast
import datetime as dt
from pathlib import Path

import pandas as pd


def validate(source: Path, frames: dict):
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_daily_scan")
    loop = next(n for n in function.body if isinstance(n, ast.For)
                and ast.unparse(n.target) == "(ticker, df)"
                and any(isinstance(x, ast.Name) and x.id == "_stale_drop" for x in ast.walk(n)))
    scope = dict(master_dict=frames, expected_data_date=dt.date(2026, 9, 4),
                 _stale_floor=dt.date(2026, 9, 4), validated_dict={}, _stale_drop={})
    exec(compile(ast.Module(body=[loop], type_ignores=[]), str(source), "exec"), scope)
    return scope["validated_dict"], scope["_stale_drop"]


def test_labor_day_crypto_fx_and_equity_share_friday_settlement():
    def frame(dates):
        return pd.DataFrame({"Close": [100] * len(dates)}, index=pd.to_datetime(dates))
    frames = {
        "SPY": frame(["2026-09-03", "2026-09-04"]),
        "BTC-USD": frame(["2026-09-04", "2026-09-05", "2026-09-06", "2026-09-07"]),
        "USDBRL=X": frame(["2026-09-04", "2026-09-07", "2026-09-08"]),
        "STALE": frame(["2026-09-03"]),
        "FUTURE_ONLY": frame(["2026-09-07", "2026-09-08"]),
    }
    valid, stale = validate(Path(__file__).resolve().parents[1] / "daily_scan.py", frames)
    assert set(valid) == {"SPY", "BTC-USD", "USDBRL=X"}
    assert {df.index[-1].date() for df in valid.values()} == {dt.date(2026, 9, 4)}
    assert stale == {"STALE": dt.date(2026, 9, 3)}
    assert len(frames["BTC-USD"]) == 4  # history itself is preserved
