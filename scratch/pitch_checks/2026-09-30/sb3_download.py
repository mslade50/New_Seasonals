import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import cache_io  # noqa: E402

out = Path(__file__).parent / "sb3_ledger_r2.parquet"
ok = cache_io.download_to_local("seasonal_ideas_log.parquet", str(out))
print("ok", ok, cache_io.last_download_error())
print(cache_io.head("seasonal_ideas_log.parquet"))
