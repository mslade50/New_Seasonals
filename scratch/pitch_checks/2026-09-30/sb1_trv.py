"""sb1: kill check for board ticket TRV 21d Oct long (gate: 10d return <= 15th pctile)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sb1_engine import run

run("TRV", 10)
