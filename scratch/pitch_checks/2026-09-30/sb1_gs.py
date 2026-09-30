"""sb1: kill check for board ticket GS 21d Oct long (gate: 21d return <= 15th pctile)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sb1_engine import run

run("GS", 21)
