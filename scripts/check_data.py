"""First thing to run with your REAL data: validates every table/column the project needs.

    python scripts/check_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cottonil import config as cfg  # noqa: E402
from cottonil.data_io import SCHEMA, load_table  # noqa: E402

d = cfg.find_data_dir()
print(f"Data folder: {d}   [{cfg.data_source(d).upper()}]\n")
bad = 0
for name in SCHEMA:
    try:
        df = load_table(name, d)
        print(f"  OK   {name:<22s} {len(df):>10,} rows  {df.shape[1]:>3} cols")
    except Exception as e:
        bad += 1
        print(f"  FAIL {name:<22s} {str(e)[:150]}")
print("\nAll tables valid - you can run the notebooks." if not bad else f"\n{bad} table(s) need attention before running the notebooks.")
sys.exit(1 if bad else 0)
