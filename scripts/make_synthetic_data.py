"""Create a synthetic, schema-identical Cottonil dataset for testing.

    python scripts/make_synthetic_data.py --scale 1.0      # full size (540K orders, a few minutes)
    python scripts/make_synthetic_data.py --scale 0.1      # quick (54K orders)

Output: data/synthetic/COTTONIL_Data/  (never overwrites data/COTTONIL_Data = your real data)
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cottonil.synth import generate  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0, help="1.0 = 540K orders like the real data")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "synthetic" / "COTTONIL_Data"))
    a = ap.parse_args()
    t = time.time()
    generate(a.out, scale=a.scale, seed=a.seed)
    print(f"done in {time.time() - t:.0f}s -> {a.out}")
