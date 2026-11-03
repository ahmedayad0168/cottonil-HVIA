"""Execute the notebooks in order (train -> evaluate -> save artifacts), writing outputs back into each .ipynb.

    python scripts/run_all_notebooks.py                 # everything (~8-12 min on full-size data, 1 CPU)
    python scripts/run_all_notebooks.py --only 01 03    # just these (by numeric prefix)

Needs:  pip install -r requirements-dev.txt
Data :  real data in data/COTTONIL_Data/  (else synthetic from data/synthetic/ if present)
"""
import argparse
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="numeric prefixes, e.g. 00 03")
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()
    from cottonil import config as cfg
    print(f"Data folder: {cfg.find_data_dir()}  [{cfg.data_source().upper()}]\n")
    nbs = sorted((ROOT / "notebooks").glob("[0-9][0-9]_*.ipynb"))
    if a.only:
        nbs = [n for n in nbs if n.name[:2] in a.only]
    t_all = time.time()
    for path in nbs:
        t0 = time.time()
        nb = nbformat.read(path, as_version=4)
        try:
            NotebookClient(nb, timeout=a.timeout, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}).execute()
        except Exception as e:  # keep the partial outputs so the failure is debuggable
            nbformat.write(nb, path)
            print(f"FAILED {path.name} after {time.time() - t0:.0f}s\n{str(e)[-1800:]}")
            sys.exit(1)
        nbformat.write(nb, path)
        print(f"OK  {path.name:<42s} {time.time() - t0:5.0f}s")
    print(f"\nAll done in {time.time() - t_all:.0f}s. Artifacts are in artifacts/. Start the app with:  python app.py")


if __name__ == "__main__":
    main()
