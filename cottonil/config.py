"""Central configuration: paths, data discovery and artifact helpers.

A data folder is valid if it contains 02_TABLES/sales_orders.csv.
Synthetic data folders contain a SYNTHETIC_DATA_NOTICE.txt marker, which is how
every saved model/metric file knows whether it was trained on real or fake data.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
SERVING_DIR = ARTIFACTS_DIR / "serving"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
SYNTHETIC_MARKER = "SYNTHETIC_DATA_NOTICE.txt"

for _d in (MODELS_DIR, SERVING_DIR, METRICS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# data discovery
# ----------------------------------------------------------------------------
def _is_valid(p: Path) -> bool:
    return (p / "02_TABLES" / "sales_orders.csv").exists()


def find_data_dir() -> Path:
    cands = []
    env = os.environ.get("COTTONIL_DATA_DIR")
    if env:
        cands.append(Path(env))
    cands += [
        PROJECT_ROOT / "data" / "COTTONIL_Data",
        PROJECT_ROOT / "COTTONIL_Data",
        Path.cwd() / "COTTONIL_Data",
        PROJECT_ROOT / "data" / "synthetic" / "COTTONIL_Data",
    ]
    for c in cands:
        if _is_valid(c):
            return c
    raise FileNotFoundError(
        "Could not find the Cottonil data.\n"
        "  - Put your real data in  data/COTTONIL_Data/02_TABLES/*.csv  OR\n"
        "  - run  python scripts/make_synthetic_data.py  to create a synthetic test set OR\n"
        "  - set the COTTONIL_DATA_DIR environment variable."
    )


def data_source(data_dir: Path | None = None) -> str:
    """'synthetic' if the folder carries the synthetic marker, else 'real'."""
    data_dir = data_dir or find_data_dir()
    return "synthetic" if (data_dir / SYNTHETIC_MARKER).exists() else "real"


# ----------------------------------------------------------------------------
# artifact helpers (used by every notebook so the app can find things)
# ----------------------------------------------------------------------------
def save_model(name: str, obj) -> Path:
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(obj, path, compress=3)
    return path


def load_model(name: str):
    return joblib.load(MODELS_DIR / f"{name}.joblib")


def save_metrics(name: str, metrics: dict, data_dir: Path | None = None) -> Path:
    """Persist metrics + provenance (real vs synthetic data, timestamp)."""
    payload = {
        "model": name,
        "data_source": data_source(data_dir),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **metrics,
    }
    path = METRICS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=_json_default, ensure_ascii=False))
    return path


def load_metrics(name: str) -> dict:
    p = METRICS_DIR / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_serving(name: str, df, index: bool = False) -> Path:
    """Small tables the app reads at runtime (gz-compressed CSV, no extra deps)."""
    path = SERVING_DIR / f"{name}.csv.gz"
    df.to_csv(path, index=index, compression="gzip")
    return path


def load_serving(name: str, **kw):
    import pandas as pd

    return pd.read_csv(SERVING_DIR / f"{name}.csv.gz", **kw)


def _json_default(o):
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if hasattr(o, "isoformat"):
        return o.isoformat()
    return str(o)
