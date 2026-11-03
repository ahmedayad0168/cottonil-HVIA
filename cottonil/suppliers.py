"""Features for the purchase-order risk models (shared by notebook 06 and the app).
Only information available WHEN THE PO IS DRAFTED is used (no delivery dates, no status)."""
from __future__ import annotations

import numpy as np
import pandas as pd

CAT_COLS = ["supplier_id", "material_type", "supplier_city", "receiving_warehouse_id", "currency", "po_purpose"]
NUM_COLS = ["log_subtotal", "payment_terms_days", "month", "n_lines", "avg_qty", "log_avg_unit_cost"]
FEATURES = CAT_COLS + NUM_COLS


def build_po_table(po: pd.DataFrame, items: pd.DataFrame, sup: pd.DataFrame) -> pd.DataFrame:
    g = items.groupby("po_id").agg(n_lines=("po_line_id", "size"), avg_qty=("quantity", "mean"),
                                   avg_unit_cost=("unit_cost_EGP", "mean"), last_delivery=("delivery_date", "max"))
    d = po.merge(g, on="po_id", how="left").merge(
        sup[["supplier_id", "material_type", "city_country"]].rename(columns={"city_country": "supplier_city"}), on="supplier_id", how="left")
    d["lead_days"] = (d["last_delivery"] - d["po_date"]).dt.days
    d["partial"] = (d["po_status"] == "Partial receipt").astype(int)
    return d


def fit_maps(d: pd.DataFrame) -> dict:
    return {c: sorted(d[c].astype(str).unique().tolist()) for c in CAT_COLS}


def encode(d: pd.DataFrame, maps: dict) -> pd.DataFrame:
    X = pd.DataFrame(index=d.index)
    for c in CAT_COLS:
        m = {v: i for i, v in enumerate(maps[c])}
        X[c] = d[c].astype(str).map(m).fillna(-1).astype(np.int16)
    X["log_subtotal"] = np.log1p(d["po_subtotal_EGP"]).astype(np.float32)
    X["payment_terms_days"] = d["payment_terms_days"].astype(np.float32)
    X["month"] = pd.to_datetime(d["po_date"]).dt.month.astype(np.int8)
    X["n_lines"] = d["n_lines"].astype(np.float32)
    X["avg_qty"] = d["avg_qty"].astype(np.float32)
    X["log_avg_unit_cost"] = np.log1p(d["avg_unit_cost"]).astype(np.float32)
    return X[FEATURES]


def wilson(k, n, z=1.96):
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = np.where(n > 0, k / np.maximum(n, 1), 0)
    den = 1 + z ** 2 / np.maximum(n, 1)
    c = (p + z ** 2 / (2 * np.maximum(n, 1))) / den
    h = z * np.sqrt(p * (1 - p) / np.maximum(n, 1) + z ** 2 / (4 * np.maximum(n, 1) ** 2)) / den
    return np.clip(c - h, 0, 1), np.clip(c + h, 0, 1)
