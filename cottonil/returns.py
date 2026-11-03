"""Feature encoding for the return-risk model (shared by notebook 05 and the app)."""
from __future__ import annotations

import numpy as np
import pandas as pd

CAT_COLS = ["category", "size", "sales_channel", "payment_method", "branch_id", "age_band", "customer_segment"]
NUM_COLS = ["quantity", "unit_price_EGP", "discount_pct", "log_line_value", "items_count", "identified", "month", "dow"]
FEATURES = CAT_COLS + NUM_COLS


def fit_maps(df: pd.DataFrame) -> dict:
    return {c: sorted(df[c].astype(str).fillna("NA").unique().tolist()) for c in CAT_COLS}


def encode(df: pd.DataFrame, maps: dict) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in CAT_COLS:
        m = {v: i for i, v in enumerate(maps[c])}
        X[c] = df[c].astype(str).map(m).fillna(-1).astype(np.int16)    # unseen value -> -1 (treated as missing)
    X["quantity"] = df["quantity"].astype(np.float32)
    X["unit_price_EGP"] = df["unit_price_EGP"].astype(np.float32)
    X["discount_pct"] = (df["discount_line_EGP"] / df["gross_line_EGP"].replace(0, np.nan)).fillna(0).astype(np.float32)
    X["log_line_value"] = np.log1p(df["net_line_EGP"]).astype(np.float32)
    X["items_count"] = df["items_count"].astype(np.float32)
    X["identified"] = df["customer_id"].notna().astype(np.int8)
    X["month"] = pd.to_datetime(df["order_date"]).dt.month.astype(np.int8)
    X["dow"] = pd.to_datetime(df["order_date"]).dt.dayofweek.astype(np.int8)
    return X[FEATURES]
