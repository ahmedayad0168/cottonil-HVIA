"""Loading + validating the Cottonil tables.

Schemas below are taken from the real data profile in test.ipynb. If a real
file is missing a column, `load_table` fails early with a clear message
instead of a confusing KeyError 40 cells later.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .config import find_data_dir

# required columns per table (only what this project uses)
SCHEMA: dict[str, list[str]] = {
    "sales_orders": ["sales_order_id", "order_date", "branch_id", "customer_id", "sales_channel",
                     "payment_method", "items_count", "gross_EGP", "discount_EGP",
                     "net_before_tax_EGP", "tax_EGP", "paid_total_EGP", "status"],
    "sales_order_items": ["sales_line_id", "sales_order_id", "order_date", "branch_id", "sku_id",
                          "product_id", "quantity", "unit_price_EGP", "gross_line_EGP",
                          "discount_line_EGP", "net_line_EGP", "unit_cost_EGP", "margin_EGP"],
    "products_catalog": ["product_id", "product_name_ar", "category_en", "audience", "listed_price_EGP"],
    "sku_variants": ["sku_id", "product_id", "size", "color_ar", "price_multiplier"],
    "customers": ["customer_id", "customer_segment", "governorate_ar", "joined_date", "age_band"],
    "branches": ["branch_id", "branch_name_ar", "governorate_ar", "city_ar", "retail_location_type",
                 "opening_date"],
    "monthly_stock": ["year_month", "stock_location_id", "stock_location_type", "product_category",
                      "closing_units", "closing_stock_value_EGP", "stockout_risk_score"],
    "sales_returns": ["return_id", "sales_order_id", "sales_line_id", "return_date", "sku_id",
                      "return_quantity", "refund_amount_EGP", "reason", "refund_method"],
    "purchase_orders": ["po_id", "po_date", "supplier_id", "receiving_warehouse_id", "po_status",
                        "currency", "po_subtotal_EGP", "payment_terms_days", "po_purpose"],
    "purchase_order_items": ["po_line_id", "po_id", "sku_id", "quantity", "unit_cost_EGP",
                             "line_cost_EGP", "delivery_date"],
    "suppliers": ["supplier_id", "supplier_name_ar", "material_type", "city_country",
                  "payment_terms_days"],
    "warehouses": ["warehouse_id", "warehouse_label", "governorate_ar", "warehouse_kind",
                   "capacity_units"],
    "marketing_campaigns": ["campaign_id", "start_date", "end_date", "channel", "target_audience",
                            "governorate_ar", "budget_EGP", "spend_EGP", "impressions", "clicks",
                            "orders_attributed"],
    "inventory_transfers": ["transfer_id", "transfer_date", "source_warehouse_id",
                            "destination_branch_id", "transfer_status", "transport_cost_EGP"],
}

# columns stored as month/day/year in the real files (everything else is ISO yyyy-mm-dd)
_MDY_COLUMNS = {"opening_date", "joined_date"}
_DATE_COLUMNS = {"order_date", "return_date", "po_date", "delivery_date", "start_date", "end_date",
                 "transfer_date", "opening_date", "joined_date"}


def tables_dir(data_dir: Path | None = None) -> Path:
    return (data_dir or find_data_dir()) / "02_TABLES"


def load_table(name: str, data_dir: Path | None = None, usecols: list[str] | None = None,
               parse_dates: bool = True) -> pd.DataFrame:
    """Read `<data>/02_TABLES/<name>.csv`, validate required columns, parse dates."""
    path = tables_dir(data_dir) / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing table file: {path}")
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [c for c in SCHEMA.get(name, []) if c not in header]
    if missing:
        raise ValueError(f"{path.name}: missing required columns {missing}. Found: {header}")
    cols = usecols if usecols is not None else [c for c in header if c in SCHEMA.get(name, header)]
    df = pd.read_csv(path, usecols=cols)
    if parse_dates:
        for c in df.columns:
            if c in _DATE_COLUMNS:
                fmt = "%m/%d/%Y" if c in _MDY_COLUMNS else None
                df[c] = pd.to_datetime(df[c], format=fmt, errors="coerce")
    return df


# ---------------------------------------------------------------------------
# master tables
# ---------------------------------------------------------------------------
def load_product_master(data_dir: Path | None = None) -> pd.DataFrame:
    """One row per SKU with product + category info."""
    prod = load_table("products_catalog", data_dir)
    sku = load_table("sku_variants", data_dir)
    m = sku.merge(prod, on="product_id", how="left")
    m = m.rename(columns={"category_en": "category"})
    return m


def load_sales_lines(data_dir: Path | None = None) -> pd.DataFrame:
    """Sales lines enriched with category/size and the order's channel + customer.

    This is the workhorse table (1.1M rows on the real data), so we keep
    memory low by using categorical dtypes for ids.
    """
    lines = load_table("sales_order_items", data_dir)
    prod = load_table("products_catalog", data_dir)[["product_id", "category_en", "audience",
                                                     "product_name_ar", "listed_price_EGP"]]
    sku = load_table("sku_variants", data_dir)[["sku_id", "size", "color_ar"]]
    orders = load_table("sales_orders", data_dir)[["sales_order_id", "customer_id", "sales_channel",
                                                   "payment_method", "items_count", "status"]]
    lines = lines.merge(prod, on="product_id", how="left").rename(columns={"category_en": "category"})
    lines = lines.merge(sku, on="sku_id", how="left")
    lines = lines.merge(orders, on="sales_order_id", how="left")
    for c in ("branch_id", "category", "audience", "size", "sales_channel", "payment_method", "status"):
        lines[c] = lines[c].astype("category")
    return lines


def week_start(dates: pd.Series) -> pd.Series:
    """Monday of the ISO week containing each date."""
    d = pd.to_datetime(dates)
    return (d - pd.to_timedelta(d.dt.dayofweek, unit="D")).dt.normalize()


def quick_profile(df: pd.DataFrame, name: str = "") -> pd.DataFrame:
    """Compact profile used in the exploration notebook."""
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "nulls": df.isna().sum(),
        "null_%": (df.isna().mean() * 100).round(2),
        "unique": df.nunique(),
    }).rename_axis(name or "column")


@lru_cache(maxsize=1)
def default_data_dir() -> Path:
    return find_data_dir()
