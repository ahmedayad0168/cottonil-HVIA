"""Restock planning: turn (forecast + current stock + warehouse stock + transport cost)
into a prioritised list of transfers. Pure pandas, no model needed, so the app can
re-run it instantly with different policy settings ("what-if").

Policy (kept simple on purpose, every number is explainable to a store manager):
    target_stock = cover_weeks/4 * ( mean_4wk + safety * (P90_4wk - mean_4wk) )
    order_qty    = max(0, ceil(target_stock - closing_units))
    action       = URGENT if < 1.5 weeks of cover, RESTOCK if order_qty > 0, OVERSTOCK if > 12 weeks
    source       = cheapest eligible warehouse that still has enough stock (greedy, by priority)
Granularity is branch x category because `monthly_stock` is category level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DC_KINDS = ("Central DC", "Regional DC", "Finished goods")


def aggregate_forecast(fl: pd.DataFrame) -> pd.DataFrame:
    """4-week demand per branch x category from the weekly forecast table.

    P90 of the 4-week total is approximated as mean + sqrt(sum of squared weekly (P90-mean)),
    i.e. assuming independent weekly errors (slightly optimistic, documented in notebook 02).
    """
    f = fl.copy()
    f["up"] = (f["p90"] - f["mean"]).clip(lower=0) ** 2
    g = f.groupby(["branch_id", "category"]).agg(mean_4wk=("mean", "sum"), up=("up", "sum")).reset_index()
    g["p90_4wk"] = g["mean_4wk"] + np.sqrt(g["up"])
    g["weekly_mean"] = g["mean_4wk"] / 4.0
    return g.drop(columns="up")


def transfer_cost_table(transfers: pd.DataFrame) -> pd.DataFrame:
    """Median historical transport cost per (warehouse, branch) pair."""
    return (transfers.groupby(["source_warehouse_id", "destination_branch_id"])["transport_cost_EGP"]
            .median().rename("cost").reset_index()
            .rename(columns={"source_warehouse_id": "warehouse_id", "destination_branch_id": "branch_id"}))


def build_restock_plan(stock: pd.DataFrame, fc4: pd.DataFrame, wh_stock: pd.DataFrame, tcost: pd.DataFrame,
                       price: pd.DataFrame, cover_weeks: float = 4.0, safety: float = 1.0) -> pd.DataFrame:
    """
    stock    : branch_id, category, closing_units, [p_high_risk_next]
    fc4      : output of aggregate_forecast
    wh_stock : warehouse_id, category, closing_units   (only distribution-capable warehouses)
    tcost    : warehouse_id, branch_id, cost
    price    : category, avg_price
    """
    d = stock.merge(fc4, on=["branch_id", "category"], how="left").merge(price, on="category", how="left")
    d["avg_price"] = d["avg_price"].fillna(d["avg_price"].median())
    if "p_high_risk_next" not in d:
        d["p_high_risk_next"] = np.nan
    d[["mean_4wk", "p90_4wk", "weekly_mean"]] = d[["mean_4wk", "p90_4wk", "weekly_mean"]].fillna(0)
    d["weeks_of_cover"] = np.where(d["weekly_mean"] > 0, d["closing_units"] / d["weekly_mean"].where(d["weekly_mean"] > 0), np.inf)
    d["target_stock"] = cover_weeks / 4.0 * (d["mean_4wk"] + safety * (d["p90_4wk"] - d["mean_4wk"]))
    d["order_qty"] = np.ceil((d["target_stock"] - d["closing_units"]).clip(lower=0)).astype(int)
    d["shortfall_units"] = (d["mean_4wk"] - d["closing_units"]).clip(lower=0)
    d["shortfall_value_egp"] = d["shortfall_units"] * d["avg_price"]
    d["priority_score"] = d["shortfall_value_egp"] + d["order_qty"] * d["avg_price"] * 0.05 \
        + d["p_high_risk_next"].fillna(0) * d["avg_price"]
    d["action"] = np.select(
        [d["weeks_of_cover"] < 1.5, d["order_qty"] > 0, d["weeks_of_cover"] > 12],
        ["URGENT", "RESTOCK", "OVERSTOCK"], default="OK")

    # --- greedy source allocation (highest priority first, warehouse stock is consumed) ---
    avail = {(r.warehouse_id, r.category): float(r.closing_units) for r in wh_stock.itertuples()}
    cost = {(r.warehouse_id, r.branch_id): float(r.cost) for r in tcost.itertuples()}
    wh_med = tcost.groupby("warehouse_id")["cost"].median().to_dict()
    glob_med = float(tcost["cost"].median()) if len(tcost) else 0.0
    whs = sorted({w for w, _ in avail})
    src, ship, tc = [], [], []
    todo = d.index[d["order_qty"] > 0]
    order_idx = d.loc[todo].sort_values("priority_score", ascending=False).index
    res = {}
    for i in order_idx:
        row = d.loc[i]
        cands = [(cost.get((w, row.branch_id), wh_med.get(w, glob_med)), w) for w in whs
                 if avail.get((w, row.category), 0) >= row.order_qty]
        if cands:
            c, w = min(cands)
            qty = row.order_qty
        else:   # no single warehouse covers it: ship what the best-stocked one has (partial)
            w = max(whs, key=lambda x: avail.get((x, row.category), 0), default=None)
            qty = min(row.order_qty, avail.get((w, row.category), 0)) if w else 0
            c = cost.get((w, row.branch_id), wh_med.get(w, glob_med)) if w else 0
        if w and qty > 0:
            avail[(w, row.category)] -= qty
        res[i] = (w if qty > 0 else None, int(qty), float(c) if qty > 0 else 0.0)
    d["source_warehouse"] = pd.Series({i: v[0] for i, v in res.items()})
    d["ship_qty"] = pd.Series({i: v[1] for i, v in res.items()}).reindex(d.index).fillna(0).astype(int)
    d["est_transfer_cost_egp"] = pd.Series({i: v[2] for i, v in res.items()}).reindex(d.index).fillna(0.0)
    d["weeks_of_cover"] = d["weeks_of_cover"].replace(np.inf, 99.0).round(1)
    d = d.sort_values("priority_score", ascending=False).reset_index(drop=True)
    d["priority_rank"] = np.arange(1, len(d) + 1)
    return d
