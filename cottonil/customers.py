"""Customer features (RFM+), churn labels and rule-based RFM segments.

Churn definition (editable): an *active* customer (bought in the last 365 days) is
"churned" at snapshot date t if they place NO order in (t, t + CHURN_DAYS].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CHURN_DAYS = 120
ACTIVE_WINDOW = 365
FEATURES = ["recency_days", "freq_total", "freq_90", "freq_365", "monetary_total", "monetary_365", "aov", "avg_items",
            "tenure_days", "avg_gap_days", "recency_ratio", "share_ecom", "share_wholesale", "days_since_join",
            "age_code", "gov_code", "is_reseller"]
AGE_ORDER = ["18-24", "25-34", "35-44", "45-54", "55+"]


def snapshot_features(o: pd.DataFrame, snap: pd.Timestamp) -> pd.DataFrame:
    """o: identified orders (customer_id, order_date, net, items, channel). Uses orders <= snap only."""
    x = o[o["order_date"] <= snap]
    if x.empty:
        return pd.DataFrame()
    age = (snap - x["order_date"]).dt.days
    x = x.assign(w90=(age < 90).astype(np.int8), w365=(age < ACTIVE_WINDOW).astype(np.int8),
                 net365=np.where(age < ACTIVE_WINDOW, x["net"], 0.0),
                 ecom=(x["channel"] == "E-commerce").astype(np.int8),
                 whole=(x["channel"] == "Wholesale / reseller").astype(np.int8))
    g = x.groupby("customer_id", sort=False).agg(
        last=("order_date", "max"), first=("order_date", "min"), freq_total=("order_date", "size"),
        freq_90=("w90", "sum"), freq_365=("w365", "sum"), monetary_total=("net", "sum"), monetary_365=("net365", "sum"),
        avg_items=("items", "mean"), share_ecom=("ecom", "mean"), share_wholesale=("whole", "mean"))
    g["recency_days"] = (snap - g["last"]).dt.days
    g["tenure_days"] = (snap - g["first"]).dt.days
    g["aov"] = g["monetary_total"] / g["freq_total"]
    g["avg_gap_days"] = np.where(g["freq_total"] > 1, g["tenure_days"] / (g["freq_total"] - 1).clip(lower=1), np.nan)
    g["recency_ratio"] = g["recency_days"] / (g["avg_gap_days"] + 1)
    return g.drop(columns=["last", "first"])


def add_static(g: pd.DataFrame, customers: pd.DataFrame, snap: pd.Timestamp) -> pd.DataFrame:
    c = customers.set_index("customer_id")
    g = g.join(c[["joined_date", "age_band", "governorate_ar", "customer_segment"]], how="left")
    g["days_since_join"] = (snap - g["joined_date"]).dt.days
    g["age_code"] = g["age_band"].map({a: i for i, a in enumerate(AGE_ORDER)}).fillna(-1)
    g["gov_code"] = pd.factorize(g["governorate_ar"], sort=True)[0]
    g["is_reseller"] = (g["customer_segment"].astype(str).str.lower().str.contains("resell|b2b|wholesale")).astype(int)
    return g


def churn_labels(o: pd.DataFrame, customers_idx: pd.Index, snap: pd.Timestamp, days: int = CHURN_DAYS) -> pd.Series:
    """1 = no order in (snap, snap+days]."""
    fut = o[(o["order_date"] > snap) & (o["order_date"] <= snap + pd.Timedelta(days=days))]["customer_id"].unique()
    return pd.Series(~pd.Index(customers_idx).isin(fut), index=customers_idx).astype(int)


def rfm_segments(df: pd.DataFrame) -> pd.Series:
    """Classic, explainable RFM segments (quintile scores 1-5)."""
    r = pd.qcut(df["recency_days"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)       # recent = 5
    f = pd.qcut(df["freq_total"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    m = pd.qcut(df["monetary_total"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    seg = np.select(
        [(r >= 4) & (f >= 4), (r >= 3) & (f >= 4), (r >= 4) & (f == 1), (r >= 4) & (f >= 2),
         (r <= 2) & (f >= 4) & (m >= 4), (r <= 2) & (f >= 3), (r <= 2) & (f <= 2)],
        ["Champions", "Loyal", "New customers", "Potential loyalists", "Can't lose them", "At risk", "Hibernating"],
        default="Needs attention")
    return pd.Series(seg, index=df.index)


ACTIONS = {
    "Champions": "Reward: early access / loyalty perks; ask for referrals",
    "Loyal": "Upsell: cross-sell bundles, higher-margin categories",
    "Potential loyalists": "Nurture: 2nd/3rd-purchase offer within 30 days",
    "New customers": "Onboard: welcome message + incentive for 2nd order",
    "Needs attention": "Re-engage: reminder with their favourite category",
    "At risk": "Win-back: personalised WhatsApp/SMS offer now",
    "Can't lose them": "Priority win-back: call or VIP offer (high lifetime value)",
    "Hibernating": "Low-cost reactivation only (SMS/email), no heavy discount",
    "Lapsed (>365d)": "Reactivation campaign / exclude from active reporting",
}


def recommended_action(segment: str, p_churn: float) -> str:
    base = ACTIONS.get(segment, "")
    if segment in ("Champions", "Loyal", "Potential loyalists", "New customers") and p_churn >= .6:
        return "Early-warning: usually active but likely to lapse -> proactive nudge. " + base
    return base
