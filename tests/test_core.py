"""Unit + integration tests. Run:  python -m pytest -q tests"""
import numpy as np
import pandas as pd
import pytest

from cottonil import crosssell as XS, customers as C, forecast as F, marketing as MK, restock as RS, suppliers as SP
from cottonil.calendar_features import black_friday, calendar_features
from cottonil.data_io import load_sales_lines, load_table


# --------------------------------------------------------------------------- calendar
def test_calendar_flags():
    f = calendar_features(["2025-03-10", "2025-03-25", "2025-08-01"])
    assert list(f.is_ramadan) == [1, 1, 0]
    assert f.days_to_eid_fitr.iloc[1] == 5                       # Eid al-Fitr 2025-03-30
    assert black_friday(2026) == pd.Timestamp("2026-11-27")


# --------------------------------------------------------------------------- data + schema
def test_schema_validation_catches_missing_column(path):
    t = path / "02_TABLES"
    t.mkdir()
    pd.DataFrame({"sales_order_id": ["a"], "order_date": ["2020-01-01"]}).to_csv(t / "sales_orders.csv", index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_table("sales_orders", path)


def test_synthetic_matches_headline_stats(tiny_data):
    so, soi = load_table("sales_orders"), load_table("sales_order_items")
    assert 0.15 < so.customer_id.isna().mean() < 0.23                         # ~19.1% anonymous
    assert 2.0 < so.items_count.mean() < 2.5
    assert (soi.groupby("sales_order_id").size().reindex(so.sales_order_id).values == so.items_count.values).all()
    assert np.allclose(soi.margin_EGP, soi.net_line_EGP - soi.quantity * soi.unit_cost_EGP, atol=.02)


# --------------------------------------------------------------------------- forecasting
@pytest.fixture(scope="module")
def panel(tiny_data):
    return F.make_panel(load_sales_lines(), load_table("branches"))


def test_panel_units_match_ledger(panel, tiny_data):
    total = load_sales_lines().quantity.sum()
    assert abs(panel["U"].sum() - total) / total < 0.01                        # only the partial first week is dropped
    assert panel["U"].shape[0] == panel["B"] * panel["K"]


def test_features_cannot_see_the_future(panel):
    """Changing every unit AFTER origin w must not change that origin's features."""
    W = panel["U"].shape[1]
    w = W - 30
    X1, _, m1 = F.build_features(panel, 2, np.array([w]))
    p2 = dict(panel)
    U2 = panel["U"].copy()
    U2[:, w + 1:] = np.random.default_rng(0).integers(0, 500, U2[:, w + 1:].shape)
    p2["U"] = U2
    X2, _, m2 = F.build_features(p2, 2, np.array([w]))
    pd.testing.assert_frame_equal(X1, X2)


def test_forecast_roundtrip_and_interval_order(panel):
    from sklearn.ensemble import HistGradientBoostingRegressor
    W = panel["U"].shape[1]
    models, phi = {}, {}
    for h in F.HORIZONS:
        X, y, _ = F.build_features(panel, h, np.arange(F.MIN_HISTORY_WEEKS - 1, W - h))
        models[h] = HistGradientBoostingRegressor(loss="poisson", max_iter=15).fit(X, y)
        phi[h] = 0.5
    fl = F.latest_forecast({"models": models, "features": F.feature_names(), "phi": phi}, panel)
    assert len(fl) == panel["U"].shape[0] * 4 and (fl.p10 <= fl["mean"] + 1e-6).all() and (fl["mean"] <= fl.p90 + 1e-6).all()
    assert fl.week_start.min() == panel["weeks"][-1] + pd.Timedelta(days=7)


def test_overdispersion_and_quantiles():
    rng = np.random.default_rng(0)
    mu = np.full(20000, 10.0)
    y = rng.negative_binomial(5, 5 / (5 + 10), 20000)                          # true phi = 1/5 = 0.2
    phi = F.estimate_overdispersion(y, mu)
    assert 0.15 < phi < 0.25
    lo, hi = F.nb_quantile(mu, phi, .1), F.nb_quantile(mu, phi, .9)
    assert 0.75 < ((y >= lo) & (y <= hi)).mean() < 0.9


# --------------------------------------------------------------------------- restock
def test_restock_policy_and_source_choice():
    stock = pd.DataFrame({"branch_id": ["B1", "B2"], "category": ["Socks", "Socks"], "closing_units": [10, 10], "p_high_risk_next": [.9, .1]})
    fc4 = pd.DataFrame({"branch_id": ["B1", "B2"], "category": ["Socks", "Socks"], "mean_4wk": [80.0, 80.0], "p90_4wk": [100.0, 100.0], "weekly_mean": [20.0, 20.0]})
    wh = pd.DataFrame({"warehouse_id": ["W1", "W2"], "category": ["Socks", "Socks"], "closing_units": [100, 1000]})
    tc = pd.DataFrame({"warehouse_id": ["W1", "W2", "W1", "W2"], "branch_id": ["B1", "B1", "B2", "B2"], "cost": [50, 500, 50, 500]})
    price = pd.DataFrame({"category": ["Socks"], "avg_price": [100.0]})
    p = RS.build_restock_plan(stock, fc4, wh, tc, price, cover_weeks=4, safety=1.0)
    assert (p.order_qty == 90).all()                                          # target 100 - stock 10
    first = p.iloc[0]
    assert first.branch_id == "B1" and first.source_warehouse == "W1"         # higher priority, cheap WH has enough stock
    second = p.iloc[1]
    assert second.source_warehouse == "W2"                                    # W1 only has 10 left -> falls to W2
    assert (p.action == "URGENT").all()                                       # 0.5 weeks of cover


def test_restock_no_demand_no_order():
    stock = pd.DataFrame({"branch_id": ["B1"], "category": ["Socks"], "closing_units": [50]})
    fc4 = pd.DataFrame({"branch_id": ["B1"], "category": ["Socks"], "mean_4wk": [0.0], "p90_4wk": [0.0], "weekly_mean": [0.0]})
    p = RS.build_restock_plan(stock, fc4, pd.DataFrame({"warehouse_id": ["W"], "category": ["Socks"], "closing_units": [9]}),
                              pd.DataFrame({"warehouse_id": ["W"], "branch_id": ["B1"], "cost": [1]}), pd.DataFrame({"category": ["Socks"], "avg_price": [1.0]}))
    assert p.order_qty.iloc[0] == 0 and p.action.iloc[0] == "OVERSTOCK"


# --------------------------------------------------------------------------- cross-sell
def test_crosssell_learns_pairs_and_respects_masks():
    orders = np.repeat(np.arange(300), 2)
    prods = np.tile([0, 1], 300)                                                # A and B always together
    extra = np.concatenate([np.repeat(np.arange(300, 400), 1)]); extra_p = np.full(100, 2)   # C always alone
    B = XS.basket_matrix(np.concatenate([orders, extra]), np.concatenate([prods, extra_p]), 3)
    m = XS.fit(B, k_shrink=1)
    bundle = {**m, "product_ids": ["A", "B", "C"], "margin_per_unit": np.array([10, 10, 99], dtype=np.float32)}
    r = XS.recommend(bundle, ["A"], k=3)
    assert r.product_id.tolist() == ["B"]                                       # only B ever co-occurs; basket item and C excluded
    assert XS.recommend(bundle, ["A"], k=3, available={"C"}).empty               # stock mask removes B


# --------------------------------------------------------------------------- customers
def test_churn_labels_and_features():
    o = pd.DataFrame({"customer_id": ["a", "a", "b"], "order_date": pd.to_datetime(["2020-01-01", "2020-03-01", "2020-01-10"]),
                      "net": [100.0, 50.0, 70.0], "items": [1, 2, 1], "channel": ["Retail POS"] * 3})
    snap = pd.Timestamp("2020-02-01")
    g = C.snapshot_features(o, snap)
    assert g.loc["a", "freq_total"] == 1 and g.loc["a", "recency_days"] == 31 and g.loc["b", "monetary_total"] == 70
    y = C.churn_labels(o, g.index, snap, days=120)
    assert y["a"] == 0 and y["b"] == 1                                           # a buys again on 03-01, b never does


def test_wilson_interval_widens_for_small_n():
    lo1, hi1 = SP.wilson(5, 10)
    lo2, hi2 = SP.wilson(500, 1000)
    assert (hi1 - lo1) > (hi2 - lo2) * 5


# --------------------------------------------------------------------------- marketing
def test_marketing_optimizer_improves_and_respects_constraints():
    p = {"channels": ["a", "b"], "beta": {"a": .9, "b": .5}, "c0": {"a": 1.0, "b": 1.0}, "n": {"a": 10, "b": 10}, "spend_current": {"a": 1e6, "b": 1e6}}
    r = MK.optimize(p, lo=.5, hi=1.5)
    assert r["orders"] >= r["orders_current"]
    assert abs(sum(r["alloc"].values()) - 2e6) < 1                               # same total budget
    assert 0.5e6 - 1 <= r["alloc"]["b"] <= 1.5e6 + 1 and r["alloc"]["a"] > r["alloc"]["b"]   # moves money to the less saturated channel


# --------------------------------------------------------------------------- shipped artifacts + services
def test_services_smoke_on_shipped_artifacts():
    from cottonil import services as V
    ok, missing = V.artifacts_ready()
    assert ok, missing
    b0 = V.branch_choices()[0][1]
    pc = V.product_choices()
    V.overview_view("All"); V.forecast_view(b0, "ALL"); V.restock_view("ALL", 4, 1.0, ["URGENT"])
    till, cards, fig = V.crosssell_view(b0, [pc[40][1]], 3, "Margin-aware", True)
    assert "AI SUGGESTED" in till
    cid = V.random_customer("valuable at-risk")
    html, _ = V.customer_lookup(cid)
    assert "Segment" in html or "SEGMENT" in html.upper()
    ro = V.return_options(); cat = "Men boxers"
    V.return_view(cat, ro["sizes_by_cat"][cat][0], "E-commerce", "Cash", b0, "25-34", "Retail", 1, 200, 0, 1, True, 5)
    sm = V.supplier_options(); V.supplier_view(sm["supplier_id"][0], sm["receiving_warehouse_id"][0], "EGP", sm["po_purpose"][0], 5e5, 60, 2, 3000, 60, 6)
    n = len(V.marketing_channels()); V.marketing_scenario([1.0] * n)
    assert all(isinstance(x, float) for x in V.marketing_optimum(.5, 1.5))
