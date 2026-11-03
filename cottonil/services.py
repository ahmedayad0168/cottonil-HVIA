"""All app logic lives here (no Gradio imports) so every function can be unit-tested.

Each *_view function returns plain Python/pandas/plotly objects (+ HTML strings).
"""
from __future__ import annotations

import html
import json
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from . import config as cfg
from . import crosssell as XS
from . import marketing as MK
from . import restock as RS
from . import returns as RT
from . import suppliers as SP
from .visuals import (NAVY, TEAL, ORANGE, RED, badge, banner, fmt_money, kpi_cards, product_image_html, rtl)

PALETTE = [NAVY, TEAL, ORANGE, RED, "#7C6A9C", "#5B8E7D", "#9AA5B1"]
ACTION_COLORS = {"URGENT": RED, "RESTOCK": ORANGE, "OK": TEAL, "OVERSTOCK": "#7C6A9C"}


# ----------------------------------------------------------------------------
# artifact store (lazy + cached)
# ----------------------------------------------------------------------------
class Store:
    def __init__(self):
        self._serving, self._models, self._metrics = {}, {}, {}

    def serving(self, name: str) -> pd.DataFrame:
        if name not in self._serving:
            self._serving[name] = cfg.load_serving(name)
        return self._serving[name]

    def model(self, name: str):
        if name not in self._models:
            self._models[name] = cfg.load_model(name)
        return self._models[name]

    def metrics(self, name: str) -> dict:
        if name not in self._metrics:
            self._metrics[name] = cfg.load_metrics(name)
        return self._metrics[name]

    def reload(self):
        self.__init__()
        _plan_cache.cache_clear()


S = Store()
REQUIRED = ["00_data_audit", "01_demand_forecast", "02_stockout_risk", "03_churn", "04_crosssell", "05_return_risk",
            "06_supplier_risk", "07_marketing"]


def artifacts_ready() -> tuple[bool, list[str]]:
    missing = [m for m in REQUIRED if not (cfg.METRICS_DIR / f"{m}.json").exists()]
    return (not missing), missing


def data_source_banner() -> str:
    srcs = {S.metrics(m).get("data_source") for m in REQUIRED if S.metrics(m)}
    if "synthetic" in srcs:
        return banner("<b>Demo mode - SYNTHETIC data.</b> These models were trained on a generated dataset that mimics the Cottonil schema, "
                      "so every number, chart and score here is for demonstration only. Put the real CSVs in <code>data/COTTONIL_Data</code>, run "
                      "<code>python scripts/run_all_notebooks.py</code>, and this banner disappears.", "warn")
    return banner("Models trained on <b>real Cottonil data</b>.", "ok")


def _style(fig: go.Figure, h: int = 380, legend: bool = True) -> go.Figure:
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=48, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, Segoe UI, Arial", size=13, color="#1f2d3a"), title_font=dict(size=15, color=NAVY),
                      showlegend=legend, legend=dict(orientation="h", y=-0.18), colorway=PALETTE, hoverlabel=dict(font_size=12))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(0,0,0,.07)", zeroline=False)
    return fig


def _empty(msg: str, h: int = 300) -> go.Figure:
    f = go.Figure()
    f.add_annotation(text=msg, showarrow=False, font=dict(size=15, color="#6b7a8a"))
    f.update_xaxes(visible=False)
    f.update_yaxes(visible=False)
    return _style(f, h, False)


def branch_choices() -> list[tuple[str, str]]:
    d = S.serving("dim_branches")
    return [(f"{r.branch_id} · {r.branch_name_ar} ({r.governorate_ar})", r.branch_id) for r in d.itertuples()]


def category_choices() -> list[str]:
    return sorted(S.serving("dim_products").category_en.unique())


def category_choices_all() -> list[tuple[str, str]]:
    return [("All categories", "ALL")] + [(c, c) for c in category_choices()]


# ----------------------------------------------------------------------------
# 1) overview
# ----------------------------------------------------------------------------
def overview_years() -> list[str]:
    return ["All"] + sorted(S.serving("kpi_monthly").ym.str[:4].unique())


def overview_view(year: str = "All"):
    km, kc = S.serving("kpi_monthly"), S.serving("kpi_category_month")
    kb, kr = S.serving("kpi_branch"), S.serving("kpi_returns")
    k0 = S.metrics("00_data_audit").get("kpis", {})
    if year != "All":
        km, kc = km[km.ym.str[:4] == year], kc[kc.ym.str[:4] == year]
    rev, orders = km.revenue.sum(), km.orders.sum()
    cards = kpi_cards([
        ("Net revenue", fmt_money(rev), f"{'all years' if year == 'All' else year}"),
        ("Orders", f"{orders:,.0f}", f"AOV {fmt_money(rev / max(orders, 1))}"),
        ("Gross margin", f"{kc.margin.sum() / max(kc.revenue.sum(), 1):.1%}", f"{fmt_money(kc.margin.sum())}"),
        ("Units sold", f"{kc.units.sum():,.0f}", f"{kc.units.sum() / max(orders, 1):.1f} per order"),
        ("Anonymous orders", f"{k0.get('anonymous_share', np.nan):.1%}", "no customer ID (all years)"),
        ("Return rate", f"{k0.get('return_rate_orders', np.nan):.1%}", f"refunds {fmt_money(k0.get('refunds_egp', 0))}"),
    ])
    f1 = px.bar(km, x="ym", y="revenue", color="sales_channel", title="Monthly net revenue by channel", labels={"ym": "", "revenue": "EGP", "sales_channel": ""},
                color_discrete_sequence=PALETTE)
    f1.update_xaxes(type="category", nticks=14)
    ch = km.groupby("sales_channel").revenue.sum().reset_index()
    f2 = px.pie(ch, names="sales_channel", values="revenue", hole=.55, title="Revenue mix by channel", color_discrete_sequence=PALETTE)
    f2.update_traces(textinfo="percent+label")
    cat = kc.groupby("category").agg(revenue=("revenue", "sum"), margin=("margin", "sum"), units=("units", "sum")).reset_index()
    cat["margin_pct"] = cat.margin / cat.revenue.replace(0, np.nan)
    cat = cat.sort_values("revenue").tail(12)
    f3 = px.bar(cat, x="revenue", y="category", orientation="h", color="margin_pct", color_continuous_scale=["#F4A259", "#1D8A99"],
                title="Revenue by category (colour = gross margin %)", labels={"margin_pct": "margin", "category": ""}, hover_data={"units": ":,.0f"})
    f3.update_coloraxes(colorbar=dict(tickformat=".0%", len=.6))
    top = kb.sort_values("revenue_12m", ascending=False).head(12).sort_values("revenue_12m")
    top["label"] = top.branch_id + " · " + top.branch_name_ar
    f4 = px.bar(top, x="revenue_12m", y="label", orientation="h", title="Top branches - last 12 months", color_discrete_sequence=[NAVY], labels={"revenue_12m": "EGP", "label": ""})
    rr = kr.groupby("reason").agg(returns=("returns", "sum"), refund=("refund", "sum")).reset_index()
    f5 = px.pie(rr, names="reason", values="returns", hole=.55, title="Why customers return (all years)", color_discrete_sequence=PALETTE)
    return cards, _style(f1, 390), _style(f2, 390, False), _style(f3, 420, False), _style(f4, 420, False), _style(f5, 390)


# ----------------------------------------------------------------------------
# 2) demand forecast
# ----------------------------------------------------------------------------
def forecast_view(branch: str, category: str, weeks_back: int = 52):
    hist, fl = S.serving("weekly_history"), S.serving("forecast_latest")
    hist["week_start"] = pd.to_datetime(hist["week_start"])
    fl["week_start"] = pd.to_datetime(fl["week_start"])
    hb = hist[hist.branch_id == ("ALL" if branch == "ALL" else branch)]
    fb = fl if branch == "ALL" else fl[fl.branch_id == branch]
    if category != "ALL":
        hb, fb = hb[hb.category == category], fb[fb.category == category]
    h = hb.groupby("week_start").units.sum().sort_index().tail(weeks_back)
    fb = fb.assign(up=(fb.p90 - fb["mean"]).clip(lower=0) ** 2, dn=(fb["mean"] - fb.p10).clip(lower=0) ** 2)
    f = fb.groupby("week_start").agg(mean=("mean", "sum"), up=("up", "sum"), dn=("dn", "sum")).sort_index()
    f["p90"], f["p10"] = f["mean"] + np.sqrt(f.up), (f["mean"] - np.sqrt(f.dn)).clip(lower=0)
    if f.empty or h.empty:
        return kpi_cards([("No data", "-")]), _empty("No forecast for this selection"), pd.DataFrame()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h.index, y=h.values, name="Actual units / week", mode="lines", line=dict(color=NAVY, width=2.4)))
    xs = [h.index[-1]] + list(f.index)
    fig.add_trace(go.Scatter(x=xs + xs[::-1], y=[h.values[-1]] + list(f.p90) + ([h.values[-1]] + list(f.p10))[::-1], fill="toself", mode="none",
                             fillcolor="rgba(244,162,89,.25)", name="P10-P90 range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[h.values[-1]] + list(f["mean"]), name="Forecast", mode="lines+markers", line=dict(color=ORANGE, width=3, dash="dot"), marker=dict(size=8)))
    names = {v: k for k, v in branch_choices()}
    title_b = "Whole network" if branch == "ALL" else names.get(branch, branch)
    fig.update_layout(title=f"Weekly units - {title_b} · {'all categories' if category == 'ALL' else category}", yaxis_title="units / week")
    last4, next4 = float(h.tail(4).sum()), float(f["mean"].sum())
    p90_4 = float(f["mean"].sum() + np.sqrt((f.up).sum()))
    cards = kpi_cards([("Next 4 weeks (expected)", f"{next4:,.0f}", "units"), ("Planning level (P90)", f"{p90_4:,.0f}", "units - plan stock to this if stock-outs are costly"),
                       ("Last 4 weeks (actual)", f"{last4:,.0f}", "units"), ("Change vs last 4 weeks", f"{(next4 / max(last4, 1) - 1):+.1%}", "seasonality + trend")])
    tbl = f.reset_index()[["week_start", "p10", "mean", "p90"]].rename(columns={"week_start": "Week starting", "p10": "Low (P10)", "mean": "Expected", "p90": "High (P90)"})
    tbl["Week starting"] = tbl["Week starting"].dt.strftime("%Y-%m-%d")
    return cards, _style(fig, 430), tbl.round(0)


def forecast_accuracy_html() -> str:
    m = S.metrics("01_demand_forecast")
    if not m:
        return ""
    w, b = m["test_wape_by_horizon"], m["best_baseline_wape_by_horizon"]
    agg = m.get("aggregated_wape", {})
    net = agg.get("1", {}).get("network x week")
    return kpi_cards([("Error, branch x category x week", f"{w['1']:.0%}", f"best naive baseline {b['1']:.0%} (WAPE, test)"),
                      ("Error, category x week", f"{agg.get('1', {}).get('category x week', float('nan')):.0%}", "aggregated"),
                      ("Error, whole network", f"{net:.0%}" if net is not None else "-", "aggregated"),
                      ("P10-P90 coverage", f"{m['interval_coverage'].get('1', float('nan')):.0%}", "target 80%")])


# ----------------------------------------------------------------------------
# 3) stock & restock
# ----------------------------------------------------------------------------
@lru_cache(maxsize=64)
def _plan_cache(cover_weeks: float, safety: float) -> pd.DataFrame:
    fc4 = RS.aggregate_forecast(S.serving("forecast_latest"))
    return RS.build_restock_plan(S.serving("stock_latest"), fc4, S.serving("warehouse_stock_latest"), S.serving("transfer_costs"),
                                 S.serving("category_price"), cover_weeks=cover_weeks, safety=safety)


def restock_view(branch: str, cover_weeks: float, safety: float, actions: list[str]):
    plan = _plan_cache(float(cover_weeks), float(safety))
    br = S.serving("dim_branches").set_index("branch_id")
    plan = plan.assign(branch_name=plan.branch_id.map(br.branch_name_ar), governorate=plan.branch_id.map(br.governorate_ar))
    scope = plan if branch == "ALL" else plan[plan.branch_id == branch]
    if actions:
        scope = scope[scope.action.isin(actions)]
    allp = plan if branch == "ALL" else plan[plan.branch_id == branch]
    cards = kpi_cards([
        ("Urgent pairs", f"{(allp.action == 'URGENT').sum():,}", "< 1.5 weeks of cover"),
        ("To restock", f"{(allp.order_qty > 0).sum():,}", f"{allp.ship_qty.sum():,.0f} units to move"),
        ("Transfer cost", fmt_money(allp.est_transfer_cost_egp.sum()), "median historical lane cost"),
        ("Demand at risk", fmt_money(allp.shortfall_value_egp.sum()), "if nothing is done (4 wk)"),
        ("Early warnings", f"{int((allp.p_high_risk_next >= S.model('stockout_risk')['threshold']).sum()):,}", "ML: high risk next month"),
    ])
    sc = scope.assign(cover=scope.weeks_of_cover.clip(upper=16), size=(scope.shortfall_value_egp + 1).clip(upper=scope.shortfall_value_egp.quantile(.99) + 1))
    if sc.empty:
        f1 = _empty("Nothing matches these filters")
    else:
        f1 = px.scatter(sc, x="cover", y="p_high_risk_next", size="size", color="action", color_discrete_map=ACTION_COLORS, size_max=28,
                        hover_data={"branch_id": True, "category": True, "closing_units": True, "order_qty": True, "cover": False, "size": False},
                        labels={"cover": "weeks of cover (capped at 16)", "p_high_risk_next": "P(high stock-out risk next month)"},
                        title="Where the risk is: low cover + high ML risk = act first")
        f1.add_vline(x=1.5, line_dash="dash", line_color=RED, opacity=.5)
        f1.update_yaxes(tickformat=".0%")
    cat = scope.groupby("category").order_qty.sum().sort_values().tail(12).reset_index()
    f2 = px.bar(cat, x="order_qty", y="category", orientation="h", color_discrete_sequence=[TEAL], title="Units to restock by category", labels={"order_qty": "units", "category": ""}) if len(cat) else _empty("-")
    t = scope.sort_values("priority_rank").head(300)[["priority_rank", "branch_id", "branch_name", "category", "closing_units", "weekly_mean", "weeks_of_cover",
                                                    "p_high_risk_next", "order_qty", "source_warehouse", "ship_qty", "est_transfer_cost_egp", "action"]].copy()
    t["p_high_risk_next"] = (t.p_high_risk_next * 100).round(0)
    t.columns = ["#", "Branch", "Name", "Category", "Stock", "Forecast/wk", "Weeks cover", "Risk next mo %", "Order qty", "From WH", "Ship qty", "Cost EGP", "Action"]
    return cards, _style(f1, 430), _style(f2, 430, False), t.round(1), scope


def stock_ml_html() -> str:
    m = S.metrics("02_stockout_risk")
    if not m:
        return ""
    a, b = m["model"], m["persistence_baseline"]
    return banner(f"<b>Early-warning model</b> (test on the last 10 months): PR-AUC <b>{a['PR-AUC']:.2f}</b> vs {b['PR-AUC']:.2f} for the 'same as this month' baseline; "
                  f"ROC-AUC {a['ROC-AUC']:.2f} vs {b['ROC-AUC']:.2f}. Base rate of high-risk pairs: {m['test_base_rate']:.1%}.", "info")


# ----------------------------------------------------------------------------
# 4) cross-sell
# ----------------------------------------------------------------------------
def product_choices() -> list[tuple[str, str]]:
    d = S.serving("dim_products").sort_values(["category_en", "product_id"])
    return [(f"{r.category_en} · {r.product_name_ar} · EGP {r.avg_price_12m:,.0f}", r.product_id) for r in d.itertuples()]


def crosssell_view(branch: str, basket: list[str], k: int, mode: str, use_stock: bool, min_stock: int = 10):
    if not basket:
        return banner("Add one or more items to the basket to see suggestions.", "info"), "", _empty("No basket yet", 280)
    bundle = S.model("crosssell")
    dp = S.serving("dim_products").set_index("product_id")
    stock = S.serving("stock_latest")
    st = stock[stock.branch_id == branch].set_index("category").closing_units.to_dict() if branch != "ALL" else {}
    avail = None
    if use_stock and st:
        avail = {p for p, c in zip(bundle["product_ids"], bundle["category"]) if st.get(c, 0) >= min_stock}
    rec = XS.recommend(bundle, basket, k=int(k), available=avail, by="margin" if str(mode).startswith("Margin") else "prob")
    total = sum(dp.loc[p, "avg_price_12m"] for p in basket if p in dp.index)
    lines = "".join(f'<div>  1 x {rtl(dp.loc[p, "product_name_ar"])} <span class="dim">[{dp.loc[p, "category_en"]}]</span> ........ <span class="hl">EGP {dp.loc[p, "avg_price_12m"]:,.0f}</span></div>' for p in basket if p in dp.index)
    sug = ""
    if len(rec):
        r0 = rec.iloc[0]
        sug = (f'<div class="dim">{"-" * 54}</div><div>AI SUGGESTED ADD-ON <span class="hl">(bought together in {r0.prob:.0%} of similar baskets)</span></div>'
               f'<div>-&gt; "Add {rtl(dp.loc[r0.product_id, "product_name_ar"])} for EGP {dp.loc[r0.product_id, "avg_price_12m"]:,.0f}"</div>'
               f'<div class="dim">expected margin EGP {r0.expected_margin:,.1f} · [ONE-CLICK ACCEPT]  [SKIP]</div>')
    till = f'<div class="till"><div class="dim">[COTTONIL POS · BRANCH {branch}]</div><div>Current order:</div>{lines}<div>BASKET TOTAL ........ <span class="hl">EGP {total:,.0f}</span></div>{sug}</div>'
    if rec.empty:
        return till, banner("No in-stock suggestion available for this basket.", "warn"), _empty("No suggestions", 280)
    cards = []
    for r in rec.itertuples():
        cat = dp.loc[r.product_id, "category_en"]
        stock_txt = f"{cat} stock at branch: {st[cat]:,.0f}" if cat in st else "stock: n/a"
        cards.append(f'<div class="rec"><div>{product_image_html(cat, 84)}</div><div><div class="t">{rtl(dp.loc[r.product_id, "product_name_ar"])}</div>'
                     f'<div class="m">{cat} · EGP {dp.loc[r.product_id, "avg_price_12m"]:,.0f}</div>'
                     f'<div class="p">{badge(f"bought together {r.prob:.0%}", "ok")}{badge(f"+EGP {r.margin_per_unit:,.0f} margin", "info")}</div><div class="m">{stock_txt}</div></div></div>')
    cards_html = f'<div class="rec-grid">{"".join(cards)}</div>'
    fig = px.bar(rec.assign(name=[dp.loc[p, "category_en"] + " · " + p for p in rec.product_id]).iloc[::-1], x="expected_margin", y="name", orientation="h",
                 title="Expected margin per suggestion (probability x margin)", labels={"expected_margin": "EGP", "name": ""}, color_discrete_sequence=[TEAL])
    return till, cards_html, _style(fig, 300, False)


def category_lift_fig() -> go.Figure:
    cl = S.serving("category_lift")
    piv = cl.pivot(index="seed_category", columns="add_on_category", values="lift")
    f = px.imshow(piv, color_continuous_scale=["#ffffff", "#F4A259", "#0B3C5D"], aspect="auto", title="Category lift: how much more often B is bought when A is in the basket (1 = no relation)",
                  labels=dict(color="lift", x="add-on", y="in basket"))
    return _style(f, 520, False)


def crosssell_metrics_html() -> str:
    m = S.metrics("04_crosssell")
    if not m:
        return ""
    t = m["test"]
    return banner(f"<b>Offline test</b> ({m['n_test_baskets']:,} held-out baskets): top-3 hit-rate <b>{t['co-occurrence (by probability)']['hit@3']:.0%}</b> vs {t['popularity baseline']['hit@3']:.0%} for 'most popular items'. "
                  f"Margin-aware ranking: {t['co-occurrence (by expected margin)']['hit@3']:.0%} hit-rate but avg margin/unit EGP {m['avg_margin_top3']['margin']:.0f} vs {m['avg_margin_top3']['probability']:.0f}. "
                  f"Validate with an A/B test on real tills before trusting uplift.", "info")


# ----------------------------------------------------------------------------
# 5) customers
# ----------------------------------------------------------------------------
def _gauge(value: float, title: str, steps=((0, .35, "#cfeedd"), (.35, .65, "#fde3b8"), (.65, 1, "#f6b8b0")), max_pct: float = 100.0) -> go.Figure:
    """value is a probability (0-1); `steps` are fractions of the axis range [0, max_pct]."""
    f = go.Figure(go.Indicator(mode="gauge+number", value=value * 100, number=dict(suffix="%", valueformat=".1f", font=dict(size=38, color=NAVY)), title=dict(text=title, font=dict(size=14)),
                               gauge=dict(axis=dict(range=[0, max_pct]), bar=dict(color=NAVY), steps=[dict(range=[a * max_pct, b * max_pct], color=c) for a, b, c in steps])))
    return _style(f, 260, False)


def customer_lookup(cid: str):
    df = S.serving("customer_scores")
    cid = (cid or "").strip().upper()
    r = df[df.customer_id == cid]
    if r.empty:
        return banner(f"Customer <b>{html.escape(cid)}</b> not found among scored customers (only customers with at least one identified order are scored).", "warn"), _empty("-", 260)
    r = r.iloc[0]
    kind = "bad" if r.p_churn >= .65 else "warn" if r.p_churn >= .35 else "ok"
    lapsed = r.segment.startswith("Lapsed")
    cards = kpi_cards([("Segment", r.segment, f"{r.customer_segment} · {r.age_band}"), ("Churn risk", "lapsed" if lapsed else f"{r.p_churn:.0%}", f"no order in next {S.model('churn_model')['churn_days']} days"),
                       ("Value at risk", fmt_money(r.value_at_risk_egp), "P(churn) x 12-month spend"), ("Last order", f"{int(r.recency_days)} days ago", str(r.last_order_date)),
                       ("Orders (12m / total)", f"{int(r.freq_365)} / {int(r.freq_total)}", f"AOV {fmt_money(r.aov)}"), ("Favourite category", r.fav_category, r.governorate_ar)])
    action = banner(f"{badge('Suggested action', kind)} {html.escape(r.recommended_action)}", kind)
    return cards + action, _gauge(1.0 if lapsed else r.p_churn, "Probability of churn")


def random_customer(kind: str) -> str:
    df = S.serving("customer_scores")
    act = df[~df.segment.str.startswith("Lapsed")]
    if kind == "valuable at-risk":
        c = act[act.p_churn > .55].sort_values("monetary_365", ascending=False).head(50)
    elif kind == "loyal champion":
        c = act[act.segment == "Champions"].head(200)
    else:
        c = act
    return str(c.sample(1).customer_id.iloc[0]) if len(c) else ""


def segments_view():
    df = S.serving("customer_scores")
    act = df[~df.segment.str.startswith("Lapsed")]
    g = act.groupby("segment").agg(customers=("customer_id", "size"), churn=("p_churn", "mean"), spend=("monetary_365", "sum"), var=("value_at_risk_egp", "sum")).reset_index()
    f1 = px.bar(g.sort_values("var"), x="var", y="segment", orientation="h", color="churn", color_continuous_scale=["#1D8A99", "#F4A259", "#E4572E"],
                title="Value at risk by RFM segment (colour = average churn probability)", labels={"var": "EGP at risk", "segment": "", "churn": "churn"})
    f1.update_coloraxes(colorbar=dict(tickformat=".0%", len=.6))
    s = act.sample(min(4000, len(act)), random_state=1)
    f2 = px.scatter(s, x="recency_days", y="p_churn", color="segment", size=np.log1p(s.monetary_365).clip(lower=1), size_max=12, opacity=.6, color_discrete_sequence=PALETTE + ["#999"],
                    title="Recency vs churn probability (bubble = spend)", labels={"recency_days": "days since last order", "p_churn": "P(churn)"})
    f2.update_yaxes(tickformat=".0%")
    ab = S.serving("anonymous_by_branch").sort_values("anon_revenue", ascending=False).head(12).iloc[::-1]
    ab["label"] = ab.branch_id + " · " + ab.branch_name_ar
    f3 = px.bar(ab, x="anon_revenue", y="label", orientation="h", color="anon_share", color_continuous_scale=["#cfe8ec", "#0B3C5D"],
                title="Where identity is lost: anonymous revenue by branch (colour = share of orders)", labels={"anon_revenue": "EGP anonymous", "label": "", "anon_share": "share"})
    f3.update_coloraxes(colorbar=dict(tickformat=".0%", len=.6))
    return _style(f1, 400, False), _style(f2, 400), _style(f3, 400, False)


def call_list(segment: str, min_p: float, n: int) -> pd.DataFrame:
    df = S.serving("customer_scores")
    d = df[~df.segment.str.startswith("Lapsed")]
    if segment != "All":
        d = d[d.segment == segment]
    d = d[d.p_churn >= min_p].sort_values("value_at_risk_egp", ascending=False).head(int(n))
    out = d[["customer_id", "segment", "p_churn", "value_at_risk_egp", "recency_days", "freq_365", "monetary_365", "fav_category", "governorate_ar", "recommended_action"]].copy()
    out["p_churn"] = (out.p_churn * 100).round(0)
    out.columns = ["Customer", "Segment", "Churn %", "Value at risk EGP", "Days since order", "Orders 12m", "Spend 12m EGP", "Favourite", "Governorate", "Suggested action"]
    return out.round(0)


def segment_choices() -> list[str]:
    return ["All"] + sorted(S.serving("customer_scores").segment.unique().tolist())


def churn_metrics_html() -> str:
    m = S.metrics("03_churn")
    if not m:
        return ""
    a, b = m["model"], m["baseline_logit"]
    return banner(f"<b>Churn model</b> (test on the latest snapshots): ROC-AUC <b>{a['ROC-AUC']:.2f}</b> (simple recency/frequency/spend baseline: {b['ROC-AUC']:.2f}); "
                  f"top-decile lift {a['Top-decile lift']:.1f}x. Active customers scored: {m['n_active']:,}.", "info")


# ----------------------------------------------------------------------------
# 6) return risk
# ----------------------------------------------------------------------------
def return_options() -> dict:
    b = S.model("return_risk")
    rr = S.serving("return_rates")
    order = ["Free", "S", "M", "L", "XL", "2XL", "3XL", "4Y", "6Y", "8Y", "10Y", "12Y"]
    key = lambda x: order.index(x) if x in order else 99
    return {"maps": b["maps"], "sizes_by_cat": rr.groupby("category")["size"].apply(lambda s: sorted(s.astype(str).unique(), key=key)).to_dict()}


def return_view(category, size, channel, payment, branch, age, segment, qty, price, disc_pct, items, identified, month):
    b = S.model("return_risk")
    gross = float(qty) * float(price)
    disc = gross * float(disc_pct) / 100
    row = pd.DataFrame([{"category": category, "size": size, "sales_channel": channel, "payment_method": payment, "branch_id": branch, "age_band": age,
                         "customer_segment": segment, "quantity": qty, "unit_price_EGP": price, "discount_line_EGP": disc, "gross_line_EGP": gross,
                         "net_line_EGP": gross - disc, "items_count": items, "customer_id": "C1" if identified else None,
                         "order_date": pd.Timestamp(2026, int(month), 15)}])
    p = float(b["model"].predict_proba(RT.encode(row, b["maps"]))[0, 1])
    base = b["base_rate"]
    rb = S.serving("return_baseline_rates")
    hit = rb[(rb.sales_channel == channel) & (rb.category == category)]
    p_simple = float(hit.p.iloc[0]) if len(hit) else base
    top = p >= b["top5_threshold"]
    kind = "bad" if top else "warn" if p > base * 1.5 else "ok"
    cards = kpi_cards([("Return probability", f"{p:.1%}", f"average line: {base:.1%}"), ("Relative risk", f"{p / base:.1f}x", "vs average line"),
                       (f"Simple rate ({channel[:10]} x {category[:14]})", f"{p_simple:.1%}", "historical average"), ("Refund at stake", fmt_money(p * (gross - disc)), "probability x line value")])
    msg = {"bad": "Top-5% risk line: confirm size by phone/WhatsApp before dispatch, or add an extra QA check.", "warn": "Above-average risk: consider a size-guide reminder.",
           "ok": "Typical risk - no special action."}[kind]
    verdict = banner(f"<b>Model verdict from training:</b> {html.escape(b['verdict'])}", "info")
    return cards + banner(msg, kind) + verdict, _gauge(min(p, .30), "Probability this line is returned", ((0, .25, "#cfeedd"), (.25, .6, "#fde3b8"), (.6, 1, "#f6b8b0")), max_pct=30)


def return_heatmap() -> tuple[go.Figure, go.Figure]:
    rr = S.serving("return_rates")
    rr["size_issue_rate"] = rr.get("Size issue", 0) / rr.lines
    ap = rr[rr.category.isin(["Men boxers", "Men tops", "Men underwear", "Women tops", "Women bras", "Women pajamas", "Girls pajamas", "Boys pajamas"])]
    piv = ap.pivot(index="category", columns="size", values="size_issue_rate")
    order = ["Free", "S", "M", "L", "XL", "2XL", "3XL", "4Y", "6Y", "8Y", "10Y", "12Y"]
    piv = piv[[c for c in order if c in piv.columns] + [c for c in piv.columns if c not in order]]
    f1 = px.imshow(piv, color_continuous_scale="Oranges", aspect="auto", title="'Size issue' returns per sold line, by category and size", labels=dict(color="rate"), text_auto=".1%")
    c = rr.groupby("category").agg(lines=("lines", "sum"), returns=("returns", "sum")).assign(rate=lambda d: d.returns / d.lines).sort_values("rate").tail(12).reset_index()
    f2 = px.bar(c, x="rate", y="category", orientation="h", title="Return rate by category", color_discrete_sequence=[RED], labels={"rate": "returned lines / sold lines", "category": ""})
    f2.update_xaxes(tickformat=".1%")
    return _style(f1, 400, False), _style(f2, 400, False)


# ----------------------------------------------------------------------------
# 7) supplier / PO risk
# ----------------------------------------------------------------------------
def supplier_options() -> dict:
    b = S.model("supplier_risk")
    return b["maps"]


def supplier_view(supplier, warehouse, currency, purpose, subtotal, terms, n_lines, avg_qty, avg_cost, month):
    b = S.model("supplier_risk")
    sc = S.serving("supplier_scorecard").set_index("supplier_id")
    sup_tab = S.serving("supplier_scorecard")
    mat = sc.loc[supplier, "material_type"]
    maps = b["maps"]
    row = pd.DataFrame([{"supplier_id": supplier, "material_type": mat, "supplier_city": sc.loc[supplier, "supplier_city"], "receiving_warehouse_id": warehouse,
                         "currency": currency, "po_purpose": purpose, "po_subtotal_EGP": subtotal, "payment_terms_days": terms, "n_lines": n_lines,
                         "avg_qty": avg_qty, "avg_unit_cost": avg_cost, "po_date": pd.Timestamp(2026, int(month), 15)}])
    X = SP.encode(row, maps)
    p = float(b["clf"].predict_proba(X)[0, 1])
    lead = float(b["reg"].predict(X)[0])
    typ = b["typical_lead_days"].get(supplier, lead)
    risky = p >= b["threshold"]
    kind = "bad" if risky else "warn" if p > b["overall_partial_rate"] else "ok"
    cards = kpi_cards([("P(partial receipt)", f"{p:.0%}", f"company average {b['overall_partial_rate']:.0%}"), ("Expected lead time", f"{lead:.0f} days", f"supplier typical {typ:.0f} days"),
                       ("Suggested buffer", f"+{max(lead - typ, 0):.0f} days", "predicted minus typical lead time"), ("Supplier history", f"{sc.loc[supplier, 'partial_rate']:.0%}", f"{int(sc.loc[supplier, 'pos'])} POs · {mat}")])
    msg = (f"P(partial) &ge; {b['threshold']:.0%}: <b>split this PO</b> with a backup supplier of the same material and raise the safety buffer." if risky
           else "Within normal range - proceed, monitor delivery.")
    alt = sup_tab[(sup_tab.material_type == mat) & (sup_tab.supplier_id != supplier) & (sup_tab.pos >= 30)
                   & (sup_tab.partial_rate <= b["overall_partial_rate"])].sort_values("ci_high").head(4)
    alt_df = alt[["supplier_id", "pos", "partial_rate", "ci_low", "ci_high", "median_lead_days"]].copy()
    alt_df.columns = ["Backup supplier", "POs", "Partial rate", "CI low", "CI high", "Median lead days"]
    alt_df[["Partial rate", "CI low", "CI high"]] = (alt_df[["Partial rate", "CI low", "CI high"]] * 100).round(0)
    return cards + banner(msg, kind), _gauge(p, "Probability of partial receipt"), alt_df


def supplier_scorecard_fig() -> go.Figure:
    sc = S.serving("supplier_scorecard").sort_values("partial_rate").reset_index(drop=True)
    f = go.Figure(go.Scatter(x=sc.supplier_id, y=sc.partial_rate, mode="markers", marker=dict(color=NAVY, size=8), customdata=np.c_[sc.pos, sc.material_type],
                             error_y=dict(type="data", symmetric=False, array=sc.ci_high - sc.partial_rate, arrayminus=sc.partial_rate - sc.ci_low, color="#9bb7c9"),
                             hovertemplate="%{x}<br>%{customdata[1]}<br>partial rate %{y:.0%}<br>%{customdata[0]} POs<extra></extra>"))
    f.add_hline(y=float((sc.partial.sum()) / sc.pos.sum()), line_dash="dash", line_color=RED, annotation_text="overall")
    f.update_layout(title="Supplier partial-receipt rate with 95% confidence interval", yaxis_tickformat=".0%")
    f.update_xaxes(showticklabels=False)
    return _style(f, 380, False)


# ----------------------------------------------------------------------------
# 8) marketing
# ----------------------------------------------------------------------------
def marketing_channels() -> list[str]:
    return list(S.model("marketing_response")["channels"])


def marketing_scenario(mults: list[float]):
    p = S.model("marketing_response")
    ch = p["channels"]
    cur = {c: p["spend_current"][c] for c in ch}
    new = {c: cur[c] * m for c, m in zip(ch, mults)}
    o_cur = {c: MK.channel_orders(p, c, cur[c]) for c in ch}
    o_new = {c: MK.channel_orders(p, c, new[c]) for c in ch}
    tot_c, tot_n = sum(cur.values()), sum(new.values())
    # uncertainty: re-evaluate with bootstrap parameters
    lifts = []
    for b in range(len(p["beta_boot"][ch[0]])):
        pb = {**p, "beta": {c: p["beta_boot"][c][b] for c in ch}, "c0": {c: p["c0_boot"][c][b] for c in ch}}
        lifts.append(MK.total_orders(pb, new) / MK.total_orders(pb, cur) - 1)
    lifts = np.array(lifts)
    d_budget = tot_n / tot_c - 1
    d_orders = sum(o_new.values()) / sum(o_cur.values()) - 1
    cards = kpi_cards([("Budget", fmt_money(tot_n), f"{d_budget:+.1%} vs current {fmt_money(tot_c)}"), ("Predicted orders", f"{d_orders:+.1%}", "platform-attributed, relative"),
                       ("Uncertainty (90%)", f"{np.percentile(lifts, 5):+.1%} to {np.percentile(lifts, 95):+.1%}", "sampling noise only"),
                       ("Cost per order", f"{tot_n / sum(o_new.values()):.2f}", f"now {tot_c / sum(o_cur.values()):.2f} EGP")])
    df = pd.DataFrame({"channel": ch * 2, "scenario": ["Current"] * len(ch) + ["Scenario"] * len(ch), "spend": [cur[c] / 1e6 for c in ch] + [new[c] / 1e6 for c in ch],
                       "orders": [o_cur[c] / 1e3 for c in ch] + [o_new[c] / 1e3 for c in ch]})
    f1 = px.bar(df, x="channel", y="spend", color="scenario", barmode="group", title="Annual spend by channel (EGP M)", color_discrete_sequence=[NAVY, ORANGE], labels={"spend": "EGP M", "channel": ""})
    f2 = px.bar(df, x="channel", y="orders", color="scenario", barmode="group", title="Predicted attributed orders (thousand)", color_discrete_sequence=[NAVY, ORANGE], labels={"orders": "thousand", "channel": ""})
    f3 = go.Figure()
    for i, c in enumerate(ch):
        s = np.linspace(.2, 2.0, 40) * cur[c]
        f3.add_trace(go.Scatter(x=s / 1e6, y=[MK.channel_orders(p, c, x) / 1e3 for x in s], name=c, line=dict(color=PALETTE[i % len(PALETTE)])))
        f3.add_trace(go.Scatter(x=[new[c] / 1e6], y=[o_new[c] / 1e3], mode="markers", marker=dict(size=11, color=PALETTE[i % len(PALETTE)], line=dict(color="white", width=2)), showlegend=False, hoverinfo="skip"))
    f3.update_layout(title="Response curves (dots = your scenario) - flatter curve = more saturated", xaxis_title="channel spend, EGP M / year", yaxis_title="orders (thousand)")
    return cards, _style(f1, 340), _style(f2, 340), _style(f3, 420)


def marketing_optimum(lo: float, hi: float) -> list[float]:
    p = S.model("marketing_response")
    r = MK.optimize(p, lo=lo, hi=hi)
    return [float(round(r["alloc"][c] / p["spend_current"][c], 2)) for c in p["channels"]]


def marketing_table() -> pd.DataFrame:
    p = S.model("marketing_response")
    rows = []
    for c in p["channels"]:
        lo, hi = p["beta_ci"][c]
        S_c = p["spend_current"][c]
        o = MK.channel_orders(p, c, S_c)
        rows.append({"Channel": c, "Annual spend EGP M": round(S_c / 1e6, 1), "Avg cost/order": round(S_c / o, 2), "Marginal cost/order": round(S_c / o / p["beta"][c], 2),
                     "Elasticity (beta)": round(p["beta"][c], 2), "95% CI": f"{lo:.2f} - {hi:.2f}"})
    return pd.DataFrame(rows).sort_values("Marginal cost/order")


# ----------------------------------------------------------------------------
# 9) model cards
# ----------------------------------------------------------------------------
def model_cards_html() -> str:
    def f(x, fmt=".2f"):
        try:
            return format(float(x), fmt)
        except Exception:
            return "-"
    rows = []
    m1, m2, m3, m4, m5, m6, m7 = (S.metrics(k) for k in ["01_demand_forecast", "02_stockout_risk", "03_churn", "04_crosssell", "05_return_risk", "06_supplier_risk", "07_marketing"])
    if m1:
        rows.append(("Demand forecast", "HistGradientBoosting (Poisson), 4 horizons", f"WAPE {f(m1['test_wape_by_horizon']['1'])} vs naive {f(m1['best_baseline_wape_by_horizon']['1'])}", "Plan restocks / production", m1["data_source"]))
    if m2:
        rows.append(("Stock-out early warning", "HistGradientBoosting classifier", f"PR-AUC {f(m2['model']['PR-AUC'])} vs persistence {f(m2['persistence_baseline']['PR-AUC'])}", "Flag branch x category one month ahead", m2["data_source"]))
    if m3:
        rows.append(("Customer churn", "HistGradientBoosting + RFM rules", f"ROC-AUC {f(m3['model']['ROC-AUC'])} vs baseline {f(m3['baseline_logit']['ROC-AUC'])}", "Win-back lists, value at risk", m3["data_source"]))
    if m4:
        rows.append(("Cross-sell", "Shrunk item-item confidence, noisy-or", f"hit@3 {f(m4['test']['co-occurrence (by probability)']['hit@3'])} vs popularity {f(m4['test']['popularity baseline']['hit@3'])}", "Checkout add-on suggestion", m4["data_source"]))
    if m5:
        rows.append(("Return risk", "HistGradientBoosting classifier", f"ROC-AUC {f(m5['model']['ROC-AUC'])} vs simple rate {f(m5['baseline']['ROC-AUC'])}", "Prioritise QA / size checks", m5["data_source"]))
    if m6:
        rows.append(("Supplier / PO risk", "HistGB classifier + lead-time regressor", f"ROC-AUC {f(m6['model']['ROC-AUC'])} vs supplier rate {f(m6['baseline']['ROC-AUC'])}", "Split risky POs, set buffers", m6["data_source"]))
    if m7:
        rows.append(("Marketing optimizer", "Log-log response per channel + SLSQP", f"hold-out R2 {f(np.mean(list(m7['holdout_r2'].values())))}", "Re-split budget (test it!)", m7["data_source"]))
    body = "".join(f"<tr><td><b>{a}</b></td><td>{b}</td><td>{c}</td><td>{d}</td><td>{badge(e, 'warn' if e == 'synthetic' else 'ok')}</td></tr>" for a, b, c, d, e in rows)
    return ('<table style="width:100%;border-collapse:collapse;font-size:14px"><thead><tr style="text-align:left;border-bottom:2px solid #0B3C5D"><th>Model</th><th>Method</th><th>Test result (time-based split)</th><th>Use</th><th>Trained on</th></tr></thead>'
            f'<tbody>{body}</tbody></table>')


def data_quality_html() -> str:
    m = S.metrics("00_data_audit")
    if not m:
        return ""
    rows = "".join(f"<tr><td>{badge(r['status'], 'ok' if r['status'] == 'PASS' else 'warn')}</td><td>{html.escape(r['check'])}</td><td>{html.escape(str(r['detail']))}</td></tr>" for r in m["data_quality"])
    return f'<table style="width:100%;border-collapse:collapse;font-size:13px"><tbody>{rows}</tbody></table>'
