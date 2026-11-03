"""Cottonil AI Operations Hub - Gradio app (entry point for Hugging Face Spaces).

Run locally:  python app.py          ->  http://127.0.0.1:7860  &&  http://127.0.0.1:7860/?__theme=light
All logic lives in cottonil/services.py; this file only wires the UI.
"""
from __future__ import annotations

import os
import tempfile

import gradio as gr

from cottonil import services as V
from cottonil.visuals import CSS, banner, hero_html

THEME = gr.themes.Soft(primary_hue="teal", secondary_hue="orange", neutral_hue="slate").set(body_background_fill="#fbfaf7")


def build_app() -> gr.Blocks:
    ready, missing = V.artifacts_ready()
    with gr.Blocks(title="Cottonil AI Operations Hub") as demo:
        gr.HTML(hero_html())
        if not ready:
            gr.HTML(banner(f"<b>Models not found.</b> Missing artifacts for: {', '.join(missing)}. Run <code>python scripts/run_all_notebooks.py</code> first.", "bad"))
            return demo
        gr.HTML(V.data_source_banner())
        branches = [("Whole network", "ALL")] + V.branch_choices()
        cats = V.category_choices_all()

        with gr.Tabs():
            # ------------------------------------------------------------------ overview
            with gr.Tab("📊 Overview"):
                year = gr.Dropdown(V.overview_years(), value="All", label="Year", scale=1)
                o_cards = gr.HTML()
                with gr.Row():
                    o1 = gr.Plot(show_label=False)
                    o2 = gr.Plot(show_label=False)
                with gr.Row():
                    o3 = gr.Plot(show_label=False)
                    o4 = gr.Plot(show_label=False)
                o5 = gr.Plot(show_label=False)
                outs = [o_cards, o1, o2, o3, o4, o5]
                year.change(V.overview_view, year, outs)
                demo.load(V.overview_view, year, outs)

            # ------------------------------------------------------------------ forecast
            with gr.Tab("📈 Forecast"):
                gr.Markdown("**How many units will we sell in the next 4 weeks?** Pick a branch and category. The orange band is the P10-P90 range - plan to the top of it where a stock-out is costly.")
                gr.HTML(V.forecast_accuracy_html())
                with gr.Row():
                    fb = gr.Dropdown(branches, value="ALL", label="Branch")
                    fc = gr.Dropdown(cats, value="ALL", label="Category")
                f_cards = gr.HTML()
                f_plot = gr.Plot(show_label=False)
                f_tbl = gr.Dataframe(interactive=False, label="Forecast by week")
                for c in (fb, fc):
                    c.change(V.forecast_view, [fb, fc], [f_cards, f_plot, f_tbl])
                demo.load(V.forecast_view, [fb, fc], [f_cards, f_plot, f_tbl])

            # ------------------------------------------------------------------ restock
            with gr.Tab("📦 Restock"):
                gr.Markdown("**What should we move, from where, and in what order?** Forecast + current stock + warehouse stock + historical transport cost → a prioritised transfer list. Change the policy and the plan recomputes.")
                gr.HTML(V.stock_ml_html())
                with gr.Row():
                    rb = gr.Dropdown(branches, value="ALL", label="Branch", scale=2)
                    rcw = gr.Slider(1, 8, value=4, step=.5, label="Target cover (weeks of demand to hold)", scale=2)
                    rsf = gr.Slider(0, 1, value=1, step=.25, label="Safety (0 = mean forecast, 1 = P90)", scale=2)
                ract = gr.CheckboxGroup(["URGENT", "RESTOCK", "OK", "OVERSTOCK"], value=["URGENT", "RESTOCK"], label="Show")
                r_cards = gr.HTML()
                with gr.Row():
                    r1 = gr.Plot(show_label=False)
                    r2 = gr.Plot(show_label=False)
                r_tbl = gr.Dataframe(interactive=False, label="Restock plan (top 300 by priority)", max_height=460)
                r_state = gr.State()
                with gr.Row():
                    r_btn = gr.Button("⬇ Export full filtered plan (CSV)")
                    r_file = gr.File(label="", visible=False)

                def _restock(b, cw, sf, a):
                    cards, f1, f2, t, scope = V.restock_view(b, cw, sf, a)
                    return cards, f1, f2, t, scope

                def _export(scope):
                    p = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="restock_plan_")
                    scope.to_csv(p.name, index=False)
                    return gr.File(value=p.name, visible=True)
                rins, routs = [rb, rcw, rsf, ract], [r_cards, r1, r2, r_tbl, r_state]
                for c in (rb, ract):
                    c.change(_restock, rins, routs)
                for c in (rcw, rsf):
                    c.change(_restock, rins, routs, trigger_mode="always_last")
                demo.load(_restock, rins, routs)
                r_btn.click(_export, r_state, r_file)

            # ------------------------------------------------------------------ cross-sell
            with gr.Tab("🛒 Cross-sell"):
                gr.Markdown("**What should the cashier offer next?** Add the items scanned so far; the engine suggests the add-on most often bought with those items (weighted by margin) that is also in stock at the branch.")
                gr.HTML(V.crosssell_metrics_html())
                pchoices = V.product_choices()
                with gr.Row():
                    xb = gr.Dropdown(branches[1:], value=branches[1][1], label="Branch (for stock check)", scale=1)
                    xbasket = gr.Dropdown(pchoices, multiselect=True, label="Basket (type to search products)", scale=3)
                with gr.Row():
                    xmode = gr.Radio(["Margin-aware (probability x margin)", "Most often bought together"], value="Margin-aware (probability x margin)", label="Rank by")
                    xk = gr.Slider(1, 8, value=4, step=1, label="How many suggestions")
                    xstock = gr.Checkbox(True, label="Only suggest items in stock at the branch")
                with gr.Row():
                    ex1 = gr.Button("Example: boxer")
                    ex2 = gr.Button("Example: women's top")
                    ex3 = gr.Button("Example: kids pajamas")
                x_till = gr.HTML()
                x_cards = gr.HTML()
                x_plot = gr.Plot(show_label=False)
                with gr.Accordion("Which categories go together? (lift heatmap)", open=False):
                    gr.Plot(value=V.category_lift_fig(), show_label=False)
                xins = [xb, xbasket, xk, xmode, xstock]
                for c in (xb, xbasket, xmode, xstock):
                    c.change(V.crosssell_view, xins, [x_till, x_cards, x_plot])
                xk.change(V.crosssell_view, xins, [x_till, x_cards, x_plot], trigger_mode="always_last")

                def _example(cat):
                    ids = [v for lbl, v in pchoices if lbl.startswith(cat + " ·")]
                    return [ids[0]] if ids else []
                demo.load(lambda: _example("Men boxers"), None, xbasket)
                ex1.click(lambda: _example("Men boxers"), None, xbasket)
                ex2.click(lambda: _example("Women tops"), None, xbasket)
                ex3.click(lambda: _example("Girls pajamas"), None, xbasket)

            # ------------------------------------------------------------------ customers
            with gr.Tab("👥 Customers"):
                gr.Markdown("**Who is about to leave, and what should we do about it?**")
                gr.HTML(V.churn_metrics_html())
                with gr.Row():
                    cid = gr.Textbox(label="Customer ID (e.g. C0012345)", scale=2)
                    cgo = gr.Button("Look up", variant="primary", scale=1)
                    cr1 = gr.Button("🎲 Valuable at-risk customer", scale=1)
                    cr2 = gr.Button("🎲 A champion", scale=1)
                with gr.Row():
                    c_html = gr.HTML()
                    c_gauge = gr.Plot(show_label=False)
                demo.load(lambda: V.random_customer("valuable at-risk"), None, cid).then(V.customer_lookup, cid, [c_html, c_gauge])
                cgo.click(V.customer_lookup, cid, [c_html, c_gauge])
                cid.submit(V.customer_lookup, cid, [c_html, c_gauge])
                cr1.click(lambda: V.random_customer("valuable at-risk"), None, cid).then(V.customer_lookup, cid, [c_html, c_gauge])
                cr2.click(lambda: V.random_customer("loyal champion"), None, cid).then(V.customer_lookup, cid, [c_html, c_gauge])
                gr.Markdown("### Portfolio view")
                with gr.Row():
                    s1 = gr.Plot(show_label=False)
                    s2 = gr.Plot(show_label=False)
                s3 = gr.Plot(show_label=False)
                demo.load(V.segments_view, None, [s1, s2, s3])
                gr.Markdown("### Win-back call list (sorted by value at risk)")
                with gr.Row():
                    cl_seg = gr.Dropdown(V.segment_choices(), value="All", label="Segment")
                    cl_p = gr.Slider(0, 1, value=.5, step=.05, label="Minimum churn probability")
                    cl_n = gr.Slider(10, 200, value=40, step=10, label="Rows")
                cl_tbl = gr.Dataframe(interactive=False, max_height=420)
                for c in (cl_seg,):
                    c.change(V.call_list, [cl_seg, cl_p, cl_n], cl_tbl)
                for c in (cl_p, cl_n):
                    c.change(V.call_list, [cl_seg, cl_p, cl_n], cl_tbl, trigger_mode="always_last")
                demo.load(V.call_list, [cl_seg, cl_p, cl_n], cl_tbl)

            # ------------------------------------------------------------------ returns
            with gr.Tab("↩️ Returns"):
                gr.Markdown("**Which order lines are likely to come back, and where do returns concentrate?**")
                ro = V.return_options()
                maps = ro["maps"]
                with gr.Row():
                    with gr.Column(scale=1):
                        rcat = gr.Dropdown(maps["category"], value="Men boxers" if "Men boxers" in maps["category"] else maps["category"][0], label="Category")
                        rsize = gr.Dropdown(ro["sizes_by_cat"].get("Men boxers", maps["size"]), value=(ro["sizes_by_cat"].get("Men boxers") or maps["size"])[0], label="Size")
                        rch = gr.Dropdown(maps["sales_channel"], value="E-commerce" if "E-commerce" in maps["sales_channel"] else maps["sales_channel"][0], label="Channel")
                        rpay = gr.Dropdown(maps["payment_method"], value=maps["payment_method"][0], label="Payment method")
                        rbr = gr.Dropdown(maps["branch_id"], value=maps["branch_id"][0], label="Fulfilling branch")
                    with gr.Column(scale=1):
                        rage = gr.Dropdown(maps["age_band"], value="25-34" if "25-34" in maps["age_band"] else maps["age_band"][0], label="Customer age band")
                        rseg = gr.Dropdown(maps["customer_segment"], value="Retail" if "Retail" in maps["customer_segment"] else maps["customer_segment"][0], label="Customer type")
                        rqty = gr.Slider(1, 20, value=2, step=1, label="Quantity")
                        rprice = gr.Slider(20, 2500, value=300, step=5, label="Unit price (EGP)")
                        rdisc = gr.Slider(0, 40, value=10, step=1, label="Discount %")
                        ritems = gr.Slider(1, 5, value=2, step=1, label="Items in order")
                        rid = gr.Checkbox(True, label="Customer identified")
                        rmon = gr.Slider(1, 12, value=11, step=1, label="Month")
                    with gr.Column(scale=1):
                        r_out = gr.HTML()
                        r_g = gr.Plot(show_label=False)
                rin = [rcat, rsize, rch, rpay, rbr, rage, rseg, rqty, rprice, rdisc, ritems, rid, rmon]
                rcat.change(lambda c: gr.Dropdown(choices=ro["sizes_by_cat"].get(c, maps["size"]), value=(ro["sizes_by_cat"].get(c) or maps["size"])[0]), rcat, rsize)
                for c in (rcat, rsize, rch, rpay, rbr, rage, rseg, rid):
                    c.change(V.return_view, rin, [r_out, r_g])
                for c in (rqty, rprice, rdisc, ritems, rmon):
                    c.change(V.return_view, rin, [r_out, r_g], trigger_mode="always_last")
                demo.load(V.return_view, rin, [r_out, r_g])
                h1, h2 = gr.Plot(show_label=False), gr.Plot(show_label=False)
                demo.load(V.return_heatmap, None, [h1, h2])

            # ------------------------------------------------------------------ suppliers
            with gr.Tab("🚚 Suppliers"):
                gr.Markdown("**Will this purchase order arrive complete, and when?** Enter the draft PO. Above 40% partial-receipt risk, the hub suggests splitting it.")
                sm = V.supplier_options()
                with gr.Row():
                    with gr.Column():
                        sv = gr.Dropdown(sm["supplier_id"], value=sm["supplier_id"][0], label="Supplier")
                        swh = gr.Dropdown(sm["receiving_warehouse_id"], value=sm["receiving_warehouse_id"][0], label="Receiving warehouse")
                        scur = gr.Dropdown(sm["currency"], value=sm["currency"][0], label="Currency")
                        spur = gr.Dropdown(sm["po_purpose"], value=sm["po_purpose"][0], label="Purpose")
                    with gr.Column():
                        ssub = gr.Slider(10000, 5000000, value=700000, step=10000, label="PO value (EGP)")
                        sterm = gr.Dropdown([30, 45, 60, 90], value=60, label="Payment terms (days)")
                        snl = gr.Slider(1, 5, value=2, step=1, label="Lines")
                        sq = gr.Slider(100, 6500, value=3000, step=100, label="Avg quantity per line")
                        sc = gr.Slider(5, 400, value=60, step=5, label="Avg unit cost (EGP)")
                        smon = gr.Slider(1, 12, value=6, step=1, label="Month")
                    with gr.Column():
                        s_out = gr.HTML()
                        s_g = gr.Plot(show_label=False)
                s_alt = gr.Dataframe(interactive=False, label="Better-than-average backup suppliers for the same material")
                sin = [sv, swh, scur, spur, ssub, sterm, snl, sq, sc, smon]
                for c in (sv, swh, scur, spur, sterm):
                    c.change(V.supplier_view, sin, [s_out, s_g, s_alt])
                for c in (ssub, snl, sq, sc, smon):
                    c.change(V.supplier_view, sin, [s_out, s_g, s_alt], trigger_mode="always_last")
                demo.load(V.supplier_view, sin, [s_out, s_g, s_alt])
                gr.Plot(value=V.supplier_scorecard_fig(), show_label=False)

            # ------------------------------------------------------------------ marketing
            with gr.Tab("📣 Marketing"):
                gr.Markdown("**What if we moved budget between channels?** Drag each channel's annual budget (100% = last 12 months). Orders are *platform-attributed*, so read the change in %, not the absolute numbers.")
                chs = V.marketing_channels()
                with gr.Row():
                    msl = [gr.Slider(.5, 1.5, value=1.0, step=.05, label=c) for c in chs]
                with gr.Row():
                    mopt = gr.Button("✨ Suggest best split (same total budget, ±50%)", variant="primary")
                    mrst = gr.Button("Reset to current")
                m_cards = gr.HTML()
                with gr.Row():
                    m1, m2 = gr.Plot(show_label=False), gr.Plot(show_label=False)
                m3 = gr.Plot(show_label=False)
                m_tbl = gr.Dataframe(value=V.marketing_table(), interactive=False, label="Channel efficiency (elasticity < 1 = diminishing returns)")

                def _mk(*m):
                    return V.marketing_scenario(list(m))
                for s in msl:
                    s.change(_mk, msl, [m_cards, m1, m2, m3], trigger_mode="always_last")
                demo.load(_mk, msl, [m_cards, m1, m2, m3])
                mopt.click(lambda: V.marketing_optimum(.5, 1.5), None, msl).then(_mk, msl, [m_cards, m1, m2, m3])
                mrst.click(lambda: [1.0] * len(chs), None, msl).then(_mk, msl, [m_cards, m1, m2, m3])

            # ------------------------------------------------------------------ about
            with gr.Tab("ℹ️ About"):
                gr.Markdown("### Models in this hub")
                gr.HTML(V.model_cards_html())
                gr.Markdown("### Data-quality checks (notebook 00)")
                gr.HTML(V.data_quality_html())
                gr.Markdown(
                    "### Known limitations\n"
                    "- Stock data is **category-level**, so restock plans are per category, not per size/colour.\n"
                    "- Marketing-attributed orders are platform claims and exceed the sales ledger; use relative efficiency only and **validate with a holdout test**.\n"
                    "- Model quality numbers are from a **time-based test split**, not from live use. Track acceptance of suggestions and forecast error after launch.\n"
                    "- Retrain monthly: `python scripts/run_all_notebooks.py` (about 10 minutes on a laptop with the full data).")
    return demo


demo = build_app()

if __name__ == "__main__":
    demo.queue().launch(theme=THEME, css=CSS, server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
