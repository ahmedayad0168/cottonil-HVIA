"""Synthetic Cottonil dataset generator -- SCHEMA-IDENTICAL to the real tables.

WHY THIS EXISTS
The real CSVs are confidential, so this generator lets anybody run and test the
whole project end to end. It copies the real table names, column names, id
formats, date formats and the headline statistics reported in the two PDFs
(80 branches, 16 categories, 139 products / 695 SKUs, 19.1% anonymous orders,
~2.1 items/order, ~38% gross margin, ~5.3% returns, ~33% partial POs ...).

IMPORTANT: it also *injects mild, plausible structure* (basket affinities,
churn behaviour, return drivers, supplier reliability, diminishing returns in
marketing) so that the models have something to learn. That means metrics
measured on synthetic data say NOTHING about real-world accuracy. Every saved
artifact is tagged data_source="synthetic" until you retrain on real data.

Usage:  python scripts/make_synthetic_data.py --scale 1.0
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .calendar_features import black_friday, calendar_features
from .config import SYNTHETIC_MARKER

START = pd.Timestamp("2018-10-19")
END = pd.Timestamp("2026-10-18")

# ----------------------------------------------------------------------------
# static reference data
# ----------------------------------------------------------------------------
GOVS = [("القاهرة", 21), ("الجيزة", 8), ("الإسكندرية", 8), ("الدقهلية", 5), ("الشرقية", 4),
        ("الغربية", 4), ("المنوفية", 3), ("القليوبية", 4), ("البحيرة", 3), ("كفر الشيخ", 2),
        ("دمياط", 2), ("بورسعيد", 2), ("الإسماعيلية", 2), ("السويس", 1), ("الفيوم", 1),
        ("بني سويف", 1), ("المنيا", 2), ("أسيوط", 2), ("سوهاج", 2), ("قنا", 1), ("أسوان", 2)]
CAIRO_AREAS = ["الزمالك", "المعادي", "مدينة نصر", "مصر الجديدة", "التجمع الخامس", "الزيتون", "المقطم",
               "وسط البلد", "شبرا", "حلوان", "القطامية", "مدينتي", "السيدة زينب", "الرحاب"]
WAREHOUSES = [
    ("WH01", "مخزن مركزي القاهرة", "القاهرة", "Central DC", 262541),
    ("WH02", "مخزن التوزيع الإسكندرية", "الإسكندرية", "Regional DC", 119583),
    ("WH03", "مخزن التوزيع الجيزة", "الجيزة", "Regional DC", 143948),
    ("WH04", "مخزن التوزيع الدلتا", "الدقهلية", "Regional DC", 114899),
    ("WH05", "مخزن التوزيع الصعيد", "المنيا", "Regional DC", 96050),
    ("WH06", "مخزن القطن الخام", "القاهرة", "Raw materials", 138566),
    ("WH07", "مخزن المنتج التام", "القاهرة", "Finished goods", 307157),
    ("WH08", "مخزن المرتجعات", "القاهرة", "Returns", 332021),
    ("WH09", "مخزن توزيع الشرقية", "الشرقية", "Regional DC", 159485),
    ("WH10", "مخزن توزيع البحيرة", "البحيرة", "Regional DC", 111800),
    ("WH11", "مخزن توزيع الغربية", "الغربية", "Regional DC", 332684),
    ("WH12", "مخزن توزيع القناة", "الإسماعيلية", "Regional DC", 299463),
]
GOV_TO_DC = {  # nearest distribution warehouse (used only to shape synthetic transport costs)
    "القاهرة": "WH01", "الجيزة": "WH03", "القليوبية": "WH01", "الإسكندرية": "WH02", "البحيرة": "WH10",
    "الدقهلية": "WH04", "دمياط": "WH04", "كفر الشيخ": "WH04", "الغربية": "WH11", "المنوفية": "WH11",
    "الشرقية": "WH09", "الإسماعيلية": "WH12", "بورسعيد": "WH12", "السويس": "WH12", "الفيوم": "WH05",
    "بني سويف": "WH05", "المنيا": "WH05", "أسيوط": "WH05", "سوهاج": "WH05", "قنا": "WH05", "أسوان": "WH05",
}
APPAREL_SIZES_M = ["S", "M", "L", "XL", "2XL", "3XL"]
APPAREL_SIZES_W = ["S", "M", "L", "XL"]
KID_SIZES = ["4Y", "6Y", "8Y", "10Y", "12Y"]
SOCK_SIZES = ["Free", "M", "L", "XL"]
COLORS = ["أبيض", "أسود", "كحلي", "رمادي", "بيج", "أزرق", "نبيتي", "وردي", "أخضر"]
SIZE_POP = {"S": .6, "M": 1.0, "L": 1.0, "XL": .7, "2XL": .35, "3XL": .2, "Free": .9,
            "4Y": .8, "6Y": 1, "8Y": 1, "10Y": .8, "12Y": .6}

# category: n_products, price lo/hi, share of lines, poisson lambda for extra qty, audience,
#           sizes, base product names, apparel-with-sizes flag
CATS = {
    "Accessories": dict(n=5, lo=103, hi=400, w=.01, lam=.3, aud="Unisex", sizes=["Free"], sz=False,
                        names=["حقيبة سفر قطن", "حزام جلد", "مناديل قطن"]),
    "Boys pajamas": dict(n=8, lo=300, hi=700, w=.05, lam=.2, aud="Kids", sizes=KID_SIZES, sz=True,
                         names=["بيجامة ولادي قطن", "بيجامة ولادي شتوي"]),
    "Girls pajamas": dict(n=8, lo=300, hi=700, w=.05, lam=.2, aud="Kids", sizes=KID_SIZES, sz=True,
                          names=["بيجامة بناتي قطن", "بيجامة بناتي شتوي"]),
    "Girls underwear": dict(n=4, lo=90, hi=200, w=.02, lam=.8, aud="Kids", sizes=KID_SIZES, sz=True,
                            names=["كيلوت بناتي قطن"]),
    "Home textiles": dict(n=8, lo=400, hi=2270, w=.03, lam=.3, aud="Home", sizes=["Free"], sz=False,
                          names=["فوطة قطن", "روب حمام", "ملاءة سرير"]),
    "Kids underwear": dict(n=6, lo=90, hi=250, w=.04, lam=.8, aud="Kids", sizes=KID_SIZES, sz=True,
                           names=["فانلة اطفال قطن", "سليب اطفال"]),
    "Men bottoms": dict(n=6, lo=250, hi=520, w=.04, lam=.3, aud="Men", sizes=APPAREL_SIZES_M, sz=True,
                        names=["شورت رجالي صيفي", "بنطلون بيجامة رجالي"]),
    "Men boxers": dict(n=16, lo=190, hi=520, w=.15, lam=.8, aud="Men", sizes=APPAREL_SIZES_M, sz=True,
                       names=["بوكسر رجالي قطن", "بوكسر رجالي ليكرا", "بوكسر رجالي مخطط"]),
    "Men tops": dict(n=16, lo=250, hi=665, w=.13, lam=.5, aud="Men", sizes=APPAREL_SIZES_M, sz=True,
                     names=["حمالة رجالي مرسريزية", "فانلة رجالي نص كم", "فانلة رجالي كت"]),
    "Men underwear": dict(n=8, lo=85, hi=200, w=.08, lam=.8, aud="Men", sizes=APPAREL_SIZES_M, sz=True,
                          names=["سليب رجالي قطن", "كيلوت رجالي"]),
    "Socks": dict(n=24, lo=85, hi=290, w=.21, lam=2.0, aud="Men", sizes=SOCK_SIZES, sz=False,
                  names=["3 * شراب رجالي كلاسيك-لايكرا", "3*شراب قدم", "شراب رياضي", "شراب حريمي"]),
    "Women bottoms": dict(n=4, lo=200, hi=400, w=.01, lam=.3, aud="Women", sizes=APPAREL_SIZES_W, sz=True,
                          names=["ليجنج حريمي قطن", "شورت حريمي"]),
    "Women bras": dict(n=6, lo=190, hi=420, w=.05, lam=.4, aud="Women", sizes=APPAREL_SIZES_W, sz=True,
                       names=["ستيان حريمي قطن", "ستيان حريمي سيمليس"]),
    "Women pajamas": dict(n=8, lo=450, hi=900, w=.04, lam=.2, aud="Women", sizes=APPAREL_SIZES_W, sz=True,
                          names=["بيجامة حريمي قطن", "بيجامة حريمي شتوي"]),
    "Women tops": dict(n=8, lo=150, hi=400, w=.08, lam=.5, aud="Women", sizes=APPAREL_SIZES_W, sz=True,
                       names=["تيشيرت حريمي قطن", "كامي حريمي"]),
    "Women underwear": dict(n=4, lo=100, hi=250, w=.01, lam=.8, aud="Women", sizes=APPAREL_SIZES_W, sz=True,
                            names=["كيلوت حريمي قطن"]),
}
CAT_LIST = sorted(CATS)
K = len(CAT_LIST)
CAT_IDX = {c: i for i, c in enumerate(CAT_LIST)}

# who gets bought together (seed category -> {partner: weight})
PARTNERS = {
    "Men boxers": {"Socks": 5, "Men tops": 4, "Men underwear": 2, "Men bottoms": 1},
    "Men tops": {"Men boxers": 4, "Socks": 4, "Men bottoms": 2, "Men underwear": 1},
    "Socks": {"Men boxers": 4, "Men tops": 3, "Women tops": 1},
    "Men underwear": {"Men tops": 3, "Socks": 3, "Men boxers": 2},
    "Men bottoms": {"Men tops": 3, "Men boxers": 2},
    "Women tops": {"Women bras": 4, "Women underwear": 3, "Women pajamas": 2, "Women bottoms": 2},
    "Women bras": {"Women underwear": 5, "Women tops": 3},
    "Women underwear": {"Women bras": 5, "Women tops": 2},
    "Women pajamas": {"Women tops": 3, "Women bottoms": 2, "Socks": 1},
    "Women bottoms": {"Women tops": 4, "Women pajamas": 2},
    "Girls pajamas": {"Girls underwear": 4, "Boys pajamas": 2, "Kids underwear": 2},
    "Boys pajamas": {"Kids underwear": 4, "Girls pajamas": 2},
    "Girls underwear": {"Girls pajamas": 4, "Kids underwear": 2},
    "Kids underwear": {"Boys pajamas": 3, "Girls underwear": 2, "Socks": 2},
    "Home textiles": {"Home textiles": 4, "Women pajamas": 1},
    "Accessories": {"Men boxers": 1, "Home textiles": 1},
}
FIRST = ["أحمد", "محمد", "محمود", "مصطفى", "علي", "عمر", "كريم", "خالد", "يوسف", "إبراهيم", "طارق", "شريف",
         "مريم", "فاطمة", "منى", "رانيا", "أمل", "أسماء", "دعاء", "يارا", "نور", "هبة", "عبدالله", "عبدالرحمن"]
LAST = ["محمد", "علي", "حسن", "إبراهيم", "السيد", "حمادة", "خليل", "منصور", "شعبان", "جابر", "الدسوقي",
        "عامر", "فتحى", "سعيد", "محمود", "مصطفى", "الشريف", "النجار", "عبدالعزيز", "عبدالفتاح"]
MATERIALS = ["Packaging", "Accessories", "Finished garments", "Elastic & Lycra", "Labels",
             "Machinery services", "Knitted fabric", "Dyes", "Logistics", "Cotton yarn"]
SUP_CITIES = ["بورسعيد، مصر", "الجيزة، مصر", "القاهرة، مصر", "الإسكندرية، مصر", "دمياط، مصر",
              "المحلة الكبرى، مصر"]
YEAR_ORDERS_K = {2018: 8.6, 2019: 41.5, 2020: 42.8, 2021: 46.7, 2022: 61.9, 2023: 72.0, 2024: 82.0,
                 2025: 93.4, 2026: 78.2}
CHANNEL_CPO = {"Meta Ads": 8.4, "In-store": 8.4, "SMS / CRM": 9.2, "Google Ads": 9.4,
               "Influencers": 9.5, "Outdoor": 10.0}
CHANNEL_BETA = {"Meta Ads": .88, "In-store": .90, "SMS / CRM": .80, "Google Ads": .82,
                "Influencers": .72, "Outdoor": .58}   # outdoor saturates fastest
CHANNEL_SHARE = {"Meta Ads": .22, "Google Ads": .19, "In-store": .15, "SMS / CRM": .13,
                 "Influencers": .15, "Outdoor": .16}


def price_index(day_idx: np.ndarray) -> np.ndarray:
    """Nominal price level (EGP devaluation): ~0.15 in Oct-2018 -> 1.0 in late 2026."""
    knots_d = np.array([0, 365, 730, 1095, 1460, 1825, 2190, 2555, 2922])
    knots_v = np.array([.15, .20, .25, .30, .38, .55, .75, .92, 1.02])
    return np.interp(day_idx, knots_d, knots_v)


def _log(verbose, msg):
    if verbose:
        print(f"[synth] {msg}", flush=True)


# ----------------------------------------------------------------------------
def generate(out_dir: str | Path, scale: float = 1.0, seed: int = 7, verbose: bool = True) -> Path:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    (out / "02_TABLES").mkdir(parents=True, exist_ok=True)
    (out / "01_CATALOG").mkdir(parents=True, exist_ok=True)
    T = out / "02_TABLES"
    days = pd.date_range(START, END, freq="D")
    nd = len(days)

    # ------------------------------------------------------------- branches
    rows, k = [], 0
    for gov, n in GOVS:
        for i in range(n):
            k += 1
            city = CAIRO_AREAS[i % len(CAIRO_AREAS)] if gov == "القاهرة" else gov
            rows.append((f"BR{k:03d}", f"{city} {i + 1}", gov, city, f"شارع الجمهورية {i + 1}، {city}"))
    branches = pd.DataFrame(rows, columns=["branch_id", "branch_name_ar", "governorate_ar", "city_ar", "address_ar"])
    perm = rng.permutation(len(branches))  # mix governorates across ids like the real file
    branches = branches.iloc[perm].reset_index(drop=True)
    branches["branch_id"] = [f"BR{i + 1:03d}" for i in range(len(branches))]
    branches["retail_location_type"] = rng.choice(["Retail branch", "Point of sale / reseller"],
                                                  len(branches), p=[.65, .35])
    open_days = rng.integers(0, 660, len(branches))
    open_days[:10] = rng.integers(0, 30, 10)
    branches["opening_date"] = START + pd.to_timedelta(open_days, unit="D")
    B = len(branches)
    branch_w = rng.lognormal(0, .05, B)
    branch_open_idx = open_days

    wh = pd.DataFrame(WAREHOUSES, columns=["warehouse_id", "warehouse_label", "governorate_ar",
                                           "warehouse_kind", "capacity_units"])

    # ------------------------------------------------------------- catalog
    prods, skus = [], []
    pid = 0
    for cat in CAT_LIST:
        sp = CATS[cat]
        for j in range(sp["n"]):
            pid += 1
            base = sp["names"][j % len(sp["names"])]
            price = int(round(rng.uniform(sp["lo"], sp["hi"]) / 5) * 5)
            prods.append((f"PR{pid:04d}", f"{base} {j + 1}", cat, sp["aud"], price,
                          "https://cottonilegypt.com/"))
    products = pd.DataFrame(prods, columns=["product_id", "product_name_ar", "category_en", "audience",
                                            "listed_price_EGP", "source_url"])
    P = len(products)
    mult5 = [.95, .975, 1.0, 1.025, 1.05]
    sid = 0
    for i, r in products.iterrows():
        sizes = CATS[r.category_en]["sizes"]
        off = int(rng.integers(0, len(sizes)))
        coff = int(rng.integers(0, len(COLORS)))
        for s in range(5):
            sid += 1
            skus.append((f"SKU{sid:06d}", r.product_id, sizes[(off + s) % len(sizes)],
                         COLORS[(coff + s) % len(COLORS)], mult5[s]))
    sku_df = pd.DataFrame(skus, columns=["sku_id", "product_id", "size", "color_ar", "price_multiplier"])
    prod_cat = products["category_en"].map(CAT_IDX).values
    listed = products["listed_price_EGP"].values.astype(float)
    sku_by_prod = np.arange(len(sku_df)).reshape(P, 5)
    sku_mult = sku_df["price_multiplier"].values
    sku_w = sku_df["size"].map(SIZE_POP).values.reshape(P, 5)
    sku_cw = np.cumsum(sku_w / sku_w.sum(1, keepdims=True), axis=1)
    prod_margin = rng.normal(.425, .02, P).clip(.34, .46)
    prods_in_cat = [np.where(prod_cat == c)[0] for c in range(K)]
    zipf_w = []
    for c in range(K):
        w = 1.0 / np.arange(1, len(prods_in_cat[c]) + 1) ** 0.8
        rng.shuffle(w)
        zipf_w.append(w / w.sum())
    lam = np.array([CATS[c]["lam"] for c in CAT_LIST])
    cat_w = np.array([CATS[c]["w"] for c in CAT_LIST])
    cat_w = cat_w / cat_w.sum()
    cat_has_size = np.array([CATS[c]["sz"] for c in CAT_LIST])

    # category x month seasonality
    CM = np.ones((K, 13))
    for c in CAT_LIST:
        i = CAT_IDX[c]
        if "pajamas" in c:
            CM[i, [11, 12, 1, 2]] *= 1.6
            CM[i, [6, 7, 8]] *= .7
        if c == "Home textiles":
            CM[i, [11, 12, 1, 2]] *= 1.5
            CM[i, [6, 7, 8]] *= .8
        if c == "Socks":
            CM[i, [11, 12, 1, 2]] *= 1.25
        if c in ("Men bottoms",):
            CM[i, [6, 7, 8]] *= 1.3
        if c in ("Kids underwear", "Girls underwear"):
            CM[i, 9] *= 1.5
        if c == "Women tops":
            CM[i, [6, 7, 8]] *= 1.1
    catp = CM[:, 1:].T * cat_w[None, :]            # (12, K)
    catp = catp / catp.sum(1, keepdims=True)
    Pm = np.zeros((K, K))
    for c, d in PARTNERS.items():
        for c2, w in d.items():
            Pm[CAT_IDX[c], CAT_IDX[c2]] = w
    Pm = Pm / Pm.sum(1, keepdims=True)

    # ------------------------------------------------------------- customers
    n_cust = max(600, int(round(85000 * scale)))
    join_day = np.sort(np.where(rng.random(n_cust) < .08, rng.integers(0, 20, n_cust),
                                rng.integers(0, nd - 20, n_cust)))
    is_reseller = np.zeros(n_cust, bool)
    n_res = int(round(n_cust * 1666 / 85000))
    is_reseller[rng.choice(n_cust, n_res, replace=False)] = True
    u = rng.random(n_cust)
    life = np.where(u < .20, rng.integers(0, 20, n_cust), np.where(u < .35, 20000, rng.exponential(650, n_cust)))
    life = np.where(is_reseller, 20000, life)
    churn_day = join_day + life
    activity = np.maximum(rng.gamma(.8, 1.5, n_cust), .03)
    activity = np.where(is_reseller, activity * 3 + .5, activity)
    cust_gov = rng.choice([g for g, _ in GOVS], n_cust, p=np.array([n for _, n in GOVS]) / 80)
    customers = pd.DataFrame({
        "customer_id": [f"C{i + 1:07d}" for i in range(n_cust)],
        "customer_segment": np.where(is_reseller, "Reseller", "Retail"),
        "customer_name_ar": [f"{a} {b}" for a, b in zip(rng.choice(FIRST, n_cust), rng.choice(LAST, n_cust))],
        "governorate_ar": cust_gov,
        "joined_date": START + pd.to_timedelta(join_day, unit="D"),
        "age_band": rng.choice(["18-24", "25-34", "35-44", "45-54", "55+"], n_cust),
    })
    _log(verbose, f"dims done: {B} branches, {P} products, {len(sku_df)} SKUs, {n_cust} customers")

    # ------------------------------------------------------------- orders
    N = int(round(540000 * scale))
    cf = calendar_features(days)
    dow = days.dayofweek.values
    dow_mult = np.array([.9, .9, .95, 1.1, 1.3, 1.25, 1.0])[dow]
    season = dow_mult * (1 + .25 * cf["is_ramadan"].values + .5 * (cf["days_to_eid_fitr"].values <= 10)
                         + .25 * (cf["days_to_eid_adha"].values <= 7)
                         + .7 * (cf["days_to_black_friday"].values <= 3))
    season *= np.where(np.isin(days.month, [6, 7, 8]), .92, 1) * np.where(days.month == 9, 1.08, 1) \
        * np.where(days.month == 12, 1.08, 1)
    yr = days.year.values
    day_w = np.zeros(nd)
    tot_k = sum(YEAR_ORDERS_K.values())
    for y, v in YEAR_ORDERS_K.items():
        m = yr == y
        day_w[m] = season[m] / season[m].sum() * (v / tot_k)
    day_w /= day_w.sum()
    day_of_order = np.sort(rng.choice(nd, N, p=day_w))
    od = day_of_order
    yfrac = od / 365.25
    p_ecom = np.interp(yfrac, [0, 3.2, 8.0], [.06, .15, .24])
    p_whole = np.full(N, .10)
    ur = rng.random(N)
    channel = np.where(ur < p_whole, "Wholesale / reseller",
                       np.where(ur < p_whole + p_ecom, "E-commerce", "Retail POS"))
    # branch (must be open)
    branch_idx = np.zeros(N, int)
    months_o = pd.PeriodIndex(days[od], freq="M")
    for per in months_o.unique():
        sel = np.where(months_o == per)[0]
        mid_day = od[sel].max()
        act = np.where(branch_open_idx <= mid_day)[0]
        w = branch_w[act] / branch_w[act].sum()
        branch_idx[sel] = rng.choice(act, len(sel), p=w)
    # customers: weekly eligibility sampling
    p_anon = np.where(channel == "Retail POS", .25, np.where(channel == "E-commerce", .06, 0.0))
    anon = rng.random(N) < p_anon
    cust_idx = np.full(N, -1)
    week_of_order = od // 7
    res_ids = np.where(is_reseller)[0]
    ret_ids_all = np.where(~is_reseller)[0]
    order_by_week = np.split(np.arange(N), np.searchsorted(week_of_order, np.arange(1, week_of_order.max() + 1)))
    for w_i, idx in enumerate(order_by_week):
        if len(idx) == 0:
            continue
        ws, we = w_i * 7, w_i * 7 + 6
        elig = (join_day <= we) & (churn_day >= ws)
        wh_sel = idx[(channel[idx] == "Wholesale / reseller")]
        rt_sel = idx[(channel[idx] != "Wholesale / reseller") & (~anon[idx])]
        for sel, pool in ((wh_sel, res_ids), (rt_sel, ret_ids_all)):
            if len(sel) == 0:
                continue
            cand = pool[elig[pool]]
            if len(cand) == 0:
                anon[sel] = True if pool is ret_ids_all else anon[sel]
                continue
            pw = activity[cand]
            cust_idx[sel] = rng.choice(cand, len(sel), p=pw / pw.sum())
    # can't order before joining within the join week
    has_c = cust_idx >= 0
    od = np.where(has_c, np.maximum(od, join_day[np.where(has_c, cust_idx, 0)]), od)
    order = np.argsort(od, kind="stable")           # keep ids chronological
    od, channel, branch_idx, cust_idx = od[order], channel[order], branch_idx[order], cust_idx[order]
    _log(verbose, f"orders drafted: {N:,}")

    # ------------------------------------------------------------- lines
    p_items = np.where((channel == "Wholesale / reseller")[:, None],
                       np.array([[0, .4, .3, .2, .1]]), np.array([[.34, .41, .15, .06, .04]]))
    cum = np.cumsum(p_items, axis=1)
    items = (rng.random(N)[:, None] > cum).sum(1) + 1
    items = np.minimum(items, 5)
    o_of_l = np.repeat(np.arange(N), items)
    L = o_of_l.size
    start_idx = np.cumsum(items) - items
    pos = np.arange(L) - np.repeat(start_idx, items)
    mon_line = days.month.values[od][o_of_l]
    cat_idx = np.zeros(L, int)
    first = pos == 0
    for m in range(1, 13):
        sel = first & (mon_line == m)
        if sel.any():
            cat_idx[sel] = rng.choice(K, sel.sum(), p=catp[m - 1])
    seed_cat = cat_idx[start_idx][o_of_l]
    rest = ~first
    aff = rest & (rng.random(L) < .55)
    for c in range(K):
        sel = aff & (seed_cat == c)
        if sel.any():
            cat_idx[sel] = rng.choice(K, sel.sum(), p=Pm[c])
    ind = rest & ~aff
    for m in range(1, 13):
        sel = ind & (mon_line == m)
        if sel.any():
            cat_idx[sel] = rng.choice(K, sel.sum(), p=catp[m - 1])

    def draw_products(mask):
        out_p = np.zeros(L, int)
        for c in range(K):
            sel = mask & (cat_idx == c)
            if sel.any():
                out_p[sel] = rng.choice(prods_in_cat[c], sel.sum(), p=zipf_w[c])
        return out_p

    prod_idx = draw_products(np.ones(L, bool))
    for _ in range(5):                                # remove same-product duplicates in an order
        dup = pd.Series(o_of_l * 1000 + prod_idx).duplicated().values
        if not dup.any():
            break
        prod_idx[dup] = draw_products(dup)[dup]
    keep = ~pd.Series(o_of_l * 1000 + prod_idx).duplicated().values
    o_of_l, prod_idx, cat_idx = o_of_l[keep], prod_idx[keep], cat_idx[keep]
    L = o_of_l.size
    items = np.bincount(o_of_l, minlength=N)
    u = rng.random(L)[:, None]
    sku_pos = (u > sku_cw[prod_idx]).sum(1).clip(0, 4)
    sku_idx = sku_by_prod[prod_idx, sku_pos]
    line_day = od[o_of_l]
    pidx = price_index(line_day)
    ch_line = channel[o_of_l]
    qty = 1 + rng.poisson(lam[cat_idx])
    qty = np.where(ch_line == "Wholesale / reseller", qty * rng.integers(3, 13, L), qty).clip(1, 135)
    unit_nom = listed[prod_idx] * sku_mult[sku_idx] * pidx
    unit_price = np.round(unit_nom * (1 + rng.normal(0, .02, L)), 2)
    unit_cost = np.round(unit_nom * (1 - prod_margin[prod_idx]) * (1 + rng.normal(0, .01, L)), 2)
    disc_pct = np.where(rng.random(L) < np.where(ch_line == "E-commerce", .65, .5), rng.uniform(.02, .20, L), 0.0)
    gross = np.round(qty * unit_price, 2)
    disc = np.round(gross * disc_pct, 2)
    net = np.round(gross - disc, 2)
    margin = np.round(net - qty * unit_cost, 2)
    _log(verbose, f"lines drafted: {L:,}")

    # ------------------------------------------------------------- returns
    size_sku = sku_df["size"].values[sku_idx]
    ecom = (ch_line == "E-commerce").astype(float)
    whole = (ch_line == "Wholesale / reseller").astype(float)
    cat_fx = np.zeros(K)
    for c, v in {"Women bras": .5, "Men tops": .3, "Socks": -.3, "Home textiles": -.3,
                 "Men boxers": .1, "Girls pajamas": .2}.items():
        cat_fx[CAT_IDX[c]] = v
    size_fx = np.where(np.isin(size_sku, ["2XL", "3XL"]), .35, np.where(np.isin(size_sku, ["S"]), .15, 0.0))
    z0 = 1.3 * ecom - 1.0 * whole + cat_fx[cat_idx] + 1.2 * disc_pct + size_fx + .12 * np.log1p(qty)
    target = 28500 / 1138536
    lo_a, hi_a = -9.0, 0.0
    for _ in range(40):
        a = (lo_a + hi_a) / 2
        mp = (1 / (1 + np.exp(-(a + z0)))).mean()
        lo_a, hi_a = (a, hi_a) if mp < target else (lo_a, a)
    pret = 1 / (1 + np.exp(-(a + z0)))
    returned = rng.random(L) < pret
    R = np.where(returned)[0]
    reasons = np.array(["Size issue", "Defect claim", "Other", "Wrong item", "Exchange"])
    rw = np.ones((len(R), 5))
    rw[:, 0] = np.where(cat_has_size[cat_idx[R]], 1.6, .15)
    rw[:, 3] = np.where(ecom[R] > 0, 1.5, .6)
    rw[:, 1] = np.where(whole[R] > 0, 1.8, 1.0)
    rw /= rw.sum(1, keepdims=True)
    reason = reasons[(rng.random(len(R))[:, None] > np.cumsum(rw, 1)).sum(1).clip(0, 4)]
    rq = np.where(qty[R] == 1, 1, np.where(rng.random(len(R)) < .55, 1, rng.integers(1, qty[R] + 1)))
    refund = np.round(net[R] * rq / qty[R], 2)
    ret_day = np.minimum(line_day[R] + rng.integers(1, 31, len(R)), nd - 1)
    _log(verbose, f"returns drafted: {len(R):,}")

    # ------------------------------------------------------------- write sales tables
    so_ids = np.array([f"SO{i + 1:08d}" for i in range(N)])
    sl_ids = np.array([f"SL{i + 1:09d}" for i in range(L)])
    br_ids = branches["branch_id"].values
    sku_ids = sku_df["sku_id"].values
    prod_ids = products["product_id"].values
    odate = days[od]
    gross_o = np.round(np.bincount(o_of_l, gross, N), 2)
    disc_o = np.round(np.bincount(o_of_l, disc, N), 2)
    net_o = np.round(gross_o - disc_o, 2)
    tax_o = np.round(net_o * .14, 2)
    status = np.where(channel == "E-commerce", "Delivered", "Completed").astype(object)
    returned_orders = np.unique(o_of_l[R])
    status[returned_orders] = "Partially returned"
    pay_choices = np.array(["Cash", "Card POS", "Wallet", "Cash on delivery", "Bank transfer"])
    pay_p = {"Retail POS": [.38, .38, .14, 0, .10], "E-commerce": [0, .25, .35, .40, 0],
             "Wholesale / reseller": [.20, .05, .05, .30, .40]}
    pay = np.empty(N, object)
    for ch, pp in pay_p.items():
        sel = np.where(channel == ch)[0]
        pay[sel] = rng.choice(pay_choices, len(sel), p=np.array(pp) / np.sum(pp))
    cust_ids = np.array([""] + list(customers["customer_id"]), dtype=object)
    so = pd.DataFrame({
        "sales_order_id": so_ids, "order_date": odate.strftime("%Y-%m-%d"), "branch_id": br_ids[branch_idx],
        "customer_id": np.where(cust_idx >= 0, cust_ids[cust_idx + 1], np.nan), "sales_channel": channel,
        "payment_method": pay, "items_count": items, "gross_EGP": gross_o, "discount_EGP": disc_o,
        "net_before_tax_EGP": net_o, "tax_EGP": tax_o, "paid_total_EGP": np.round(net_o + tax_o, 2),
        "status": status})
    soi = pd.DataFrame({
        "sales_line_id": sl_ids, "sales_order_id": so_ids[o_of_l], "order_date": odate[o_of_l].strftime("%Y-%m-%d"),
        "branch_id": br_ids[branch_idx[o_of_l]], "sku_id": sku_ids[sku_idx], "product_id": prod_ids[prod_idx],
        "quantity": qty, "unit_price_EGP": unit_price, "gross_line_EGP": gross, "discount_line_EGP": disc,
        "net_line_EGP": net, "unit_cost_EGP": unit_cost, "margin_EGP": margin})
    ret = pd.DataFrame({
        "return_id": [f"RET{i + 1:07d}" for i in range(len(R))], "sales_order_id": so_ids[o_of_l[R]],
        "sales_line_id": sl_ids[R], "return_date": days[ret_day].strftime("%Y-%m-%d"), "sku_id": sku_ids[sku_idx[R]],
        "return_quantity": rq, "refund_amount_EGP": refund, "reason": reason,
        "refund_method": rng.choice(["Cash", "Wallet", "Card reversal"], len(R), p=[.35, .35, .30])})
    ret = ret.sample(frac=1, random_state=seed).reset_index(drop=True)
    ret["return_id"] = [f"RET{i + 1:07d}" for i in range(len(ret))]
    so.to_csv(T / "sales_orders.csv", index=False)
    soi.to_csv(T / "sales_order_items.csv", index=False)
    ret.to_csv(T / "sales_returns.csv", index=False)
    _log(verbose, "wrote sales_orders / sales_order_items / sales_returns")

    # ------------------------------------------------------------- monthly stock
    periods = pd.period_range("2018-10", "2026-10", freq="M")
    M = len(periods)
    ym_line = (pd.PeriodIndex(odate[o_of_l], freq="M") - periods[0]).map(lambda x: x.n).values
    ym_line = np.asarray(ym_line, dtype=int)
    sold = np.zeros((B, K, M))
    np.add.at(sold, (branch_idx[o_of_l], cat_idx, ym_line), qty)
    open_m = np.array([(pd.Period(branches["opening_date"].iloc[b], "M") - periods[0]).n for b in range(B)])
    for b in range(B):
        for c in range(K):
            row = sold[b, c]
            pos_mean = row[open_m[b]:].mean() if open_m[b] < M else row.mean()
            row[:open_m[b]] = pos_mean
    dem = .5 * sold + .5 * np.concatenate([sold[:, :, :1], sold[:, :, :-1]], axis=2)
    chronic = rng.normal(0, .55, (B, K, 1))
    cover = np.exp(np.log(1.8) + chronic + rng.normal(0, .45, (B, K, M)))     # months of cover
    closing = np.maximum(8, np.round(dem * cover + rng.normal(0, 2, (B, K, M)))).astype(int)
    risk = np.clip(1 / (1 + (cover / 1.0) ** 2.4) + rng.normal(0, .06, (B, K, M)), 0, .968)
    avg_price = np.array([listed[prod_cat == c].mean() for c in range(K)])
    pidx_m = price_index((periods.to_timestamp() - START).days.values.clip(0))
    wh_base = np.array([w[4] for w in WAREHOUSES])
    stock_rows = []
    cat_names = np.array(CAT_LIST)
    for m in range(M):
        ym = str(periods[m])
        b_df = pd.DataFrame({
            "year_month": ym, "stock_location_id": np.repeat(br_ids, K), "stock_location_type": "Branch",
            "product_category": np.tile(cat_names, B), "closing_units": closing[:, :, m].ravel(),
            "closing_stock_value_EGP": np.round(closing[:, :, m].ravel() * np.tile(avg_price, B) * pidx_m[m] * .6, 2),
            "stockout_risk_score": np.round(risk[:, :, m].ravel(), 3)})
        wu = np.round(np.outer(wh_base * rng.uniform(.5, 1.2, 12), cat_w * K) / K * rng.lognormal(0, .3, (12, K)) / 6).astype(int)
        w_df = pd.DataFrame({
            "year_month": ym, "stock_location_id": np.repeat([w[0] for w in WAREHOUSES], K), "stock_location_type": "Warehouse",
            "product_category": np.tile(cat_names, 12), "closing_units": wu.ravel().clip(100),
            "closing_stock_value_EGP": np.round(wu.ravel().clip(100) * np.tile(avg_price, 12) * pidx_m[m] * .6, 2),
            "stockout_risk_score": np.round(rng.beta(1.5, 7, 12 * K), 3)})
        stock_rows += [b_df, w_df]
    stock = pd.concat(stock_rows, ignore_index=True)
    stock.to_csv(T / "monthly_stock.csv", index=False)
    _log(verbose, f"wrote monthly_stock: {len(stock):,} rows")

    # ------------------------------------------------------------- suppliers / purchase orders
    S = 48
    mat = [MATERIALS[i % 10] for i in range(S)]
    suppliers = pd.DataFrame({
        "supplier_id": [f"SUP{i + 1:03d}" for i in range(S)], "supplier_name_ar": [f"مورد خامات رقم {i + 1:03d}" for i in range(S)],
        "material_type": mat, "city_country": rng.choice(SUP_CITIES, S),
        "payment_terms_days": rng.choice([15, 30, 45, 60, 90], S)})
    n_po = int(round(41500 * scale))
    po_day = np.sort(rng.choice(nd, n_po, p=day_w))
    sup_idx = rng.choice(S, n_po, p=np.random.default_rng(1).dirichlet(np.full(S, 6)))
    rec_wh = rng.choice(12, n_po)
    n_lines = (rng.random(n_po)[:, None] > np.cumsum([.22, .32, .26, .12, .08])).sum(1).clip(0, 4) + 1
    o_po = np.repeat(np.arange(n_po), n_lines)
    nl = o_po.size
    pol_day = po_day[o_po]
    pol_q = rng.integers(90, 6501, nl)
    pol_cost = np.round(rng.lognormal(np.log(55), .85, nl) * price_index(pol_day) / .7, 2)
    pol_line = np.round(pol_q * pol_cost, 2)
    subtotal = np.round(np.bincount(o_po, pol_line, n_po), 2)
    sup_fx = np.random.default_rng(2).normal(0, .55, S)
    mat_fx = {"Cotton yarn": .25, "Dyes": .35, "Elastic & Lycra": .1, "Packaging": -.2, "Labels": -.2}
    purpose = rng.choice(["Store network stock", "Seasonal procurement", "Finished goods replenishment"], n_po)
    currency = rng.choice(["EGP", "USD"], n_po, p=[.9, .1])
    zp = (sup_fx[sup_idx] + np.array([mat_fx.get(m, 0) for m in np.array(mat)[sup_idx]])
          + .30 * (np.log(subtotal + 1) - np.log(7e5)) + np.where(rec_wh == 4, .35, 0)
          + np.where(purpose == "Seasonal procurement", .2, 0) + np.where(currency == "USD", .3, 0))
    lo_a, hi_a = -5, 5
    for _ in range(40):
        a = (lo_a + hi_a) / 2
        mp = (1 / (1 + np.exp(-(a + zp)))).mean()
        lo_a, hi_a = (a, hi_a) if mp < .33 else (lo_a, a)
    partial = rng.random(n_po) < 1 / (1 + np.exp(-(a + zp)))
    mat_lead = {"Cotton yarn": 18, "Dyes": 14, "Machinery services": 25, "Finished garments": 10}
    lead = 12 + np.array([mat_lead.get(m, 6) for m in np.array(mat)[sup_idx]])[o_po] \
        + np.where(partial[o_po], 8, 0) + rng.normal(0, 4, nl)
    lead = np.maximum(3, np.round(lead)).astype(int)
    po_id_arr = np.array([f"PO{i + 1:07d}" for i in range(n_po)])
    pos_df = pd.DataFrame({
        "po_line_id": [f"POL{i + 1:08d}" for i in range(nl)], "po_id": po_id_arr[o_po],
        "sku_id": rng.choice(sku_ids, nl),
        "quantity": pol_q, "unit_cost_EGP": pol_cost, "line_cost_EGP": pol_line,
        "delivery_date": (days[0] + pd.to_timedelta(pol_day + lead, unit="D")).strftime("%Y-%m-%d")})
    pos = pd.DataFrame({
        "po_id": po_id_arr, "po_date": days[po_day].strftime("%Y-%m-%d"),
        "supplier_id": suppliers["supplier_id"].values[sup_idx],
        "receiving_warehouse_id": [WAREHOUSES[i][0] for i in rec_wh],
        "po_status": np.where(partial, "Partial receipt", "Received"), "currency": currency,
        "po_subtotal_EGP": subtotal, "payment_terms_days": rng.choice([30, 45, 60, 90], n_po),
        "po_purpose": purpose})
    pos.to_csv(T / "purchase_orders.csv", index=False)
    pos_df.to_csv(T / "purchase_order_items.csv", index=False)
    suppliers.to_csv(T / "suppliers.csv", index=False)

    # ------------------------------------------------------------- marketing
    n_mk = 2200
    chans = rng.choice(list(CHANNEL_SHARE), n_mk, p=np.array(list(CHANNEL_SHARE.values())) / sum(CHANNEL_SHARE.values()))
    start_d = rng.choice(nd - 75, n_mk, p=day_w[:nd - 75] / day_w[:nd - 75].sum())
    dur = rng.integers(3, 71, n_mk)
    budget = rng.integers(1000, 250000, n_mk)
    spend = np.round(budget * rng.uniform(.55, 1.0, n_mk), 2)
    beta = np.array([CHANNEL_BETA[c] for c in chans])
    cpo = np.array([CHANNEL_CPO[c] for c in chans])
    k_ch = (1e5 / cpo) / (1e5 ** beta)
    orders_attr = np.maximum(10, np.round(k_ch * spend ** beta * rng.lognormal(0, .35, n_mk) * (dur / 30) ** .15)).astype(int)
    ctr = {"Meta Ads": .045, "Google Ads": .06, "In-store": .03, "SMS / CRM": .02, "Influencers": .05, "Outdoor": .01}
    cpm = {"Meta Ads": 22, "Google Ads": 35, "In-store": 15, "SMS / CRM": 8, "Influencers": 18, "Outdoor": 12}
    impressions = np.round(spend / np.array([cpm[c] for c in chans]) * 1000 * rng.lognormal(0, .2, n_mk)).astype(int)
    clicks = np.round(impressions * np.array([ctr[c] for c in chans]) * rng.lognormal(0, .2, n_mk)).astype(int)
    mk = pd.DataFrame({
        "campaign_id": [f"MKT{i + 1:05d}" for i in range(n_mk)], "start_date": days[start_d].strftime("%Y-%m-%d"),
        "end_date": (days[start_d] + pd.to_timedelta(dur, unit="D")).strftime("%Y-%m-%d"), "channel": chans,
        "target_audience": rng.choice(["Men", "Women", "Kids", "Families"], n_mk),
        "governorate_ar": rng.choice([g for g, _ in GOVS], n_mk), "budget_EGP": budget, "spend_EGP": spend,
        "impressions": impressions, "clicks": clicks, "orders_attributed": orders_attr})
    mk.to_csv(T / "marketing_campaigns.csv", index=False)

    # ------------------------------------------------------------- inventory transfers
    n_tr = int(round(36500 * scale))
    tr_wh = rng.choice(12, n_tr)
    tr_br = rng.choice(B, n_tr)
    tiers = np.array([95, 155, 280, 450, 800, 1200, 2500])
    pair_tier = np.random.default_rng(3).integers(0, 7, (12, B))
    own_dc = np.array([[GOV_TO_DC[branches["governorate_ar"].iloc[b]] == WAREHOUSES[w][0] for b in range(B)] for w in range(12)])
    pair_tier = np.where(own_dc, np.random.default_rng(4).integers(0, 2, (12, B)), pair_tier)
    tcost = tiers[pair_tier[tr_wh, tr_br]]
    transfers = pd.DataFrame({
        "transfer_id": [f"TRF{i + 1:07d}" for i in range(n_tr)],
        "transfer_date": days[rng.choice(nd, n_tr)].strftime("%Y-%m-%d"),
        "source_warehouse_id": [WAREHOUSES[i][0] for i in tr_wh], "destination_branch_id": br_ids[tr_br],
        "transfer_status": rng.choice(["Completed", "In transit"], n_tr, p=[.95, .05]), "transport_cost_EGP": tcost})
    transfers.to_csv(T / "inventory_transfers.csv", index=False)

    # ------------------------------------------------------------- remaining dims
    b_out = branches.copy()
    b_out["opening_date"] = b_out["opening_date"].dt.strftime("%-m/%-d/%Y")
    b_out.to_csv(T / "branches.csv", index=False)
    c_out = customers.copy()
    c_out["joined_date"] = c_out["joined_date"].dt.strftime("%-m/%-d/%Y")
    c_out.to_csv(T / "customers.csv", index=False)
    wh.to_csv(T / "warehouses.csv", index=False)
    products.to_csv(T / "products_catalog.csv", index=False)
    sku_df.to_csv(T / "sku_variants.csv", index=False)
    products[["product_id", "product_name_ar", "category_en", "listed_price_EGP"]].rename(
        columns={"category_en": "category"}).to_csv(out / "01_CATALOG" / "PRODUCT_CATALOG_PUBLIC_REFERENCE.csv", index=False)
    (out / SYNTHETIC_MARKER).write_text(
        "THIS DATA IS SYNTHETIC.\nIt was generated by cottonil/synth.py to mirror the schema of the real Cottonil "
        f"tables (scale={scale}, seed={seed}).\nAny model trained on it is for pipeline testing only.\n")
    _log(verbose, f"DONE -> {out}  (orders={N:,}, lines={L:,}, returns={len(R):,}, POs={n_po:,})")
    return out
