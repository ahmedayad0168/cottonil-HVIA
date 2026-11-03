"""Generated visuals: product tiles (SVG), hero banner, KPI cards, badges.

No product photos were provided with the data, so each category gets a clean SVG illustration.
To use REAL photos: drop  assets/products/<Category name>.jpg|png|webp  (e.g. "Men boxers.jpg");
the app picks them up automatically (see `product_image_html`).
"""
from __future__ import annotations

import base64
import html
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "products"

NAVY, TEAL, ORANGE, RED, SAND = "#0B3C5D", "#1D8A99", "#F4A259", "#E4572E", "#F6F1E7"

# category -> (garment shape, tile colour)
STYLE = {
    "Men boxers": ("boxer", "#0B3C5D"), "Men underwear": ("brief", "#2F5D7C"), "Men tops": ("tank", "#1D8A99"),
    "Men bottoms": ("shorts", "#4C6E91"), "Socks": ("sock", "#F4A259"), "Women tops": ("tank", "#C06C84"),
    "Women bras": ("bra", "#B5527B"), "Women underwear": ("brief", "#D17A9B"), "Women bottoms": ("shorts", "#9A6FB0"),
    "Women pajamas": ("pajama", "#7C6A9C"), "Girls pajamas": ("pajama", "#E58FAE"), "Boys pajamas": ("pajama", "#4F9DBA"),
    "Girls underwear": ("brief", "#E58FAE"), "Kids underwear": ("tank", "#5BAE9B"), "Home textiles": ("towel", "#5B8E7D"),
    "Accessories": ("bag", "#8A7F72"),
}

SHAPES = {
    "boxer": '<path d="M28 34 H92 L96 58 L74 92 L60 66 L46 92 L24 58 Z" fill="white" fill-opacity=".95"/>'
             '<path d="M28 34 H92" stroke="{c}" stroke-width="7" stroke-opacity=".35"/>',
    "brief": '<path d="M26 36 H94 Q90 66 66 90 H54 Q30 66 26 36 Z" fill="white" fill-opacity=".95"/>'
             '<path d="M26 36 H94" stroke="{c}" stroke-width="7" stroke-opacity=".35"/>',
    "tank": '<path d="M42 20 H52 Q60 40 68 20 H78 L94 42 L83 51 V98 H37 V51 L26 42 Z" fill="white" fill-opacity=".95"/>',
    "shorts": '<path d="M30 30 H90 L98 90 H66 L60 58 L54 90 H22 Z" fill="white" fill-opacity=".95"/>'
              '<path d="M30 30 H90" stroke="{c}" stroke-width="7" stroke-opacity=".35"/>',
    "sock": '<path d="M48 18 H76 V62 L98 76 Q106 92 90 98 H58 Q44 98 46 82 Z" fill="white" fill-opacity=".95"/>'
            '<path d="M48 28 H76 M48 38 H76" stroke="{c}" stroke-width="5" stroke-opacity=".4"/>',
    "bra": '<path d="M24 38 Q42 34 52 52 Q60 62 68 52 Q78 34 96 38 Q94 70 76 74 Q64 76 60 66 Q56 76 44 74 Q26 70 24 38 Z" fill="white" fill-opacity=".95"/>'
           '<path d="M32 30 L40 38 M88 30 L80 38" stroke="white" stroke-width="4"/>',
    "pajama": '<path d="M34 16 H48 Q56 28 64 16 H78 L92 34 L83 42 V60 H37 V42 L28 34 Z" fill="white" fill-opacity=".95"/>'
              '<path d="M37 64 H83 L88 104 H66 L60 78 L54 104 H32 Z" fill="white" fill-opacity=".8"/>',
    "towel": '<rect x="26" y="26" width="68" height="72" rx="6" fill="white" fill-opacity=".95"/>'
             '<path d="M26 44 H94 M26 80 H94" stroke="{c}" stroke-width="5" stroke-opacity=".35"/>',
    "bag": '<path d="M30 46 H90 L96 102 H24 Z" fill="white" fill-opacity=".95"/>'
           '<path d="M46 46 V36 Q60 20 74 36 V46" fill="none" stroke="white" stroke-width="6"/>',
}


def slug(cat: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in cat).lower()


def product_svg(category: str, size: int = 96) -> str:
    shape, col = STYLE.get(category, ("tank", "#6B7A8F"))
    gid = f"g{abs(hash(category)) % 10_000}"
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 120 120" role="img" aria-label="{html.escape(category)}">'
            f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{col}"/><stop offset="1" stop-color="{col}" stop-opacity=".72"/></linearGradient></defs>'
            f'<rect width="120" height="120" rx="22" fill="url(#{gid})"/>'
            f'<circle cx="98" cy="22" r="26" fill="white" fill-opacity=".10"/><circle cx="16" cy="108" r="30" fill="white" fill-opacity=".08"/>'
            f'{SHAPES[shape].replace("{c}", col)}</svg>')


def product_image_html(category: str, size: int = 96) -> str:
    """Real photo if present in assets/products/, else the generated SVG tile."""
    for ext in ("jpg", "jpeg", "png", "webp"):
        for name in (category, slug(category)):
            p = ASSETS / f"{name}.{ext}"
            if p.exists():
                mime = "jpeg" if ext in ("jpg", "jpeg") else ext
                b64 = base64.b64encode(p.read_bytes()).decode()
                return f'<img src="data:image/{mime};base64,{b64}" width="{size}" height="{size}" style="border-radius:18px;object-fit:cover"/>'
    return product_svg(category, size)


def hero_html(subtitle: str = "Forecasting · Restocking · Cross-sell · Customer intelligence · Risk · Marketing") -> str:
    dots = "".join(f'<circle cx="{x}" cy="{y}" r="{r}" fill="white" fill-opacity="{o}"/>'
                   for x, y, r, o in [(60, 40, 38, .06), (170, 120, 54, .05), (880, 30, 70, .06), (760, 150, 40, .05), (480, 10, 26, .05), (980, 140, 46, .05)])
    return f'''
<div class="hero">
  <svg class="hero-bg" viewBox="0 0 1040 170" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg">{dots}</svg>
  <div class="hero-row">
    <div class="logo">{product_svg("Men boxers", 64)}</div>
    <div><div class="hero-title">Cottonil <span>AI Operations Hub</span></div><div class="hero-sub">{html.escape(subtitle)}</div></div>
  </div>
</div>'''


def kpi_cards(items: list[tuple[str, str, str]] | list[tuple[str, str]], cols: int | None = None) -> str:
    out = []
    for it in items:
        label, value = it[0], it[1]
        sub = it[2] if len(it) > 2 else ""
        out.append(f'<div class="kpi"><div class="kl">{html.escape(label)}</div><div class="kv">{value}</div><div class="ks">{sub}</div></div>')
    return f'<div class="kpi-grid">{"".join(out)}</div>'


def badge(text: str, kind: str = "info") -> str:
    return f'<span class="badge b-{kind}">{html.escape(text)}</span>'


def banner(text: str, kind: str = "warn") -> str:
    return f'<div class="banner bn-{kind}">{text}</div>'


def fmt_money(x: float, unit: str = "EGP") -> str:
    ax = abs(x)
    if ax >= 1e9:
        return f"{unit} {x / 1e9:,.2f}B"
    if ax >= 1e6:
        return f"{unit} {x / 1e6:,.1f}M"
    if ax >= 1e3:
        return f"{unit} {x / 1e3:,.1f}K"
    return f"{unit} {x:,.0f}"


def rtl(text: str) -> str:
    return f'<span dir="rtl" style="unicode-bidi:isolate">{html.escape(str(text))}</span>'


CSS = """
:root{--navy:#0B3C5D;--teal:#1D8A99;--orange:#F4A259;--red:#E4572E;--sand:#F6F1E7;--ink:#1f2d3a;--muted:#6b7a8a}
.gradio-container{max-width:1280px !important;margin:auto}
.hero{position:relative;overflow:hidden;border-radius:20px;padding:22px 26px;margin-bottom:10px;
      background:linear-gradient(120deg,var(--navy) 0%,#0f5a73 55%,var(--teal) 100%);color:#fff;box-shadow:0 8px 24px rgba(11,60,93,.25)}
.hero-bg{position:absolute;inset:0;width:100%;height:100%}
.hero-row{position:relative;display:flex;align-items:center;gap:18px}
.hero, .hero *{color:#fff !important}
.hero-title{font-size:30px;font-weight:800;letter-spacing:.2px}.hero-title span{font-weight:500;opacity:.9}
.hero-sub{opacity:.85;font-size:14px;margin-top:2px}
.logo svg{filter:drop-shadow(0 4px 8px rgba(0,0,0,.25))}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:6px 0 10px}
.kpi{background:var(--sand);border:1px solid #e8dfcc;border-left:5px solid var(--teal);border-radius:14px;padding:12px 14px}
.kpi:nth-child(4n+2){border-left-color:var(--orange)}.kpi:nth-child(4n+3){border-left-color:var(--navy)}.kpi:nth-child(4n+4){border-left-color:var(--red)}
.kl{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.6px;font-weight:600}
.kv{font-size:24px;font-weight:800;color:var(--navy) !important;line-height:1.2;white-space:nowrap}.ks{font-size:12px;color:var(--muted)}
.banner{border-radius:12px;padding:10px 14px;margin:6px 0;font-size:14px}
.bn-warn{background:#fff4e0;border:1px solid #f4c77d;color:#7a4b00}.bn-info{background:#e6f3f6;border:1px solid #a9d6df;color:#0b4d5a}
.bn-ok{background:#e8f6ee;border:1px solid #a8dcbc;color:#14532d}.bn-bad{background:#fdeceb;border:1px solid #f2b4ad;color:#7f1d1d}
.badge{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;font-weight:700;margin-right:6px}
.b-info{background:#e6f3f6;color:#0b4d5a}.b-warn{background:#fff4e0;color:#7a4b00}.b-bad{background:#fdeceb;color:#9b1c1c}.b-ok{background:#e8f6ee;color:#14532d}
.rec-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px;margin-top:8px}
.rec{display:flex;gap:12px;align-items:center;background:#fff;border:1px solid #e3e8ee;border-radius:16px;padding:12px;box-shadow:0 2px 8px rgba(16,24,40,.06)}
.rec .t{font-weight:700;color:var(--navy);font-size:15px}.rec .m{font-size:12px;color:var(--muted)}
.rec .p{font-size:13px;margin-top:4px}.rec.out{opacity:.45}
.till{background:#0d1b2a;color:#d9f1ff;font-family:ui-monospace,Menlo,monospace;border-radius:14px;padding:14px 16px;font-size:13px;line-height:1.7}
.till, .till div, .till span{color:#d9f1ff !important}.till .hl{color:#7ee0c3 !important}.till .dim{color:#8fa7bb !important}
.tabs button{font-weight:600 !important}
footer{display:none !important}
"""
