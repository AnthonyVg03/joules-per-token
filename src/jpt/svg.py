"""Tiny SVG chart writer (no matplotlib needed). Readable on GitHub."""

from __future__ import annotations

import math
from html import escape

PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#be185d", "#4d7c0f"]
W, H = 760, 420
ML, MR, MT, MB = 70, 210, 50, 55  # margins (right side holds the legend)


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=mag * 10)
    start = math.floor(lo / step + 1e-9) * step
    end = math.ceil(hi / step - 1e-9) * step  # last tick at or above the max: no point outside the frame
    k = round((end - start) / step)
    return [round(start + i * step, 10) for i in range(k + 1)]


def _fmt(v: float) -> str:
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}".rstrip("0").rstrip(".")
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _frame(title: str, xlabel: str, ylabel: str, xt: list[float], yt: list[float], sx, sy) -> list[str]:
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           'font-family="-apple-system, Helvetica, Arial, sans-serif" font-size="12">',
           f'<rect width="{W}" height="{H}" fill="white"/>',
           f'<text x="{ML}" y="28" font-size="16" font-weight="600" fill="#111">{escape(title)}</text>']
    for t in yt:
        y = sy(t)
        out.append(f'<line x1="{ML}" x2="{W - MR}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        out.append(f'<text x="{ML - 8}" y="{y + 4:.1f}" text-anchor="end" fill="#6b7280">{_fmt(t)}</text>')
    for t in xt:
        x = sx(t)
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{MT}" y2="{H - MB}" stroke="#f3f4f6"/>')
        out.append(f'<text x="{x:.1f}" y="{H - MB + 18}" text-anchor="middle" fill="#6b7280">{_fmt(t)}</text>')
    out.append(f'<line x1="{ML}" x2="{W - MR}" y1="{H - MB}" y2="{H - MB}" stroke="#9ca3af"/>')
    out.append(f'<text x="{(ML + W - MR) / 2}" y="{H - 12}" text-anchor="middle" fill="#374151">{escape(xlabel)}</text>')
    out.append(f'<text x="16" y="{(MT + H - MB) / 2}" text-anchor="middle" fill="#374151" '
               f'transform="rotate(-90 16 {(MT + H - MB) / 2})">{escape(ylabel)}</text>')
    return out


def _scales(xs: list[float], ys: list[float], y_zero: bool):
    xlo, xhi = min(xs), max(xs)
    ylo, yhi = (0 if y_zero else min(ys)), max(ys)
    if xlo == xhi:
        xlo, xhi = xlo - 1, xhi + 1
    if ylo == yhi:
        ylo, yhi = ylo - 1, yhi + 1
    xt, yt = _nice_ticks(xlo, xhi), _nice_ticks(ylo, yhi)
    x0, x1, y0, y1 = xt[0], xt[-1], yt[0], yt[-1]
    sx = lambda v: ML + (v - x0) / (x1 - x0) * (W - ML - MR)
    sy = lambda v: H - MB - (v - y0) / (y1 - y0) * (H - MT - MB)
    return xt, yt, sx, sy


def line_chart(series: list[tuple[str, list[tuple[float, float]]]], title: str, xlabel: str, ylabel: str,
               y_zero: bool = True) -> str:
    series = [(n, sorted(p)) for n, p in series if p]
    if not series:
        return ""
    xs = [x for _, p in series for x, _ in p]
    ys = [y for _, p in series for _, y in p]
    xt, yt, sx, sy = _scales(xs, ys, y_zero)
    out = _frame(title, xlabel, ylabel, xt, yt, sx, sy)
    for i, (name, pts) in enumerate(series):
        c = PALETTE[i % len(PALETTE)]
        d = " ".join(f"{'M' if j == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for j, (x, y) in enumerate(pts))
        out.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        for x, y in pts:
            out.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.5" fill="{c}"><title>{escape(name)}: '
                       f'{_fmt(x)} → {_fmt(y)}</title></circle>')
        ly = MT + 10 + i * 20
        out.append(f'<rect x="{W - MR + 16}" y="{ly - 9}" width="12" height="12" rx="2" fill="{c}"/>')
        out.append(f'<text x="{W - MR + 34}" y="{ly + 1}" fill="#111">{escape(name)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def scatter(points: list[dict], title: str, xlabel: str, ylabel: str, highlight: set[str] | None = None) -> str:
    """points: [{'label', 'x', 'y', 'group'}]. Highlighted labels are drawn bold and joined."""
    pts = [p for p in points if p["x"] is not None and p["y"] is not None]
    if not pts:
        return ""
    highlight = highlight or set()
    xt, yt, sx, sy = _scales([p["x"] for p in pts], [p["y"] for p in pts], y_zero=False)
    out = _frame(title, xlabel, ylabel, xt, yt, sx, sy)
    groups = sorted({p["group"] for p in pts})
    color = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(groups)}
    front = sorted((p for p in pts if p["label"] in highlight), key=lambda p: p["x"])
    if len(front) > 1:
        d = " ".join(f"{'M' if j == 0 else 'L'}{sx(p['x']):.1f},{sy(p['y']):.1f}" for j, p in enumerate(front))
        out.append(f'<path d="{d}" fill="none" stroke="#111" stroke-width="1.2" stroke-dasharray="5 4"/>')
    for p in pts:
        hl = p["label"] in highlight
        out.append(f'<circle cx="{sx(p["x"]):.1f}" cy="{sy(p["y"]):.1f}" r="{6 if hl else 4.5}" '
                   f'fill="{color[p["group"]]}" stroke="{"#111" if hl else "white"}" stroke-width="{1.5 if hl else 1}">'
                   f'<title>{escape(p["label"])}</title></circle>')
        out.append(f'<text x="{sx(p["x"]) + 8:.1f}" y="{sy(p["y"]) - 6:.1f}" font-size="10" fill="#374151">'
                   f'{escape(p.get("short", p["label"]))}</text>')
    for i, g in enumerate(groups):
        ly = MT + 10 + i * 20
        out.append(f'<rect x="{W - MR + 16}" y="{ly - 9}" width="12" height="12" rx="6" fill="{color[g]}"/>')
        out.append(f'<text x="{W - MR + 34}" y="{ly + 1}" fill="#111">{escape(g)}</text>')
    if highlight:
        ly = MT + 10 + len(groups) * 20 + 10
        out.append(f'<line x1="{W - MR + 16}" x2="{W - MR + 30}" y1="{ly - 3}" y2="{ly - 3}" stroke="#111" stroke-dasharray="5 4"/>')
        out.append(f'<text x="{W - MR + 34}" y="{ly + 1}" fill="#111">Pareto frontier</text>')
    out.append("</svg>")
    return "\n".join(out)
