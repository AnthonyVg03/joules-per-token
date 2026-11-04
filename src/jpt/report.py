"""Turn the SQLite results into tables, charts and a Pareto frontier."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from . import svg
from .stats import cv, median, stability_label
from .store import Store

DECODE_PROMPTS = ("short", "code")  # long generations: decode speed and energy per token
PREFILL_PROMPT = "long_ctx"  # long prompt, short answer: prefill speed and time to first token


def summarize(store: Store, session: str) -> list[dict]:
    """One row per variant with the headline numbers."""
    runs = defaultdict(list)
    for r in store.query("SELECT * FROM runs WHERE session=?", (session,)):
        runs[r["variant"]].append(dict(r))
    qual = defaultdict(dict)
    for q in store.query("SELECT * FROM quality WHERE session=? ORDER BY ts", (session,)):
        qual[q["variant"]][q["task"]] = dict(q)

    out = []
    for vid in sorted(set(runs) | set(qual)):
        rs = runs.get(vid, [])
        dec = [r for r in rs if r["prompt_id"] in DECODE_PROMPTS]
        pre = [r for r in rs if r["prompt_id"] == PREFILL_PROMPT]
        any_row = (rs or [None])[0] or next(iter(qual[vid].values()))
        g, p = qual[vid].get("gsm8k"), qual[vid].get("perplexity")
        decode = [r["decode_tps"] for r in dec]
        out.append({
            "variant": vid, "model": any_row["model"], "engine": any_row["engine"], "quant": any_row["quant"],
            "bits": any_row["bits"], "bpw": next((r["bpw"] for r in rs if r.get("bpw")), None),
            "decode_tps": median(decode), "decode_cv": cv(decode), "stability": stability_label(cv(decode)),
            "prefill_tps": median(r["prefill_tps"] for r in pre),
            "ttft_ms": median(r["ttft_ms"] for r in pre),
            "j_per_token": median(r["j_per_token"] for r in dec),
            "j_per_token_dyn": median(r["j_per_token_dyn"] for r in dec),
            "avg_power_w": median(r["avg_power_w"] for r in dec),
            "peak_mem_mb": max((r["peak_mem_mb"] for r in rs if r["peak_mem_mb"]), default=None),
            "gsm8k": g["score"] if g else None, "gsm8k_lo": g["ci_low"] if g else None,
            "gsm8k_hi": g["ci_high"] if g else None, "gsm8k_n": g["n"] if g else None,
            "gsm8k_j": g["energy_j"] if g else None,
            "ppl": p["score"] if p else None,
        })
    return out


def pareto(rows: list[dict]) -> set[str]:
    """Variants not dominated on (GSM8K ↑, decode tok/s ↑, J/token ↓).

    A dominates B if A is at least as good on all three and strictly better on one.
    """
    cand = [r for r in rows if None not in (r["gsm8k"], r["decode_tps"], r["j_per_token"])]

    def key(r):
        return (r["gsm8k"], r["decode_tps"], -r["j_per_token"])

    front = set()
    for a in cand:
        ka = key(a)
        dominated = any(all(x >= y for x, y in zip(key(b), ka)) and key(b) != ka for b in cand if b is not a)
        if not dominated:
            front.add(a["variant"])
    return front


def _x(r: dict) -> float:
    return r["bpw"] or r["bits"]


def _series(rows: list[dict], field: str) -> list[tuple[str, list[tuple[float, float]]]]:
    by = defaultdict(list)
    for r in rows:
        if r[field] is not None:
            by[f"{r['model']} · {r['engine']}"].append((_x(r), r[field]))
    return sorted(by.items())


def _f(v, spec: str, pct: bool = False) -> str:
    if v is None:
        return "—"
    return format(v * 100 if pct else v, spec) + ("%" if pct else "")


def build(store: Store, session: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = summarize(store, session)
    front = pareto(rows)

    with open(out_dir / "summary.csv", "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]) + ["pareto"])
            w.writeheader()
            for r in rows:
                w.writerow({**r, "pareto": r["variant"] in front})

    charts = {
        "gsm8k.svg": svg.line_chart([(n, [(x, y * 100) for x, y in p]) for n, p in _series(rows, "gsm8k")],
                                    "GSM8K accuracy vs compression", "bits per weight", "accuracy (%)"),
        "decode.svg": svg.line_chart(_series(rows, "decode_tps"), "Generation speed",
                                     "bits per weight", "tokens / s"),
        "energy.svg": svg.line_chart(_series(rows, "j_per_token"), "Energy per generated token",
                                     "bits per weight", "joules / token"),
        "memory.svg": svg.line_chart([(n, [(x, y / 1024) for x, y in p]) for n, p in _series(rows, "peak_mem_mb")],
                                     "Peak memory", "bits per weight", "GB"),
        "perplexity.svg": svg.line_chart(_series(rows, "ppl"), "Perplexity (WikiText-2) — lower is better",
                                         "bits per weight", "perplexity", y_zero=False),
        "pareto.svg": svg.scatter(
            [{"label": r["variant"], "short": r["quant"], "group": f"{r['model']} · {r['engine']}",
              "x": r["j_per_token"], "y": r["gsm8k"] * 100 if r["gsm8k"] is not None else None} for r in rows],
            "Quality vs energy", "joules / token (lower is better)", "GSM8K accuracy (%)", highlight=front),
    }
    th = store.query("SELECT variant, t_s, decode_tps FROM thermal WHERE session=? ORDER BY ts", (session,))
    if th:
        by = defaultdict(list)
        for r in th:
            if r["decode_tps"]:
                by[r["variant"]].append((r["t_s"] / 60, r["decode_tps"]))
        charts["thermal.svg"] = svg.line_chart(sorted(by.items()), "Sustained load: speed over time",
                                               "minutes", "tokens / s")
    for name, content in charts.items():
        if content:
            (out_dir / name).write_text(content)

    md = [f"# Results — session `{session}`", ""]
    if rows:
        md += ["| Variant | Bits/weight | Decode (tok/s) | Stability | Prefill (tok/s) | TTFT (ms) | "
               "J/token | J/token (idle-subtracted) | Memory (GB) | GSM8K | Perplexity | Pareto |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(rows, key=lambda r: (r["model"], r["engine"], -_x(r))):
            md.append("| " + " | ".join([
                r["variant"], _f(_x(r), ".2f"), _f(r["decode_tps"], ".1f"), r["stability"],
                _f(r["prefill_tps"], ".0f"), _f(r["ttft_ms"], ".0f"), _f(r["j_per_token"], ".3f"),
                _f(r["j_per_token_dyn"], ".3f"), _f(r["peak_mem_mb"] / 1024 if r["peak_mem_mb"] else None, ".2f"),
                (_f(r["gsm8k"], ".1f", True) + (f" ({_f(r['gsm8k_lo'], '.0f', True)}–{_f(r['gsm8k_hi'], '.0f', True)})"
                                                 if r["gsm8k"] is not None else "")),
                _f(r["ppl"], ".2f"), "★" if r["variant"] in front else "",
            ]) + " |")
        md += ["", "★ = on the Pareto frontier: no other variant is at once more accurate, faster and more "
                   "energy-efficient.", ""]
    for name in charts:
        if (out_dir / name).exists():
            md += [f"![{name}]({name})", ""]
    md += ["## How to read", "",
           "- **Bits/weight**: actual file size × 8 / parameter count (more reliable than the quantization label).",
           "- **J/token**: whole-SoC energy (CPU + GPU + ANE + DRAM) during the request, divided by generated tokens.",
           "- **Idle-subtracted**: the idle draw measured just before is removed — the cost of inference itself.",
           "- **GSM8K**: in brackets, the 95% Wilson confidence interval. Variants whose intervals overlap are not "
           "significantly different.", ""]
    path = out_dir / "report.md"
    path.write_text("\n".join(md))
    return path
