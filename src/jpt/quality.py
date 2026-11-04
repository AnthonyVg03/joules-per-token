"""What quantization costs in quality.

* **GSM8K accuracy** — grade-school maths problems with a single numeric answer:
  automatic, unambiguous scoring, and multi-step reasoning degrades early when a
  model is compressed. Run through ``/v1/chat/completions`` so both engines are
  scored the same way.
* **Perplexity** (llama.cpp only) — how surprised the model is by a reference
  text (WikiText-2). Lower is better. Sensitive to small quantization damage
  that accuracy can miss, but says nothing about whether answers are right.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path

from .power import PowerMeter
from .stats import wilson_interval

GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
WIKITEXT_URL = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"

SYSTEM = ("Solve the math problem. Reason step by step, then write the final answer "
          "on the last line in the form '#### <number>'.")


# --------------------------------------------------------------------------- datasets


def gsm8k(data_dir: Path) -> list[dict]:
    path = data_dir / "gsm8k_test.jsonl"
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(GSM8K_URL, path)
    items = []
    for line in path.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            items.append({"question": d["question"], "answer": d["answer"].split("####")[-1].strip()})
    return items


def wikitext(data_dir: Path) -> Path:
    raw = data_dir / "wikitext-2-raw" / "wiki.test.raw"
    if not raw.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        z = data_dir / "wikitext-2-raw-v1.zip"
        urllib.request.urlretrieve(WIKITEXT_URL, z)
        with zipfile.ZipFile(z) as f:
            f.extractall(data_dir)
    return raw


# --------------------------------------------------------------------------- GSM8K

_NUM = r"-?\$?\s*[\d,]*\.?\d+"


def _to_float(s: str) -> float | None:
    s = s.replace(",", "").replace("$", "").replace(" ", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def extract_answer(text: str) -> float | None:
    """'#### 42' if present, else 'the answer is 42', else the last number in the text."""
    for pattern in (rf"####\s*({_NUM})", rf"answer is[:\s]*\**\s*({_NUM})", rf"\\boxed\{{({_NUM})\}}"):
        hits = re.findall(pattern, text, flags=re.IGNORECASE)
        if hits:
            return _to_float(hits[-1])
    nums = re.findall(_NUM, text)
    return _to_float(nums[-1]) if nums else None


def is_correct(pred: float | None, gold: str) -> bool:
    g = _to_float(gold)
    return pred is not None and g is not None and abs(pred - g) < 1e-6


def run_gsm8k(client, data_dir: Path, *, n: int, max_tokens: int, meter: PowerMeter | None = None,
              log=print) -> dict:
    items = gsm8k(data_dir)[:n]  # the first n problems: same subset for every variant
    details, correct, tokens = [], 0, 0
    a = meter.snapshot() if meter else None
    for i, it in enumerate(items):
        text, n_tok = client.chat([{"role": "system", "content": SYSTEM},
                                   {"role": "user", "content": it["question"]}], max_tokens)
        pred = extract_answer(text)
        ok = is_correct(pred, it["answer"])
        correct += ok
        tokens += n_tok or 0
        details.append({"i": i, "gold": it["answer"], "pred": pred, "ok": ok, "tokens": n_tok})
        if (i + 1) % 10 == 0:
            log(f"  GSM8K {i + 1}/{len(items)}: {correct}/{i + 1} correct")
    energy = meter.energy_between(a, meter.snapshot()) if meter else None
    lo, hi = wilson_interval(correct, len(items))
    return {"task": "gsm8k", "n": len(items), "score": correct / len(items) if items else 0.0,
            "ci_low": lo, "ci_high": hi, "gen_tokens": tokens,
            "energy_j": energy.total_j if energy else None, "details": details}


# --------------------------------------------------------------------------- perplexity


def run_perplexity(model_path: Path, data_dir: Path, *, chunks: int, ctx: int, log_path: Path) -> dict:
    exe = shutil.which("llama-perplexity")
    if not exe:
        raise RuntimeError("llama-perplexity not found (brew install llama.cpp)")
    text = wikitext(data_dir)
    t0 = time.monotonic()
    out = subprocess.run([exe, "-m", str(model_path), "-f", str(text), "-c", str(ctx), "--chunks", str(chunks),
                          "-ngl", "999"], capture_output=True, text=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(out.stdout + "\n" + out.stderr)
    m = re.search(r"Final estimate:\s*PPL\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+)", out.stdout + out.stderr)
    if not m:
        raise RuntimeError(f"could not parse perplexity, see {log_path}")
    ppl, err = float(m.group(1)), float(m.group(2))
    return {"task": "perplexity", "n": chunks, "score": ppl, "ci_low": ppl - 1.96 * err, "ci_high": ppl + 1.96 * err,
            "details": {"seconds": round(time.monotonic() - t0, 1), "ctx": ctx}}
