"""Speed + energy + memory benchmark of one running server.

Method (inspired by asiai):
* one unmeasured warm-up generation (loads weights in memory, compiles kernels);
* ``runs`` trials per prompt, greedy decoding (temperature 0);
* the median is the headline number, the coefficient of variation tells how
  much to trust it;
* energy counters are read just before and just after each request.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from . import thermal
from .engine import Generation
from .memory import PeakMemory
from .power import PowerMeter

# An original, fixed text used to build a long prompt (prefill measurement).
_FILLER = (
    "The river town woke slowly. Fishermen checked their nets, the baker opened the shutters, "
    "and the schoolteacher wrote the date on the blackboard. A delivery van stopped near the bridge, "
    "its driver unloading crates of apples, flour and lamp oil. Children ran along the quay, "
    "counting the boats, arguing about which one was the fastest and which one had the oldest sail. "
)


@dataclass(frozen=True)
class Prompt:
    id: str
    text: str
    max_tokens: int | None = None  # None -> settings.max_tokens


PROMPTS = [
    Prompt("short", "Write a detailed explanation of how a refrigerator works, step by step."),
    Prompt("code", "Write a Python function that parses a CSV file and returns the average of each numeric column. "
                   "Include docstrings and handle errors."),
    Prompt("long_ctx", "Read the following text, then summarise it in three sentences.\n\n" + _FILLER * 12 + "\nSummary:",
           max_tokens=64),
]


def _measure(client, prompt: str, max_tokens: int, meter: PowerMeter, pid: int | None,
             baseline_wired: int | None = None):
    a = meter.snapshot()
    with PeakMemory(pid, baseline_wired) as mem:
        gen: Generation = client.generate(prompt, max_tokens)
    b = meter.snapshot()
    energy = meter.energy_between(a, b)
    for s in (a, b):
        if hasattr(meter, "release"):
            meter.release(s)
    return gen, energy, mem.peak_mb


def run(client, meter: PowerMeter, *, runs: int, max_tokens: int, idle_power_w: float | None,
        pid: int | None = None, baseline_wired: int | None = None, prompts: list[Prompt] = PROMPTS,
        trial_pause: float = 0.0, log=print) -> list[dict]:
    """Benchmark a ready server. Returns one dict per trial (to be stored)."""
    log("  warm-up…")
    client.generate("Hello, my name is", 16)
    rows = []
    for p in prompts:
        n_tok = p.max_tokens or max_tokens
        for trial in range(1, runs + 1):
            if trial_pause:
                time.sleep(trial_pause)
            # A unique first line defeats any prompt cache, so every trial pays the full prefill.
            text = f"[{uuid.uuid4().hex[:8]}]\n{p.text}"
            gen, energy, peak = _measure(client, text, n_tok, meter, pid, baseline_wired)
            row = {
                "prompt_id": p.id, "trial": trial,
                "prompt_tokens": gen.prompt_tokens, "gen_tokens": gen.gen_tokens,
                "ttft_ms": round(gen.ttft_s * 1000, 1), "prefill_tps": gen.prefill_tps, "decode_tps": gen.decode_tps,
                "wall_s": gen.wall_s, "timing_source": gen.timing_source,
                "peak_mem_mb": peak, "thermal_state": thermal.thermal_state(), "power_backend": meter.name,
                "idle_power_w": idle_power_w,
            }
            if energy is not None:
                dyn = energy.total_j - (idle_power_w or 0) * energy.seconds
                row.update(
                    energy_j=energy.total_j, energy_dyn_j=dyn, avg_power_w=energy.avg_power_w,
                    cpu_j=energy.joules.get("cpu"), gpu_j=energy.joules.get("gpu"),
                    ane_j=energy.joules.get("ane"), dram_j=energy.joules.get("dram"),
                    j_per_token=energy.total_j / gen.gen_tokens if gen.gen_tokens else None,
                    j_per_token_dyn=dyn / gen.gen_tokens if gen.gen_tokens else None,
                )
            rows.append(row)
            dec = f"{gen.decode_tps:.1f}" if gen.decode_tps else "—"
            jpt = f"{row['j_per_token']:.3f} J/tok" if row.get("j_per_token") else ""
            log(f"  {p.id:8s} trial {trial}: {dec} tok/s  TTFT {row['ttft_ms']:.0f} ms  {jpt}")
    return rows


def run_thermal(client, meter: PowerMeter, *, minutes: float, max_tokens: int = 512, log=print) -> list[dict]:
    """Generate non-stop for ``minutes`` and record how speed and power evolve."""
    import time

    rows, t0 = [], time.monotonic()
    prompt = PROMPTS[0].text
    while (elapsed := time.monotonic() - t0) < minutes * 60:
        gen, energy, _ = _measure(client, f"[{uuid.uuid4().hex[:8]}]\n{prompt}", max_tokens, meter, None)
        state = thermal.thermal_state()
        row = {"t_s": round(elapsed, 1), "decode_tps": gen.decode_tps,
               "avg_power_w": energy.avg_power_w if energy else None, "thermal_state": state}
        rows.append(row)
        log(f"  t={elapsed / 60:5.1f} min  {gen.decode_tps or 0:6.1f} tok/s  "
            f"{(row['avg_power_w'] or 0):5.1f} W  thermique={thermal.label(state)}")
    return rows
