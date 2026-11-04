"""Run the whole study: every variant × (speed + energy, GSM8K, perplexity).

Resumable: a variant/task already stored for this session is skipped, so an
interrupted sweep restarts where it stopped.
"""

from __future__ import annotations

import time
from pathlib import Path

from . import bench, quality, system, thermal
from .engine import llama_server, mlx_server
from .models import Config, Variant, fetch
from .power import PowerMeter
from .store import Store


class Paths:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.models = root / "models"
        self.data = root / "data"
        self.results = root / "results"
        self.logs = root / "results" / "logs"
        self.db = root / "results" / "jpt.db"


def server_for(v: Variant, paths: Paths, ctx: int):
    log = paths.logs / f"{v.id.replace('/', '__')}.log"
    if v.engine == "llama.cpp":
        return llama_server(v.local_path(paths.models), log, ctx=ctx)
    if v.engine == "mlx":
        return mlx_server(v.repo, log)
    raise ValueError(v.engine)


def _context(v: Variant, sysinfo: dict, session: str, bpw: float | None) -> dict:
    return {"session": session, "chip": sysinfo["chip"], "os": sysinfo["os"], "engine": v.engine,
            "engine_version": sysinfo["llama_server"] if v.engine == "llama.cpp" else sysinfo["mlx_lm"],
            "model": v.model, "quant": v.quant, "bits": v.bits, "bpw": bpw, "variant": v.id}


def bits_per_weight(meta: dict) -> float | None:
    size, n = meta.get("size"), meta.get("n_params")
    return round(size * 8 / n, 2) if size and n else None


def run_variant(v: Variant, cfg: Config, store: Store, meter: PowerMeter, paths: Paths, *, session: str,
                tasks: set[str], download: bool, log=print) -> None:
    s = cfg.settings
    if not v.is_downloaded(paths.models):
        if not download:
            log(f"[{v.id}] model not downloaded — run `jpt fetch` or add --download")
            return
        fetch(v, paths.models)
    sysinfo = system.describe()
    need_bench = "bench" in tasks and not store.has_runs(session, v.id)
    need_gsm = "gsm8k" in tasks and not store.has_quality(session, v.id, "gsm8k")
    need_ppl = "perplexity" in tasks and v.engine == "llama.cpp" and not store.has_quality(session, v.id, "perplexity")
    if not (need_bench or need_gsm or need_ppl):
        log(f"[{v.id}] already measured, skipped")
        return

    log(f"\n=== {v.id} ===")
    wait_cool(s.cool_timeout_seconds, log)
    bpw = None
    if need_bench or need_gsm:
        srv = server_for(v, paths, s.ctx)
        with srv:
            bpw = bits_per_weight(srv.client.meta())
            ctx = _context(v, sysinfo, session, bpw)
            log(f"  idle power ({s.idle_seconds:.0f} s, once the machine is quiet)…")
            idle = meter.wait_quiet(max_cpu_w=s.quiet_cpu_w, window_s=s.idle_seconds, log=log)
            idle_w = idle.avg_power_w if idle else None
            if need_bench:
                for row in bench.run(srv.client, meter, runs=s.runs, max_tokens=s.max_tokens,
                                     idle_power_w=idle_w, pid=srv.pid,
                                     baseline_wired=srv.baseline_wired, trial_pause=s.trial_pause_seconds,
                                     log=log):
                    store.add_run(**ctx, **row)
            if need_gsm:
                res = quality.run_gsm8k(srv.client, paths.data, n=s.gsm8k_n, max_tokens=s.gsm8k_max_tokens,
                                        meter=meter, log=log)
                details = res.pop("details")
                store.add_quality(details=details, **_qctx(ctx), **res)
                log(f"  GSM8K: {res['score']:.1%} ({res['ci_low']:.1%}–{res['ci_high']:.1%})")
    if need_ppl:
        log("  perplexity…")
        res = quality.run_perplexity(v.local_path(paths.models), paths.data, chunks=s.perplexity_chunks,
                                     ctx=s.perplexity_ctx, log_path=paths.logs / f"{v.id.replace('/', '__')}__ppl.log")
        details = res.pop("details")
        store.add_quality(details=details, **_qctx(_context(v, sysinfo, session, bpw)), **res)
        log(f"  perplexity: {res['score']:.3f}")
    if s.cooldown_seconds:
        log(f"  cool-down {s.cooldown_seconds:.0f} s…")
        time.sleep(s.cooldown_seconds)


def wait_cool(timeout_s: float, log=print) -> None:
    """Start every variant from a cool chip: wait until the thermal state is back to nominal."""
    t0 = time.monotonic()
    while (state := thermal.thermal_state()) not in (None, 0):
        if time.monotonic() - t0 > timeout_s:
            log(f"  warning: still hot after {timeout_s:.0f} s ({thermal.label(state)}), measuring anyway")
            return
        log(f"  chip is warm ({thermal.label(state)}), waiting for it to cool down…")
        time.sleep(30)


def _qctx(ctx: dict) -> dict:
    keys = ("session", "engine", "model", "quant", "bits", "variant")
    return {k: ctx[k] for k in keys}


def run(cfg: Config, store: Store, meter: PowerMeter, paths: Paths, *, session: str, only: str | None,
        tasks: set[str], download: bool, log=print) -> None:
    variants = cfg.select(only)
    log(f"{len(variants)} variant(s) — session \"{session}\" — energy backend: {meter.name}")
    for v in variants:
        try:
            run_variant(v, cfg, store, meter, paths, session=session, tasks=tasks, download=download, log=log)
        except KeyboardInterrupt:
            raise
        except Exception as e:  # one broken variant must not kill a multi-hour sweep
            log(f"[{v.id}] FAILED: {e}")
