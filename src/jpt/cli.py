"""Command line: ``jpt info | list | fetch | bench | quality | sweep | thermal | report``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, bench, report, system, thermal
from .models import disk_free_gb, fetch, load_config
from .power import get_meter
from .store import Store
from .sweep import Paths, run, server_for

DEFAULT_CONFIG = "configs/sweep.toml"


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--root", type=Path, default=Path.cwd(), help="project directory (default: current directory)")
    p.add_argument("--config", type=Path, default=None, help=f"TOML config (default: {DEFAULT_CONFIG})")
    p.add_argument("--session", default="main", help="name of the measurement series (default: main)")


def _cfg(a):
    return load_config(a.config or a.root / DEFAULT_CONFIG)


def cmd_info(a) -> None:
    info = system.describe()
    for k, v in info.items():
        print(f"{k:14s} {v}")
    meter = get_meter(a.power)
    print(f"{'energy':14s} backend = {meter.name}")
    if meter.available:
        e = meter.measure_idle(2.0)
        parts = "  ".join(f"{c}={w / e.seconds:.2f} W" for c, w in e.joules.items())
        print(f"{'idle power':14s} {e.avg_power_w:.2f} W  ({parts})")
    print(f"{'thermal':14s} {thermal.label(thermal.thermal_state())}")
    print(f"{'free disk':14s} {disk_free_gb(a.root / 'models'):.0f} GB")


def cmd_list(a) -> None:
    cfg = _cfg(a)
    for v in cfg.variants:
        mark = "✓" if v.is_downloaded(a.root / "models") else " "
        print(f"[{mark}] {v.id:40s} {v.repo}")


def cmd_fetch(a) -> None:
    cfg = _cfg(a)
    for v in cfg.select(a.variant):
        if not v.is_downloaded(a.root / "models") or v.engine == "mlx":
            fetch(v, a.root / "models")
    print("Downloads complete.")


def _run(a, tasks: set[str]) -> None:
    cfg = _cfg(a)
    if getattr(a, "runs", None):
        cfg.settings.runs = a.runs
    if getattr(a, "n", None):
        cfg.settings.gsm8k_n = a.n
    paths = Paths(a.root)
    store = Store(paths.db)
    meter = get_meter(a.power)
    try:
        run(cfg, store, meter, paths, session=a.session, only=a.variant, tasks=tasks, download=a.download)
    finally:
        meter.close()
        store.close()
    print(f"\nDone. Report: jpt report --session {a.session}")


def cmd_thermal(a) -> None:
    cfg = _cfg(a)
    variants = cfg.select(a.variant)
    if len(variants) != 1:
        sys.exit(f"--variant must match exactly one variant ({len(variants)} matched). See `jpt list`.")
    v = variants[0]
    paths = Paths(a.root)
    if not v.is_downloaded(paths.models):
        sys.exit("Model not downloaded: run `jpt fetch --variant ...` first.")
    store, meter = Store(paths.db), get_meter(a.power)
    print(f"Sustained load: {a.minutes:.0f} min on {v.id} — thermal state: {thermal.label(thermal.thermal_state())}")
    try:
        with server_for(v, paths, cfg.settings.ctx) as srv:
            for row in bench.run_thermal(srv.client, meter, minutes=a.minutes, max_tokens=a.max_tokens):
                store.add_thermal(session=a.session, variant=v.id, **row)
    finally:
        meter.close()
        store.close()


def cmd_report(a) -> None:
    paths = Paths(a.root)
    store = Store(paths.db)
    path = report.build(store, a.session, paths.results / a.session)
    store.close()
    print(f"Report written: {path}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="jpt", description="What quantization really costs: speed, memory, quality and energy of local LLMs on Apple Silicon.")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_, variant=True, power=True):
        sp = sub.add_parser(name, help=help_)
        _common(sp)
        if variant:
            sp.add_argument("--variant", help="filter on the variant id (e.g. 'qwen2.5-1.5b/llama.cpp/Q4_K_M' or 'Q4')")
        if power:
            sp.add_argument("--power", default="auto", choices=["auto", "ioreport", "powermetrics", "none"])
        sp.set_defaults(fn=fn)
        return sp

    add("info", cmd_info, "machine, energy backend, idle power, thermal state", variant=False)
    add("list", cmd_list, "list the variants in the config", power=False)
    add("fetch", cmd_fetch, "download the models", power=False)
    for name, tasks, help_ in (("bench", {"bench"}, "speed, memory, energy"),
                               ("quality", {"gsm8k", "perplexity"}, "GSM8K accuracy + perplexity"),
                               ("sweep", {"bench", "gsm8k", "perplexity"}, "the full study (bench + quality)")):
        sp = add(name, lambda a, t=tasks: _run(a, t), help_)
        sp.add_argument("--download", action="store_true", help="download missing models")
        sp.add_argument("--runs", type=int, help="trials per prompt (overrides the config)")
        sp.add_argument("--n", type=int, help="number of GSM8K questions (overrides the config)")
    sp = add("thermal", cmd_thermal, "sustained load: thermal throttling")
    sp.add_argument("--minutes", type=float, default=10)
    sp.add_argument("--max-tokens", type=int, default=512)
    add("report", cmd_report, "tables, charts, Pareto frontier", variant=False, power=False)

    sys.stdout.reconfigure(line_buffering=True)  # progress lines appear immediately, even in a log file
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
