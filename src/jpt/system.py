"""Describe the machine the benchmark runs on."""

from __future__ import annotations

import platform
import shutil
import subprocess


def _sysctl(name: str) -> str | None:
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def chip() -> str:
    return _sysctl("machdep.cpu.brand_string") or platform.processor() or "unknown"


def memory_gb() -> float | None:
    raw = _sysctl("hw.memsize")
    return round(int(raw) / 2**30, 1) if raw and raw.isdigit() else None


def os_version() -> str:
    mac = platform.mac_ver()[0]
    return f"macOS {mac}" if mac else platform.platform()


def tool_version(cmd: list[str]) -> str | None:
    if not shutil.which(cmd[0]):
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        text = (out.stdout or out.stderr).strip().splitlines()
        return text[0][:120] if text else None
    except (OSError, subprocess.SubprocessError):
        return None


def describe() -> dict:
    return {
        "chip": chip(),
        "memory_gb": memory_gb(),
        "os": os_version(),
        "python": platform.python_version(),
        "llama_server": tool_version(["llama-server", "--version"]),
        "mlx_lm": _mlx_version(),
    }


def _mlx_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("mlx-lm")
    except Exception:
        return None
