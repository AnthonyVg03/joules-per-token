"""Memory used by the model.

On Apple Silicon, llama.cpp and MLX put the weights (and the KV cache) in GPU
buffers. macOS counts that memory as **wired** system memory and does *not*
attribute it to the process: Activity Monitor shows ~200 MB for a server
holding a 3 GB model. Measured on an M5 with Qwen2.5-1.5B F16: wired memory
went from 2.2 GB to 5.4 GB at load time (+3.2 GB = the model file).

So the model's memory is measured as:
    (peak wired memory − wired memory before the server started)
  + peak process footprint (CPU-side buffers, ``ri_phys_footprint``)

System-wide wired memory also moves with other apps: close them for clean numbers.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import re
import subprocess
import threading
import time


class _RusageInfoV2(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (n, ctypes.c_uint64)
        for n in (
            "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups", "ri_pageins",
            "ri_wired_size", "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime",
            "ri_proc_exit_abstime", "ri_child_user_time", "ri_child_system_time",
            "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups", "ri_child_pageins",
            "ri_child_elapsed_abstime", "ri_diskio_bytesread", "ri_diskio_byteswritten",
        )
    ]


_libc = None


def footprint_bytes(pid: int) -> int | None:
    """Process footprint (what Activity Monitor shows), GPU weights excluded."""
    global _libc
    try:
        if _libc is None:
            _libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.dylib")
            _libc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(_RusageInfoV2)]
            _libc.proc_pid_rusage.restype = ctypes.c_int
        info = _RusageInfoV2()
        if _libc.proc_pid_rusage(pid, 2, ctypes.byref(info)) == 0:
            return int(info.ri_phys_footprint)
    except (OSError, AttributeError):
        pass
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=5).stdout
        return int(out.strip()) * 1024 if out.strip() else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def parse_vm_stat(text: str) -> int | None:
    """Wired memory in bytes from ``vm_stat`` output."""
    page = re.search(r"page size of (\d+) bytes", text)
    wired = re.search(r"Pages wired down:\s+(\d+)", text)
    if not (page and wired):
        return None
    return int(wired.group(1)) * int(page.group(1))


def wired_bytes() -> int | None:
    try:
        return parse_vm_stat(subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout)
    except (OSError, subprocess.SubprocessError):
        return None


class PeakMemory:
    """Context manager sampling footprint and wired memory in the background."""

    def __init__(self, pid: int | None, baseline_wired: int | None = None, interval: float = 0.25) -> None:
        self.pid, self.baseline_wired, self.interval = pid, baseline_wired, interval
        self.peak_footprint: int | None = None
        self.peak_wired: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        f = footprint_bytes(self.pid) if self.pid else None
        if f is not None and (self.peak_footprint is None or f > self.peak_footprint):
            self.peak_footprint = f
        if self.baseline_wired is not None:
            w = wired_bytes()
            if w is not None and (self.peak_wired is None or w > self.peak_wired):
                self.peak_wired = w

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            time.sleep(self.interval)

    def __enter__(self) -> "PeakMemory":
        if self.pid:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    @property
    def peak_mb(self) -> float | None:
        """Model memory = wired increase since before the server started + process footprint."""
        if self.peak_footprint is None and self.peak_wired is None:
            return None
        gpu = max(0, (self.peak_wired or 0) - (self.baseline_wired or 0)) if self.peak_wired else 0
        return round(((self.peak_footprint or 0) + gpu) / 2**20, 1)
