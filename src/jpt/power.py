"""Measure SoC energy on Apple Silicon.

Two backends:

* ``ioreport`` (default) — reads the "Energy Model" counters of Apple's private
  IOReport library through ctypes. No sudo needed. The counters are cumulative
  energy per block (CPU, GPU, ANE, DRAM…), so the energy of a request is simply
  ``counters(after) - counters(before)``: no sampling, no integration error.
* ``powermetrics`` — fallback that runs ``sudo -n powermetrics`` in the
  background and integrates the power samples over time. Needs passwordless sudo.

Both measure the whole chip, not just the model process: close other apps, and
use the idle baseline (``measure_idle``) to compute the *dynamic* energy.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import plistlib
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field

from .stats import trapezoid_energy

COMPONENTS = ("cpu", "gpu", "ane", "dram")


@dataclass
class Energy:
    seconds: float
    joules: dict[str, float] = field(default_factory=dict)  # per component

    @property
    def total_j(self) -> float:
        return sum(self.joules.values())

    @property
    def avg_power_w(self) -> float:
        return self.total_j / self.seconds if self.seconds > 0 else 0.0


class PowerMeter:
    name = "none"
    available = False

    def snapshot(self):  # opaque marker
        return time.monotonic()

    def energy_between(self, a, b) -> Energy | None:
        return None

    def measure_idle(self, seconds: float = 5.0) -> Energy | None:
        a = self.snapshot()
        time.sleep(seconds)
        return self.energy_between(a, self.snapshot())

    def wait_quiet(self, max_cpu_w: float = 1.5, window_s: float = 3.0, timeout_s: float = 180.0,
                   log=print) -> Energy | None:
        """Wait until background activity has settled, then return that quiet idle measurement.

        Why: right after a multi-GB download, macOS (Spotlight, XProtect…) scans the new file
        and the CPU draws several watts for a minute or two. Measuring then would charge that
        noise to the model. We re-measure the idle CPU power until it drops below ``max_cpu_w``.
        """
        t0 = time.monotonic()
        while True:
            e = self.measure_idle(window_s)
            if e is None:
                return None
            cpu_w = e.joules.get("cpu", 0.0) / e.seconds if e.seconds else 0.0
            if cpu_w <= max_cpu_w:
                return e
            if time.monotonic() - t0 > timeout_s:
                log(f"  warning: machine still busy after {timeout_s:.0f} s (CPU {cpu_w:.1f} W), measuring anyway")
                return e
            log(f"  background activity (CPU {cpu_w:.1f} W), waiting…")

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- IOReport

_UNITS = {"mJ": 1e-3, "uJ": 1e-6, "µJ": 1e-6, "nJ": 1e-9, "J": 1.0}


def _component(name: str) -> str | None:
    """Map an IOReport 'Energy Model' channel name to a component.

    Checked on an M5: 'CPU Energy' (mJ) is the CPU total, 'GPU Energy' (nJ) the
    precise GPU total, 'DRAM' (mJ) the memory. ANE channels start with 'ANE'.
    Per-core channels (PCPU0, ECPU…) are already included in 'CPU Energy'.
    """
    if name == "CPU Energy":
        return "cpu"
    if name == "GPU Energy":
        return "gpu"
    if name.startswith("ANE"):
        return "ane"
    if name == "DRAM":
        return "dram"
    return None


class IOReportMeter(PowerMeter):
    name = "ioreport"

    def __init__(self) -> None:
        vp = ctypes.c_void_p
        cf = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreFoundation"))
        ior = ctypes.cdll.LoadLibrary("/usr/lib/libIOReport.dylib")
        cf.CFStringCreateWithCString.restype = vp
        cf.CFStringCreateWithCString.argtypes = [vp, ctypes.c_char_p, ctypes.c_uint32]
        cf.CFStringGetCString.restype = ctypes.c_bool
        cf.CFStringGetCString.argtypes = [vp, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
        cf.CFDictionaryGetValue.restype = vp
        cf.CFDictionaryGetValue.argtypes = [vp, vp]
        cf.CFArrayGetCount.restype = ctypes.c_long
        cf.CFArrayGetCount.argtypes = [vp]
        cf.CFArrayGetValueAtIndex.restype = vp
        cf.CFArrayGetValueAtIndex.argtypes = [vp, ctypes.c_long]
        cf.CFRelease.argtypes = [vp]
        ior.IOReportCopyChannelsInGroup.restype = vp
        ior.IOReportCopyChannelsInGroup.argtypes = [vp, vp, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        ior.IOReportCreateSubscription.restype = vp
        ior.IOReportCreateSubscription.argtypes = [vp, vp, ctypes.POINTER(vp), ctypes.c_uint64, vp]
        ior.IOReportCreateSamples.restype = vp
        ior.IOReportCreateSamples.argtypes = [vp, vp, vp]
        ior.IOReportCreateSamplesDelta.restype = vp
        ior.IOReportCreateSamplesDelta.argtypes = [vp, vp, vp]
        for fn in ("IOReportChannelGetChannelName", "IOReportChannelGetUnitLabel"):
            getattr(ior, fn).restype = vp
            getattr(ior, fn).argtypes = [vp]
        ior.IOReportSimpleGetIntegerValue.restype = ctypes.c_int64
        ior.IOReportSimpleGetIntegerValue.argtypes = [vp, ctypes.c_int32]
        self._cf, self._ior = cf, ior

        channels = ior.IOReportCopyChannelsInGroup(self._s("Energy Model"), None, 0, 0, 0)
        if not channels:
            raise RuntimeError("IOReport 'Energy Model' group not found")
        self._sub_channels = vp()
        self._sub = ior.IOReportCreateSubscription(None, channels, ctypes.byref(self._sub_channels), 0, None)
        if not self._sub:
            raise RuntimeError("IOReport subscription failed")
        self._key_channels = self._s("IOReportChannels")
        self.available = True

    def _s(self, text: str):
        return self._cf.CFStringCreateWithCString(None, text.encode(), 0x08000100)

    def _str(self, ref) -> str:
        if not ref:
            return ""
        buf = ctypes.create_string_buffer(256)
        self._cf.CFStringGetCString(ref, buf, 256, 0x08000100)
        return buf.value.decode(errors="replace")

    def snapshot(self):
        return (time.monotonic(), self._ior.IOReportCreateSamples(self._sub, self._sub_channels, None))

    def energy_between(self, a, b) -> Energy:
        (t0, s0), (t1, s1) = a, b
        delta = self._ior.IOReportCreateSamplesDelta(s0, s1, None)
        joules = dict.fromkeys(COMPONENTS, 0.0)
        try:
            arr = self._cf.CFDictionaryGetValue(delta, self._key_channels)
            for i in range(self._cf.CFArrayGetCount(arr)):
                item = self._cf.CFArrayGetValueAtIndex(arr, i)
                comp = _component(self._str(self._ior.IOReportChannelGetChannelName(item)))
                if comp is None:
                    continue
                unit = _UNITS.get(self._str(self._ior.IOReportChannelGetUnitLabel(item)), 1e-3)
                joules[comp] += self._ior.IOReportSimpleGetIntegerValue(item, 0) * unit
        finally:
            self._cf.CFRelease(delta)
        return Energy(seconds=t1 - t0, joules=joules)

    def release(self, snap) -> None:
        if snap and snap[1]:
            self._cf.CFRelease(snap[1])


# --------------------------------------------------------------------------- powermetrics


class PowermetricsMeter(PowerMeter):
    """Background ``sudo -n powermetrics`` sampler. Requires passwordless sudo."""

    name = "powermetrics"

    def __init__(self, interval_ms: int = 200) -> None:
        if not shutil.which("powermetrics"):
            raise RuntimeError("powermetrics not found")
        self._samples: list[tuple[float, dict[str, float]]] = []
        self._lock = threading.Lock()
        self._proc = subprocess.Popen(
            ["sudo", "-n", "powermetrics", "--samplers", "cpu_power,gpu_power,ane_power", "-i", str(interval_ms), "-f", "plist"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        time.sleep(interval_ms / 1000 * 3)
        if self._proc.poll() is not None:
            err = self._proc.stderr.read().decode(errors="replace") if self._proc.stderr else ""
            raise RuntimeError(f"powermetrics failed (passwordless sudo needed): {err.strip()[:200]}")
        self.available = True

    def _read(self) -> None:
        buf = b""
        assert self._proc.stdout
        for chunk in iter(lambda: self._proc.stdout.read(4096), b""):
            buf += chunk
            while b"\x00" in buf:
                doc, buf = buf.split(b"\x00", 1)
                try:
                    p = plistlib.loads(doc.strip())
                except Exception:
                    continue
                proc = p.get("processor", {})
                watts = {
                    "cpu": proc.get("cpu_power", 0) / 1000,
                    "gpu": proc.get("gpu_power", p.get("gpu", {}).get("gpu_power", 0)) / 1000,
                    "ane": proc.get("ane_power", 0) / 1000,
                    "dram": 0.0,
                }
                with self._lock:
                    self._samples.append((time.monotonic(), watts))

    def energy_between(self, a, b) -> Energy:
        t0, t1 = a, b
        with self._lock:
            window = [(t, w) for t, w in self._samples if t0 <= t <= t1]
        joules = {c: trapezoid_energy([(t, w[c]) for t, w in window]) for c in COMPONENTS}
        return Energy(seconds=t1 - t0, joules=joules)

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()


def get_meter(backend: str = "auto") -> PowerMeter:
    """Return the best available meter. ``backend``: auto | ioreport | powermetrics | none."""
    if backend in ("auto", "ioreport"):
        try:
            return IOReportMeter()
        except Exception:
            if backend == "ioreport":
                raise
    if backend in ("auto", "powermetrics"):
        try:
            return PowermetricsMeter()
        except Exception:
            if backend == "powermetrics":
                raise
    return PowerMeter()
