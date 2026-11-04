"""Small statistics helpers (standard library only)."""

from __future__ import annotations

import math
import statistics
from typing import Iterable, Sequence


def median(values: Iterable[float]) -> float | None:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return statistics.median(vals) if vals else None


def cv(values: Iterable[float]) -> float | None:
    """Coefficient of variation (stdev / mean). None if fewer than 2 values."""
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if len(vals) < 2:
        return None
    m = statistics.fmean(vals)
    if m == 0:
        return None
    return statistics.stdev(vals) / m


def stability_label(c: float | None) -> str:
    """Same thresholds as asiai: <5% stable, <10% variable, else unstable."""
    if c is None:
        return "n/a"
    if c < 0.05:
        return "stable"
    if c < 0.10:
        return "variable"
    return "unstable"


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (better than p ± 1.96·se for small n)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def trapezoid_energy(samples: Sequence[tuple[float, float]]) -> float:
    """Integrate power samples [(t_seconds, watts), ...] into joules."""
    e = 0.0
    for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
        e += (t1 - t0) * (p0 + p1) / 2
    return e
