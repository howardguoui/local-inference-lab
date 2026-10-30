"""GPU telemetry through NVML (the library behind nvidia-smi).

`GpuSampler` polls in a background thread while a benchmark runs and reports peak
VRAM, mean utilization, peak and mean power, and the energy the board used. Without an
NVIDIA driver it degrades to "no data" instead of failing, so benchmarks still run on any
machine.

Energy comes from NVML's total energy counter (millijoules since the driver loaded) when the
GPU has one, else from integrating the power samples over time (trapezoid rule). Both are the
whole board's draw, idle baseline included, not just the inference server's share.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from itertools import pairwise

GIB = 1024**3


def _nvml(nvml=None):
    if nvml is not None:
        return nvml
    import pynvml  # from nvidia-ml-py

    return pynvml


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def gpu_snapshot(index: int = 0, nvml=None) -> dict | None:
    """One reading of the GPU: name, memory, utilization, temperature, power. None without NVML."""
    try:
        nv = _nvml(nvml)
        nv.nvmlInit()
    except Exception:
        return None
    try:
        h = nv.nvmlDeviceGetHandleByIndex(index)
        mem = nv.nvmlDeviceGetMemoryInfo(h)
        util = nv.nvmlDeviceGetUtilizationRates(h)
        out = {
            "name": _text(nv.nvmlDeviceGetName(h)),
            "driver": _text(nv.nvmlSystemGetDriverVersion()),
            "memory_total_gib": round(mem.total / GIB, 2),
            "memory_used_gib": round(mem.used / GIB, 2),
            "memory_free_gib": round(mem.free / GIB, 2),
            "utilization_pct": util.gpu,
        }
        for key, fn, scale in (
            ("temperature_c", lambda: nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU), 1),
            ("power_w", lambda: nv.nvmlDeviceGetPowerUsage(h), 1000),
            ("power_limit_w", lambda: nv.nvmlDeviceGetEnforcedPowerLimit(h), 1000),
        ):
            try:
                out[key] = round(fn() / scale, 1)
            except Exception:
                pass  # not every GPU or driver reports every field
        return out
    finally:
        try:
            nv.nvmlShutdown()
        except Exception:
            pass


@dataclass
class GpuStats:
    samples: int
    peak_memory_gib: float | None = None
    mean_utilization_pct: float | None = None
    peak_power_w: float | None = None
    mean_power_w: float | None = None
    energy_j: float | None = None
    energy_source: str | None = None  # "nvml_counter" or "power_samples"

    def as_dict(self) -> dict:
        return asdict(self)


def energy_from_samples(times_s: list[float], watts: list[float]) -> float | None:
    """Joules under a power curve: trapezoid rule over (time, watts) samples. None with fewer than two."""
    if len(times_s) < 2 or len(times_s) != len(watts):
        return None
    points = list(zip(times_s, watts, strict=True))
    return sum((t1 - t0) * (w0 + w1) / 2 for (t0, w0), (t1, w1) in pairwise(points))


class GpuSampler:
    """Samples memory, utilization and power every `interval` seconds between start() and stop()."""

    def __init__(self, index: int = 0, interval: float = 0.25, nvml=None):
        self.index = index
        self.interval = interval
        self._nvml_arg = nvml
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.mem: list[int] = []
        self.util: list[int] = []
        self.power: list[float] = []
        self.power_t: list[float] = []  # time.monotonic() of each power sample
        self.available = False
        self._t0 = self._t1 = 0.0
        self._energy0_mj: int | None = None
        self._energy1_mj: int | None = None

    def _energy_mj(self) -> int | None:
        try:
            return int(self.nv.nvmlDeviceGetTotalEnergyConsumption(self.handle))
        except Exception:
            return None  # older GPUs and some drivers have no energy counter

    def start(self) -> GpuSampler:
        try:
            self.nv = _nvml(self._nvml_arg)
            self.nv.nvmlInit()
            self.handle = self.nv.nvmlDeviceGetHandleByIndex(self.index)
            self.available = True
        except Exception:
            self.available = False
            return self
        self._energy0_mj = self._energy_mj()
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="gpu-sampler", daemon=True)
        self._thread.start()
        return self

    def _sample(self) -> None:
        nv, h = self.nv, self.handle
        self.mem.append(nv.nvmlDeviceGetMemoryInfo(h).used)
        self.util.append(nv.nvmlDeviceGetUtilizationRates(h).gpu)
        try:
            watts = nv.nvmlDeviceGetPowerUsage(h) / 1000
        except Exception:
            return
        self.power.append(watts)
        self.power_t.append(time.monotonic())

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample()
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self) -> GpuStats:
        if not self.available:
            return GpuStats(samples=0)
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        try:
            self._sample()  # one final reading so short runs still report
        except Exception:
            pass
        self._t1 = time.monotonic()
        self._energy1_mj = self._energy_mj()
        try:
            self.nv.nvmlShutdown()
        except Exception:
            pass
        energy, source, span = self._energy()
        return GpuStats(
            samples=len(self.mem),
            peak_memory_gib=round(max(self.mem) / GIB, 2) if self.mem else None,
            mean_utilization_pct=round(sum(self.util) / len(self.util), 1) if self.util else None,
            peak_power_w=round(max(self.power), 1) if self.power else None,
            mean_power_w=round(energy / span, 1) if energy is not None and span > 0 else None,
            energy_j=round(energy, 1) if energy is not None else None,
            energy_source=source,
        )

    def _energy(self) -> tuple[float | None, str | None, float]:
        """(joules, source, seconds covered): the hardware counter if it advanced, else the power samples."""
        e0, e1 = self._energy0_mj, self._energy1_mj
        if e0 is not None and e1 is not None and e1 > e0:
            return (e1 - e0) / 1000, "nvml_counter", self._t1 - self._t0
        energy = energy_from_samples(self.power_t, self.power)
        if energy is None:
            return None, None, 0.0
        return energy, "power_samples", self.power_t[-1] - self.power_t[0]


def default_gpu_gib(which: str, fallback: float = 16.0, nvml=None) -> tuple[float, str]:
    """VRAM to plan against: "total" (vLLM sizes from total memory) or "free" (llama.cpp
    must fit next to whatever else is running, such as a Windows desktop)."""
    snap = gpu_snapshot(nvml=nvml)
    if not snap:
        return fallback, f"{fallback} GiB (no NVML; pass --gpu-gib)"
    key = "memory_total_gib" if which == "total" else "memory_free_gib"
    return snap[key], f"{snap[key]} GiB {which} on {snap['name']} (NVML)"
