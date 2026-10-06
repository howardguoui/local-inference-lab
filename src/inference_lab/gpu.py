"""GPU telemetry through NVML (the library behind nvidia-smi).

`GpuSampler` polls in a background thread while a benchmark runs and reports peak
VRAM, mean utilization and peak power. Without an NVIDIA driver it degrades to
"no data" instead of failing, so benchmarks still run on any machine.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass

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

    def as_dict(self) -> dict:
        return asdict(self)


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
        self.available = False

    def start(self) -> GpuSampler:
        try:
            self.nv = _nvml(self._nvml_arg)
            self.nv.nvmlInit()
            self.handle = self.nv.nvmlDeviceGetHandleByIndex(self.index)
            self.available = True
        except Exception:
            self.available = False
            return self
        self._thread = threading.Thread(target=self._run, name="gpu-sampler", daemon=True)
        self._thread.start()
        return self

    def _sample(self) -> None:
        nv, h = self.nv, self.handle
        self.mem.append(nv.nvmlDeviceGetMemoryInfo(h).used)
        self.util.append(nv.nvmlDeviceGetUtilizationRates(h).gpu)
        try:
            self.power.append(nv.nvmlDeviceGetPowerUsage(h) / 1000)
        except Exception:
            pass

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
        try:
            self.nv.nvmlShutdown()
        except Exception:
            pass
        return GpuStats(
            samples=len(self.mem),
            peak_memory_gib=round(max(self.mem) / GIB, 2) if self.mem else None,
            mean_utilization_pct=round(sum(self.util) / len(self.util), 1) if self.util else None,
            peak_power_w=round(max(self.power), 1) if self.power else None,
        )


def default_gpu_gib(which: str, fallback: float = 16.0, nvml=None) -> tuple[float, str]:
    """VRAM to plan against: "total" (vLLM sizes from total memory) or "free" (llama.cpp
    must fit next to whatever else is running, such as a Windows desktop)."""
    snap = gpu_snapshot(nvml=nvml)
    if not snap:
        return fallback, f"{fallback} GiB (no NVML; pass --gpu-gib)"
    key = "memory_total_gib" if which == "total" else "memory_free_gib"
    return snap[key], f"{snap[key]} GiB {which} on {snap['name']} (NVML)"
