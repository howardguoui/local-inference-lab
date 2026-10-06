from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.fake_server import ServerThread, build_app


@pytest.fixture(scope="session")
def server():
    with ServerThread(build_app()) as s:
        yield s


class FakeNvml:
    """Enough of pynvml for the sampler: memory climbs 1 GiB per reading up to 12 GiB."""

    NVML_TEMPERATURE_GPU = 0

    def __init__(self, fail_init: bool = False):
        self.fail_init = fail_init
        self.reads = 0

    def nvmlInit(self):
        if self.fail_init:
            raise RuntimeError("NVML Shared Library Not Found")

    def nvmlShutdown(self):
        pass

    def nvmlDeviceGetHandleByIndex(self, i):
        return i

    def nvmlDeviceGetMemoryInfo(self, h):
        self.reads += 1
        used = min(12, 4 + self.reads) * 1024**3
        return SimpleNamespace(total=16 * 1024**3, used=used, free=16 * 1024**3 - used)

    def nvmlDeviceGetUtilizationRates(self, h):
        return SimpleNamespace(gpu=90, memory=60)

    def nvmlDeviceGetPowerUsage(self, h):
        return 250_000  # milliwatts

    def nvmlDeviceGetEnforcedPowerLimit(self, h):
        return 300_000

    def nvmlDeviceGetTemperature(self, h, sensor):
        return 64

    def nvmlDeviceGetName(self, h):
        return b"NVIDIA GeForce RTX 5070 Ti"

    def nvmlSystemGetDriverVersion(self):
        return "580.00"


@pytest.fixture
def nvml():
    return FakeNvml()
