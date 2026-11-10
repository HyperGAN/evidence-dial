"""One-GPU lease, explicit owned caps and live floors; never signal peers."""
from __future__ import annotations
import contextlib
import ctypes
import fcntl
import os
import signal
import threading
import time
from pathlib import Path

GIB = 1024**3


class Resources:
    def __init__(self, gpu: int, status: Path):
        self.gpu, self.status = gpu, Path(status)
        self.stop = threading.Event()
        self.failure = None
        self.allocator_cap = 22 * GIB
        self.nvml_cap = 24 * GIB
        self.rss_cap = 32 * GIB
        self.gpu_floor = 5 * GIB
        self.host_floor = 47 * GIB

    def sample(self):
        import psutil
        import pynvml as nv
        handle = nv.nvmlDeviceGetHandleByIndex(self.gpu)
        mem = nv.nvmlDeviceGetMemoryInfo(handle)
        own = sum(p.usedGpuMemory for p in nv.nvmlDeviceGetComputeRunningProcesses(handle) if p.pid == os.getpid())
        import torch
        allocated = torch.cuda.memory_reserved(0) if torch.cuda.is_initialized() else 0
        return {"global_gpu_free": mem.free, "own_nvml": own,
            "allocator_reserved": allocated, "rss": psutil.Process().memory_info().rss,
            "host_available": psutil.virtual_memory().available}

    def check(self):
        if self.failure:
            raise RuntimeError(self.failure)
        s = self.sample()
        checks = [(s["global_gpu_free"] >= self.gpu_floor, "GPU free floor"),
                  (s["host_available"] >= self.host_floor, "host available floor"),
                  (s["own_nvml"] <= self.nvml_cap, "own NVML cap"),
                  (s["allocator_reserved"] <= self.allocator_cap, "allocator cap"),
                  (s["rss"] <= self.rss_cap, "own RSS cap")]
        for ok, name in checks:
            if not ok:
                raise RuntimeError(f"Resource refusal: {name}: {s}")
        return s

    def __enter__(self):
        import pynvml as nv
        from .io import write_json
        nv.nvmlInit()
        # Existing host admission locks are shared for bounded coexistence;
        # our private exclusive lease sequences every Evidence Dial GPU job.
        self.stack = contextlib.ExitStack()
        for path, mode in [(f"/mnt/nvme/sliders/qualification/admission-gpu-{self.gpu}.lock", fcntl.LOCK_SH),
                           (f"/mnt/nvme/sliders/locks/gpu-{self.gpu}.lock", fcntl.LOCK_SH),
                           (str(Path(__file__).resolve().parents[2] / "artifacts/evidence-dial-gpu.lock"), fcntl.LOCK_EX)]:
            f = self.stack.enter_context(open(path, "a+"))
            fcntl.flock(f, mode | fcntl.LOCK_NB)
        # Death of the actual exec parent must not leave an orphan GPU worker.
        parent = os.getppid()
        if ctypes.CDLL(None).prctl(1, signal.SIGTERM) != 0 or os.getppid() != parent:
            raise RuntimeError("Parent-death protection failed")
        initial = self.check()
        self.status.parent.mkdir(parents=True, exist_ok=True)
        write_json(self.status, {"state": "admitted", "pid": os.getpid(), "gpu": self.gpu,
            "caps": {k: getattr(self, k) for k in ["allocator_cap", "nvml_cap", "rss_cap", "gpu_floor", "host_floor"]},
            "initial": initial, "lease": "SH admission/GPU; EX Evidence Dial", "time": time.time()})
        def watch():
            while not self.stop.wait(1):
                try:
                    self.check()
                except Exception as e:
                    self.failure = str(e)
                    write_json(self.status.with_suffix(".failure.json"), {"failure": self.failure, "time": time.time()})
                    os.kill(os.getpid(), signal.SIGTERM)
                    return
        self.thread = threading.Thread(target=watch, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=2)
        self.stack.close()
