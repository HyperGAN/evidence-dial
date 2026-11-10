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
        self.peaks = {}
        self.minima = {}

    def sample(self):
        import subprocess
        try:
            import pynvml as nv
        except ModuleNotFoundError:
            # Captured observer environments stay immutable. nvidia-smi reads
            # the same physical NVML telemetry without installing a package.
            free = int(subprocess.check_output(['nvidia-smi',f'--id={self.gpu}',
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*1024**2
            lines = subprocess.check_output(['nvidia-smi',f'--id={self.gpu}',
                '--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).splitlines()
            own = sum(int(line.split(',')[1].strip())*1024**2 for line in lines
                if line.split(',')[0].strip() == str(os.getpid()))
        else:
            handle = nv.nvmlDeviceGetHandleByIndex(self.gpu)
            mem = nv.nvmlDeviceGetMemoryInfo(handle)
            free = mem.free
            own = sum(p.usedGpuMemory for p in nv.nvmlDeviceGetComputeRunningProcesses(handle) if p.pid == os.getpid())
        info = {line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
        rss = next(int(line.split()[1])*1024 for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:'))
        import torch
        allocated = torch.cuda.memory_reserved(0) if torch.cuda.is_initialized() else 0
        return {"global_gpu_free": free, "own_nvml": own,
            "allocator_reserved": allocated, "rss": rss,
            "host_available": info['MemAvailable']}

    def check(self):
        if self.failure:
            raise RuntimeError(self.failure)
        budget=Path(__file__).resolve().parents[2]/'artifacts/execution-budget.json'
        if budget.exists():
            import json
            from datetime import datetime, timezone
            deadline=datetime.fromisoformat(json.loads(budget.read_text())['deadline_utc'])
            if datetime.now(timezone.utc)>=deadline:
                raise RuntimeError('Declared one-day execution budget exhausted; preserve actual state')
        s = self.sample()
        for key in ['own_nvml','allocator_reserved','rss']:
            self.peaks[key]=max(self.peaks.get(key,0),s[key])
        for key in ['global_gpu_free','host_available']:
            self.minima[key]=min(self.minima.get(key,s[key]),s[key])
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
        from .io import write_json
        try:
            import pynvml as nv
        except ModuleNotFoundError:
            pass
        else:
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
        self.started = time.perf_counter()
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
        from .io import write_json
        self.stop.set()
        self.thread.join(timeout=2)
        try:
            write_json(self.status.with_suffix('.completed.json'), {
                'state':'exception' if args and args[0] else 'complete',
                'pid':os.getpid(),'gpu':self.gpu,'seconds':time.perf_counter()-self.started,
                'gpu_lease_hours':(time.perf_counter()-self.started)/3600,
                'sampled_peaks':self.peaks,'sampled_minima':self.minima,
                'time':time.time(),'failure':self.failure})
        finally:
            self.stack.close()
