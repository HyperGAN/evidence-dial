"""Execution derivatives: live memory/ownership limits, no elapsed-work kill.

The Nov6 standing no-time-limit instruction applies to future wrappers. Existing
captured attempts/deadlines remain unchanged. No optimizer/model code changes.
"""
import contextlib
import ctypes
import fcntl
import os
from pathlib import Path
import signal
import threading
import time

from .followup_resources import FollowupResources
from .gpu0_resources import GPU0FollowupResources
from .io import write_json


def memory_check(resource):
    if resource.failure:
        raise RuntimeError(resource.failure)
    sample = resource.sample()
    for key in ['own_nvml', 'allocator_reserved', 'rss']:
        resource.peaks[key] = max(resource.peaks.get(key, 0), sample[key])
    for key in ['global_gpu_free', 'host_available']:
        resource.minima[key] = min(resource.minima.get(key, sample[key]), sample[key])
    for valid, name in [
        (sample['global_gpu_free'] >= resource.gpu_floor, 'GPU free floor'),
        (sample['host_available'] >= resource.host_floor, 'host available floor'),
        (sample['own_nvml'] <= resource.nvml_cap, 'own NVML cap'),
        (sample['allocator_reserved'] <= resource.allocator_cap, 'allocator cap'),
        (sample['rss'] <= resource.rss_cap, 'own RSS cap')]:
        if not valid:
            raise RuntimeError(f'Resource refusal: {name}: {sample}')
    return sample


class ContinuedGPU1Resources(FollowupResources):
    def check(self):
        return memory_check(self)


class ContinuedGPU0Resources(GPU0FollowupResources):
    def check(self):
        return memory_check(self)


class CPUObserverResources:
    """Separate CPU observer, host floor47GiB and tighter own RSS24GiB.

    No CUDA model work or GPU lease is used. The private EX lease prevents two
    Evidence Dial CPU observers from loading concurrently; it reserves no RAM.
    """
    def __init__(self, status):
        self.status = Path(status)
        self.stop = threading.Event()
        self.failure = None
        self.peak_rss = 0
        self.minimum_available = None
        self.rss_cap = 24 * 1024**3
        self.host_floor = 47 * 1024**3

    def check(self):
        if self.failure:
            raise RuntimeError(self.failure)
        info = {line.split(':')[0]: int(line.split()[1]) * 1024
                for line in Path('/proc/meminfo').read_text().splitlines()}
        rss = next(int(line.split()[1]) * 1024 for line in Path('/proc/self/status').read_text().splitlines()
                   if line.startswith('VmRSS:'))
        self.peak_rss = max(self.peak_rss, rss)
        self.minimum_available = min(self.minimum_available or info['MemAvailable'], info['MemAvailable'])
        if rss > self.rss_cap or info['MemAvailable'] < self.host_floor:
            raise RuntimeError(f'CPU observer memory refusal: RSS={rss}, available={info["MemAvailable"]}')
        return {'rss': rss, 'host_available': info['MemAvailable']}

    def __enter__(self):
        self.stack = contextlib.ExitStack()
        try:
            lock = self.stack.enter_context(open(Path(__file__).resolve().parents[2] / 'artifacts/evidence-dial-cpu-observer.lock', 'a+'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            parent = os.getppid()
            assert ctypes.CDLL(None).prctl(1, signal.SIGTERM) == 0 and os.getppid() == parent
            self.started = time.monotonic()
            initial = self.check()
            write_json(self.status, {'state': 'admitted', 'pid': os.getpid(), 'placement': 'CPU only,4threads',
                'initial': initial, 'own_rss_cap': self.rss_cap, 'host_floor': self.host_floor,
                'elapsed_work_deadline': None, 'time': time.time()})
            def watch():
                while not self.stop.wait(1):
                    try:
                        self.check()
                    except Exception as error:
                        self.failure = str(error)
                        write_json(self.status.with_suffix('.failure.json'), {'failure': self.failure, 'time': time.time()})
                        os.kill(os.getpid(), signal.SIGTERM)
                        return
            self.thread = threading.Thread(target=watch, daemon=True)
            self.thread.start()
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=2)
        try:
            write_json(self.status.with_suffix('.completed.json'), {'state': 'exception' if args and args[0] else 'complete',
                'failure': self.failure, 'peak_rss': self.peak_rss, 'minimum_available': self.minimum_available,
                'seconds': time.monotonic() - self.started, 'time': time.time()})
        finally:
            self.stack.close()
