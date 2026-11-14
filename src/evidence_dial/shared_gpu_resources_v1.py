"""Execution-only shared GPU lanes with exclusive ownership per candidate.

The Nov12 owner request removes whole-job serialization between our different
jobs. Original live floors, own caps, telemetry and parent-death cleanup remain.
No model, optimizer, saved state or semantic observer implementation changes.
"""
import contextlib
import ctypes
import fcntl
import os
from pathlib import Path
import signal
import threading
import time

from .continuation_resources_v1 import ContinuedGPU0Resources, ContinuedGPU1Resources
from .io import write_json


class SharedGPUEntry:
    def __enter__(self):
        try:
            import pynvml as nv
        except ModuleNotFoundError:
            pass
        else:
            nv.nvmlInit()
        self.stack = contextlib.ExitStack()
        self.status.parent.mkdir(parents=True, exist_ok=True)
        project = Path(__file__).resolve().parents[2]
        lane = project / ('artifacts/evidence-dial-gpu0.lock' if self.gpu == 0
                          else 'artifacts/evidence-dial-gpu.lock')
        # Training, recovery, inference and the observer all use the same
        # candidate directory. Distinct candidates can share either GPU.
        job = self.status.parent / '.shared-gpu-job.lock'
        leases = [(Path(f'/mnt/nvme/sliders/qualification/admission-gpu-{self.gpu}.lock'), fcntl.LOCK_SH),
                  (Path(f'/mnt/nvme/sliders/locks/gpu-{self.gpu}.lock'), fcntl.LOCK_SH),
                  (lane, fcntl.LOCK_SH), (job, fcntl.LOCK_EX)]
        try:
            for path, mode in leases:
                handle = self.stack.enter_context(open(path, 'a+'))
                fcntl.flock(handle, mode | fcntl.LOCK_NB)
            parent = os.getppid()
            if ctypes.CDLL(None).prctl(1, signal.SIGTERM) != 0 or os.getppid() != parent:
                raise RuntimeError('Parent-death protection failed')
            initial = self.check()
            self.started = time.perf_counter()
            write_json(self.status, {'state': 'admitted', 'pid': os.getpid(),
                'pid_birth_ticks': Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
                'gpu': self.gpu, 'parent_pid': parent, 'initial': initial,
                'caps': {k: getattr(self, k) for k in
                    ['allocator_cap', 'nvml_cap', 'rss_cap', 'gpu_floor', 'host_floor']},
                'leases': [{'path': str(path), 'mode': 'SH' if mode == fcntl.LOCK_SH else 'EX'}
                    for path, mode in leases], 'elapsed_work_deadline': None, 'time': time.time()})
            def watch():
                while not self.stop.wait(1):
                    try:
                        self.check()
                    except Exception as error:
                        self.failure = str(error)
                        write_json(self.status.with_suffix('.failure.json'),
                            {'failure': self.failure, 'pid': os.getpid(), 'time': time.time()})
                        os.kill(os.getpid(), signal.SIGTERM)
                        return
            self.thread = threading.Thread(target=watch, daemon=True)
            self.thread.start()
            return self
        except BaseException as error:
            write_json(self.status.with_suffix('.failure.json'), {'failure': str(error),
                'state': 'entry-refused', 'pid': os.getpid(), 'time': time.time()})
            self.stack.close()
            raise


class SharedGPU0Resources(SharedGPUEntry, ContinuedGPU0Resources):
    pass


class SharedGPU1Resources(SharedGPUEntry, ContinuedGPU1Resources):
    pass
