"""Owner-requested GPU0 parallel lane; legacy GPU1 lease stays untouched."""
import contextlib
import ctypes
import fcntl
import os
from pathlib import Path
import signal
import threading
import time

from .followup_resources import FollowupResources
from .io import write_json


class GPU0FollowupResources(FollowupResources):
    def __init__(self, gpu, status, *, plan):
        super().__init__(gpu, status, plan=plan)
        if gpu != 0 or self.plan['physical_gpu_uuid'] != 'GPU-72c1b506-891d-b8bc-b353-e020585e1c47':
            raise ValueError('This separately leased lane is restricted to physical GPU0')

    def __enter__(self):
        try:
            import pynvml as nv
        except ModuleNotFoundError:
            pass
        else:
            nv.nvmlInit()
        self.stack = contextlib.ExitStack()
        try:
            for path, mode in [('/mnt/nvme/sliders/qualification/admission-gpu-0.lock', fcntl.LOCK_SH),
                               ('/mnt/nvme/sliders/locks/gpu-0.lock', fcntl.LOCK_SH),
                               (str(Path(__file__).resolve().parents[2] / 'artifacts/evidence-dial-gpu0.lock'), fcntl.LOCK_EX)]:
                file = self.stack.enter_context(open(path, 'a+'))
                fcntl.flock(file, mode | fcntl.LOCK_NB)
            parent = os.getppid()
            if ctypes.CDLL(None).prctl(1, signal.SIGTERM) != 0 or os.getppid() != parent:
                raise RuntimeError('Parent-death protection failed')
            initial = self.check(); self.started = time.perf_counter()
            write_json(self.status, {'state': 'admitted', 'pid': os.getpid(), 'gpu': 0,
                       'caps': {k: getattr(self, k) for k in ['allocator_cap','nvml_cap','rss_cap','gpu_floor','host_floor']},
                       'initial': initial, 'lease': 'SH host/GPU0; EX Evidence Dial GPU0; legacy GPU1 EX unchanged',
                       'time': time.time()})
            def watch():
                while not self.stop.wait(1):
                    try:
                        self.check()
                    except Exception as error:
                        self.failure = str(error)
                        write_json(self.status.with_suffix('.failure.json'), {'failure': self.failure, 'time': time.time()})
                        os.kill(os.getpid(), signal.SIGTERM)
                        return
            self.thread = threading.Thread(target=watch, daemon=True); self.thread.start()
            return self
        except BaseException:
            self.stack.close(); raise
