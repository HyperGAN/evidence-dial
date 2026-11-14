"""New execution window for the explicitly authorized November 11 follow-up.

Reuse leases, telemetry, own caps, live floors and parent-death cleanup. The
expired November 9 experiment budget remains immutable and applies to its runs.
"""
import json
import time
from pathlib import Path

from .resources import Resources


class FollowupResources(Resources):
    def __init__(self, gpu, status, *, plan):
        super().__init__(gpu, status)
        self.plan_path = Path(plan)
        self.plan = json.loads(self.plan_path.read_text())

    def check(self):
        if self.failure:
            raise RuntimeError(self.failure)
        if time.time() >= self.plan['deadline_epoch']:
            raise RuntimeError('November 11 follow-up execution window exhausted')
        s = self.sample()
        for key in ['own_nvml', 'allocator_reserved', 'rss']:
            self.peaks[key] = max(self.peaks.get(key, 0), s[key])
        for key in ['global_gpu_free', 'host_available']:
            self.minima[key] = min(self.minima.get(key, s[key]), s[key])
        checks = [(s['global_gpu_free'] >= self.gpu_floor, 'GPU free floor'),
                  (s['host_available'] >= self.host_floor, 'host available floor'),
                  (s['own_nvml'] <= self.nvml_cap, 'own NVML cap'),
                  (s['allocator_reserved'] <= self.allocator_cap, 'allocator cap'),
                  (s['rss'] <= self.rss_cap, 'own RSS cap')]
        for ok, name in checks:
            if not ok:
                raise RuntimeError(f'Resource refusal: {name}: {s}')
        return s
