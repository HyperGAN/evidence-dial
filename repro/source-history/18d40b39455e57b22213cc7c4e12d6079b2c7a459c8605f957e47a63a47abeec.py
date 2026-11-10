"""Report measured resource time and explicit legacy telemetry gaps."""
from pathlib import Path
import argparse
import json
ROOT=Path(__file__).resolve().parents[1]


def build(out):
    artifacts=ROOT/'artifacts';exact=[];failed=[];gaps=[]
    for path in artifacts.rglob('*.completed.json'):
        d=json.loads(path.read_text())
        if 'gpu_lease_hours' not in d:continue
        assert d['gpu']==1
        peak=d.get('sampled_peaks',{}).get('own_nvml')
        allocator=d.get('sampled_peaks',{}).get('allocator_reserved',0)
        exact.append({'source':str(path.relative_to(ROOT)),'seconds':d['seconds'],
            'lease_hours':d['gpu_lease_hours'],'peak_owned_nvml_bytes':peak,
            'GPU_resident':bool(peak or allocator),'physical_GPU1_verified_by_owned_NVML':bool(peak),
            'sampled_peaks':d.get('sampled_peaks'),
            'sampled_minima':d.get('sampled_minima'),'state':d['state']})
    for path in artifacts.rglob('*.failure.json'):
        d=json.loads(path.read_text());admission=path.with_name(path.name.replace('.failure.json','.json'))
        if not admission.exists() or 'time' not in d:continue
        initial=json.loads(admission.read_text())
        duration=d['time']-initial['time'] if 'time' in initial else None
        if duration is not None and duration>=0:
            failed.append({'source':str(path.relative_to(ROOT)),'observed_seconds_until_failure':duration,
                           'failure':d.get('failure'),'cleanup_tail_measured':False})
        else:gaps.append({'source':str(path.relative_to(ROOT)),'reason':'Earlier admission was replaced by a later exact-resume attempt; first duration unmeasured'})
    legacy=[]
    pairs=[('headroom','artifacts/runs/headroom-v3/resource-admission.json','artifacts/runs/headroom-v3/gate.json'),
           ('teacher_preparation','artifacts/cache/cache-resource-admission.json','artifacts/cache/teacher-v1/index.json'),
           ('original_native_proof','artifacts/runs/particle-G012-v1/native-check-resources.json','artifacts/runs/particle-G012-v1/native-check.json'),
           ('AlignTune_training_and_dev','artifacts/runs/aligntune-v2/resources.json','artifacts/runs/aligntune-v2/dev/summary.json')]
    for name,start,end in pairs:
        a,b=ROOT/start,ROOT/end
        if not (a.exists() and b.exists()):continue
        # Old wrappers did not write a terminal resource record. Do not replay
        # their completed work to fill missing telemetry or invent a peak.
        duration=b.stat().st_mtime-json.loads(a.read_text())['time']
        assert duration>=0
        legacy.append({'phase':name,'seconds_estimated_from_artifact_timestamps':duration,
            'hours_estimated':duration/3600,'start_receipt':start,'last_artifact':end,
            'exact_GPU_release_time_measured':False,'peak_owned_nvml_bytes':None})
    model_resident=sum(x['lease_hours'] for x in exact if x['GPU_resident'] and x['physical_GPU1_verified_by_owned_NVML'])
    unmapped=sum(x['lease_hours'] for x in exact if x['GPU_resident'] and not x['physical_GPU1_verified_by_owned_NVML'])
    failure_hours=sum(x['observed_seconds_until_failure'] for x in failed)/3600
    estimates=sum(x['hours_estimated'] for x in legacy)
    value={'declared_physical_GPU':1,'device':'NVIDIA RTX A6000','weight_EMA':None,
        'exact_recorded_GPU_resident_lease_hours':model_resident,
        'GPU_resident_lease_hours_with_unverified_physical_device':unmapped,
        'all_recorded_GPU_resident_phases_verified_on_GPU1':unmapped==0,
        'observed_hours_until_resource_failures':failure_hours,
        'legacy_phase_hours_estimated_from_timestamps':estimates,
        'accounted_GPU_hours_observed_or_estimated':model_resident+unmapped+failure_hours+estimates,
        'complete_cost_coverage':False,'exact_resource_records':exact,'bounded_failures':failed,
        'legacy_estimates':legacy,'gaps':gaps,
        'limits':['Failed loading/setup and cleanup tails are not all measured.',
                  'Legacy teacher-preparation peak memory is unmeasured; the completed teacher cache was not rerun.',
                  'One legacy diagnostic-only50.98s phase omitted CUDA mapping: allocator residency was measured but GPU1 owned NVML was0. It is retained as an unverified-device exception, not counted as certified GPU1 time; future direct diagnostics map the physical UUID explicitly.',
                  'Wall time is measured on a shared physical GPU and includes contention; it is not an exclusive-device throughput claim.',
                  'Per-query throughput/cost uses the separate retained native generation timings; $1/GPU-hour is illustrative, not a paid price.',
                  'The optional closed API comparison was not run; no model quality or billing comparison is fabricated.']}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({k:value[k] for k in ['accounted_GPU_hours_observed_or_estimated','complete_cost_coverage']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);build(p.parse_args().out)
