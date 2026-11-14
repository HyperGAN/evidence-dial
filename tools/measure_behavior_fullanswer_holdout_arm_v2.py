"""Generate one already-locked behavior arm, preserving unknown first calls."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--name', required=True); args = ap.parse_args()
    from evidence_dial.behavior_fullanswer_confirmation_v2 import pin, read, validate_plan
    from evidence_dial.io import read_jsonl, write_json
    plan = validate_plan(read(args.plan)); arm = next(a for a in plan['arms'] if a['name'] == args.name)
    folder = args.plan.resolve().parent / 'arms' / args.name
    assert not (folder / 'actual-owned-process-first.json').exists()
    parent = os.getppid(); assert ctypes.CDLL(None).prctl(1, signal.SIGTERM) == 0 and os.getppid() == parent
    def stop(sig, frame): raise KeyboardInterrupt('Owned holdout arm interrupted; preserve native/Jev firsts')
    signal.signal(signal.SIGTERM, stop)
    write_json(folder / 'actual-owned-process-first.json', {'time': time.time(), 'pid': os.getpid(),
        'birth': Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        'argv': [x.decode() for x in Path('/proc/self/cmdline').read_bytes().split(b'\0') if x],
        'plan': pin(args.plan), 'name': args.name})
    os.environ['CUDA_VISIBLE_DEVICES'] = plan['physical_gpu_uuids'][arm['gpu']]
    from evidence_dial.shared_gpu_resources_v1 import SharedGPU0Resources, SharedGPU1Resources
    from evidence_dial.behavior_fullanswer_generate_v2 import evaluate, MAX_NEW_TOKENS
    from evidence_dial.behavior_protocol_v1 import TARGETS
    from evidence_dial.evaluate import load_base, tokenizer
    from particle_sliders.inference import open_inference
    from evidence_dial.behavior_expanded_binding_v2 import inference_backend
    assert MAX_NEW_TOKENS == 4096
    rows = read_jsonl(plan['data']['path'])
    assert len(rows) == 32 and all(r['split'] == 'holdout' for r in rows)
    assert len({r['id'] for r in rows}) == len({r['question'] for r in rows}) == 32
    Resources = SharedGPU0Resources if arm['gpu'] == 0 else SharedGPU1Resources
    resource_plan = folder / 'resource-plan-first.json'
    write_json(resource_plan, {'physical_gpu_uuid': plan['physical_gpu_uuids'][arm['gpu']],
        'parent_plan': pin(args.plan), 'elapsed_work_deadline': None})
    try:
        with Resources(arm['gpu'], folder / 'resources-first.json', plan=resource_plan) as resource:
            tok = tokenizer()
            if arm['kind'] == 'particle':
                candidate = arm['candidate']
                label = f'particle-raw-{candidate["artifact"]["sha256"]}-strength-{candidate["strength"]:g}'
                with open_inference(candidate['artifact']['path'], backend=inference_backend(), operation='response') as inference:
                    native = inference.binding.response.__self__
                    with native.slider.scaled(candidate['strength']):
                        evaluate(rows, native.model, tok, folder, resource, arm=label)
            else:
                model = load_base(); assert not any('lora_' in n for n, _ in model.named_parameters())
                target = TARGETS[0] if arm['kind'] == 'prompt-only' else None
                evaluate(rows, model, tok, folder, resource, arm=arm['name'], target=target)
        validate_plan(read(args.plan))
        write_json(folder / 'completion-first.json', {'time': time.time(), 'complete': True,
            'name': args.name, 'cases': 32, 'plan': pin(args.plan), 'weight_EMA': False,
            'training_updates': 0, 'holdout_selection': False})
        print(json.dumps({'complete': True, 'name': args.name, 'cases': 32}))
    except BaseException as error:
        write_json(folder / 'arm-failed-first.json', {'time': time.time(), 'error': repr(error), 'firsts_preserved': True})
        raise


if __name__ == '__main__': main()
