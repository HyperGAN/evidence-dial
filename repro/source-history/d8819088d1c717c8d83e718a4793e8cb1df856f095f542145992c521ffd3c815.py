"""Native checks and orchestration for the distinct curated-success retry."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import torch

from .benchmark import file_sha
from .io import read_jsonl, write_json
from .protocol import SEED

ROOT = Path(__file__).resolve().parents[2]


def config(out, cache):
    from particle_sliders.lifecycle import SessionConfig
    from .curated_binding import NAME
    cfg = SessionConfig(NAME, 'evidence-dial-jev-success-G001-v1', steps=128, batch=2, seed=SEED,
        g_lr=.001, ema_decay=None, base_precision='bfloat16', cache=str(cache.resolve()),
        cache_index_sha256=file_sha(cache / 'index.json'),
        curated_binding_sha256=file_sha(ROOT / 'src/evidence_dial/curated_binding.py'),
        initialization_receipt=str((out / 'initialization.json').resolve()),
        cohort_id='evidence-dial-jev-success-G001-v1')
    path = out / 'session.json'
    if path.exists():assert json.loads(path.read_text()) == cfg.as_dict()
    else:write_json(path, cfg.as_dict())
    return cfg


def native_check(out, cache):
    from particle_sliders.response import open_response
    from .resources import Resources
    from .train import tensor_equal
    if (out / 'native-check.json').exists() or list(out.glob('FULL-*.pt')):
        raise FileExistsError('New-backend qualification already attempted; inspect saved state')
    cfg = config(out, cache)
    with Resources(1, out / 'native-check-resources.json') as resource:
        with open_response(cfg) as session:
            native = session.game.backend; row = session.game.rows[0]
            resource.check()
            with torch.no_grad():
                off = native._response(row.row.neutral_ids, scale=0.)
                on = native._response(row.row.neutral_ids, scale=1.)
            assert torch.equal(off, row.neutral) and torch.equal(on, off)
            value = native.generated(row); parameters = list(native.network.parameters())
            gradients = torch.autograd.grad(value.sum(), parameters, allow_unused=True)
            assert len(parameters) == 2593
            assert all(g is not None and torch.isfinite(g).all() for g in gradients)
            assert any(torch.count_nonzero(g) for g in gradients)
            del value, gradients
            packet = session.checkpoint(out / 'FULL-000000.pt')
            session.restore(packet)
            assert tensor_equal(packet['payload']['game'], session.checkpoint()['payload']['game'])
            session.export(out / 'RAW-000000.pt', selection='raw')
            session.export_inference(out / 'NATIVE-000000.pt', selection='raw')
            write_json(out / 'native-check.json', {'protocol': native.metadata()['protocol'],
                'new_backend_and_target_integration_check': True, 'off_matches_new_cached_pair': True,
                'initial_ON_equals_OFF': True, 'connected_trainable_tensors': len(parameters),
                'full_restore_tensor_exact': True, 'raw_exports_saved': True, 'new_training_updates': 0,
                'ema': session.ema.state_dict(), 'manifest': session.manifest,
                'cache_index_sha256': file_sha(cache / 'index.json')})


def main():
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=['cache', 'native-check', 'train', 'raw-check'])
    p.add_argument('--curation', type=Path, default=ROOT / 'artifacts/curation-success-v3')
    p.add_argument('--cache', type=Path, default=ROOT / 'artifacts/cache/curated-success-v1')
    p.add_argument('--run', type=Path, default=ROOT / 'artifacts/runs/particle-curated-G001-v1')
    p.add_argument('--stop', type=int); p.add_argument('--attempt', default='first'); args = p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = subprocess.check_output(['nvidia-smi', '--id=1', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    # Each attempted phase retains its exact new implementation before execution.
    sources = {}
    for name in ['curated_train', 'curated_binding', 'curation', 'binding', 'train', 'protocol', 'resources']:
        file = ROOT / f'src/evidence_dial/{name}.py'; sha = file_sha(file)
        captured = ROOT / f'artifacts/execution-sources/{name}-{sha}.py'
        if captured.exists():assert captured.read_bytes() == file.read_bytes()
        else:captured.parent.mkdir(parents=True, exist_ok=True); captured.write_bytes(file.read_bytes())
        sources[name] = sha
    receipt = ROOT / f'artifacts/method-repair-v1/{args.phase}-{args.stop or 0}-{args.attempt}.json'
    if receipt.exists():raise FileExistsError('Attempt already exists; inspect/reap before another derivative')
    write_json(receipt, {'state': 'started', 'sources': sources, 'pid': os.getpid(), 'phase': args.phase,
                         'stop': args.stop, 'cache': str(args.cache), 'run': str(args.run)})
    try:
        if args.phase == 'cache':
            from .curated_binding import build_cache
            build_cache(args.curation, args.cache, attempt=args.attempt)
        elif args.phase == 'native-check':native_check(args.run, args.cache)
        elif args.phase == 'train':
            from .train import train_chunk
            train_chunk(args.run, args.stop, ROOT / 'artifacts/data/policy-v1', 1)
        else:
            from .train import check_raw_generation
            check_raw_generation(args.run / 'NATIVE-000000.pt', ROOT / 'artifacts/data/policy-v1',
                                 args.run / 'initial-native-inference', 1)
    except BaseException as error:
        write_json(receipt.with_suffix('.terminal.json'), {'state': 'failed', 'error': repr(error)})
        raise
    write_json(receipt.with_suffix('.terminal.json'), {'state': 'complete'})


if __name__ == '__main__':main()
