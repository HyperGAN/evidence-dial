"""Audit paired abstention on saved Tiny Aya grounding predictions."""

import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from evidence_dial.benchmark import file_sha
from evidence_dial.io import read_jsonl
from evidence_dial.report import paired_abstention


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True, help='published benchmark directory')
    parser.add_argument('--dataset', required=True, help='dataset name in the release results')
    parser.add_argument('--data', type=Path, required=True, help='frozen dataset JSONL')
    parser.add_argument('--on', required=True, help='candidate arm in the published benchmark')
    parser.add_argument('--off', default='off', help='reference arm in the published benchmark')
    args = parser.parse_args(argv)
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+', name) for name in (args.dataset, args.on, args.off)):
        parser.error('dataset and arm names must be simple identifiers')
    try:
        results_path = args.release / 'results.json'
        dataset = json.loads(results_path.read_text())['datasets'][args.dataset]
        if file_sha(args.data) != dataset['data_sha256']:
            raise ValueError('Frozen dataset bytes differ from the release')
        arms = []
        for arm in (args.on, args.off):
            path = args.release / 'predictions' / f'{args.dataset}-{arm}.jsonl'
            if file_sha(path) != dataset['arms'][arm]['public_predictions_sha256']:
                raise ValueError(f'Saved prediction bytes differ from the release: {arm}')
            arms.append(read_jsonl(path))
        audit = paired_abstention(read_jsonl(args.data), *arms)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(dict(dataset=args.dataset, on_arm=args.on, off_arm=args.off,
                          results_sha256=file_sha(results_path), **audit), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
