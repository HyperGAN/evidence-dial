"""Plot only completed, admitted DEV quarters; never read confirmation rows."""
import argparse
from decimal import Decimal
import hashlib
import json
import os
import math
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/behavior-promptfix-20261114-v4'
DIAG = ROOT / 'artifacts/behavior-prompt-diagnosis-20261114-v3'


def read(path):
    return json.loads(Path(path).read_text())


def pin(path):
    path = Path(path)
    return {'path': str(path.resolve()),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    assert not args.output.exists(), 'Use a new figure path; preserve earlier captured figures'
    declaration_path = BASE / 'training-declaration-first.json'
    declaration = read(declaration_path)
    diagnostic_path = DIAG / 'independent-audit-first.json'
    diagnostic = read(diagnostic_path)
    inputs = [pin(declaration_path), pin(diagnostic_path)]

    def mean(predictions):
        assert pin(predictions['path']) == predictions
        rows = [json.loads(line) for line in Path(predictions['path']).read_text().splitlines()]
        assert len(rows) == 16 and len({r['id'] for r in rows}) == 16
        assert all(r['native']['output_ids'][-1] == 3 for r in rows)
        inputs.append(predictions)
        return sum(Decimal(str(r['scores']['effect'])) for r in rows) / 64

    controls = {a['name']: a for a in diagnostic['arms']}
    off = mean(controls['off']['predictions'])
    prompt = mean(controls['target0-unrestricted']['predictions'])
    cells = []
    for cohort in declaration['cohorts']:
        for step in [256, 512, 768, 1024]:
            path = BASE / f'GPU{cohort["gpu"]}-quarter{step}-independent-admission-first.json'
            if not path.exists():
                continue
            quarter = read(path)
            assert quarter['state'] == 'declared-complete-behavior-quarter-independently-readmitted'
            assert quarter['plan'] == cohort['plan'] and quarter['step'] == step
            assert quarter['all48_answers_native_EOS'] and quarter['no_weight_EMA']
            inputs.append(pin(path))
            for arm in quarter['arms']:
                value = mean(arm['predictions'])
                cells.append({'gpu': cohort['gpu'], 'g_lr': cohort['g_lr'],
                    'step': step, 'strength': arm['strength'],
                    'mean_effect': str(value), 'effect_ON_minus_OFF': str(value - off)})
    assert 0 < len(cells) <= 24
    metadata = BASE / f'DEV-training-curve-{len(cells)}cells-first.json'
    assert not metadata.exists(), 'Reuse the existing completed plot observation'
    maximum = max(Decimal(c['effect_ON_minus_OFF']) for c in cells)
    plotted = [0., float(prompt - off)] + [float(c['effect_ON_minus_OFF']) for c in cells]
    y_min = math.floor(min(plotted) * 100) / 100 - 0.01
    y_max = math.ceil(max(plotted) * 100) / 100 + 0.01
    colors = {0.25: '#0072b2', 0.5: '#d55e00', 1.0: '#7b2cbf'}
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.1), sharex=True, sharey=True)
    for ax, cohort in zip(axes, declaration['cohorts']):
        ax.axhline(0, color='#626b73', linewidth=1.4, linestyle='--', label='OFF')
        ax.axhline(float(prompt - off), color='#1c2730', linewidth=1.4,
                   linestyle=':', label='Unrestricted prompt-only')
        for strength in [0.25, 0.5, 1.0]:
            series = [c for c in cells if c['gpu'] == cohort['gpu'] and c['strength'] == strength]
            ax.plot([c['step'] for c in series],
                    [float(c['effect_ON_minus_OFF']) for c in series],
                    marker='o', color=colors[strength], linewidth=2,
                    label=f'Strength {strength:g}')
            for c in series:
                if Decimal(c['effect_ON_minus_OFF']) == maximum:
                    ax.scatter(c['step'], float(maximum), marker='*', s=220,
                               facecolor=colors[strength], edgecolor='white', zorder=5)
        ax.set(title=f'G learning rate {cohort["g_lr"]:g}', xlabel='Completed training updates',
               xticks=[256, 512, 768, 1024], xlim=(220, 1060), ylim=(y_min, y_max))
        ax.grid(axis='y', alpha=0.18)
    axes[0].set_ylabel('Mean intended behavior ON minus matched OFF')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.045),
               frameon=False)
    fig.suptitle(f'Completed DEV results: {len(cells)} of 24 declared settings',
                 fontsize=16, fontweight='bold', y=0.98)
    fig.text(0.5, 0.012,
        '16 matched DEV questions; fixed Jev rubric, not accuracy. Only admitted complete answers shown. '
        'Fresh confirmation remains separate.', ha='center', fontsize=9, color='#47535d')
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170, facecolor='white')
    plt.close(fig)
    metadata.write_text(json.dumps({'state': 'completed-DEV-curve-only',
        'source': pin(__file__), 'inputs': inputs, 'output': pin(args.output),
        'cells': cells, 'OFF_mean': str(off), 'prompt_mean': str(prompt),
        'all_maximum_ties': [c for c in cells if Decimal(c['effect_ON_minus_OFF']) == maximum],
        'holdout_rows_read': 0, 'new_model_or_Jev_calls': 0,
        'unfinished_quarters_included': False}, indent=2) + '\n')
    print(json.dumps({'observation': pin(metadata), 'figure': pin(args.output),
                      'DEV_cells': len(cells), 'new_model_or_Jev_calls': 0}))


if __name__ == '__main__':
    main()
