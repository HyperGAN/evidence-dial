"""Canonical TRAIN256 with batch8/fixed128; original ConditionalPG09 game.

This is a new data/batch hyperparameter cohort, not a replay or reset of any
completed LR study. Native forwards, critic, conditions and optimizer persist.
"""
from collections import Counter
import json
from pathlib import Path
import torch

from particle_sliders.pg09 import ConditionalPG09
from particle_sliders.pg09_audio import PreparedAudioResponse, AudioResponseCritic, audio_conditions
from particle_sliders.response import ResponseBackend
from particle_sliders.lifecycle import Binding
from .binding import initialize, CAPS, MODEL_IDENTITY
from .curated_first_answer_binding_v1 import Native, NAME, TRAIN_SHA, inference_backend
from .benchmark import file_sha
from .io import read_jsonl,write_json
from .protocol import SEED,normalize,success

STEPS=128
BATCH=8
RATES=(.001,)

def validate(options):
    if options.get('ema_decay') is not None or options.get('base_precision') != 'bfloat16':
        raise ValueError('RAW/noEMA and native BF16 required')
    if (options['mode'],options['steps'],options['batch']) != ('first-answer',STEPS,BATCH):
        raise ValueError('Declared first-answer128/batch8 cohort required')
    if options['g_lr'] not in RATES or options['binding_sha256'] != file_sha(__file__):
        raise ValueError('Undeclared rate or changed binding')
    data, cache = Path(options['training_data']), Path(options['cache'])
    if options['training_data_sha256'] != TRAIN_SHA or file_sha(data) != TRAIN_SHA:
        raise ValueError('Only the frozen separately curated TRAIN256 pool is supported')
    rows = read_jsonl(data)
    assert len(rows) == len({r['id'] for r in rows}) == 256 and all(r['split']=='train' for r in rows)
    assert Counter(r['case_type'] for r in rows) == Counter({k:64 for k in ['answerable','altered','missing','conflict']})
    if options['cache_index_sha256'] != file_sha(cache/'index.json'):
        raise ValueError('Frozen target index differs')
    index = json.loads((cache/'index.json').read_text())
    assert index['mode'] == 'canonical-curated256-first' and index['cases'] == len(index['rows']) == 256
    assert index['native_terminal_positions'] == index['neutral_answer_history_tokens'] == 0
    assert index['responses_sha256'] == file_sha(cache/'responses.pt')
    by_id = {r['id']:r for r in rows}
    assert len({r['id'] for r in index['rows']}) == 256 and {r['id'] for r in index['rows']} == set(by_id)
    for item in index['rows']:
        row = by_id[item['id']]
        assert item['case_type'] == row['case_type'] and item['history_ids'] == [] and item['position'] == 0
        assert item['target_from_actual_sample'] and item['canonical_TRAIN_gold_supplied']
        assert item['target_token'] not in [261001,3]
        assert normalize(item['teacher_output']) == normalize(row['gold']) and success(row,item['teacher_output'])


def prepare(options, *, restored_from=None):
    from .evaluate import load_base
    torch.manual_seed(SEED)
    native = Native(load_base(), mode='first-answer')
    try:
        critic = AudioResponseCritic(2048,2052).to(device='cuda:0',dtype=torch.float32)
        receipt = Path(options['initialization_receipt'])
        if restored_from is None:
            if receipt.exists():
                raise ValueError('Fresh state already initialized; restore its saved state')
            value = initialize(native,critic,options['cohort_id'])
            write_json(receipt,{'sha256':value.sha256,'receipt':value.as_dict()})
        elif not receipt.exists():
            raise ValueError('Exact restore requires its original initialization receipt')
        stored = torch.load(Path(options['cache'])/'responses.pt',map_location='cpu',weights_only=False)
        assert len(stored)==256
        rows = [PreparedAudioResponse(x.row,x.target.to('cuda:0'),x.neutral.to('cuda:0')) for x in stored]
        conditions = audio_conditions(rows)
        conditions[:,2048:2050] /= 512.
        game = ConditionalPG09(native,rows,critic,conditions=conditions,particles=native.particles,
            response_domain='continuous_vector',penalty_input_domain='continuous',batch_size=BATCH,
            seed=SEED,total_steps=STEPS,response_scale=1.,
            recipe_overrides={'lr':options['g_lr'],'d_lr_mult':18.,'prior_lr_mult':30.})
        assert game.recipe.ema_decay == 0.
        spec = inference_backend('first-answer').specification(native,model_identity=MODEL_IDENTITY,
            protocol=native.metadata()['protocol'],configuration={'mode':'first-answer','rank':8,
                'alpha':8,'bank_shape':[128,4],'ema_decay':None})
        return Binding(game,native.close,inference=spec)
    except BaseException:
        native.close()
        raise


def response_backend(mode):
    assert mode=='first-answer'
    return ResponseBackend(NAME,CAPS,prepare,validate)
