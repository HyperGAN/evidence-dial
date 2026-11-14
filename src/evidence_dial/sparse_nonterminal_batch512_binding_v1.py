"""Original PG on TRAIN256 first/first/last-content views, batch8/fixed512."""
from collections import Counter
import json
from pathlib import Path
import torch

from particle_sliders.pg09 import ConditionalPG09
from particle_sliders.pg09_audio import PreparedAudioResponse, AudioResponseCritic, audio_conditions
from particle_sliders.response import ResponseBackend
from particle_sliders.lifecycle import Binding
from .binding import initialize, CAPS, MODEL_IDENTITY
from .sparse_nonterminal_native_v1 import Native, NAME, TRAIN_SHA, inference_backend
from .benchmark import file_sha
from .io import read_jsonl,write_json
from .protocol import SEED,normalize,success

STEPS=512
BATCH=8
RATES=(.001,)

def validate(options):
    if options.get('ema_decay') is not None or options.get('base_precision') != 'bfloat16':
        raise ValueError('RAW/noEMA and native BF16 required')
    if (options['mode'],options['steps'],options['batch']) != ('first-answer',STEPS,BATCH):
        raise ValueError('Declared sparse TRAIN256/768-view PG512/batch8 cohort required')
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
    assert index['mode'] == 'canonical-curated256-first-first-lastcontent-v1'
    assert index['cases'] == 256 and index['response_rows'] == len(index['rows']) == 768
    assert index['case_and_type_weight_equal'] and index['selected_targets_exclude_all_special_tokens']
    assert index['all_target_hiddens_reused_without_teacher_generation']
    assert index['existing_neutral_first_slots_reused'] == 519 and index['unique_new_neutral_prefixes'] == 249
    assert len({item['slot_id'] for item in index['rows']}) == 768
    assert Counter(item['id'] for item in index['rows']) == Counter({row['id']:3 for row in rows})
    assert Counter(item['case_type'] for item in index['rows']) == Counter({k:192 for k in ['answerable','altered','missing','conflict']})
    assert index['native_terminal_positions'] == 0
    assert index['responses_sha256'] == file_sha(cache/'responses.pt')
    by_id = {r['id']:r for r in rows}
    assert len({r['id'] for r in index['rows']}) == 256 and {r['id'] for r in index['rows']} == set(by_id)
    for item in index['rows']:
        row = by_id[item['id']]
        assert item['case_type'] == row['case_type'] and item['slot'] in [0,1,2]
        assert item['position'] == item['target_sample_position'] == len(item['history_ids'])
        assert all(token not in [3,261001] for token in item['history_ids'])
        if item['slot'] < 2: assert item['position'] == 0
        if item['position']:
            assert item['neutral_ids'][-item['position']:] == item['teacher_ids'][-item['position']:] == item['history_ids']
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
        assert len(stored)==768
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
