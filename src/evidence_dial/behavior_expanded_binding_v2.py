"""New behavior data; unchanged Tiny Aya native adapter and ConditionalPG09.

Only rows, target-prompt metadata and declared hyperparameters differ. No CE,
reconstruction, SFT initialization, EMA, critic widening or latent replacement.
"""
import json
from pathlib import Path
import torch
from particle_sliders.pg09 import ConditionalPG09
from particle_sliders.pg09_audio import PreparedAudioResponse,AudioResponseCritic,audio_conditions
from particle_sliders.response import ResponseBackend
from particle_sliders.lifecycle import Binding
from particle_sliders.inference import InferenceBackend,InferenceBinding,native_identity
from .first_answer_binding_v2 import Native as ParentNative
from .binding import initialize,CAPS,MODEL_IDENTITY
from .protocol import SEED
from .benchmark import file_sha
from .io import write_json

NAME='tiny-aya-behavior-expanded138-h1024-v2'
PROTOCOL='cohere2-behavior-allpassing-systemtarget-samequestion-emptyhistory-v2'


class Native(ParentNative):
    def metadata(self):
        value=super().metadata()
        value.update(protocol=PROTOCOL,target_convention='MetaGlimmer: frozen raw positive first-token hidden with behavior system instruction; neutral same user question without system instruction; matched empty continuation history; target selected among16 instruction variants by NanoGPT Jev')
        return value


def validate(options):
    assert options['ema_decay'] is None and options['base_precision']=='bfloat16'
    assert options['mode']=='first-answer' and options['steps']==1024 and options['batch']==8
    assert options['g_lr'] in [.001,.00025] and options['d_lr']==.018 and options['bank_lr']==.03
    assert options['binding_sha256']==file_sha(__file__)
    cache=Path(options['cache']);index=json.loads((cache/'index.json').read_text())
    assert options['cache_index_sha256']==file_sha(cache/'index.json') and index['responses_sha256']==file_sha(cache/'responses.pt')
    assert index['mode']=='behavior-all-passing-MetaGlimmer-empty-history-v1' and index['shared_history']==[]
    assert index['neutral_behavior_instruction']=='' and not index['neutral_system_message_present']
    assert options['training_data_sha256']==file_sha(options['training_data'])==index['training_data']['sha256']
    audit=Path(options['cache_admission']);assert options['cache_admission_sha256']==file_sha(audit)
    admitted=json.loads(audit.read_text());assert admitted['state']=='all1024-saved-firsts-reverified-all138-passing-contexts-admitted'
    assert admitted['training_pairs']==index['cases']==len(index['rows'])==138
    assert admitted['unique_training_questions']==index['unique_training_questions']==37


def prepare(options,*,restored_from=None):
    from .evaluate import load_base
    torch.manual_seed(SEED);native=Native(load_base(),mode='first-answer')
    try:
        critic=AudioResponseCritic(2048,2052).to(device='cuda:0',dtype=torch.float32)
        receipt=Path(options['initialization_receipt'])
        if restored_from is None:
            assert not receipt.exists(),'Fresh initialization already exists; resume saved state'
            value=initialize(native,critic,options['cohort_id']);write_json(receipt,{'sha256':value.sha256,'receipt':value.as_dict()})
        else:assert receipt.exists(),'Exact resume requires original scoped initializer receipt'
        stored=torch.load(Path(options['cache'])/'responses.pt',map_location='cpu',weights_only=False)
        rows=[PreparedAudioResponse(x.row,x.target.to('cuda:0'),x.neutral.to('cuda:0')) for x in stored]
        conditions=audio_conditions(rows);conditions[:,2048:2050]/=512.
        game=ConditionalPG09(native,rows,critic,conditions=conditions,particles=native.particles,
            response_domain='continuous_vector',penalty_input_domain='continuous',batch_size=8,seed=SEED,total_steps=1024,response_scale=1.,
            recipe_overrides={'lr':options['g_lr'],'d_lr_mult':options['d_lr']/options['g_lr'],'prior_lr_mult':options['bank_lr']/options['g_lr']})
        assert game.recipe.ema_decay==0.
        spec=inference_backend().specification(native,model_identity=MODEL_IDENTITY,protocol=PROTOCOL,
            configuration={'mode':'first-answer','rank':8,'alpha':8,'bank_shape':[128,4],'ema_decay':None})
        return Binding(game,native.close,inference=spec)
    except BaseException:native.close();raise


def response_backend():return ResponseBackend(NAME,CAPS,prepare,validate)


def validate_inference(options,spec):
    assert not options and spec.configuration=={'mode':'first-answer','rank':8,'alpha':8,'bank_shape':[128,4],'ema_decay':None}


def prepare_inference(options,spec):
    from .evaluate import load_base
    native=Native(load_base(),mode='first-answer',checkpointing=False)
    return InferenceBinding(native.network,native_identity(native,model_identity=MODEL_IDENTITY,protocol=PROTOCOL),native.close,response=native.response)


def inference_backend():return InferenceBackend(NAME,CAPS,prepare_inference,validate_inference)
