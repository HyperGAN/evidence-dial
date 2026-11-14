"""Separate diagnostic arm: scale only critic token-length coordinates.

The model, paired targets, routed bank and official PG game are unchanged.
This does not modify the completed curated-v1 backend or its checkpoints.
"""
from pathlib import Path
import json
import torch
from particle_sliders.pg09 import ConditionalPG09
from particle_sliders.pg09_audio import AudioResponseCritic, audio_conditions
from particle_sliders.response import ResponseBackend
from particle_sliders.lifecycle import Binding
from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from .binding import initialize, CAPS, MODEL_IDENTITY
from .curated_binding import Native as CuratedNative, validate as validate_curated
from .benchmark import file_sha
from .io import write_json
from .protocol import SEED

NAME='tiny-aya-evidence-dial-lengthnorm-diagnostic-v1'
CONDITIONING='native-neutral-hidden;neutral-and-positive-lengths-div512;history-and-strength-unchanged-v1'


def critic_conditions(rows):
    value=audio_conditions(rows)
    assert value.shape[1]==2052
    value[:,2048:2050]/=512.
    return value


class Native(CuratedNative):
    def metadata(self):
        value=super().metadata()
        value['critic_conditioning']=CONDITIONING
        return value


def validate(options):
    validate_curated(options)
    if options.get('conditioning')!=CONDITIONING or options.get('diagnostic_binding_sha256')!=file_sha(__file__):
        raise ValueError('Only the explicitly source-bound length-normalization diagnostic is supported')


def prepare(options, *, restored_from=None):
    from .evaluate import load_base
    torch.manual_seed(SEED);native=Native(load_base())
    try:
        critic=AudioResponseCritic(2048,2052).to(device='cuda:0',dtype=torch.float32)
        receipt=Path(options['initialization_receipt'])
        if restored_from is None:
            if receipt.exists():raise ValueError('Fresh initialization already exists; preserve and resume')
            value=initialize(native,critic,options['cohort_id'])
            write_json(receipt,{'sha256':value.sha256,'receipt':value.as_dict()})
        elif not receipt.is_file():raise ValueError('Resume requires original initialization receipt')
        cache=Path(options['cache']);index=json.loads((cache/'index.json').read_text())
        assert index['responses_sha256']==file_sha(cache/'responses.pt')
        from particle_sliders.pg09_audio import PreparedAudioResponse
        original=torch.load(cache/'responses.pt',map_location='cpu',weights_only=False)
        rows=[PreparedAudioResponse(r.row,r.target.to('cuda:0'),r.neutral.to('cuda:0')) for r in original]
        conditions=critic_conditions(rows)
        game=ConditionalPG09(native,rows,critic,conditions=conditions,particles=native.particles,
            response_domain='continuous_vector',penalty_input_domain='continuous',batch_size=2,
            seed=SEED,total_steps=128,response_scale=1.,recipe_overrides={'lr':.001,'d_lr_mult':18.,'prior_lr_mult':30.})
        assert sorted(g['lr'] for g in game.opt_g.param_groups)==[.001,.03]
        assert all(abs(g['lr']-.018)<1e-12 for g in game.opt_d.param_groups)
        assert game.recipe.ema_decay==0.
        spec=INFERENCE.specification(native,model_identity=MODEL_IDENTITY,protocol=native.metadata()['protocol'],
            configuration={'rank':8,'alpha':8,'bank_shape':[128,4],'ema_decay':None,'critic_conditioning':CONDITIONING})
        return Binding(game,native.close,inference=spec)
    except BaseException:
        native.close();raise


def validate_inference(options,spec):
    if options or spec.configuration!={'rank':8,'alpha':8,'bank_shape':[128,4],'ema_decay':None,'critic_conditioning':CONDITIONING}:
        raise ValueError('Only source-bound RAW inference supported')


def prepare_inference(options,spec):
    from .evaluate import load_base
    native=Native(load_base(),checkpointing=False)
    return InferenceBinding(native.network,native_identity(native,model_identity=MODEL_IDENTITY,protocol=spec.identity['protocol']),native.close,response=native.response)


BACKEND=ResponseBackend(NAME,CAPS,prepare,validate)
INFERENCE=InferenceBackend(NAME,CAPS,prepare_inference,validate_inference)
