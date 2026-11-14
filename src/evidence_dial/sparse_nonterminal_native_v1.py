"""Metadata-only TRAIN256 sparse nonterminal Native; all numerical methods inherited unchanged."""
from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from .binding import CAPS, MODEL_IDENTITY
from .curated_first_answer_binding_v1 import Native as ParentNative

NAME = 'tiny-aya-evidence-dial-sparse-nonterminal256-v1'
TRAIN_SHA = '1768cfc2d7a3213ccba70fddca844b2e50902ceb13e525f01ca45ef45465c11e'

class Native(ParentNative):
    def metadata(self):
        value = super().metadata()
        value.update(protocol='cohere2-canonical-sparse-nonterminal256-v1',
            target_convention='actual sampled hidden states in canonical-TRAIN-gold-informed private contexts; equal first/first/last-content views per TRAIN256 case; ordinary neutral question plus identical sampled answer prefix on both sides; all special/control/terminal targets excluded')
        return value

def validate_inference(options,spec):
    if options or spec.configuration != {'mode':'first-answer','rank':8,'alpha':8,'bank_shape':[128,4],'ema_decay':None}:
        raise ValueError('Only declared RAW curated512 inference is supported')

def prepare_inference(options,spec):
    from .evaluate import load_base
    native = Native(load_base(),mode='first-answer',checkpointing=False)
    return InferenceBinding(native.network,
        native_identity(native,model_identity=MODEL_IDENTITY,protocol=native.metadata()['protocol']),
        native.close,response=native.response)

def inference_backend(mode):
    assert mode == 'first-answer'
    return InferenceBackend(NAME,CAPS,prepare_inference,validate_inference)
