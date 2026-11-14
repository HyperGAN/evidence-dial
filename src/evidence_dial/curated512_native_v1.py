"""Metadata-only TRAIN512 Native; all numerical methods inherited unchanged."""
from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from .binding import CAPS, MODEL_IDENTITY
from .curated_first_answer_binding_v1 import Native as ParentNative

NAME = 'tiny-aya-evidence-dial-curated512-first-v1'
TRAIN_SHA = 'e7e0b57cc4e4338e3dad1040ac9b06886d3b4fd7648a4c3701f1685ffdbec5f7'

class Native(ParentNative):
    def metadata(self):
        value = super().metadata()
        value.update(protocol='cohere2-canonical-curated512-first-v1',
            target_convention='actual sampled first-answer hidden state in explicitly canonical-TRAIN-gold-informed private context; ordinary neutral question, no answer prefix, no terminal target; one target per separately curated512 TRAIN case')
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
