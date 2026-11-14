"""Native export namespace for the unchanged first-answer long cohorts.

Only the registry name differs from the original first-answer inference backend.
Preparation, validation, native adapter, tensor schema and forward are reused.
"""
from particle_sliders.inference import InferenceBackend
from .binding import CAPS
from .first_answer_binding_v2 import prepare_inference, validate_inference


def inference_backend(mode):
    assert mode == 'first-answer'
    return InferenceBackend('tiny-aya-evidence-dial-first-answer-long-v1', CAPS,
        prepare_inference, validate_inference)
