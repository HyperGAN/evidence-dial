"""Complete-answer inference for the repaired behavior artifact.

Use inside an active bounded runner, passing its resource monitor. Configure
CUDA visibility before importing/loading the native model. This owner calls
the same full-answer generator as the experiments and never the legacy QA
response callback. It performs no training, initialization or Jev calls.
"""
from contextlib import contextmanager
import hashlib
import math
from pathlib import Path

from particle_sliders.inference import open_inference, read_inference
from .behavior_fullanswer_generate_v2 import generate
from .behavior_promptfix_binding_v4 import inference_backend
from .evaluate import tokenizer


def _sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


class BehaviorInference:
    """One loaded raw adapter; create through open_behavior_inference()."""

    def __init__(self, native, tok, capabilities, resource, artifact_sha256):
        self._native, self._tok = native, tok
        self._capabilities, self._resource = capabilities, resource
        self.artifact_sha256 = artifact_sha256
        self._closed = False

    def capture_identity(self, strength):
        """Bind a saved answer to the exact artifact and round-trip strength."""
        value = float(strength)
        if not math.isfinite(value):
            raise ValueError('A finite strength is required')
        self._capabilities.check('response', strength=value)
        return f'behavior-raw-{self.artifact_sha256}-strength-{value.hex()}'

    def respond(self, *, text, strength=1., capture_path):
        """Return a durable complete native capture, requiring an EOS answer.

        A 4,096-token safety cutoff is saved unfinished and raises. Existing
        captures must match the question, artifact, strength and decoder;
        unresolved native firsts are retained without automatic replay.
        """
        if self._closed:
            raise RuntimeError('Behavior inference context is closed')
        if not isinstance(text, str) or not text.strip():
            raise ValueError('An ordinary user question is required')
        value = float(strength)
        arm = self.capture_identity(value)
        self._resource.check()
        with self._native.slider.scaled(value):
            return generate(self._native.model, self._tok, text, Path(capture_path),
                self._resource, arm=arm, target=None)


@contextmanager
def open_behavior_inference(artifact, *, resource, expected_sha256=None):
    """Load a raw repaired-study artifact without reconstructing training.

    ``resource`` is the active runner monitor and must expose ``check()``.
    ``respond()`` retains the ordinary question and uses no behavior prompt.
    The underlying native model is released when the context closes.
    """
    if not callable(getattr(resource, 'check', None)):
        raise TypeError('An active resource monitor with check() is required')
    resource.check()
    path = Path(artifact)
    sha = _sha256(path)
    if expected_sha256 is not None and sha != expected_sha256:
        raise ValueError('Behavior artifact SHA-256 differs from the expected raw artifact')
    packet = read_inference(path)
    if _sha256(path) != sha:
        raise ValueError('Behavior artifact changed while it was being read')
    if packet['selection'] != 'raw' or packet['specification']['configuration'].get('ema_decay') is not None:
        raise ValueError('Raw weights with weight EMA disabled are required')
    # Pass the already validated packet, so model loading uses those exact
    # tensors rather than reading a potentially changed path again.
    with open_inference(packet, backend=inference_backend(), operation='response') as inference:
        native = inference.binding.response.__self__
        owner = BehaviorInference(native, tokenizer(), inference.capabilities, resource, sha)
        try:
            resource.check()
            yield owner
        finally:
            owner._closed = True
    resource.check()
