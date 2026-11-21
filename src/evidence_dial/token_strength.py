"""Request-scoped gains for Tiny Aya prompt prefill and cached decoding."""

from contextlib import ExitStack, contextmanager
from math import isfinite
from sys import float_info


@contextmanager
def token_strengths(model, scale_adapter, *, prefix_tokens, max_tokens, strength, step_gains):
    if type(strength) not in (float, int) or not -float_info.max <= strength <= float_info.max:
        raise ValueError("Tiny Aya nominal strength must be finite")
    if not isinstance(step_gains, (list, tuple)) or len(step_gains) != max_tokens:
        raise ValueError("Tiny Aya requires one gain per possible generated token")
    if any(type(gain) not in (int, float) or not 0 <= gain <= float_info.max for gain in step_gains):
        raise ValueError("Tiny Aya token gains must be finite and nonnegative")
    values = [strength * gain for gain in step_gains]
    if any(not isfinite(value) for value in values):
        raise ValueError("Tiny Aya effective strengths must be finite")

    scope = ExitStack()
    applied = []

    def before(_module, _args, kwargs):
        position = len(applied)
        ids, mask = kwargs["input_ids"], kwargs["attention_mask"]
        expected_width = 1 if position else prefix_tokens
        if (position >= max_tokens or ids.shape != (1, expected_width)
                or mask.shape != (1, prefix_tokens + position)):
            raise RuntimeError("Tiny Aya cached generation position differs from the requested schedule")
        scope.enter_context(scale_adapter(values[position]))
        applied.append(values[position])

    def after(_module, _args, _output):
        scope.close()

    with ExitStack() as handles:
        handles.callback(model.register_forward_pre_hook(before, with_kwargs=True).remove)
        handles.callback(model.register_forward_hook(after, always_call=True).remove)
        handles.callback(scope.close)
        yield applied
