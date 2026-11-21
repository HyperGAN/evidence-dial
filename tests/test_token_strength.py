from contextlib import contextmanager
from threading import RLock

import torch
from transformers import BatchEncoding, Cohere2Config, Cohere2ForCausalLM

from evidence_dial import portable


def test_portable_raw_generation_uses_cached_token_gains(monkeypatch):
    monkeypatch.setattr(portable, "MAX_NEW_TOKENS", 3)
    model = Cohere2ForCausalLM(Cohere2Config(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, pad_token_id=0,
        bos_token_id=1, eos_token_id=3, sliding_window=16,
    )).eval()

    class Tokenizer:
        pad_token_id = 0

        def apply_chat_template(self, _messages, **_kwargs):
            return BatchEncoding({"input_ids": torch.tensor([[1, 4, 5]]),
                                  "attention_mask": torch.ones(1, 3, dtype=torch.long)})

        def decode(self, ids, **_kwargs):
            return ",".join(str(value) for value in ids)

    class Slider:
        active = 0.0

        @contextmanager
        def scaled(self, value):
            previous = self.active
            self.active = value
            try:
                yield
            finally:
                self.active = previous

    instance = portable.Portable.__new__(portable.Portable)
    instance.model, instance.tokenizer, instance.slider = model, Tokenizer(), Slider()
    instance._inference_lock = RLock()
    request = {"context": "Blue is a color.", "question": "What is blue?", "strength": 0.5}
    original = instance.generate(**request)
    constant = instance.generate(**request, step_gains=[1.0] * 3)
    assert constant["output_ids"] == original["output_ids"]
    assert constant["effective_strengths"] == [0.5] * constant["output_tokens"]
    observed = []
    probe = model.model.register_forward_pre_hook(lambda *_: observed.append(instance.slider.active))
    try:
        timed = instance.generate(**request, step_gains=[1.5, 1, 0.5])
    finally:
        probe.remove()
    assert observed == timed["effective_strengths"] == [0.75, 0.5, 0.25][:timed["output_tokens"]]
    assert instance.slider.active == 0 and not model._forward_pre_hooks and not model._forward_hooks
