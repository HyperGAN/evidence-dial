from threading import RLock

import pytest
import torch
from transformers import BatchEncoding, Cohere2Config, Cohere2ForCausalLM

from evidence_dial import binding, portable
from evidence_dial.adaptive_strength import CachedGate, CachedPromptSteering
from particle_sliders.adaptive_steering import CLABSParameters
from particle_sliders.prompt_steering import PromptSteering


@pytest.fixture
def native(monkeypatch):
    monkeypatch.setattr(portable, "MAX_NEW_TOKENS", 3)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        model = Cohere2ForCausalLM(Cohere2Config(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, pad_token_id=0,
            bos_token_id=1, eos_token_id=3, sliding_window=16,
        )).eval()
        model.generation_config.min_new_tokens = 2
        monkeypatch.setattr(binding, "targets", lambda model: {
            "model.layers.0.self_attn.o_proj": model.model.layers[0].self_attn.o_proj,
        })
        slider = binding.Slider(model)
        # The trained RAW projection would have a nonzero learned up branch.
        with torch.no_grad():
            slider.hosts[0][1].lora_up.weight.normal_()

    class Tokenizer:
        pad_token_id = 0

        def apply_chat_template(self, _messages, **_kwargs):
            return BatchEncoding({"input_ids": torch.tensor([[1, 4, 5]]),
                                  "attention_mask": torch.ones(1, 3, dtype=torch.long)})

        def decode(self, ids, **_kwargs):
            return ",".join(str(value) for value in ids)

    instance = portable.Portable.__new__(portable.Portable)
    instance._inference_lock = RLock()
    instance.model, instance.slider, instance.tokenizer = model, slider, Tokenizer()
    instance.config = {"base_revision": "test-model", "weights_sha256": "test-raw"}
    try:
        yield instance
    finally:
        instance.close()


def _run(native, **options):
    logits = []
    handle = native.model.register_forward_hook(
        lambda _module, _args, output: logits.append(output.logits.clone()))
    try:
        result = native.generate("Blue is a color.", "What is blue?", **options)
    finally:
        handle.remove()
    return result, torch.cat(logits, dim=1)


def _axis_direction():
    direction = torch.zeros(16)
    direction[0] = 1
    return direction


def test_raw_gate_matches_native_off_and_fixed_cached_logits_exactly(native):
    off, off_logits = _run(native, strength=0)
    fixed, fixed_logits = _run(native, strength=0.5, step_gains=[0, 1, 1])
    assert not torch.equal(off_logits[:, 1:], fixed_logits[:, 1:])
    for threshold, expected, expected_logits, effective in [
        (-1e30, off, off_logits, 0.0), (1e30, fixed, fixed_logits, 0.5),
    ]:
        result, logits = _run(native, strength=0.5, adaptive=CachedGate(
            layer=0, method="cfp", parameter=threshold, intervention="raw_adapter",
            direction=_axis_direction(), evidence_ref="test fixture"))
        assert torch.equal(logits, expected_logits)
        assert result["output_ids"] == expected["output_ids"]
        assert result["effective_strengths"] == [0, effective, effective]
        trace = result["adaptive_steering"]
        assert trace["usage"] == {"fired": 2 if effective else 0, "eligible": 2}
        assert trace["raw_sha256"] == "test-raw"
        assert [step["output_index"] for step in trace["cached_steps"]] == [1, 2]
        assert [step["cache_position"] for step in trace["cached_steps"]] == [3, 4]
    assert native.slider.hosts[0][1].multiplier == 0
    assert not native.model._forward_pre_hooks and not native.model._forward_hooks
    assert not native.model.model.layers[0]._forward_pre_hooks


def test_direct_edit_reads_its_contextual_residual_before_intervention(native):
    projection = _axis_direction()
    policy = CachedGate(layer=1, method="cfp", parameter=1e30, intervention="residual",
                        direction=projection, evidence_ref="test fixture")
    inputs = []
    handle = native.model.model.layers[1].register_forward_pre_hook(
        lambda _module, args: inputs.append(args[0].clone()))
    try:
        off, off_logits = _run(native, strength=0)
        inputs.clear()
        result, logits = _run(native, strength=-0.5, adaptive=policy)
    finally:
        handle.remove()
    trace = result["adaptive_steering"]
    assert [step["alignment"] for step in trace["cached_steps"]] == [
        -input[0, 0, 0].item() for input in inputs[1:]]
    assert trace["intervention"] == "residual" and "raw_sha256" not in trace
    assert result["effective_strengths"] == [0, -0.5, -0.5]
    torch.testing.assert_close(logits[:, :1], off_logits[:, :1], rtol=0, atol=0)
    assert not torch.equal(logits[:, 1:], off_logits[:, 1:])
    assert result["output_tokens"] == off["output_tokens"] == 3


def test_clabs_state_is_request_local_and_psr_applies_fitted_coefficients(native):
    clabs = CLABSParameters(mean=0, std=1, epsilon=1e-6, setpoint=0.75,
                            proportional_gain=1, integral_gain=0.2, beta=0.2)
    policy = CachedGate(layer=1, method="clabs", parameter=0.5, intervention="residual",
                        direction=_axis_direction(), evidence_ref="test fixture", seed=23, clabs=clabs)
    first, _ = _run(native, strength=0.5, adaptive=policy)
    second, _ = _run(native, strength=0.5, adaptive=policy)
    assert first["adaptive_steering"] == second["adaptive_steering"]
    fitted = PromptSteering(16).eval().requires_grad_(False)
    with torch.no_grad():
        fitted.probe.weight.zero_()
        fitted.probe.bias.fill_(2)
        fitted.direction.weight.copy_(_axis_direction().unsqueeze(0))
    result, logits = _run(native, strength=0.5, adaptive=CachedPromptSteering(
        layer=1, fitted=fitted, evidence_ref="test fixture"))
    _, residual_logits = _run(native, strength=1, adaptive=CachedGate(
        layer=1, method="cfp", parameter=1e30, intervention="residual",
        direction=_axis_direction(), evidence_ref="test fixture"))
    assert result["effective_strengths"] == [0, 1, 1]
    assert result["adaptive_steering"]["method"] == "s_psr"
    assert torch.equal(logits, residual_logits)


def test_adaptive_requires_explicit_nonconflicting_input_and_cleans_up_failure(native):
    policy = CachedGate(layer=0, method="random", parameter=1, intervention="raw_adapter",
                        evidence_ref="test fixture", seed=2)
    with pytest.raises(ValueError, match="explicit strength"):
        _run(native, adaptive=policy)
    with pytest.raises(ValueError, match="no fixed step_gains"):
        _run(native, strength=0.5, adaptive=policy, step_gains=[0, 1, 1])

    host, adapter = native.slider.hosts[0]

    def fail_while_scaled(_module, _args, _result):
        if adapter.multiplier:
            raise RuntimeError("test fault after adaptive activation")

    handle = host.register_forward_hook(fail_while_scaled)
    try:
        with pytest.raises(RuntimeError, match="test fault"):
            _run(native, strength=0.5, adaptive=policy)
    finally:
        handle.remove()
    assert adapter.multiplier == 0
    assert not native.model._forward_pre_hooks and not native.model._forward_hooks
    assert not native.model.model.layers[0]._forward_pre_hooks
    _run(native, strength=0)


def test_native_eos_in_prefill_leaves_the_gate_unused(native):
    native.model.generation_config.min_new_tokens = 0

    def emit_eos(_module, _args, output):
        output.logits.zero_()
        output.logits[..., 3] = 1

    handle = native.model.register_forward_hook(emit_eos)
    try:
        result, _ = _run(native, strength=0.5, adaptive=CachedGate(
            layer=0, method="random", parameter=1, intervention="raw_adapter",
            evidence_ref="test fixture", seed=2))
    finally:
        handle.remove()
    assert result["output_ids"] == [3]
    assert result["effective_strengths"] == [0]
    assert result["adaptive_steering"]["cached_steps"] == []
    assert result["adaptive_steering"]["usage"] == {"fired": 0, "eligible": 0}
