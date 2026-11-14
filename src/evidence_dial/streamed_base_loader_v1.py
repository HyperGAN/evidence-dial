"""Load the same complete native BF16 TinyAya directly onto its one GPU.

Execution-only alternative to constructing all weights on CPU then .to(cuda).
The model, snapshot, precision, SDPA implementation and seed remain unchanged.
No quantization, CPU offload, tensor slicing for inference or model forward.
"""
from pathlib import Path


def load_base():
    import torch
    from transformers import AutoModelForCausalLM
    from .evaluate import base_snapshot
    from .protocol import SEED
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.cuda.set_per_process_memory_fraction(22 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    model, loading = AutoModelForCausalLM.from_pretrained(base_snapshot(),
        torch_dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True,
        device_map={'': 'cuda:0'}, low_cpu_mem_usage=True, output_loading_info=True)
    assert not any(loading.get(key) for key in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']), loading
    assert all(value.device == torch.device('cuda:0') for value in model.parameters())
    assert all(value.device == torch.device('cuda:0') for value in model.buffers())
    model.requires_grad_(False)
    return model.eval()


def verify_snapshot_weights(model, check):
    """Bitwise compare every stored tensor in bounded CPU chunks; no forward."""
    import json
    import torch
    from safetensors import safe_open
    from .evaluate import base_snapshot
    from .benchmark import file_sha
    snapshot = Path(base_snapshot())
    index = snapshot / 'model.safetensors.index.json'
    value = json.loads(index.read_text())
    actual = model.state_dict()
    checked = set()
    elements = 0
    shards = []
    for name in sorted(set(value['weight_map'].values())):
        path = snapshot / name
        check()
        with safe_open(path, framework='pt', device='cpu') as file:
            for key in file.keys():
                assert value['weight_map'][key] == name and key in actual
                expected = file.get_tensor(key)
                tensor = actual[key]
                assert expected.shape == tensor.shape and tensor.dtype == torch.bfloat16
                a, b = expected.reshape(-1), tensor.reshape(-1)
                # At most8MiB of BF16 weights copied from GPU per comparison.
                for offset in range(0, a.numel(), 4 * 1024**2):
                    check()
                    host = b[offset:offset + 4 * 1024**2].detach().cpu()
                    assert torch.equal(host, a[offset:offset + 4 * 1024**2].to(dtype=tensor.dtype)), key
                checked.add(key)
                elements += tensor.numel()
        shards.append({'path': str(path), 'sha256': file_sha(path)})
    assert checked == set(value['weight_map'])
    extra = set(actual) - checked
    # HF can materialize the tied output head without a duplicate stored tensor.
    if extra:
        assert extra == {'lm_head.weight'} and model.config.tie_word_embeddings
        assert actual['lm_head.weight'].data_ptr() == actual['model.embed_tokens.weight'].data_ptr()
    assert all(not parameter.requires_grad for parameter in model.parameters()) and not model.training
    return {'stored_tensor_count': len(checked), 'materialized_tied_keys': sorted(extra),
        'checked_elements': elements, 'all_pretrained_tensor_values_bitwise_equal_after_declared_BF16_load': True,
        'index_sha256': file_sha(index), 'shards': shards, 'model_forward_calls': 0}
