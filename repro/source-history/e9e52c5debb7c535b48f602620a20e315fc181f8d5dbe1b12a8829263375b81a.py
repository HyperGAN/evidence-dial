"""Explicit RAW distribution export; preserves original strict FULL/v2 packets.

The shared native-v2 packet is intentionally tied to its exact source/runtime
and local model configuration. This separate safetensors format distributes
only the same learned parameter values, by logical name, with their provenance.
It cannot resume training or silently migrate a scientific checkpoint.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import time

from .protocol import MODEL, REVISION, MAX_NEW_TOKENS, messages
from .io import write_json
from .benchmark import file_sha

FORMAT='evidence-dial-raw-routed-adapter-v1'


def export(full: Path, native: Path, out: Path, *, guard: Path):
    import torch
    from safetensors.torch import save_file
    from particle_sliders.lifecycle import read_packet
    from particle_sliders.inference import read_inference
    from particle_sliders.pg09 import _hash
    from particle_sliders.routed_projection import TinyParticleProjection
    from particle_sliders.reference import RoutedMLP
    if out.exists():
        raise FileExistsError('Distribution export is immutable')
    q=json.loads(guard.read_text())
    if q['status']!='pass' or q['checkpoint']['sha256']!=file_sha(full):
        raise ValueError('Actual exported FULL itself requires a qualified passing guard')
    checkpoint=read_packet(full)
    packet=read_inference(native)
    state=checkpoint['payload']['game']
    if packet['selection']!='raw' or checkpoint['payload']['ema']!={'decay':None,'values':None}:
        raise ValueError('Only explicit RAW/no-weight-EMA source states can be distributed')
    assert packet['origin']['training_manifest_sha256']==_hash(checkpoint['manifest'])
    assert packet['origin']['completed_steps']==state['completed_steps']==128
    names=state['config']['backend']['adapter_names']
    assert len(names)==len(packet['network'])
    tensors={}
    for i,name in enumerate(names):
        key=f'weights.{i}'
        value=packet['network'][key]
        assert torch.equal(value,state['network'][key]) and value.dtype==torch.float32
        tensors[name]=value.detach().cpu().contiguous()
    out.mkdir(parents=True)
    weights=out/'particle_adapter.safetensors'
    save_file(tensors,str(weights),metadata={'format':FORMAT,'weights_kind':'raw','ema_decay':'None'})
    write_json(out/'particle_adapter.json',{'format':FORMAT,'base_model':MODEL,'base_revision':REVISION,
        'weights_kind':'raw','ema_decay':None,'rank':8,'alpha':8,'bank_shape':[128,4],
        'target_modules':['q_proj','k_proj','v_proj','o_proj'],'native_hidden_width':2048,
        'weights_sha256':file_sha(weights),'keys':list(tensors),
        'tensor_schema':{k:{'shape':list(v.shape),'dtype':'float32'} for k,v in tensors.items()},
        'origin':packet['origin'],'source_full_sha256':file_sha(full),'source_native_v2_sha256':file_sha(native),
        'source_selection':'raw','inference_only':True,'training_resume_supported':False,
        'tensor_values_unchanged':True,'base_weights_included':False,
        'projection_source_sha256':file_sha(inspect.getsourcefile(TinyParticleProjection)),
        'router_source_sha256':file_sha(inspect.getsourcefile(RoutedMLP)),
        'verified_export_environment':{p:importlib.metadata.version(p) for p in ['torch','transformers','particle-sliders-core','particlegan']},
        'licence':'CC-BY-NC-4.0, inherited from Tiny Aya base',
        'qualification':'Distribution tensors equal committed FULL/RAW v2 exactly. Native generation equivalence is recorded separately.'})


class Portable:
    def __init__(self, directory, *, device='cuda:0'):
        import torch
        from safetensors.torch import load_file
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from particle_sliders.routed_projection import TinyParticleProjection
        from particle_sliders.reference import RoutedMLP
        from .binding import Slider
        if str(device).startswith('cuda'):
            gpu_device=torch.device(device)
            torch.cuda.set_per_process_memory_fraction(22*1024**3/torch.cuda.get_device_properties(gpu_device).total_memory,gpu_device)
        if Path(directory).is_dir():
            root=Path(directory)
        else:
            from huggingface_hub import snapshot_download
            root=Path(snapshot_download(str(directory),allow_patterns=['particle_adapter.json','particle_adapter.safetensors']))
        config=json.loads((root/'particle_adapter.json').read_text())
        if (config['format'],config['base_model'],config['base_revision'],config['weights_kind'],config['ema_decay'])!=(FORMAT,MODEL,REVISION,'raw',None):
            raise ValueError('Unknown adapter/base identity or averaged weights refused')
        if importlib.metadata.version('transformers')!='4.57.6':
            raise ValueError('Use the pinned native Cohere2 Transformers4.57.6 implementation')
        assert config['projection_source_sha256']==file_sha(inspect.getsourcefile(TinyParticleProjection))
        assert config['router_source_sha256']==file_sha(inspect.getsourcefile(RoutedMLP))
        assert config['weights_sha256']==file_sha(root/'particle_adapter.safetensors')
        from huggingface_hub import snapshot_download
        base=snapshot_download(MODEL,revision=REVISION,
            allow_patterns=['*.json','*.safetensors','*.jinja','*.model','*.txt'])
        self.tokenizer=AutoTokenizer.from_pretrained(base)
        self.model=AutoModelForCausalLM.from_pretrained(base,
            torch_dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval().requires_grad_(False)
        self.slider=Slider(self.model)
        try:
            values=load_file(str(root/'particle_adapter.safetensors'),device=device)
            assert set(values)==set(config['keys'])==set(self.slider.state_dict())
            for name,value in values.items():
                assert list(value.shape)==config['tensor_schema'][name]['shape'] and value.dtype==torch.float32
                assert torch.isfinite(value).all()
            self.slider.load_state_dict(values,strict=True)
            self.slider.eval().requires_grad_(False)
            for name,value in self.slider.state_dict().items():
                assert torch.equal(value,values[name])
        except BaseException:
            self.slider.close()
            raise
        self.config=config

    def generate(self, context, question, *, strength=1., options=None):
        import torch
        row={'context':context,'question':question,'options':options}
        device=next(self.model.parameters()).device
        inputs=self.tokenizer.apply_chat_template(messages(row),add_generation_prompt=True,
            tokenize=True,return_dict=True,return_tensors='pt').to(device)
        inputs.pop('token_type_ids',None)
        count=inputs['input_ids'].shape[-1]
        if count>2048:
            raise ValueError('Input exceeds2048 tokens; evidence is never truncated')
        start=time.perf_counter()
        with self.slider.scaled(strength),torch.inference_mode():
            output=self.model.generate(**inputs,do_sample=False,max_new_tokens=MAX_NEW_TOKENS,
                use_cache=True,eos_token_id=3,pad_token_id=self.tokenizer.pad_token_id)
        ids=output[0,count:].tolist()
        return {'output':self.tokenizer.decode(ids,skip_special_tokens=True),'output_ids':ids,
            'input_tokens':count,'output_tokens':len(ids),'seconds':time.perf_counter()-start,
            'strength':strength,'weights_kind':'raw','ema_decay':None}

    def close(self):
        self.slider.close()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
