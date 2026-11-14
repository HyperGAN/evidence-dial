"""Prospective matched TRAIN512 SFT control, fresh fixed1536 with guarded segments.

Pinned AlignTune RAFT/SFT trainer on canonical, untruncated prompt rows.

RAFT's optional citation branch is disabled (upstream placeholder). We feed
pretokenized prompt/completion rows to preserve the exact comparison protocol.
"""
from __future__ import annotations
from collections import deque
import json
import os
from pathlib import Path
import sys
import time

from .io import read_jsonl, write_json
from .protocol import SEED, digest, messages


def train(data: Path, out: Path, gpu: int, *, recipe_path: Path | None = None, stop=1536, attempt="first"):
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from peft import LoraConfig, get_peft_model
    from trl import SFTConfig
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "vendor/aligntune/src"))
    from aligntune.backends.trl.raft.raft_trainer import RaftTrainer, RaftTrainerConfig
    from .evaluate import load_base, tokenizer
    from .durable_evaluate_v1 import NativeGenerator,evaluate_rows
    from .shared_gpu_resources_v1 import SharedGPU1Resources as FollowupResources
    from .qualified_text_guard_admission_v1 import passing_guard
    from .benchmark import file_sha

    recipe = json.loads((recipe_path or root / "configs/aligntune.json").read_text())
    assert recipe['train_cases'] == len(read_jsonl(data / 'train.jsonl')) == 512
    assert recipe['steps'] == 1536 and stop in [2,32,64,384,768,1152,1536]
    assert gpu == 1 and file_sha(data/'train.jsonl') == 'e7e0b57cc4e4338e3dad1040ac9b06886d3b4fd7648a4c3701f1685ffdbec5f7'
    out.mkdir(parents=True, exist_ok=True)
    ident = {"recipe": recipe, "train_sha256":digest(read_jsonl(data / "train.jsonl")),
        "source_sha256":digest(Path(__file__).read_text()),
        "training_module": "aligntune.backends.trl.raft.raft_trainer.RaftTrainer",
        "environment": {p: __import__('importlib.metadata',fromlist=['version']).version(p) for p in ["torch","transformers","peft","trl","datasets","accelerate","pyarrow","fsspec","dill","multiprocess"]}}
    if (out / "identity.json").exists():
        if json.loads((out / "identity.json").read_text()) != ident:
            raise ValueError("Immutable baseline identity changed")
    else:
        write_json(out / "identity.json", ident)
    completed = sorted(out.glob("checkpoint-*"), key=lambda p:int(p.name.split("-")[-1]))
    if (out / "training-result.json").exists():
        raise ValueError("Baseline completed; use the saved adapter rather than retrain")

    if (out/'candidate-failed.json').exists() or list(out.glob('candidate-confirmed-failed-*.json')):
        raise RuntimeError('Qualified catastrophe stops this SFT control')
    if completed:
        current = int(completed[-1].name.split('-')[-1])
        assert current < stop
        artifact = out/f'adapter-{current:06d}'/'adapter_model.safetensors'
        assert passing_guard(out,current,artifact) is not None
        journal = read_jsonl(out/'updates.jsonl')
        assert [r['step'] for r in journal] == list(range(1,current+1))
    else:
        assert not (out/'updates.jsonl').exists()
    session = {'backend':'AlignTune-RaftTrainer-SFT-curated512', 'steps':1536,
        'batch':2,'g_lr':.0002,'ema_decay':None,'training_sha256':file_sha(data/'train.jsonl')}
    if (out/'session.json').exists():
        assert json.loads((out/'session.json').read_text()) == session
    else:
        write_json(out/'session.json',session)
    with FollowupResources(gpu, out / f"resources-{stop}-{attempt}.json", plan=out.parent/"execution-plan.json") as resource:
        tok = tokenizer()
        base = load_base()
        model = get_peft_model(base, LoraConfig(r=8, lora_alpha=8, lora_dropout=0.,
            target_modules=["q_proj","k_proj","v_proj","o_proj"], bias="none", task_type="CAUSAL_LM"))
        # Preserve the host LoRA law; all pretrained tensors stay frozen.
        assert all(torch.count_nonzero(p) == 0 for n,p in model.named_parameters() if "lora_B" in n)
        train_rows = read_jsonl(data / "train.jsonl")
        prepared = []
        for row in train_rows:
            ids = tok.apply_chat_template(messages(row), add_generation_prompt=True)
            completion = tok.encode(row["gold"], add_special_tokens=False) + [261001, 3]
            assert len(ids)+len(completion) <= 768
            prepared.append({"id":row["id"], "input_ids":ids+completion,
                "labels":[-100]*len(ids)+completion, "attention_mask":[1]*(len(ids)+len(completion))})
        queue = deque()
        def collate(examples):
            width = max(len(x["input_ids"]) for x in examples)
            queue.append([x["id"] for x in examples])
            return {key:torch.tensor([x[key]+[pad]*(width-len(x[key])) for x in examples])
                for key,pad in [("input_ids",0),("attention_mask",0),("labels",-100)]}
        pending = []
        class Journal(TrainerCallback):
            def on_train_begin(self,args,state,control,**kwargs):
                if completed:
                    from safetensors.torch import load_file
                    from peft import get_peft_model_state_dict
                    from .train import tensor_equal
                    def cpu(value):
                        if isinstance(value,torch.Tensor): return value.detach().cpu()
                        if isinstance(value,dict): return {k:cpu(v) for k,v in value.items()}
                        if isinstance(value,(list,tuple)): return type(value)(cpu(v) for v in value)
                        return value
                    saved = completed[-1]
                    expected = load_file(str(saved/'adapter_model.safetensors'))
                    actual = get_peft_model_state_dict(kwargs['model'])
                    assert actual.keys() == expected.keys()
                    assert all(torch.equal(actual[k].detach().cpu(),expected[k]) for k in expected)
                    optimizer = torch.load(saved/'optimizer.pt',map_location='cpu',weights_only=False)
                    assert tensor_equal(cpu(kwargs['optimizer'].state_dict()),cpu(optimizer))
                    assert state.global_step == int(saved.name.split('-')[-1])
                    write_json(out/f'restore-before-{stop}-{attempt}.json',{'restored_step':state.global_step,
                        'adapter_values_exact':True,'optimizer_values_exact':True,'fresh_updates_before_witness':0,'ema_decay':None})
                else:
                    assert state.global_step == 0
                    assert all(torch.count_nonzero(p)==0 for n,p in kwargs['model'].named_parameters() if 'lora_B' in n)
                    write_json(out/f'fresh-before-{stop}-{attempt}.json',{'step':0,'zero_output_LoRA':True,'ema_decay':None})
                return control
            def on_step_end(self, args, state, control, **kwargs):
                resource.check()
                pending.append({"step":state.global_step, "case_ids":queue.popleft(), "ema_decay":None})
                if state.global_step >= stop:
                    control.should_training_stop = True
                return control
            def on_save(self, args, state, control, **kwargs):
                # The shared trainer saved weights/optimizer/RNG before a row
                # exposure becomes committed. Resume only that saved prefix.
                with (out / "updates.jsonl").open("a") as f:
                    for item in pending:
                        f.write(json.dumps(item)+"\n")
                    f.flush(); os.fsync(f.fileno())
                pending.clear()
                return control
        args = SFTConfig(output_dir=str(out), max_steps=1536, per_device_train_batch_size=2,
            gradient_accumulation_steps=1, learning_rate=.0002, lr_scheduler_type="constant",
            warmup_steps=0, weight_decay=0., max_grad_norm=1., optim="adamw_torch",
            seed=SEED, data_seed=SEED, bf16=True, fp16=False, report_to=[],
            save_steps=1, save_total_limit=None, logging_steps=1,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant":False},
            packing=False, max_length=None, dataset_kwargs={"skip_prepare_dataset":True},
            remove_unused_columns=False, dataloader_num_workers=0, push_to_hub=False,
            dataloader_drop_last=recipe.get('dataloader_drop_last', False),
            eos_token="<EOS_TOKEN>", completion_only_loss=True)
        trainer = RaftTrainer(model=model, args=args, train_dataset=Dataset.from_list(prepared),
            processing_class=tok, data_collator=collate, callbacks=[Journal()],
            raft_config=RaftTrainerConfig(use_citation_loss=False, use_doc_ranking_loss=False))
        started = time.perf_counter()
        result = trainer.train(resume_from_checkpoint=str(completed[-1]) if completed else None)
        model.save_pretrained(out / f"adapter-{trainer.state.global_step:06d}")
        tok.save_pretrained(out / f"adapter-{trainer.state.global_step:06d}")
        write_json(out / ("training-result.json" if trainer.state.global_step == 1536 else f"segment-{trainer.state.global_step:06d}.json"), {"steps":trainer.state.global_step,
            "seconds_this_execution":time.perf_counter()-started, "metrics":result.metrics,
            "ema_decay":None, "weight_selection":"raw", "recipe":recipe,
            "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes":torch.cuda.max_memory_reserved()})
        model.gradient_checkpointing_disable()
        model.eval()
        gen = NativeGenerator(model, tok)
        from .protocol import prompt
        probes = []
        for kind in ["answerable", "missing", "conflict", "altered"]:
            row = next(r for r in read_jsonl(data/"dev.jsonl") if r["case_type"] == kind)
            gen.capture_path = out/"probe-native"/f"{trainer.state.global_step:06d}-{row['id']}.json"
            text = gen([prompt(row)])[0]
            probes.append({"id":row["id"],"output":text,"native":gen.last})
        write_json(out/f"probes-{trainer.state.global_step:06d}.json", {"checkpoint":trainer.state.global_step,"strength":1,"probes":probes,"selection_metric_used":False})
