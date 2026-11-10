"""Pinned AlignTune RAFT/SFT trainer on our canonical, untruncated prompt rows.

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


def train(data: Path, out: Path, gpu: int, *, recipe_path: Path | None = None):
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from peft import LoraConfig, get_peft_model
    from trl import SFTConfig
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "vendor/aligntune/src"))
    from aligntune.backends.trl.raft.raft_trainer import RaftTrainer, RaftTrainerConfig
    from .evaluate import load_base, tokenizer, NativeGenerator, evaluate_rows
    from .resources import Resources

    recipe = json.loads((recipe_path or root / "configs/aligntune.json").read_text())
    assert recipe['train_cases'] == len(read_jsonl(data / 'train.jsonl'))
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

    with Resources(gpu, out / "resources.json") as resource:
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
            assert len(ids)+len(completion) <= 512
            prepared.append({"id":row["id"], "input_ids":ids+completion,
                "labels":[-100]*len(ids)+completion, "attention_mask":[1]*(len(ids)+len(completion))})
        queue = deque()
        def collate(examples):
            width = max(len(x["input_ids"]) for x in examples)
            queue.append([x["id"] for x in examples])
            return {key:torch.tensor([x[key]+[pad]*(width-len(x[key])) for x in examples])
                for key,pad in [("input_ids",0),("attention_mask",0),("labels",-100)]}
        class Journal(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                resource.check()
                with (out / "updates.jsonl").open("a") as f:
                    f.write(json.dumps({"step":state.global_step, "case_ids":queue.popleft(), "ema_decay":None})+"\n")
                    f.flush();os.fsync(f.fileno())
                return control
        args = SFTConfig(output_dir=str(out), max_steps=128, per_device_train_batch_size=2,
            gradient_accumulation_steps=1, learning_rate=.0002, lr_scheduler_type="constant",
            warmup_steps=0, weight_decay=0., max_grad_norm=1., optim="adamw_torch",
            seed=SEED, data_seed=SEED, bf16=True, fp16=False, report_to=[],
            save_steps=32, save_total_limit=None, logging_steps=1,
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
        model.save_pretrained(out / "adapter")
        tok.save_pretrained(out / "adapter")
        write_json(out / "training-result.json", {"steps":trainer.state.global_step,
            "seconds_this_execution":time.perf_counter()-started, "metrics":result.metrics,
            "ema_decay":None, "weight_selection":"raw", "recipe":recipe,
            "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes":torch.cuda.max_memory_reserved()})
        model.gradient_checkpointing_disable()
        model.eval()
        gen = NativeGenerator(model, tok)
        evaluate_rows(read_jsonl(data / "dev.jsonl"), out / "dev", gen, arm="aligntune-lora", resource=resource)
