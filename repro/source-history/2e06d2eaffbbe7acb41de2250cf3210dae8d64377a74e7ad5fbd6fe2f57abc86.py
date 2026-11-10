"""Native HF generation through AuditKIT, with durable per-case commits."""
from __future__ import annotations
import collections
import json
import os
import time
from pathlib import Path

from .io import read_jsonl, write_json
from .protocol import MODEL, REVISION, MAX_NEW_TOKENS, SEED, digest, prompt, success, faith_diagnostics


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(base_snapshot(), local_files_only=True)


def base_snapshot():
    # Use the hub's actual resolved snapshot, independently of legacy
    # TRANSFORMERS_CACHE settings inherited from other projects on this host.
    from huggingface_hub import snapshot_download
    return snapshot_download(MODEL, revision=REVISION, local_files_only=True)


def load_base():
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.cuda.set_per_process_memory_fraction(22 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    model = AutoModelForCausalLM.from_pretrained(base_snapshot(),
        torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True)
    model.requires_grad_(False)
    return model.to("cuda:0").eval()


class NativeGenerator:
    def __init__(self, model, tok):
        self.model, self.tok, self.last = model, tok, None
        self.capture_path = None

    def __call__(self, texts):
        import torch
        outputs = []
        for text in texts:
            if self.capture_path and self.capture_path.exists():
                captured = json.loads(self.capture_path.read_text())
                if captured["prompt_sha256"] != digest(text):
                    raise ValueError("Native capture prompt identity changed")
                self.last = captured["native"]
                outputs.append(captured["output"])
                continue
            inputs = self.tok.apply_chat_template([{"role": "user", "content": text}],
                add_generation_prompt=True, tokenize=True, return_dict=True,
                return_token_type_ids=False, return_tensors="pt").to("cuda:0")
            # Cohere's tokenizer emits this BERT-style field even when the
            # chat-template helper ignores return_token_type_ids; Cohere2's
            # causal model has no such input. Token IDs/mask remain untouched.
            inputs.pop("token_type_ids", None)
            n = inputs["input_ids"].shape[-1]
            if n > 2048:
                raise ValueError("Frozen eligibility violation; evidence must never be truncated")
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                generated = self.model.generate(**inputs, do_sample=False,
                    max_new_tokens=MAX_NEW_TOKENS, use_cache=True,
                    pad_token_id=self.tok.pad_token_id,
                    eos_token_id=self.model.generation_config.eos_token_id)
            torch.cuda.synchronize()
            ids = generated[0, n:].tolist()
            output = self.tok.decode(ids, skip_special_tokens=True)
            self.last = {"input_ids": inputs["input_ids"][0].tolist(), "output_ids": ids,
                "input_tokens": n, "output_tokens": len(ids),
                "seconds": time.perf_counter() - started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
            if self.capture_path:
                write_json(self.capture_path, {"output": output, "native": self.last, "prompt_sha256": digest(text)})
            outputs.append(output)
        return outputs


def identity(rows, arm, teacher):
    import importlib.metadata as md
    from . import protocol
    return {"base": MODEL, "revision": REVISION, "arm": arm, "teacher": teacher,
        "data_sha256": digest(rows), "prompt_sha256": digest(protocol.PROMPT),
        "scorer_source_sha256": digest(Path(protocol.__file__).read_text()),
        "generation_source_sha256": digest(Path(__file__).read_text()),
        "decoding": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS, "native_eos": True},
        "ema_decay": None, "weights": "raw-pretrained" if arm == "off" else arm,
        "environment": {p: md.version(p) for p in ["torch", "transformers", "auditkit", "curatorkit"]}}


def evaluate_rows(rows, out: Path, generator, *, arm="off", teacher=False, resource=None):
    import auditkit as ak
    from auditkit.model import CallableModel
    out.mkdir(parents=True, exist_ok=True)
    ident = identity(rows, arm, teacher)
    idpath = out / "identity.json"
    if idpath.exists():
        if json.loads(idpath.read_text()) != ident:
            raise ValueError("Run identity changed; use a new immutable run directory")
    else:
        write_json(idpath, ident)
    saved = read_jsonl(out / "predictions.jsonl") if (out / "predictions.jsonl").exists() else []
    done = {r["id"] for r in saved}
    if len(done) != len(saved) or not done <= {r["id"] for r in rows}:
        raise ValueError("Duplicate or foreign committed case IDs")

    @ak.scorer(direction=ak.Direction.MAXIMIZE)
    def strict_grounded_success(sample, output):
        return float(success(sample.metadata, output))

    model = CallableModel(generator, name="evidence-dial-" + digest(ident))
    started = time.perf_counter()
    for i, row in enumerate(rows):
        if row["id"] in done:
            continue
        if resource:
            resource.check()
        text = prompt(row, teacher=teacher)
        sample = ak.Sample(id=row["id"], input=text, target=row["gold"], metadata=row)
        generator.capture_path = out / "native" / f"{row['id']}.json"
        result = ak.evaluate([sample], model=model, scorers=[strict_grounded_success],
            experiment_name=digest({"run": ident, "id": row["id"], "prompt": text}),
            config=ak.RunConfig(max_tokens=MAX_NEW_TOKENS, temperature=0, max_retries=0, concurrency=1))
        if result.failed_count or result.errors or len(result.predictions) != 1:
            write_json(out / f"unmeasured-{row['id']}.json", result.to_dict())
            raise RuntimeError(f"Generation/scoring unmeasured for {row['id']}; result retained")
        pred = result.predictions[0]
        # Retain complete AuditKIT output as evidence, plus our native timing/tokens.
        raw = result.to_dict()
        output = pred.raw_output
        if generator.capture_path.exists():
            capture = json.loads(generator.capture_path.read_text())
            assert capture["output"] == output and capture["prompt_sha256"] == digest(text)
            generator.last = capture["native"]
        if generator.last is None:
            raise RuntimeError("Cached result has no authoritative native capture")
        record = {"id": row["id"], "family": row["family"], "dataset": row["dataset"],
            "case_type": row["case_type"], "arm": arm, "teacher": teacher,
            "output": output, "success": success(row, output), "gold": row["gold"],
            "prompt_sha256": digest(text), "native": generator.last,
            "auditkit": raw}
        if row["dataset"].startswith("faith-"):
            record.update(faith_diagnostics(row, output))
        with (out / "predictions.jsonl").open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        saved.append(record)
        done.add(row["id"])
        print(json.dumps({"committed": len(saved), "total": len(rows), "id": row["id"],
            "success": record["success"], "seconds": generator.last["seconds"]}), flush=True)
    groups = collections.defaultdict(list)
    for r in saved:
        groups[r["case_type"]].append(r["success"])
    summary = {"arm": arm, "teacher": teacher, "complete": len(saved) == len(rows),
        "measured": len(saved), "declared": len(rows), "success_rate": sum(r["success"] for r in saved)/len(saved),
        "by_type": {k: {"n": len(v), "success_rate": sum(v)/len(v)} for k, v in groups.items()},
        "this_execution_seconds": time.perf_counter() - started,
        "generation_seconds": sum(r["native"]["seconds"] for r in saved),
        "generated_tokens": sum(r["native"]["output_tokens"] for r in saved)}
    write_json(out / "summary.json", summary)
    return summary


def headroom(data: Path, out: Path, gpu: int):
    from .resources import Resources
    rows = read_jsonl(data / "dev.jsonl")
    if len(rows) != 128 or any(r["split"] != "dev" for r in rows):
        raise ValueError("Headroom uses exactly the frozen 128 development cases")
    with Resources(gpu, out / "resource-admission.json") as resource:
        gen = NativeGenerator(load_base(), tokenizer())
        off = evaluate_rows(rows, out / "off", gen, resource=resource)
        teacher = evaluate_rows(rows, out / "teacher", gen, teacher=True, resource=resource)
    gate = {"off": off, "teacher": teacher,
        "teacher_beats_off": teacher["success_rate"] > off["success_rate"],
        "off_failure_count": 128 - round(128 * off["success_rate"]),
        "holdout_generated": False,
        "next": "review headroom by case type before training or binding"}
    write_json(out / "gate.json", gate)
    return gate
