"""Native first-capture execution guards, preserving original decoding and scorer.

Never repeat an unresolved native generation or accept an AuditKIT cache entry
without its exact native token capture. No model/criterion/formulation change.
"""
import collections
import json
import os
from pathlib import Path
import time

from .evaluate import NativeGenerator as ParentNativeGenerator,identity as parent_identity
from .io import read_jsonl,write_json
from .protocol import digest,prompt,success,MAX_NEW_TOKENS
from .faith_diagnostics import faith_diagnostics
from .benchmark import file_sha


class NativeGenerator(ParentNativeGenerator):
    def __call__(self,texts):
        if self.capture_path is None or len(texts)!=1:
            raise ValueError('One exact durable native capture path required per call')
        path=Path(self.capture_path)
        if not path.exists():
            pending=path.with_suffix('.pending.json')
            if pending.exists():
                raise RuntimeError('Unresolved native generation first; preserve without automatic replay')
            write_json(pending,{'prompt_sha256':digest(texts[0]),'state':'native-generation-first-pending',
                'execution_wrapper_sha256':file_sha(__file__)})
        result=super().__call__(texts)
        assert path.exists(),'A completed native call must have a durable capture'
        return result


def identity(rows,arm,teacher):
    value=parent_identity(rows,arm,teacher)
    value['native_capture_execution_source_sha256']=file_sha(__file__)
    return value


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
        generator.last = None
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
        if not generator.capture_path.exists():
            raise RuntimeError('Cached evaluation lacks its authoritative native first; leave unmeasured')
        if generator.capture_path.exists():
            capture = json.loads(generator.capture_path.read_text())
            assert capture["output"] == output and capture["prompt_sha256"] == digest(text)
            expected_ids = generator.tok.apply_chat_template([{'role':'user','content':text}], add_generation_prompt=True)
            assert capture["native"]["input_ids"] == expected_ids
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

