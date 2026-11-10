"""Freeze external transfer samples without generating any model outputs."""
from __future__ import annotations
import random
import re
from pathlib import Path
from .io import write_json, write_jsonl
from .protocol import SEED, digest, messages

REVISIONS = {
    "unanswerable": "4a14e0e9860e270a8c9ff57c511821f4f50dce57",
    "inconsistent": "e29508db3955db2c82a80a1f3c3aadc11f93fc3f",
    "counterfactual": "e655f7c8750aabe431eeeda63fa5c019a2567b18",
}


def split_control(row):
    docs = re.split(r"(?:^|\s)Document:\s*", row["context"])
    docs = [d.strip() for d in docs if d.strip()]
    if len(docs) != 2 or len(row["answers"]) != 2:
        return None, "not_two_documents_and_answers"
    a, b = row["answers"]
    def has(doc, text):
        return bool(re.search(r"(?<!\w)" + re.escape(text) + r"(?!\w)", doc, flags=re.I))
    if not (has(docs[0], a) and not has(docs[1], a) and has(docs[1], b) and not has(docs[0], b)):
        return None, "original_answer_not_exclusively_supported_by_original_document"
    return docs[0], None


def freeze(out: Path, tokenizer):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    if out.exists():
        raise FileExistsError(f"Frozen external cohort already exists: {out}")
    source, exclusions, eligible, selected, manifests = {}, [], {}, [], {}
    for name, revision in REVISIONS.items():
        repo = f"Salesforce/FaithEval-{name}-v1.0"
        path = hf_hub_download(repo, "data/test-00000-of-00001.parquet", repo_type="dataset", revision=revision)
        source[name] = pq.read_table(path).to_pylist()
        manifests[name] = {"repo": repo, "revision": revision, "total": len(source[name])}
        eligible[name] = []
        for raw in source[name]:
            qid = str(raw.get("qid", raw.get("id")))
            kind = {"unanswerable": "missing", "inconsistent": "conflict", "counterfactual": "altered"}[name]
            gold = {"missing": "UNKNOWN", "conflict": "CONFLICT"}.get(kind, raw.get("answerKey"))
            row = {"id": f"faith-{name}-{qid}", "family": "faith-question-" + digest(raw["question"])[:20],
                "dataset": f"faith-{name}", "split": "holdout", "case_type": kind,
                "context": raw["context"], "question": raw["question"], "gold": gold,
                "aliases": [gold], "source_id": qid, "source_revision": revision, "source_uri": f"hf://datasets/{repo}@{revision}/{qid}"}
            if kind == "altered":
                # ARC-derived rows sometimes store labels 1..4 but answerKey
                # remains A..D. Canonicalize by published option order and
                # require exact agreement with the published gold answer text.
                row["source_option_labels"] = raw["choices"]["label"]
                row["options"] = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", raw["choices"]["text"]))
                assert row["options"][gold] == raw["answer"]
                row["aliases"].append(raw["answer"])
            row["input_tokens"] = len(tokenizer.apply_chat_template(messages(row), add_generation_prompt=True))
            if row["input_tokens"] > 2048:
                exclusions.append({"id": row["id"], "tokens": row["input_tokens"], "reason": "input_over_2048"})
            else:
                eligible[name].append(row)
        random.Random(SEED + list(REVISIONS).index(name) + 100).shuffle(eligible[name])
        chosen = eligible[name][:300]
        for i, row in enumerate(chosen):
            row["stratum_order"] = i
        selected.extend(chosen)
        manifests[name].update(eligible=len(eligible[name]), selected=len(chosen))
    # The plan requires a full-set control-construction audit. Exact lexical
    # support is only a necessary condition; failure disables controls entirely.
    failures = []
    for raw in source["inconsistent"]:
        _, reason = split_control(raw)
        if reason:
            failures.append({"id": raw["qid"], "reason": reason})
    controls = []
    if not failures:
        # Still requires manual semantic validation before a control claim.
        control_status = "pending_semantic_support_validation"
    else:
        control_status = "unavailable_full_set_construction_failed"
    out.mkdir(parents=True)
    write_jsonl(out / "holdout.jsonl", selected)
    write_jsonl(out / "excluded.jsonl", exclusions)
    write_json(out / "control-audit.json", {"checked": len(source["inconsistent"]),
        "necessary_check": "two separable documents; each answer occurs exclusively in its corresponding document",
        "failure_count": len(failures), "failures": failures, "status": control_status,
        "controls_emitted": len(controls), "semantic_support_proven": False})
    write_json(out / "manifest.json", {"seed": SEED, "datasets": manifests,
        "selection": "eligibility before seeded shuffle and first 300; fallback first 150 per stratum",
        "frozen_ids": [r["id"] for r in selected], "rows_sha256": digest(selected),
        "excluded_count": len(exclusions), "controls": control_status, "holdout_outputs_seen": False})
