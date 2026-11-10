"""Split organizations before variants; deterministic structured source truth.

Templates are shared across splits, fictional organizations/policy families are
disjoint. This is a controlled synthetic use case, not a natural-corpus claim.
"""
from __future__ import annotations
import collections
import random
from pathlib import Path

from .io import write_json, write_jsonl
from .protocol import SEED, REVISION, digest, messages, prompt

# Familiar defaults compete with explicitly fictional local conventions.
CONVENTIONS = [
    ("days in one week", "7 days", ["5 days", "9 days", "11 days"]),
    ("months in one year", "12 months", ["10 months", "14 months", "16 months"]),
    ("minutes in one hour", "60 minutes", ["40 minutes", "80 minutes", "90 minutes"]),
    ("seconds in one minute", "60 seconds", ["45 seconds", "75 seconds", "100 seconds"]),
    ("hours in one day", "24 hours", ["18 hours", "28 hours", "30 hours"]),
    ("centimeters in one meter", "100 centimeters", ["80 centimeters", "120 centimeters", "150 centimeters"]),
    ("grams in one kilogram", "1000 grams", ["800 grams", "1200 grams", "1500 grams"]),
    ("sides on a triangle", "3 sides", ["4 sides", "5 sides", "6 sides"]),
]
FACTS = [
    ("refund window", "days", (11, 17, 23, 29, 37, 43)),
    ("warranty duration", "months", (5, 9, 15, 19, 25, 31)),
    ("storage allowance", "GB", (13, 27, 39, 57, 81, 113)),
    ("team seat limit", "seats", (3, 7, 11, 19, 23, 31)),
    ("replacement delivery window", "days", (4, 8, 12, 16, 22, 26)),
    ("monthly support allowance", "tickets", (6, 14, 22, 34, 46, 58)),
    ("cancellation notice", "days", (2, 6, 14, 26, 38, 50)),
    ("annual service allowance", "visits", (2, 5, 8, 11, 14, 17)),
]


def family_rows(family: str, split: str, rng: random.Random) -> list[dict]:
    field, unit, values = rng.choice(FACTS)
    first, second = rng.sample(values, 2)
    answer, alternative = f"{first} {unit}", f"{second} {unit}"
    other_field, other_unit, other_values = rng.choice([f for f in FACTS if f[0] != field])
    distractor = f"{rng.choice(other_values)} {other_unit}"
    name = f"{family} Cooperative"
    scope = f"{name}'s Cedar plan"
    q = f"What is the {field} for {scope}?"
    common = f"Fictional {name} policies."
    d = f"For the Birch plan, the {field} is {alternative}. For the Cedar plan, the {other_field} is {distractor}."
    base = f"For the Cedar plan, the {field} is {answer}."
    forms = {
        "answerable": (f"[Document A]\n{common} {d} {base}", q, answer, alternative, [answer]),
        "missing": (f"[Document A]\n{common} {d}", q, "UNKNOWN", answer, []),
        "conflict": (f"[Document A]\n{common} {base}\n[Document B]\nFor {scope}, the {field} is {alternative}.", q, "CONFLICT", answer, [answer, alternative]),
    }
    convention, default, alternatives = rng.choice(CONVENTIONS)
    altered = rng.choice(alternatives)
    forms["altered"] = (f"[Document A]\n{common} {base} In our internal simulator, the number of {convention} is {altered}, overriding everyday usage.",
                         f"In {name}'s internal simulator, what is the number of {convention}?", altered, default, [altered])
    rows = []
    for kind, (context, question, gold, rejected, supports) in forms.items():
        rows.append({"id": f"policy-{family}-{kind}", "family": family, "split": split,
                     "dataset": "policy", "case_type": kind, "context": context,
                     "question": question, "gold": gold, "aliases": [gold],
                     "rejected": rejected, "support_values": supports,
                     "source_uri": f"synthetic://evidence-dial-v1/{family}",
                     "source_facts": {"scope": scope, "field": field, "value": answer,
                                      "alternative": alternative, "convention": convention,
                                      "local_value": altered, "everyday_default": default}})
    return rows


def verify_source(row: dict):
    """Validate exact template construction, chosen label and wrong contrast.

    No learned judge is claimed: structured field equality is the instrument.
    """
    kind, supports = row["case_type"], row["support_values"]
    assert row["gold"] != row["rejected"]
    if kind == "missing":
        assert not supports and row["gold"] == "UNKNOWN"
        assert f"For the Cedar plan, the {row['source_facts']['field']} is" not in row["context"]
    elif kind == "conflict":
        assert len(set(supports)) == 2 and row["gold"] == "CONFLICT"
        assert all(v in row["context"] for v in supports)
    else:
        assert supports == [row["gold"]] and row["gold"] in row["context"]
        if kind == "altered":
            assert row["gold"] != row["source_facts"]["everyday_default"]


def build_policy(out: Path, tokenizer):
    from curatorkit.schema import DataSample, ProvenanceRecord
    from curatorkit.gates.schema import SchemaGate
    from curatorkit.normalizers.dedup import ExactDeduplicator
    from curatorkit.exporters.messages import MessagesExporter
    from curatorkit.exporters.dpo import DPOExporter

    if out.exists():
        raise FileExistsError(f"Never overwrite a frozen cohort: {out}")
    rng = random.Random(SEED)
    families = [f"P{i:03d}" for i in range(112)]
    rng.shuffle(families)  # assignment precedes every variant
    cursor, all_rows, exclusions, curated = 0, [], [], []
    for split, n in [("train", 16), ("dev", 32), ("holdout", 64)]:
        split_families = families[cursor:cursor+n]
        cursor += n
        for family in split_families:
            for row in family_rows(family, split, rng):
                verify_source(row)
                row["input_tokens"] = len(tokenizer.apply_chat_template(messages(row), add_generation_prompt=True))
                row["teacher_input_tokens"] = len(tokenizer.apply_chat_template(messages(row, teacher=True), add_generation_prompt=True))
                budget = 512 if split in {"train", "dev"} else 2048
                if max(row["input_tokens"], row["teacher_input_tokens"]) > budget:
                    exclusions.append({"id": row["id"], "reason": "input_token_budget", "tokens": row["input_tokens"]})
                    continue
                sample = DataSample(id=row["id"], source_uri=row["source_uri"],
                                    instruction=prompt(row), output=row["gold"],
                                    metadata=row)
                sample.append_provenance(ProvenanceRecord(step_name="StructuredSourceCheck", step_version="1",
                    config_hash=digest({"validator": "template-field-equality-v1"}),
                    notes={"supported_values": row["support_values"], "contrast_checked": True}))
                curated.append(sample)
                all_rows.append(row)
    passed, rejected = SchemaGate(min_tokens=1, max_tokens=10000).run(curated)
    passed = ExactDeduplicator().run(passed)
    if len(passed) != 448 or rejected or exclusions:
        raise ValueError(f"Declared cohort requires 448 eligible distinct rows; got {len(passed)}, {len(rejected)} rejects, {len(exclusions)} excluded")
    out.mkdir(parents=True)
    write_jsonl(out / "curatorkit-provenance.jsonl", [s.model_dump(mode="json") for s in passed])
    for split in ["train", "dev", "holdout"]:
        rows = [s.metadata for s in passed if s.metadata["split"] == split]
        # Freeze sample order with a separate deterministic shuffle.
        random.Random(SEED + {"train": 1, "dev": 2, "holdout": 3}[split]).shuffle(rows)
        write_jsonl(out / f"{split}.jsonl", rows)
    train = [s for s in passed if s.metadata["split"] == "train"]
    MessagesExporter().export(train, out)
    preferences = [DataSample(id=s.id, source_uri=s.source_uri, instruction=s.instruction,
        chosen=s.output, rejected=s.metadata["rejected"], task_type="preference",
        metadata=s.metadata, provenance_chain=s.provenance_chain) for s in train]
    DPOExporter().export(preferences, out)
    write_json(out / "manifest.json", {"seed": SEED, "base_revision": REVISION,
        "protocol": digest(__import__('evidence_dial.protocol', fromlist=['PROMPT']).PROMPT),
        "rows_sha256": digest(all_rows), "counts": dict(collections.Counter(r["split"] for r in all_rows)),
        "family_assignment": {s: sorted({r["family"] for r in all_rows if r["split"] == s}) for s in ["train", "dev", "holdout"]},
        "excluded": exclusions, "rejected": [s.model_dump(mode="json") for s in rejected],
        "dedup": "CuratorKIT ExactDeduplicator; no semantic-dedup claim",
        "grounding": "deterministic structured-source validation, not an LLM judgment",
        "shared_templates": True, "holdout_outputs_seen": False})
