"""Frozen evaluation protocol shared by curation, teachers, trainers and demo."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata

MODEL = "CohereLabs/tiny-aya-global"
REVISION = "af89d219b53ed9b13b8a4645f9c8028973510324"
SEED = 20261110
MAX_NEW_TOKENS = 64
PROMPT = """Use only the evidence, overriding everyday knowledge. Documents have equal authority regardless of order or date. Output exactly UNKNOWN for absent evidence, CONFLICT for incompatible answers, otherwise the shortest supported answer with units (or the option letter for multiple choice). No explanation.

Evidence:
{context}

Question: {question}{options}
Answer:"""


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def prompt(row: dict, *, teacher: bool = False) -> str:
    options = ""
    if row.get("options"):
        options = "\n" + "\n".join(f"{k}. {v}" for k, v in row["options"].items())
    text = PROMPT.format(context=row["context"], question=row["question"], options=options)
    if teacher:
        # Only privileged information differs; evaluated prompts never contain this.
        text = text.removesuffix("Answer:") + f"Verified answer (private teacher information): {row['gold']}\nAnswer:"
    return text


def messages(row: dict, *, teacher: bool = False) -> list[dict]:
    return [{"role": "user", "content": prompt(row, teacher=teacher)}]


def normalize(text: str) -> str:
    """Case/Unicode/whitespace and trailing sentence punctuation only.

    No substring match, article deletion, rationale removal or answer extraction.
    """
    text = unicodedata.normalize("NFKC", text).casefold().strip()
    return re.sub(r"\s+", " ", text).rstrip(".!?").strip()


def success(row: dict, output: str) -> int:
    if not isinstance(output, str):
        raise TypeError("Missing generation is unmeasured, never a zero")
    return int(normalize(output) in {normalize(a) for a in row["aliases"]})


def faitheval_normalize(text: str) -> str:
    # Published FaithEval README normalization; diagnostics only.
    import string
    text = text.lower().replace("_", " ")
    text = "".join(" " if c in string.punctuation + "‘’´`" else c for c in text)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def faith_diagnostics(row: dict, output: str) -> dict:
    text = faitheval_normalize(output)
    if row["case_type"] == "missing":
        strict, loose = ["unknown"], ["unknown", "no answer", "no information", "not", "unclear"]
    elif row["case_type"] == "conflict":
        strict = ["conflict"]
        loose = ["conflict", "multiple answers", "disagreement", "inconsistent", "contradictory", "contradiction", "inconsistency", "two answers", "2 answers", "conflicting"]
    else:
        strict = loose = [faitheval_normalize(a) for a in row["aliases"]]
    return {"faitheval_strict_substring": int(any(a in text for a in strict)),
            "faitheval_nonstrict_substring": int(any(a in text for a in loose))}
