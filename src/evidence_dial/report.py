"""Paired effect estimates, complete coverage and clustered intervals."""
from __future__ import annotations
import collections
from .protocol import normalize, SEED


def paired(rows, on, off, *, bootstrap=10000):
    import numpy as np
    declared = {r["id"] for r in rows}
    if len(declared) != len(rows):
        raise ValueError("Duplicate declared IDs")
    on_map, off_map = {r["id"]: r for r in on}, {r["id"]: r for r in off}
    if len(on_map) != len(on) or len(off_map) != len(off):
        raise ValueError("Duplicate measured IDs")
    if (set(on_map) | set(off_map)) - declared:
        raise ValueError("Measurements outside frozen cohort")
    common = declared & set(on_map) & set(off_map)
    for k in common:
        if on_map[k]["prompt_sha256"] != off_map[k]["prompt_sha256"]:
            raise ValueError("ON/OFF evaluation prompt mismatch")
        if on_map[k].get("teacher") or off_map[k].get("teacher"):
            raise ValueError("Teacher output cannot enter adapter comparison")
    groups = collections.defaultdict(list)
    for row in rows:
        if row["id"] in common:
            k = row["id"]
            groups[row["family"]].append(on_map[k]["success"] - off_map[k]["success"])
    deltas = [d for values in groups.values() for d in values]
    if not deltas:
        return {"complete": False, "measured_pairs": 0, "declared": len(rows), "effect_gain": None}
    rng = np.random.default_rng(SEED)
    sums = np.array([sum(v) for v in groups.values()])
    sizes = np.array([len(v) for v in groups.values()])
    indexes = rng.integers(0, len(groups), size=(bootstrap, len(groups)))
    estimates = sums[indexes].sum(axis=1) / sizes[indexes].sum(axis=1)
    interval = np.quantile(estimates, [0.025, 0.975]).tolist()
    gain = sum(deltas) / len(deltas)
    return {"complete": len(common) == len(rows), "measured_pairs": len(common), "declared": len(rows),
        "missing_on": sorted(declared-set(on_map)), "missing_off": sorted(declared-set(off_map)),
        "effect_gain": gain, "paired_family_bootstrap_95ci": interval, "bootstrap_replicates": bootstrap,
        "cluster_count": len(groups), "gains": deltas.count(1), "losses": deltas.count(-1),
        "unchanged": deltas.count(0),
        "claim": ("incomplete evidence" if len(common) != len(rows) else
                  "established positive gain under this protocol" if interval[0] > 0 else
                  "preliminary observed gain; interval spans zero" if gain > 0 else
                  "no positive observed gain")}


def diagnostics(rows, predictions):
    measured = {p["id"]: p for p in predictions}
    by_type = collections.defaultdict(list)
    answerable = []
    for row in rows:
        if row["id"] not in measured:
            continue
        p = measured[row["id"]]
        by_type[row["case_type"]].append(p["success"])
        if row["case_type"] in {"answerable", "altered"}:
            answerable.append(p)
    return {"by_type": {k: {"n": len(v), "success_rate": sum(v)/len(v)} for k,v in by_type.items()},
        "answerable_count": len(answerable),
        "answerable_accuracy": sum(p["success"] for p in answerable)/len(answerable) if answerable else None,
        "unnecessary_abstention_rate": sum(normalize(p["output"]) in {"unknown", "conflict"} for p in answerable)/len(answerable) if answerable else None}
