from __future__ import annotations
import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Evidence Dial reproducible pipeline")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ["build-policy", "freeze-faith"]:
        q = sub.add_parser(name)
        q.add_argument("--out", type=Path, required=True)
    q = sub.add_parser("headroom")
    q.add_argument("--data", type=Path, required=True)
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--gpu", type=int, default=1)
    q = sub.add_parser("cache")
    q.add_argument("--data", type=Path, required=True)
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--headroom", type=Path, required=True)
    q.add_argument("--gpu", type=int, default=1)
    args = p.parse_args()
    if args.command == "build-policy":
        from .data import build_policy
        from .evaluate import tokenizer
        build_policy(args.out, tokenizer())
    elif args.command == "freeze-faith":
        from .faith import freeze
        from .evaluate import tokenizer
        freeze(args.out, tokenizer())
    elif args.command in {"headroom", "cache"}:
        # Physical identity is fixed before importing torch/CUDA.
        import subprocess
        uuid = subprocess.check_output(["nvidia-smi", f"--id={args.gpu}", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
        os.environ["CUDA_VISIBLE_DEVICES"] = uuid
        if args.command == "headroom":
            from .evaluate import headroom
            print(json.dumps(headroom(args.data, args.out, args.gpu), indent=2))
        else:
            gate = json.loads(args.headroom.read_text())
            if not gate["teacher_beats_off"] or gate["off_failure_count"] < 10:
                raise ValueError("Teacher advantage and meaningful OFF headroom required before cache/binding")
            from .binding import build_cache
            from .evaluate import load_base, tokenizer
            from .resources import Resources
            with Resources(args.gpu, args.out.parent / "cache-resource-admission.json") as resource:
                build_cache(args.data, args.out, load_base(), tokenizer(), resource)


if __name__ == "__main__":
    main()
