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
    q = sub.add_parser('evaluate')
    q.add_argument('--selection',type=Path,required=True)
    q.add_argument('--data',type=Path,required=True)
    q.add_argument('--out',type=Path,required=True)
    q.add_argument('--arm',choices=['off','particle','lora'],required=True)
    q.add_argument('--adapter',type=Path)
    q.add_argument('--strength',type=float,default=1.)
    q.add_argument('--gpu',type=int,default=1)
    q = sub.add_parser('dev-particle')
    q.add_argument('--artifact',type=Path,required=True)
    q.add_argument('--data',type=Path,required=True)
    q.add_argument('--out',type=Path,required=True)
    q.add_argument('--strength',type=float,required=True)
    q.add_argument('--gpu',type=int,default=1)
    q = sub.add_parser('train-lora')
    q.add_argument('--data',type=Path,required=True)
    q.add_argument('--out',type=Path,required=True)
    q.add_argument('--gpu',type=int,default=1)
    q = sub.add_parser('infer')
    q.add_argument('--adapter',required=True,help='Portable adapter directory or HF model ID')
    q.add_argument('--row',type=Path,required=True,help='JSON object with context and question, optional options')
    q.add_argument('--out',type=Path,required=True)
    q.add_argument('--strength',type=float,default=1.)
    q.add_argument('--gpu',type=int,default=1)
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
    elif args.command in {"headroom", "cache",'evaluate','dev-particle','train-lora','infer'}:
        # Physical identity is fixed before importing torch/CUDA.
        import subprocess
        uuid = subprocess.check_output(["nvidia-smi", f"--id={args.gpu}", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
        os.environ["CUDA_VISIBLE_DEVICES"] = uuid
        if args.command=='infer':
            from .portable import Portable
            from .resources import Resources
            from .io import write_json
            if args.out.exists():raise FileExistsError('Retain the existing generation')
            row=json.loads(args.row.read_text())
            with Resources(args.gpu,args.out.with_suffix('.resources.json')):
                with Portable(args.adapter) as model:
                    result=model.generate(row['context'],row['question'],strength=args.strength,options=row.get('options'))
            write_json(args.out,result)
        elif args.command == "headroom":
            from .evaluate import headroom
            print(json.dumps(headroom(args.data, args.out, args.gpu), indent=2))
        elif args.command=='evaluate':
            from .holdout import evaluate
            print(json.dumps(evaluate(args.selection,args.data,args.out,args.arm,args.gpu,
                adapter=args.adapter,strength=args.strength),indent=2))
        elif args.command=='dev-particle':
            from .benchmark import particle
            print(json.dumps(particle(args.artifact,args.data,args.out,args.strength,args.gpu),indent=2))
        elif args.command=='train-lora':
            from .aligntune_train import train
            train(args.data,args.out,args.gpu)
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
