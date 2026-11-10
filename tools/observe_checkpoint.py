"""Launch the captured observer with its exact qualified environment."""
from pathlib import Path
import json
import os
import sys

root=Path(__file__).resolve().parents[1]
context=json.loads((root/'configs/qwen-launch-context.json').read_text())
python='/ml2/particlesliders-working/lr-work/output-effect-scoring/ming-layer1-qwen3vl8b-cuda-env-v1/bin/python'
os.execve(python,[python,str(root/'tools/check_text_guard.py'),*sys.argv[1:]],dict(os.environ,**context))
