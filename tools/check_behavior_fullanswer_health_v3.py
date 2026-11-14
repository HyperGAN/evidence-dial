"""Check actual full native-EOS DEV probes before any next training quarter."""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from evidence_dial.behavior_fullanswer_health_v3 import check
p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--step',type=int,required=True);a=p.parse_args();check(a.candidate,a.step,a.candidate/f'FULL-{a.step:06d}.pt');print('qualified full-answer categorical health pass',flush=True)
