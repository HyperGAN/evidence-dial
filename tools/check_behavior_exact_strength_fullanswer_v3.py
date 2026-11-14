from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from evidence_dial.behavior_fullanswer_health_v3 import check
p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--step',type=int,required=True);p.add_argument('--checkpoint',type=Path,required=True);a=p.parse_args();check(a.candidate,a.step,a.checkpoint);print('Exact full-answer strength health pass')
