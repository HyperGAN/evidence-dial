"""Complete-answer remeasurement of the already selected old adapter; no training/reselection."""
import argparse,ctypes,json,os,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
 from evidence_dial.behavior_fullanswer_health_v3 import pin,check,passing_guard
 from evidence_dial.io import read_jsonl,write_json
 plan=json.loads(a.plan.read_text());folder=a.plan.resolve().parent;assert not (folder/'actual-owned-process-first.json').exists()
 for item in plan['inputs']+plan['sources']:assert pin(item['path'])==item
 parent=os.getppid();assert ctypes.CDLL(None).prctl(1,signal.SIGTERM)==0 and os.getppid()==parent
 def stop(sig,frame):raise KeyboardInterrupt('Owned old-selected full-answer remeasurement interrupted; preserve firsts')
 signal.signal(signal.SIGTERM,stop);write_json(folder/'actual-owned-process-first.json',{'time':time.time(),'pid':os.getpid(),'birth':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],'argv':[x.decode() for x in Path('/proc/self/cmdline').read_bytes().split(b'\0') if x],'plan':pin(a.plan)})
 os.environ['CUDA_VISIBLE_DEVICES']=plan['physical_gpu_uuid']
 from evidence_dial.behavior_binding_v1 import inference_backend
 from evidence_dial.behavior_fullanswer_generate_v2 import evaluate
 from evidence_dial.shared_gpu_resources_v1 import SharedGPU0Resources
 from evidence_dial.evaluate import tokenizer
 from particle_sliders.inference import open_inference
 rows=read_jsonl(plan['data']['path']);assert len(rows)==16 and all(r['split']=='dev' for r in rows)
 try:
  with SharedGPU0Resources(0,folder/'resources-first.json',plan=a.plan) as resource:
   with open_inference(plan['candidate']['artifact']['path'],backend=inference_backend(),operation='response') as inference:
    native=inference.binding.response.__self__;tok=tokenizer()
    for alpha in plan['strengths']:
     out=folder/f'alpha-{alpha:g}';label=f'old-selected-raw-{plan["candidate"]["artifact"]["sha256"]}-strength-{alpha:g}'
     with native.slider.scaled(alpha):evaluate(rows,native.model,tok,out,resource,arm=label)
     records=read_jsonl(out/'predictions.jsonl');probes=[next(r for r in records if r['case_type']==kind) for kind in ['discouragement','unsupported_conclusion','reasonable_boundary','uncertainty']]
     write_json(out/'probes-000032.json',{'checkpoint':32,'strength':alpha,'probes':probes,'selection_metric_used':False});check(out,32,plan['candidate']['FULL']['path']);assert passing_guard(out,32,plan['candidate']['FULL']['path']) is not None
  write_json(folder/'completion-first.json',{'time':time.time(),'complete':True,'cases_each':16,'strengths':plan['strengths'],'old_checkpoint_fixed_before_fullanswer_measurement':True,'new_training_updates':0,'weight_EMA':False,'selection_performed':False,'fresh_holdout_access':False});print(json.dumps({'complete':True,'cases_each':16,'strengths':plan['strengths'],'new_training_updates':0}))
 except BaseException as e:write_json(folder/'measurement-failed-first.json',{'time':time.time(),'error':repr(e),'firsts_preserved':True});raise
if __name__=='__main__':main()
