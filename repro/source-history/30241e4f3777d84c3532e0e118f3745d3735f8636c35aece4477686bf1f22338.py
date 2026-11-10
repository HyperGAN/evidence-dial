"""One native replay of the first retained sample; never selects a model."""
from pathlib import Path
import argparse,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from evidence_dial.io import write_json
from evidence_dial.benchmark import file_sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--curation',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if (a.out/'call-commit.json').exists():raise FileExistsError('Proof already attempted; reuse/reconcile rather than replay again')
    selected=json.loads((a.curation/'selected.json').read_text())['selection'][0]
    source=a.curation/'generations'/(selected['generation_key']+'.json');reference=json.loads(source.read_text())
    trace_file=source.with_suffix('.safetensors');assert file_sha(trace_file)==selected['trajectory_sha256']
    os.environ['CUDA_VISIBLE_DEVICES']=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    import torch
    from safetensors.torch import load_file
    from evidence_dial.evaluate import load_base,tokenizer
    from evidence_dial.resources import Resources
    saved=load_file(str(trace_file));tok=tokenizer();inputs=tok.apply_chat_template([{'role':'user','content':reference['identity']['prompt']}],add_generation_prompt=True)
    assert inputs==reference['input_ids']
    write_json(a.out/'call-commit.json',{'selected_case':selected['id'],'generation_key':selected['generation_key'],
        'reference_sha256':file_sha(source),'trajectory_sha256':file_sha(trace_file),'source_sha256':file_sha(__file__),
        'implementation_equivalence_only':True,'used_for_selection':False})
    with Resources(1,a.out/'resources.json'):
        model=load_base();tensor=torch.tensor([inputs],device='cuda:0');states=[]
        def capture(module,args,output):states.append(output.last_hidden_state[0,-1].detach().float().cpu().clone())
        hook=model.model.register_forward_hook(capture)
        try:
            with torch.random.fork_rng(devices=[0]),torch.inference_mode():
                torch.set_rng_state(saved['rng_cpu_before']);torch.cuda.set_rng_state(saved['rng_cuda_before'],device=0)
                output=model.generate(input_ids=tensor,attention_mask=torch.ones_like(tensor),do_sample=True,
                    temperature=.7,top_p=.9,top_k=0,max_new_tokens=96,eos_token_id=3,pad_token_id=tok.pad_token_id,
                    use_cache=True,use_model_defaults=False)
            ids=output[0,len(inputs):].tolist();actual=torch.stack(states);expected=saved['last_hidden_before_each_sampled_token']
            same_shape=actual.shape==expected.shape
            result={'token_ids_equal':ids==reference['output_ids'],'hidden_values_bit_equal':same_shape and torch.equal(actual,expected),
                'maximum_absolute_hidden_difference':float((actual-expected).abs().max()) if same_shape else None,
                'input_ids_equal':True,'seed':selected['seed'],'CPU_and_CUDA_RNG_restored':True,
                'generation_key':selected['generation_key'],'native_generation_model_calls':1,
                'implementation_equivalence_only':True,'used_for_selection':False,'training_targets_replaced':False,
                'replayed_output_ids':ids,'replayed_output':tok.decode(ids,skip_special_tokens=True)}
            write_json(a.out/'result.json',result)
            if not result['token_ids_equal']:raise RuntimeError('Seed/trajectory replay differs; preserve first proof and investigate')
        finally:hook.remove()
    print(json.dumps(result))


if __name__=='__main__':main()
