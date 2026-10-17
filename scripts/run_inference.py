"""Generate candidates from a trained run; optionally evaluate with a trained TeM."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from shared.training import build_model,resolve_device,seed_everything


def load_run(path,device):
    ckpt=Path(path)/'best.pt'
    if not ckpt.is_file():raise FileNotFoundError(f'Missing {ckpt}. Train first with python scripts/train.py --model <model>.')
    state=torch.load(ckpt,map_location='cpu',weights_only=True)
    model=build_model(state['config']['model'],state['config']['model_params']).to(device)
    model.load_state_dict(state['model_state']);model.eval()
    return model,state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--predictor-run',type=Path)
    p.add_argument('--data-dir',type=Path,default=ROOT/'data')
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--targets',type=Path,help='Headerless CSV or NPY, physical [flow, efficiency, pressure ratio]')
    p.add_argument('--n-cond',type=int,default=100)
    p.add_argument('--n-samples',type=int,default=10)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='auto')
    args=p.parse_args()
    try:
        torch.set_num_threads(4)
        if args.n_cond<1 or args.n_samples<1:raise ValueError('n-cond and n-samples must be positive')
        device=resolve_device(args.device);seed_everything(args.seed)
        model,state=load_run(args.run_dir,device)
        if state['config']['model']=='performance_predictor':raise ValueError('run-dir must contain a generative model')
        stats={k:np.asarray(v) for k,v in state['normalization'].items()}
        if args.targets:
            target=np.load(args.targets,allow_pickle=False) if args.targets.suffix=='.npy' else np.loadtxt(args.targets,delimiter=',',ndmin=2)
        else:
            indices=np.load(args.run_dir/'split_indices.npz',allow_pickle=False)['test']
            fn='merged_label.npy' if state['config']['dataset']=='merged' else 'label_5200.npy'
            target=np.load(args.data_dir/fn,allow_pickle=False)[indices[:args.n_cond]]
        target=np.asarray(target)
        if target.ndim!=2 or target.shape[1]!=3 or not np.isfinite(target).all():raise ValueError('Targets must have shape (N,3) and finite values')
        cond=(target-stats['label_min'])/(stats['label_max']-stats['label_min']+1e-12)
        predictor=ps=None
        if args.predictor_run:
            predictor,ps=load_run(args.predictor_run,device)
            if ps['config']['model']!='performance_predictor':raise ValueError('predictor-run must contain a performance predictor')
        out=args.output_dir or args.run_dir/'inference'
        if out.exists() and any(out.iterdir()):raise FileExistsError(f'{out} is not empty; choose another --output-dir')
        geometry=np.concatenate([model.sample(c.astype(np.float32),n_samples=args.n_samples) for c in cond])
        physical=geometry*(stats['geom_max']-stats['geom_min'])+stats['geom_min']
        targets=np.repeat(target,args.n_samples,axis=0)
        out.mkdir(parents=True,exist_ok=True)
        for name,values in [('geometry_norm',geometry),('geometry_denorm',physical),('target_performance',targets)]:np.save(out/f'{name}.npy',values)
        np.savetxt(out/'geometry_and_targets.csv',np.column_stack([physical,targets]),delimiter=',',fmt='%.17g')
        if predictor:
            ns={k:np.asarray(v) for k,v in ps['normalization'].items()}
            x=(physical-ns['geom_min'])/(ns['geom_max']-ns['geom_min'])
            with torch.no_grad():
                predicted=np.concatenate([predictor(torch.tensor(batch,dtype=torch.float32,device=device).unsqueeze(-1)).cpu().numpy() for batch in np.array_split(x,max(1,int(np.ceil(len(x)/128))))])
            predicted=predicted*(ns['label_max']-ns['label_min'])+ns['label_min']
            np.save(out/'pred_performance.npy',predicted)
            metrics={}
            for j,name in enumerate(['flow','eff','pr']):
                error=predicted[:,j]-targets[:,j];span=np.ptp(targets[:,j]);ss=np.sum((targets[:,j]-targets[:,j].mean())**2)
                metrics[name]=dict(mae=float(np.mean(abs(error))),mre_percent=float(np.mean(abs(error)/(abs(targets[:,j])+1e-12))*100),
                    r2=float(1-np.sum(error**2)/ss) if ss>0 else None,nrmse=float(np.sqrt(np.mean(error**2))/span) if span>0 else None)
            (out/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False),encoding='utf8')
        (out/'provenance.json').write_text(json.dumps(dict(seed=args.seed,n_conditions=len(target),samples_per_condition=args.n_samples,
            model=state['config']['model'],evaluation='surrogate' if predictor else 'generation_only'),indent=2),encoding='utf8')
        print(f'Saved {len(geometry)} candidates to {out}')
    except (ValueError,FileNotFoundError,FileExistsError) as e:p.exit(2,f'Error: {e}\n')


if __name__=='__main__':main()
