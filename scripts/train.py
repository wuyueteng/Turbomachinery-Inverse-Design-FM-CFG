"""Train paper models without editing source files."""
from pathlib import Path
import argparse
from datetime import datetime
import json
import sys
import uuid
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from shared.training import ALL_MODELS,MAIN_MODELS,resolve_device,train_run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=ALL_MODELS+['all'])
    p.add_argument('--config',type=Path)
    p.add_argument('--data-dir',type=Path,default=ROOT/'data')
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--device',default='auto')
    p.add_argument('--threads',type=int,default=4,help='CPU worker threads (default: 4)')
    p.add_argument('--resume',type=Path,help='Trusted run directory or last.pt checkpoint')
    for key in ['epochs','batch-size','seed','split-seed','patience','min-epochs','max-batches']:
        p.add_argument('--'+key,type=int)
    p.add_argument('--lr',type=float)
    p.add_argument('--dataset',choices=['reference','merged'])
    p.add_argument('--stop-after',type=int,help='Stop after N epochs in this invocation; preserves the planned scheduler')
    args=p.parse_args()
    try:
        if args.threads < 1:raise ValueError('threads must be positive')
        torch.set_num_threads(args.threads)
        device=resolve_device(args.device)
        if args.resume:
            ckpt=args.resume/'last.pt' if args.resume.is_dir() else args.resume
            state=torch.load(ckpt,map_location='cpu',weights_only=False)
            if args.config or args.output_dir or any(getattr(args,k) is not None for k in ['epochs','batch_size','seed','split_seed','patience','min_epochs','max_batches','lr','dataset']):
                raise ValueError('Resume uses saved configuration; only device/data-dir/stop-after may change')
            if args.model and args.model!=state['config']['model']:raise ValueError('Resume model does not match')
            if state.get('early_stopped'):raise ValueError('This run already satisfied early stopping')
            train_run(state['config'],args.data_dir,ckpt.parent,device,resume=ckpt,stop_after=args.stop_after)
            return
        if not args.model:p.error('--model is required for a new run')
        if args.model=='all' and args.config:p.error('Use a model-specific --config with one model')
        for name in MAIN_MODELS if args.model=='all' else [args.model]:
            cfg=json.loads((ROOT/'configs'/f'{name}.json').read_text(encoding='utf8'))
            if args.config:
                custom=json.loads(args.config.read_text(encoding='utf8'))
                params=cfg['model_params']|custom.pop('model_params',{})
                cfg.update(custom);cfg['model_params']=params
            if cfg['model']!=name:raise ValueError('Configuration model does not match --model')
            for key in ['epochs','batch_size','seed','split_seed','patience','min_epochs','max_batches','lr','dataset']:
                if getattr(args,key) is not None:cfg[key]=getattr(args,key)
            if args.lr is not None and name=='cgan':cfg['model_params'].update(lr_g=args.lr,lr_d=args.lr)
            if cfg['epochs']<1 or cfg['batch_size']<1 or cfg.get('max_batches',1)<1:raise ValueError('Epochs, batch size and max-batches must be positive')
            if name=='performance_predictor' and cfg['dataset']!='reference':raise ValueError('Predictor training uses the CFD reference dataset')
            run=args.output_dir
            if run and args.model=='all':run=run/name
            if run is None:run=ROOT/'runs'/name/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
            train_run(cfg,args.data_dir,run.resolve(),device,stop_after=args.stop_after)
    except (ValueError,TypeError,FileNotFoundError,FileExistsError) as e:p.exit(2,f'Error: {e}\n')


if __name__=='__main__':main()
