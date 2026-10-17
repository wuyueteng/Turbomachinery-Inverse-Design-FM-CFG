"""Run-based training with explicit data, configuration and checkpoint state."""
from pathlib import Path
import hashlib
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset

ROOT = Path(__file__).resolve().parents[1]
MAIN_MODELS = ['fm_cfg', 'ar', 'cvae', 'cgan', 'ddpm', 'cinn']
ALL_MODELS = MAIN_MODELS + ['fm', 'performance_predictor']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_checkpoint(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def resolve_device(value='auto'):
    if value == 'auto':
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device(value)
    if device.type == 'cuda' and (not torch.cuda.is_available() or
            (device.index or 0) >= torch.cuda.device_count()):
        raise ValueError(f'Device {value} is unavailable. Use --device cpu or an available CUDA device.')
    return device


def build_model(name, params):
    if name == 'performance_predictor':
        from models.performance_predictor.vit import ViT
        return ViT(**params)
    from models.fm_cfg import ConditionalFlowMatching
    from models.fm.model import ConditionalFM
    from models.ar import ConditionalAutoregressive
    from models.cvae import ConditionalVAE
    from models.cgan import ConditionalGAN
    from models.ddpm.model import ConditionalDDPM
    from models.cinn import ConditionalGlow
    constructors = dict(fm_cfg=ConditionalFlowMatching, fm=ConditionalFM,
        ar=ConditionalAutoregressive, cvae=ConditionalVAE, cgan=ConditionalGAN,
        ddpm=ConditionalDDPM, cinn=ConditionalGlow)
    p = {k: v for k, v in params.items() if k != 'lr'}
    if name != 'cgan':
        p = {k: v for k, v in p.items() if k not in ('lr_g', 'lr_d')}
    return constructors[name](geom_dim=21, cond_dim=3, **p)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_data(cfg, data_dir):
    data_dir = Path(data_dir)
    filenames = ['parameter_geometry_5200.npy', 'label_5200.npy']
    if cfg['dataset'] == 'merged':
        filenames += ['merged_geometry.npy', 'merged_label.npy']
    for name in filenames:
        if not (data_dir/name).is_file():
            raise FileNotFoundError(f'Missing {data_dir/name}. Run python scripts/check_data.py --data-dir "{data_dir}"')
    refg, refy = [np.load(data_dir/n, allow_pickle=False) for n in filenames[:2]]
    if cfg['dataset'] == 'merged':
        g, y = [np.load(data_dir/n, allow_pickle=False) for n in filenames[2:]]
    else:
        g, y = refg, refy
    for a, cols in [(g, 21), (y, 3), (refg, 21), (refy, 3)]:
        if a.ndim != 2 or a.shape[1] != cols or not np.isfinite(a).all():
            raise ValueError(f'Invalid data shape/values: expected (*, {cols}) finite array')
    if len(g) != len(y) or len(refg) != len(refy):
        raise ValueError('Geometry and label row counts must match')
    stats = dict(geom_min=refg.min(0), geom_max=refg.max(0), label_min=refy.min(0), label_max=refy.max(0))
    if np.any(stats['geom_max'] == stats['geom_min']) or np.any(stats['label_max'] == stats['label_min']):
        raise ValueError('Reference columns must have nonzero range')
    predictor = cfg['model'] == 'performance_predictor'
    eps = 0 if predictor else 1e-12
    x = (g-stats['geom_min'])/(stats['geom_max']-stats['geom_min']+eps)
    yn = (y-stats['label_min'])/(stats['label_max']-stats['label_min']+eps)
    n, nt, nv = len(g), int(.8*len(g)), int(.1*len(g))
    if predictor:
        order = torch.randperm(n, generator=torch.Generator().manual_seed(cfg['split_seed'])).numpy()
    else:
        order = np.random.RandomState(cfg['split_seed']).permutation(n)
    splits = dict(train=order[:nt], validation=order[nt:nt+nv], test=order[nt+nv:])
    if min(map(len, splits.values())) == 0:
        raise ValueError('Dataset is too small for the 80/10/10 split')
    x = torch.tensor(x, dtype=torch.float32)
    if predictor:
        x = x.unsqueeze(-1)
    dataset = TensorDataset(x, torch.tensor(yn, dtype=torch.float32))
    return dataset, splits, stats, {n:digest(data_dir/n) for n in filenames}


def make_optimizers(model, cfg):
    if cfg['model'] == 'cgan':
        model.opt_g = torch.optim.Adam(model.generator.parameters(), lr=model.lr_g, betas=(.5,.9))
        model.opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=model.lr_d, betas=(.5,.9))
        return {'generator': model.opt_g, 'discriminator': model.opt_d}, None
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get('lr',cfg['model_params'].get('lr',1e-4)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['epochs'],eta_min=1e-6)
    return {'model': optimizer}, scheduler


def epoch_loss(model, loader, cfg, device, optimizers=None):
    training = optimizers is not None
    model.train(training)
    totals, weights = [], []
    predictor, gan = cfg['model']=='performance_predictor', cfg['model']=='cgan'
    with torch.set_grad_enabled(training):
        for i,(x,y) in enumerate(loader):
            if cfg.get('max_batches') and i >= cfg['max_batches']:
                break
            x,y=x.to(device),y.to(device)
            if predictor:
                loss=torch.nn.functional.mse_loss(model(x),y)
            elif gan and not training:
                z=torch.randn(len(x),model.noise_dim,device=device)
                loss=-model.discriminator(model.generator(z,y),y).mean()
            else:
                loss=model.training_loss(x,y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss; inspect configuration and input data')
            if training and not gan:
                optimizer=optimizers['model'];optimizer.zero_grad();loss.backward()
                if not predictor:torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
            w=1 if predictor else len(x)
            totals.append(loss.item()*w);weights.append(w)
    if not weights:raise ValueError('No batches. Reduce --batch-size.')
    return sum(totals)/sum(weights)


def train_run(cfg, data_dir, run_dir, device, resume=None, stop_after=None):
    run_dir=Path(run_dir)
    dataset,splits,stats,hashes=prepare_data(cfg,data_dir)
    predictor=cfg['model']=='performance_predictor'
    if not predictor and cfg['batch_size'] > len(splits['train']):
        raise ValueError('Batch size exceeds the training split. Reduce --batch-size.')
    seed_everything(cfg['seed'])
    model=build_model(cfg['model'],cfg['model_params']).to(device)
    optimizers,scheduler=make_optimizers(model,cfg)
    start,best,no_improve,history=0,float('inf'),0,[]
    if resume:
        # Resume files include optimizer and RNG state: load only trusted files.
        state=torch.load(resume,map_location='cpu',weights_only=False)
        if state['data_sha256'] != hashes:raise ValueError('Resume dataset differs from the saved run')
        model.load_state_dict(state['model_state'])
        for k,opt in optimizers.items():opt.load_state_dict(state['optimizers'][k])
        if scheduler:scheduler.load_state_dict(state['scheduler'])
        start,best,no_improve,history=state['epoch'],state['best_loss'],state['no_improve'],state['history']
        if cfg['model']=='cgan':model._step_count=state['gan_step_count']
        random.setstate(state['rng']['python']);np.random.set_state(state['rng']['numpy'])
        torch.set_rng_state(state['rng']['torch'])
        if device.type=='cuda' and state['rng']['cuda']:
            torch.cuda.set_rng_state_all(state['rng']['cuda'])
    else:
        if run_dir.exists() and any(run_dir.iterdir()):raise FileExistsError(f'Output directory is not empty: {run_dir}')
        run_dir.mkdir(parents=True,exist_ok=True)
        (run_dir/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf8')
        np.savez(run_dir/'split_indices.npz',**splits)
        (run_dir/'data_manifest.json').write_text(json.dumps(hashes,indent=2),encoding='utf8')
    loaders=[DataLoader(Subset(dataset,splits[k]),batch_size=cfg['batch_size'],shuffle=k=='train',
        drop_last=k=='train' and not predictor) for k in ['train','validation']]
    for epoch in range(start+1,cfg['epochs']+1):
        tr=epoch_loss(model,loaders[0],cfg,device,optimizers)
        if scheduler:scheduler.step()
        val=epoch_loss(model,loaders[1],cfg,device)
        history.append(dict(epoch=epoch,train_loss=tr,validation_loss=val))
        if val < best:
            best,no_improve=val,0
            save_checkpoint({'config':cfg,'model_state':model.state_dict(),
                'normalization':{k:v.tolist() for k,v in stats.items()},'epoch':epoch,'validation_loss':val},run_dir/'best.pt')
        else:no_improve+=1
        early=cfg['model']!='cgan' and cfg.get('patience',0)>0 and epoch>=cfg.get('min_epochs',100) and no_improve>=cfg['patience']
        save_checkpoint(dict(config=cfg,model_state=model.state_dict(),optimizers={k:v.state_dict() for k,v in optimizers.items()},
            scheduler=scheduler.state_dict() if scheduler else None,epoch=epoch,best_loss=best,no_improve=no_improve,
            history=history,data_sha256=hashes,gan_step_count=getattr(model,'_step_count',0),early_stopped=early,
            rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                     cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])),run_dir/'last.pt')
        (run_dir/'history.json').write_text(json.dumps(history,indent=2),encoding='utf8')
        print(f"{cfg['model']} epoch {epoch}/{cfg['epochs']} train={tr:.6g} validation={val:.6g}",flush=True)
        if early or (stop_after and epoch-start>=stop_after):break
    print(f'Run saved: {run_dir}',flush=True)
    return run_dir
