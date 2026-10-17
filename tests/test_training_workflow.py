"""Short software tests on released data; small networks do not represent paper results."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from shared.training import train_run,ALL_MODELS

SMALL={
    'fm_cfg':dict(hidden_dim=32,n_blocks=1,sampling_steps=2),
    'fm':dict(hidden_dim=32,n_blocks=1,sampling_steps=2),
    'ar':dict(dim=32,n_heads=2,n_layers=1),
    'cvae':dict(hidden_dim=32,latent_dim=8,n_layers=1),
    'cgan':dict(hidden_dim=32,noise_dim=8,n_g_blocks=1,n_d_blocks=1,n_critic=1),
    'ddpm':dict(hidden_dim=32,n_blocks=1,timesteps=4),
    'cinn':dict(hidden_dim=32,n_flows=2),
    'performance_predictor':dict(dim=32,depth=1,heads=2,mlp_dim=32,dim_head=16),
}


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_resume_matches_uninterrupted_for_all_models(self):
        with tempfile.TemporaryDirectory(prefix='turbo_training_') as tmp:
            for name in ALL_MODELS:
                with self.subTest(model=name):
                    cfg=json.loads((ROOT/'configs'/f'{name}.json').read_text())
                    cfg['model_params'].update(SMALL[name]);cfg.update(epochs=2,batch_size=8,max_batches=2)
                    full=Path(tmp)/name/'full';resumed=Path(tmp)/name/'resumed'
                    train_run(cfg,ROOT/'data',full,torch.device('cpu'))
                    train_run(cfg,ROOT/'data',resumed,torch.device('cpu'),stop_after=1)
                    train_run(cfg,ROOT/'data',resumed,torch.device('cpu'),resume=resumed/'last.pt')
                    a=torch.load(full/'last.pt',weights_only=False);b=torch.load(resumed/'last.pt',weights_only=False)
                    self.assertEqual(a['history'],b['history'])
                    for key in a['model_state']:self.assertTrue(torch.equal(a['model_state'][key],b['model_state'][key]),key)
                    self.assertEqual(a['gan_step_count'],b['gan_step_count'])
                    if name=='cgan':self.assertEqual(set(b['optimizers']),{'generator','discriminator'})

    def test_documented_cli_training_generation_evaluation(self):
        with tempfile.TemporaryDirectory(prefix='turbo_cli_') as tmp:
            tmp=Path(tmp)
            def call(*args,ok=True):
                p=subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,capture_output=True,text=True,encoding='utf8')
                if ok:self.assertEqual(p.returncode,0,p.stdout+p.stderr)
                else:self.assertNotEqual(p.returncode,0)
                return p
            for name in ['performance_predictor','fm_cfg']:
                config=tmp/f'{name}.json';config.write_text(json.dumps({'model_params':SMALL[name]}))
                call('scripts/train.py','--model',name,'--config',config,'--epochs','1','--batch-size','8',
                     '--max-batches','1','--device','cpu','--output-dir',tmp/name)
            call('scripts/inference.py','--run-dir',tmp/'fm_cfg','--predictor-run',tmp/'performance_predictor',
                '--n-cond','3','--n-samples','2','--device','cpu')
            self.assertTrue((tmp/'fm_cfg/inference/metrics.json').is_file())
            p=call('scripts/inference.py','--run-dir',tmp/'missing','--device','cpu',ok=False)
            self.assertIn('Train first',p.stderr)
            p=call('scripts/train.py','--model','fm_cfg','--data-dir',tmp/'no_data','--device','cpu',ok=False)
            self.assertIn('Missing',p.stderr)
            p=call('scripts/train.py','--model','fm_cfg','--batch-size','0',ok=False)
            self.assertIn('positive',p.stderr)


if __name__=='__main__':unittest.main()
