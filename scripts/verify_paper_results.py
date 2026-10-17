"""Recompute archived benchmark metrics and check the displayed paper precision."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from shared.training import MAIN_MODELS
from shared.data_utils import get_dataloaders
from scripts.plot_bubble import _compute_diversity,_compute_novelty
from scripts.plot_distribution import compute_similarity
P=ROOT/'paper_results'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'outputs/paper_metrics_verification.json');args=p.parse_args()
    _,_,_,info=get_dataloaders()
    bubble=json.loads((P/'benchmark/eval_bubble_metrics.json').read_text())
    table=json.loads((P/'benchmark/table1_display_values.json').read_text())
    calculated={};geometries={}
    for m in MAIN_MODELS:
        base=P/'models'/m/'inference';pred=np.load(base/'pred_performance.npy');target=np.load(base/'target_performance.npy');g=np.load(base/'geometry_norm.npy');geometries[m]=g
        saved=json.loads((base/'inference_metrics.json').read_text());metrics={}
        for j,name in enumerate(['flow','eff','pr']):
            err=pred[:,j]-target[:,j]
            metrics['r2_'+name]=float(1-(err**2).sum()/((target[:,j]-target[:,j].mean())**2).sum())
            metrics['nrmse_'+name]=float(np.sqrt(np.mean(err**2))/np.ptp(target[:,j]))
            metrics['mre_'+name]=float(np.mean(abs(err)/(abs(target[:,j])+1e-12))*100)
        for k,v in metrics.items():assert np.isclose(v,saved[k],rtol=1e-7,atol=1e-9),(m,k,v,saved[k])
        metrics['mean_nrmse']=float(np.mean([metrics['nrmse_'+k] for k in ['flow','eff','pr']]))
        metrics['mre']=float(np.mean([metrics['mre_'+k] for k in ['flow','eff','pr']]))
        metrics['diversity_mean']=_compute_diversity(g,10);metrics['novelty_mean']=_compute_novelty(g,info['train_geom'])
        for k in ['mre','diversity_mean','novelty_mean']:assert np.isclose(metrics[k],bubble[m][k],rtol=1e-6,atol=1e-7),(m,k)
        for key,displayed in table[m].items():
            decimals=2 if key=='mre' else 4
            assert round(metrics[key],decimals)==displayed,(m,key,metrics[key],displayed)
        calculated[m]=metrics
    distribution=compute_similarity(info['test_geom'],geometries);saved=json.loads((P/'benchmark/distribution_similarity.json').read_text())
    for m in MAIN_MODELS:
        for k in ['wd_mean','ks_mean','js_mean']:assert np.isclose(distribution[m][k],saved[m][k],rtol=1e-7,atol=1e-9),(m,k)
    from scripts.plotting import surrogate_panels as s
    from sklearn.metrics import r2_score
    checks=[]
    for j in [0,2,1]:
        t=s.TARGETS[j];true=s.y_test_all[:,j];mean=s.y_test_pred_all[:,j];sigma=s.sg_test_cal[:,j]
        checks.append(dict(r2=round(r2_score(true,mean),3),coverage2=round(float(np.mean(abs(true-mean)<=2*sigma)*100),1),
            robustness_r2=round(s.df_robust[t['r2_col']].mean(),4),robustness_nrmse=round(s.df_robust[t['nrmse_col']].mean(),4)))
    assert [c['r2'] for c in checks]==[.963,.989,.954]
    assert [c['coverage2'] for c in checks]==[97.4,94.8,94.8]
    assert [c['robustness_r2'] for c in checks]==[.9708,.9891,.9470]
    assert [c['robustness_nrmse'] for c in checks]==[.0339,.0205,.0560]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(passed=True,benchmark=calculated,surrogate=checks,
        distribution={m:{k:distribution[m][k] for k in ['wd_mean','ks_mean','js_mean']} for m in MAIN_MODELS}),indent=2),encoding='utf8')
    print('PASS: archived benchmark, Table 1 rounding, distribution distances and Figure 2 statistics')


if __name__=='__main__':main()
