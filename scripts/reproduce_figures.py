"""Redraw network-related paper results from archived numerical outputs, without weights."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from shared.training import MAIN_MODELS

P=ROOT/'paper_results'
DISPLAY=dict(zip(MAIN_MODELS,['cFM-CFG','cAR','cVAE','cWGAN-GP','cDDPM','cINN']))
COLORS=dict(zip(MAIN_MODELS,['#9467bd','#6baed6','#76b7b2','#e58038','#aaaaaa','#4d4d4d']))
VARIABLES=[('flow','Mass flow (kg/s)',0),('pr','Total pressure ratio',2),('eff','Isentropic efficiency',1)]


def read(rel):return json.loads((P/rel).read_text(encoding='utf8'))
def arr(model,name):return np.load(P/'models'/model/'inference'/f'{name}.npy',allow_pickle=False)
def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':10,'axes.titlesize':10,
        'legend.fontsize':7,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,'svg.fonttype':'none'})
def save(fig,out,name):
    for ext in ['png','pdf']:fig.savefig(out/f'{name}.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)


def fig2(out):
    from scripts.plotting import surrogate_panels as s
    for key,value in dict(FONT_STATS_BOX=6.2,FONT_LEGEND=7.5,FONT_TICKS=8,FONT_AXES_LABEL=10,
        FONT_AXES_TITLE=10,FONT_CBAR_LABEL=9,FONT_CBAR_TICK=8,FONT_MEAN_ANNOT=7,
        SCATTER_TRAIN_SIZE=2,SCATTER_TEST_SIZE=3,MU_LINE_WIDTH=1).items():setattr(s,key,value)
    style();fig,axes=plt.subplots(3,2,figsize=(8,10),layout='constrained')
    for row,j in enumerate([0,2,1]):
        target=s.TARGETS[j];s.panel_combined(axes[row,0],target,chr(97+2*row))
        ax=axes[row,1];df=s.df_robust
        v=ax.scatter(df[target['r2_col']],df[target['nrmse_col']],c=df[target['mae_col']],s=12,cmap='RdYlGn_r')
        ax.axvline(df[target['r2_col']].mean(),ls='--',lw=.7,color='grey')
        ax.axhline(df[target['nrmse_col']].mean(),ls='--',lw=.7,color='grey')
        ax.set(xlabel='$R^2$',ylabel='nRMSE',title=chr(98+2*row)+'  '+target['name']);fig.colorbar(v,ax=ax,label=target['cbar_label'])
        for a in axes[row]:a.set_box_aspect(1);a.tick_params(labelsize=8);a.xaxis.label.set_size(10);a.yaxis.label.set_size(10)
    save(fig,out,'Figure_02')


def fig3(out):
    b=read('benchmark/eval_bubble_metrics.json');cost=pd.read_csv(P/'cost/tables/single_figure_plot_data.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    nov=np.array([b[m]['novelty_mean'] for m in MAIN_MODELS])
    sizes=70+(nov-nov.min())/(np.ptp(nov)+1e-12)*220
    for i,m in enumerate(MAIN_MODELS):axes[0].scatter(b[m]['diversity_mean'],b[m]['mre'],s=sizes[i],c=COLORS[m],label=DISPLAY[m],edgecolor='grey',lw=.5)
    axes[0].set(xlabel='Diversity (k = 5)',ylabel='Mean relative error (%)',title='a  Target matching and geometric diversity');axes[0].legend()
    metrics=list(cost.metric.unique());x=np.arange(len(metrics))
    for i,m in enumerate(MAIN_MODELS):
        rows=cost[cost.model==DISPLAY[m]].set_index('metric').loc[metrics]
        bars=axes[1].bar(x+(i-2.5)*.13,rows.relative_percent,.13,color=COLORS[m],yerr=rows.relative_sd_percent)
        for bar,value in zip(bars,rows.value):axes[1].text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f'{value:.3g}',ha='center',va='bottom',rotation=90,fontsize=6)
    axes[1].set_xticks(x,['Training time\n(s/epoch)','Inference time\n(s/10 designs)','Training memory\n(MiB)','Inference memory\n(MiB)']);axes[1].set(ylabel='Normalized cost (%)',ylim=(0,145),title='b  Computational cost')
    save(fig,out,'Figure_03')


def fig6(out):
    z=np.load(P/'trajectory/trajectory_data.npz',allow_pickle=False)
    fig,axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
    for ax,key,title in zip(axes[:2],['fm_2d','dd_2d'],['cFM-CFG','cDDPM']):
        a=z[key]
        for k in range(a.shape[1]):
            ax.plot(a[:,k,0],a[:,k,1],lw=.7,alpha=.65)
            ax.scatter(a[:,k,0],a[:,k,1],c=np.linspace(0,1,len(a)),cmap='viridis',s=2)
        ax.set(xlabel='Principal component 1',ylabel='Principal component 2',title=title)
    v=np.linalg.norm(z['fm_vel'],axis=-1);t=z['fm_ts'][:-1]
    axes[2].plot(t,v,alpha=.25,lw=.6);axes[2].plot(t,v.mean(1),color='black')
    axes[2].fill_between(t,v.mean(1)-v.std(1),v.mean(1)+v.std(1),color='grey',alpha=.2)
    axes[2].set(xlabel='Integration time',ylabel='Velocity magnitude',title='cFM-CFG velocity evolution')
    save(fig,out,'Figure_06')


def fig9(out):
    from shared.data_utils import get_dataloaders
    from scripts.plot_bubble import _compute_diversity,_compute_novelty
    _,_,_,info=get_dataloaders();style()
    fig,axes=plt.subplots(1,3,figsize=(10,3),layout='constrained')
    for i,m in enumerate(['fm','fm_cfg']):
        metrics=read(f'models/{m}/inference/inference_metrics.json')
        for ax,key in zip(axes[:2],['r2','nrmse']):
            ax.bar(np.arange(3)+(i-.5)*.32,[metrics[f'{key}_{k}'] for k,_,_ in VARIABLES],.32,label=['FM','cFM-CFG'][i])
            ax.set_xticks(range(3),['Flow','Pressure ratio','Efficiency']);ax.set_ylabel('$R^2$' if key=='r2' else 'nRMSE')
        g=arr(m,'geometry_norm')
        axes[2].bar(np.arange(2)+(i-.5)*.32,[_compute_diversity(g,10),_compute_novelty(g,info['train_geom'])],.32,label=['FM','cFM-CFG'][i])
    axes[2].set_xticks(range(2),['Diversity','Novelty']);axes[2].set_ylabel('k-nearest-neighbour distance')
    for ax,label in zip(axes,'def'):ax.set_title(label);ax.legend()
    save(fig,out,'Figure_09_def')


def fig10(out):
    fig,axes=plt.subplots(3,3,figsize=(11,9),layout='constrained')
    for row,(file,key,label) in enumerate([('sweep_p_uncond.json','p_uncond','Conditional dropout probability'),('sweep_guidance_scale.json','guidance_scale','Guidance scale'),('sweep_sampling_steps.json','steps','Euler steps')]):
        values=read('ablation/'+file)['results'];x=[v[key] for v in values]
        for col,metric in enumerate(['r2','nrmse']):
            for k,name,_ in VARIABLES:axes[row,col].plot(x,[v[f'{metric}_{k}'] for v in values],marker='o',ms=3,label=name)
            axes[row,col].set_ylabel('$R^2$' if metric=='r2' else 'nRMSE')
        axes[row,2].plot(x,[v['diversity'] for v in values],color='#6baed6',marker='o',ms=3,label='Diversity')
        ax2=axes[row,2].twinx();ax2.plot(x,[v['novelty'] for v in values],color='#e58038',marker='s',ms=3,label='Novelty');ax2.set_ylabel('Novelty')
        axes[row,2].set_ylabel('Diversity')
        axes[row,2].legend(handles=axes[row,2].lines+ax2.lines,loc='best')
        for col,ax in enumerate(axes[row]):
            ax.set_xlabel(label);ax.set_title(chr(97+3*row+col))
            if col<2:ax.legend(loc='best')
            ax.axvline([.061,2,100][row],color='grey',ls='--',lw=.7)
            if row==2:ax.set_xscale('log')
    save(fig,out,'Figure_10')


def fig11(out):
    a=np.load(ROOT/'data/label_5200.npy');b=np.load(ROOT/'data/performance-10000.npy')
    fig,axes=plt.subplots(1,3,figsize=(10,3.2),layout='constrained')
    for ax,(i,j) in zip(axes,[(0,2),(0,1),(2,1)]):
        for values,color,label in [(a,'#9467bd','CFD'),(b,'#e58038','TeM')]:
            ax.scatter(values[:,i],values[:,j],s=1,alpha=.08,color=color)
            xx,yy=np.meshgrid(np.linspace(values[:,i].min(),values[:,i].max(),70),np.linspace(values[:,j].min(),values[:,j].max(),70))
            zz=gaussian_kde(values[:,[i,j]].T)(np.vstack([xx.ravel(),yy.ravel()])).reshape(xx.shape)
            ax.contour(xx,yy,zz,levels=4,colors=color,linewidths=.6);ax.plot([],[],color=color,label=label)
        names=['Mass flow (kg/s)','Isentropic efficiency','Total pressure ratio'];ax.set(xlabel=names[i],ylabel=names[j]);ax.legend()
    save(fig,out,'Figure_11_fgh')


def s1(out):
    fig,axes=plt.subplots(6,3,figsize=(10,17),layout='constrained')
    for row,m in enumerate(MAIN_MODELS):
        pred,targ=arr(m,'pred_performance'),arr(m,'target_performance')
        for col,(_,label,j) in enumerate(VARIABLES):
            ax=axes[row,col];ax.scatter(targ[:,j],pred[:,j],s=3,color=COLORS[m],alpha=.5)
            lo,hi=targ[:,j].min(),targ[:,j].max();ax.plot([lo,hi],[lo,hi],'--',color='grey',lw=.8)
            ax.set(xlabel='Target '+label,ylabel='Predicted '+label,title=DISPLAY[m])
    save(fig,out,'Supplementary_Figure_01')


def s2(out):
    from shared.data_utils import get_dataloaders,GEOM_NAMES
    _,_,_,info=get_dataloaders();style();ref=info['test_geom']
    fig,axes=plt.subplots(7,3,figsize=(11,17),layout='constrained');x=np.linspace(-.1,1.1,180)
    for j,ax in enumerate(axes.flat):
        ax.plot(x,gaussian_kde(ref[:,j])(x),color='black',label='Reference')
        for m in MAIN_MODELS:ax.plot(x,gaussian_kde(arr(m,'geometry_norm')[:,j])(x),color=COLORS[m],label=DISPLAY[m],lw=.8)
        ax.set(xlabel='Normalized '+GEOM_NAMES[j],ylabel='Density')
    axes[0,0].legend(ncols=2);save(fig,out,'Supplementary_Figure_02')


def s6(out):
    trials=read('models/fm_cfg/hpo_history_fm_cfg.json');best=min(trials,key=lambda t:t['value']);h=best['history']
    fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
    for k,label in [('train_loss','Training'),('val_loss','Validation')]:ax.plot(np.arange(1,len(h[k])+1),h[k],label=label)
    ax.set(xlabel='Epoch',ylabel='Flow-matching loss',yscale='log');ax.legend();save(fig,out,'Supplementary_Figure_06')


def s7(out):
    all_values=read('benchmark/hpo_importance.json');fig,axes=plt.subplots(2,3,figsize=(11,7),layout='constrained')
    for ax,m in zip(axes.flat,MAIN_MODELS):
        values=all_values[m]['importances'];keys=list(values);v=[values[k]['value'] for k in keys]
        ax.barh(keys,v,color=COLORS[m]);ax.invert_yaxis();ax.set(title=DISPLAY[m],xlabel='Hyperparameter importance')
    save(fig,out,'Supplementary_Figure_07')


TASKS={'2':fig2,'3':fig3,'6':fig6,'9def':fig9,'10':fig10,'11fgh':fig11,'S1':s1,'S2':s2,'S6':s6,'S7':s7}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--all',action='store_true')
    p.add_argument('--figure',choices=list(TASKS),nargs='+');p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/paper_figures');args=p.parse_args()
    if not args.all and not args.figure:p.error('Select --all or --figure')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):p.error('Output directory must be empty')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for key in TASKS if args.all else args.figure:
        style();TASKS[key](args.output_dir);print('Redrawn:',key,flush=True)
    print('Figures saved to',args.output_dir)


if __name__=='__main__':main()
