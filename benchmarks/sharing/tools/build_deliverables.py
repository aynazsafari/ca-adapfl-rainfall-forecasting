"""Build publication figures and executed analytical notebook from reviewed rows."""
from pathlib import Path
import json,sys,re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import run_experiment as r
ORDER=['LOCAL_PRIVATE','LOCAL_STATS','FEDAVG_PRIVATE_K1','FEDAVG_STATS_K1','FEDAVG_STATS_K5','FEDAVG_STATS_K20','CAAA_STATS_K1','CENTRAL_POOL_STATS','TRAIN_CLIMATOLOGY']
LABELS=['Local: own statistics','Local: shared statistics','FedAvg: own stats, k=1','FedAvg: shared stats, k=1','FedAvg: shared stats, k=5','FedAvg: shared stats, k=20','CA-AdapFL: shared, k=1','Pooled TRAIN examples','TRAIN monthly climatology']

def figures():
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    summary=pd.read_csv(ROOT/'evaluation/principal_comparison.csv').set_index('model').loc[ORDER]
    seeds=pd.read_csv(ROOT/'evaluation/macro_per_seed_ALL_endpoints.csv');seeds=seeds[seeds.endpoint=='fixed60']
    cost=pd.read_csv(ROOT/'evaluation/fit_cost_40.csv').groupby('model').mean(numeric_only=True)
    ci=pd.read_csv(ROOT/'evaluation/paired_CI_7.csv');ci=ci[ci.block_length_months==12]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.labelcolor':'#243244','text.color':'#243244','axes.edgecolor':'#243244'})
    ys=np.arange(len(ORDER));fig,ax=plt.subplots(1,2,figsize=(13.5,6.7),sharey=True)
    for k,(metric,title) in enumerate([('RMSE_mm','Macro RMSE (mm), lower is better'),('MSESS','Macro MSE skill vs TRAIN climatology')]):
        for i,m in enumerate(ORDER):
            values=seeds[seeds.model==m][metric].to_numpy();ax[k].scatter(values,np.full(len(values),i),s=23,color='#a9bacb',marker='o')
            mean=summary.loc[m,metric+'_mean'];ax[k].scatter([mean],[i],marker='D',s=50,color='#286093' if m!='TRAIN_CLIMATOLOGY' else '#243244')
            ax[k].annotate(f'{mean:.2f}' if metric=='RMSE_mm' else f'{mean:.3f}',(mean,i),xytext=(7,5),textcoords='offset points',fontsize=9)
        ax[k].set_xlabel(title);ax[k].grid(axis='x',alpha=.2);ax[k].margins(x=.2)
    ax[1].axvline(0,color='#67717d',lw=1,ls='--');ax[0].set_yticks(ys,LABELS);ax[0].invert_yaxis()
    fig.suptitle('Fixed tick 60 • five fixed regions • EVAL 2017–2024',fontsize=15)
    fig.text(.02,.02,'Dots: five paired seed estimates. Diamonds: mean. Seed spread is not a confidence interval. 480 unique client-month outcomes.',fontsize=10)
    fig.tight_layout(rect=(0,.055,1,.95));fig.savefig(out/'01_accuracy_and_climatology.png',dpi=170);fig.savefig(out/'01_accuracy_and_climatology.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(13.5,6.7))
    markers=['o','s','^','D','v','P','X','h'];short=['Local/own','Local/shared','FedAvg/own k1','FedAvg/shared k1','FedAvg/shared k5','FedAvg/shared k20','CA/shared k1','Pooled examples']
    for i,m in enumerate(ORDER[:-1]):
        x=float(cost.loc[m,'logical_required_payload_bytes']/1024**2);mean=float(summary.loc[m,'RMSE_mm_mean']);vals=seeds[seeds.model==m].RMSE_mm.to_numpy()
        ax.plot([x,x],[vals.min(),vals.max()],color='#a9bacb',lw=2)
        ax.scatter([x],[mean],marker=markers[i],s=85,color='#286093' if 'FEDAVG' in m else '#b37b24',label=short[i])
    ax.axhline(summary.loc['TRAIN_CLIMATOLOGY','RMSE_mm_mean'],color='#243244',lw=1.2,ls='--',label='TRAIN climatology (no training exchange)')
    ax.set_xscale('symlog',linthresh=.01);ax.set_xlabel('Required numeric payload (MiB): symmetric-log axis; 0 exchange retained')
    ax.set_ylabel('Macro RMSE (mm), lower is better');ax.grid(alpha=.2);ax.margins(x=.15,y=.2);ax.legend(loc='center left',bbox_to_anchor=(1.02,.5))
    ax.set_title('Skill and declared communication under the fixed update budget')
    fig.text(.02,.02,'Vertical ranges show seed min–max, not uncertainty intervals. Payload includes initial and final deployment; excludes transport and audit.',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,1));fig.savefig(out/'02_communication_and_skill.png',dpi=170);fig.savefig(out/'02_communication_and_skill.svg');plt.close(fig)
    chosen=ci[ci.metric=='macro_MSE_mm2'].sort_values('contrast_id');names=['Stats sharing in local','Model exchange / shared stats','Model exchange / own stats','Cadence5 − cadence1','Cadence20 − cadence1','Scaling × exchange interaction','CA proximal − FedAvg (secondary)']
    fig,ax=plt.subplots(figsize=(11.5,6.7));ys=np.arange(len(chosen))
    for i,q in enumerate(chosen.itertuples()):
        ax.plot([q.CI95_low,q.CI95_high],[i,i],color='#a9bacb',lw=3);ax.scatter([q.delta],[i],color='#286093',s=55,marker='D')
    ax.axvline(0,color='#243244',ls='--');ax.set_yticks(ys,names);ax.invert_yaxis();ax.set_xlabel('Contrast in macro MSE (mm²); negative favors first policy; interaction is a difference of differences')
    ax.set_title('Paired effects with 95% 12-month-block percentile intervals');ax.grid(axis='x',alpha=.2)
    fig.text(.02,.02,'Unadjusted intervals; S01–S06 form amended Holm6 annual-test family. All seven Holm7 tests are retained as sensitivity.',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,1));fig.savefig(out/'03_paired_sharing_effects.png',dpi=170);fig.savefig(out/'03_paired_sharing_effects.svg');plt.close(fig)
    # Matched candidate sets allow a visible, explicitly secondary selection sensitivity.
    both=pd.read_csv(ROOT/'evaluation/model_summary_ALL_endpoints.csv');both=both[both.endpoint.isin(['fixed60','VAL20_40_60'])]
    fig,ax=plt.subplots(figsize=(12,6.7));ys=np.arange(8)
    for i,m in enumerate(ORDER[:-1]):
        a=float(both[(both.model==m)&(both.endpoint=='fixed60')].RMSE_mm_mean.iloc[0]);b=float(both[(both.model==m)&(both.endpoint=='VAL20_40_60')].RMSE_mm_mean.iloc[0])
        ax.plot([a,b],[i,i],color='#a9bacb',lw=2);ax.scatter([a],[i],marker='D',color='#286093',s=55,label='Fixed tick60' if i==0 else None)
        ax.scatter([b],[i],marker='o',facecolors='white',edgecolors='#b37b24',s=65,label='Common VAL choice:20/40/60' if i==0 else None)
    ax.set_yticks(ys,LABELS[:-1]);ax.invert_yaxis();ax.set_xlabel('Macro RMSE (mm), lower is better');ax.grid(axis='x',alpha=.2);ax.legend();ax.set_title('Checkpoint selection is an explicitly priced secondary information policy')
    fig.text(.02,.02,'Global endpoint for every federated policy. Selecting one identity requires 280 logical summary/decision bytes; primary fitting uses no VAL feedback.',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,1));fig.savefig(out/'04_checkpoint_sensitivity.png',dpi=170);fig.savefig(out/'04_checkpoint_sensitivity.svg');plt.close(fig)

def notebook():
    out=ROOT/'notebook';out.mkdir(exist_ok=True)
    nb=nbformat.v4.new_notebook();md=nbformat.v4.new_markdown_cell;code=nbformat.v4.new_code_cell
    observed=r.load(ROOT/'reports/scientific_interpretation.json')['finding_en']
    nb.cells=[
        md('# Information sharing in Indian monthly rainfall forecasting\n\nExecuted exploratory experiment log. Forty new fits; eight declared information policies; five paired seeds. This notebook recomputes the released primary comparison and displays the reviewed figures. It does not train models when opened.\n\n## Findings\n\n'+observed+'\n\nRead these estimates together with their paired intervals and limitations. The evaluation period 2017–2024 had already been examined in the parent study.'),
        code("from pathlib import Path\nimport json, hashlib\nimport numpy as np\nimport pandas as pd\nfrom IPython.display import display, Image, Markdown\nROOT = Path.cwd().resolve()\nif not (ROOT / 'protocol.json').exists():\n    ROOT = ROOT.parent\nassert (ROOT / 'evaluation' / 'raw_predictions_PRIMARY.csv').exists()\nprotocol = json.loads((ROOT / 'protocol.json').read_text())\namendment = json.loads((ROOT / 'analysis_amendment_v1.json').read_text())\nqa = json.loads((ROOT / 'qa' / 'independent_numeric_qa.json').read_text())\nassert qa['status'] == 'PASS'\npd.set_option('display.max_columns', 12)\nprint('Independent mathematical QA:', qa['status'])"),
        md('## Context and methods\n\nTwo-by-two comparison: own versus shared TRAIN input statistics, crossed with local versus FedAvg training. Shared-statistics FedAvg also varies synchronization cadence1/5/20. CA-AdapFL and pooled TRAIN examples provide contextual comparisons. Every distributed fit retains60ticks×3epochs, identical paired local RNG and Adam resets. Pooled training has8460rather than9000optimizer steps and is contextual.\n\nPrimary endpoint: terminal tick60. Federated endpoints use the deployed global state. Mean client/seed MSE is primary; RMSE and R² are separate macro averages. Climatology is calculated only from TRAIN.\n\n### Assumptions\n\nFive fixed overlapping public regions, a common previously selected architecture, and revised source archives. Local normalization also changes parameter-coordinate compatibility. Logical payload is not WAN, energy or privacy. VAL20/40/60 selection is secondary and priced separately.\n\nThe original seven-test plan is preserved. The before-evaluation amendment separates six sharing questions from the same-payload proximal comparison. Holm7 remains visible as sensitivity; no confirmatory preregistration is claimed.'),
        md('## Data and independent recomputation'),
        code("raw = pd.read_csv(ROOT / 'evaluation' / 'raw_predictions_PRIMARY.csv', float_precision='round_trip')\nassert len(raw) == 21600\nassert raw[['client','target_month']].drop_duplicates().shape[0] == 480\nassert not raw.duplicated(['model','seed','client','target_month']).any()\nrows = []\nfor (model, seed, client), g in raw.groupby(['model','seed','client']):\n    y = g.y.to_numpy(); p = g.prediction.to_numpy(); c = g.climatology_mm.to_numpy()\n    loss = (p-y)**2\n    rows.append({'model':model, 'seed':seed, 'client':client, 'MSE_mm2':loss.mean(), 'RMSE_mm':np.sqrt(loss.mean()), 'R2':1-loss.sum()/((y-y.mean())**2).sum(), 'MSESS':1-loss.sum()/((y-c)**2).sum()})\nrecomputed = pd.DataFrame(rows).groupby('model')[['MSE_mm2','RMSE_mm','R2','MSESS']].mean()\nreleased = pd.read_csv(ROOT / 'evaluation' / 'principal_comparison.csv').set_index('model')\nfor metric in recomputed.columns:\n    assert np.allclose(recomputed[metric], released.loc[recomputed.index, metric+'_mean'], rtol=1e-12, atol=1e-10)\nprint('480 unique outcomes; seed copies are repeated forecasts.')\ndisplay(recomputed.round(4))"),
        md('## Results: accuracy and climatology'),
        code("display(Image(filename=str(ROOT / 'figures' / '01_accuracy_and_climatology.png')))"),
        md('### Required communication and computation\n\nEvery recipient is counted, including initial delivery and final global deployment. Shared scaler setup is6920bytes. Central upload counts the declared float64 engineered TRAIN example representation, not a minimal original-product record upload. Research validation exports are separate.'),
        code("cost = pd.read_csv(ROOT / 'evaluation' / 'fit_cost_40.csv')\ncolumns = ['logical_required_payload_bytes','train_wall_seconds','train_cpu_seconds','optimizer_steps','examples','three_checkpoint_VAL_selection_bytes_per_identity']\ndisplay(cost.groupby('model')[columns].mean().round(3))\ndisplay(Image(filename=str(ROOT / 'figures' / '02_communication_and_skill.png')))"),
        md('### Paired contrasts\n\nNegative MSE/RMSE differences favor the first policy. R²/skill use the opposite direction. The interaction is a difference of policy effects, not the accuracy of a model. Twelve-month intervals are primary block estimates;6/24-month sensitivities and every resample/draw are included. Intervals are unadjusted.'),
        code("ci = pd.read_csv(ROOT / 'evaluation' / 'paired_CI_7.csv')\ndisplay(ci[(ci.block_length_months==12)&(ci.metric.isin(['macro_MSE_mm2','macro_RMSE_mm']))][['contrast_id','metric','delta','CI95_low','CI95_high']].round(4))\ndisplay(Image(filename=str(ROOT / 'figures' / '03_paired_sharing_effects.png')))\ndisplay(pd.read_csv(ROOT / 'evaluation' / 'annual_MSE_sign_tests.csv')[['contrast_id','two_sided_p','holm_p_6_primary','holm_p_7_sensitivity','primary_reject_05','contrast_role']])"),
        md('### Secondary checkpoint and personal-state diagnostics\n\nA three-candidate common macro-VAL rule is distinct from the fixed-budget primary endpoint. It entails sharing validation summaries even for LOCAL_PRIVATE. This prevents treating best-VAL local results as entirely information-isolated. Personal federated forecasts are labeled separately and do not replace the global primary endpoint.'),
        code("display(Image(filename=str(ROOT / 'figures' / '04_checkpoint_sensitivity.png')))\nall_endpoints = pd.read_csv(ROOT / 'evaluation' / 'model_summary_ALL_endpoints.csv')\ndisplay(all_endpoints[['endpoint','model','RMSE_mm_mean','R2_mean','MSESS_mean']].round(4))"),
        md('## What this adds to the reviewer response\n\nThe study replaces an implicit common-preprocessing assumption with explicit policy interventions, isolates synchronization from local optimization budget, and connects effect estimates to declared communication. It supplies a substantive, bounded systems experiment even if federation remains inferior to TRAIN climatology.\n\nThese comparisons do not prove a new algorithm, general institutional benefit, privacy protection, independent external validity or operational flood-warning utility. Eight annual blocks and480client-month outcomes remain the evidence units. The before-evaluation analysis amendment and original Holm7 results stay visible.\n\nThe Persian review and English reviewer addendum accompany the executed evidence after full QA.'),
    ]
    descriptions={'01_accuracy_and_climatology.png':'Fixed tick60 macro RMSE and climatology skill with paired seed estimates',
        '02_communication_and_skill.png':'Required logical communication versus macro RMSE, including climatology',
        '03_paired_sharing_effects.png':'Seven paired MSE effects with unadjusted12-month confidence intervals',
        '04_checkpoint_sensitivity.png':'Fixed tick60 and common validation-selected global endpoints'}
    for cell in nb.cells:
        if cell.cell_type=='markdown':
            cell.source=re.sub(r'(?<=\d)(?=[^\W\d_])',' ',cell.source)
            for a,b in [('cadence1','cadence 1'),('retains60','retains 60'),('has8460','has 8460'),('than9000','than 9000'),
                ('tick60','tick 60'),('is6920','is 6920'),('and480','and 480'),('by79','by 79'),('ticks×3','ticks × 3')]:cell.source=cell.source.replace(a,b)
        else:
            for name,description in descriptions.items():
                cell.source=cell.source.replace("display(Image(filename=str(ROOT / 'figures' / '"+name+"')))",
                    "display(Image(filename=str(ROOT / 'figures' / '"+name+"'),alt="+repr(description)+"))")
    nb.metadata['kernelspec']={'display_name':'Python3','language':'python','name':'python3'}
    path=out/'Information_Sharing_Study.ipynb';nbformat.write(nb,path)
    NotebookClient(nb,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.write(nb,path)
    # A saved HTML companion keeps all outputs portable without a live kernel.
    from nbconvert import HTMLExporter
    html,_=HTMLExporter().from_notebook_node(nb)
    # nbconvert's image template does not expose IPython's nested PNG alt metadata.
    from bs4 import BeautifulSoup
    document=BeautifulSoup(html,'html.parser');images=document.find_all('img')
    assert len(images)==4
    for tag,description in zip(images,descriptions.values()):tag['alt']=description
    (out/'Information_Sharing_Study.html').write_text(str(document),encoding='utf-8')
    r.save(ROOT/'qa/notebook_execution.json',dict(status='PASS',utc=r.now(),cells=len(nb.cells),executed_code_cells=sum(c.cell_type=='code' for c in nb.cells),no_empirical_fits=True,
        ipynb_sha256=r.sha(path),html_sha256=r.sha(out/'Information_Sharing_Study.html'),visual_status='PENDING'))

def main():
    assert r.load(ROOT/'qa/independent_numeric_qa.json')['status']=='PASS'
    figures();notebook();print('FIGURES4_NOTEBOOK_EXECUTED',flush=True)

if __name__=='__main__':main()
