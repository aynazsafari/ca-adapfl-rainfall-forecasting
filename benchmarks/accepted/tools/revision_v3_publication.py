"""Publication tables and scientific figures from independently verified v3 evidence."""
from pathlib import Path
import json,hashlib,datetime,sys
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/revision_v3';EVAL=RUN/'evaluation';OUT=ROOT/'publication/revision_v3'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def make():
 qa=json.loads((RUN/'independent_qa/complete.json').read_text());assert qa['status']=='PASS' and qa['raw_predictions_sha256']==sha(EVAL/'raw_predictions.csv')
 if OUT.exists():raise RuntimeError('Publication evidence exists; preserve it before reviewed regeneration.')
 OUT.mkdir(parents=True);summary=pd.read_csv(EVAL/'seed_summary.csv',float_precision='round_trip');tail=pd.read_csv(EVAL/'tail_bias_summary.csv',float_precision='round_trip');ci=pd.read_csv(EVAL/'paired_CI_36.csv',float_precision='round_trip');frozen=json.loads((RUN/'choices_frozen.json').read_text());jobs=frozen['all_jobs'];receipts={j['id']:json.loads((RUN/'fits'/j['id']/'complete.json').read_text()) for j in jobs}
 main=summary.merge(tail,on='model',validate='one_to_one');main.to_csv(OUT/'all_model_summary.csv',index=False,float_format='%.17g')
 columns=['model','R2_mean','R2_std','RMSE_mm_mean','RMSE_mm_std','MAE_mm_mean','tail80_RMSE_macro_mm','tail80_bias_macro_mm','tail80_underprediction_macro_fraction','nonheavy_RMSE_macro_mm','MSESS_vs_climatology','negative_predictions_total_over_2400_seed_specific_rows']
 def table(name,labels):
  t=main.set_index('model').loc[labels,columns[1:]].reset_index();t.to_csv(OUT/(name+'.csv'),index=False,float_format='%.17g');return t
 candidates=['FULL_H48','FULL_H24','FULL_H12','COMPACT_H12','REGULARIZED_H12'];development=table('S11_development',candidates)
 for col in ['validation_R2_mean','validation_R2_sd','trainable_parameters','channels']:development[col]=np.nan
 for i,label in enumerate(candidates):
  rr=[receipts[j['id']] for j in jobs if j['label']==label];development.loc[i,['validation_R2_mean','validation_R2_sd','trainable_parameters','channels']]=[np.mean([x['validation_macro_r2'] for x in rr]),np.std([x['validation_macro_r2'] for x in rr],ddof=1),rr[0]['parameters']['trainable'],35 if label=='COMPACT_H12' else 43]
 development.to_csv(OUT/'S11_development.csv',index=False,float_format='%.17g')
 table('S12_components',['CAAA_RAW','EMBEDDING_OFF','ADAPTIVE_OFF','BOTH_OFF'])
 variants=['RAW','NONNEGATIVE','NONNEGATIVE_TAIL2'];methods=['CAAA','FEDAVG','FEDPROX','CENTRAL'];physical=table('S13_output_loss',[m+'_'+v for m in methods for v in variants])
 table('S14_delay',['DELAY2_REFERENCE','DELAY2_OBSERVATION'])
 ci.to_csv(OUT/'S15_new_CI_36_all_blocks.csv',index=False,float_format='%.17g')
 archived=pd.concat([pd.read_csv(EVAL/'archived_additional_CI_11_unadjusted.csv',float_precision='round_trip'),pd.read_csv(EVAL/'archived_feature_CI_2_unadjusted.csv',float_precision='round_trip')],ignore_index=True);archived.to_csv(OUT/'S16_archived_CI_13_all_blocks.csv',index=False,float_format='%.17g')
 cost=pd.DataFrame([{'id':j['id'],'label':j['label'],'seed':j['seed'],'scenario':j['scenario'],'group':j['group'],**{key:receipts[j['id']][key] for key in ['selected_round','validation_macro_r2','elapsed_seconds','optimizer_steps','examples']}} for j in jobs]);cost.to_csv(OUT/'fit_inventory_100.csv',index=False,float_format='%.17g')
 plt.rcParams.update({'font.size':11,'axes.titlesize':12,'axes.labelsize':11,'legend.fontsize':10,'savefig.dpi':250,'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
 sources={}
 def export(fig,name,description,inputs):
  fig.savefig(OUT/(name+'.png'),bbox_inches='tight');fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig);sources[name]={'description':description,'source_tables':inputs}
 short=['FULL48','FULL24','FULL12','COMPACT12','REG12'];x=np.arange(5)
 fig,ax=plt.subplots(2,1,figsize=(6.45,6.2));ax[0].errorbar(x,development.validation_R2_mean,yerr=development.validation_R2_sd,fmt='o',capsize=4,color='#265a91',label='Validation mean ± seed SD');ax[0].set_xticks(x,[f'{s}\n({int(p):,})' for s,p in zip(short,development.trainable_parameters)],fontsize=10);ax[0].set_ylabel('Macro R²');ax[0].set_title('Development selection; parameters in parentheses');ax[0].legend(loc='lower right')
 intervals=ci[(ci.block_length_months==12)&(ci.metric=='macro_R2')&ci.contrast_id.isin(['C01','C02','C03','C04'])].set_index('contrast_id').loc[['C01','C02','C03','C04']]
 for i,(_,r) in enumerate(intervals.iterrows()):ax[1].plot([r.CI95_low,r.CI95_high],[i,i],color='#265a91',lw=2);ax[1].plot(r.delta_a_minus_b,i,'o',color='#265a91')
 ax[1].set_yticks(range(4),['FULL24 − FULL48','FULL12 − FULL48','COMPACT12 − FULL12','REG12 − FULL12']);ax[1].invert_yaxis();ax[1].axvline(0,color='#555',ls='--',lw=1);ax[1].set_xlabel('Evaluation Δ macro R² (A − B)');ax[1].set_title('Exploratory paired 95% intervals');fig.tight_layout(h_pad=2)
 export(fig,'Figure_S5_capacity','Top whiskers: five-seed SD, not CI. Bottom: unadjusted paired common-date12-month block CI. Labels/parameters distinguish allfive candidates. FULL12 selected on validation, not evaluation.',['S11_development.csv','S15_new_CI_36_all_blocks.csv'])
 fig,ax=plt.subplots(figsize=(6.45,3.8));active=ci[(ci.block_length_months==12)&(ci.metric=='macro_R2')&ci.contrast_id.isin(['C05','C06','C07','C08'])].set_index('contrast_id').loc[['C05','C06','C07','C08']]
 for i,(_,r) in enumerate(active.iterrows()):ax.plot([r.CI95_low,r.CI95_high],[i,i],color='#287c69',lw=2);ax.plot(r.delta_a_minus_b,i,'o',color='#287c69')
 ax.set_yticks(range(4),['Parent − embedding off','Parent − adaptive prox off','Parent − both off','Embedding off − both off']);ax.invert_yaxis();ax.axvline(0,color='#555',ls='--');ax.set_xlabel('Δ macro R² (A − B); unadjusted paired 95% CI');ax.set_title('Selected FULL12 parent: active-component controls');fig.tight_layout();export(fig,'Figure_S6_active_components','Same selected parent and fixed sample-count aggregation. Embedding-off retains climate sequence channels/drivers; adaptive-off retains fixed proximalμ=.1.',['S12_components.csv','S15_new_CI_36_all_blocks.csv'])
 fig,axs=plt.subplots(3,1,figsize=(6.45,7.2));styles=[('RAW','#4c6686','o'),('NONNEGATIVE','#287c69','s'),('NONNEGATIVE_TAIL2','#b56a3b','^')]
 for v,color,marker in styles:
  rows=physical.set_index('model').loc[[m+'_'+v for m in methods]]
  for ax,col in zip(axs,['RMSE_mm_mean','tail80_RMSE_macro_mm','tail80_bias_macro_mm']):ax.plot(range(4),rows[col],marker=marker,color=color,label={'RAW':'Raw','NONNEGATIVE':'Nonnegative','NONNEGATIVE_TAIL2':'Nonnegative + tail ×2'}[v]);ax.set_xticks(range(4),['CAAA','FedAvg','FedProx','Central'])
 for ax,title in zip(axs,['Overall RMSE (mm)','TRAIN-q80 tail RMSE (mm)','Tail bias: forecast − observed (mm)']):ax.set_title(title);ax.grid(axis='y',alpha=.15)
 axs[2].axhline(0,color='#555',ls='--',lw=1);handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3);fig.tight_layout(rect=[0,.06,1,1],h_pad=1.2);export(fig,'Figure_S7_output_tail','Point estimates average equal-client metrics across five seeds. No uncertainty bars implied. Nonnegative outputs and fixed TRAIN tailweight2 are evaluated for allfour methods with the common validation-selected architecture.',['S13_output_loss.csv'])
 manifest={'status':'BUILT_FROM_VERIFIED_EVIDENCE','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'qa_receipt_sha256':sha(RUN/'independent_qa/complete.json'),'raw_sha256':sha(EVAL/'raw_predictions.csv'),'creator_runtime':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__},'figures':sources,'files':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}}
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print('Publication tables and3scientific figures built from verified evidence.')
if __name__=='__main__':make()
