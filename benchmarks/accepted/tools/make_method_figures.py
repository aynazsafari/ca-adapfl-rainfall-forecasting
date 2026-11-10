"""Source-backed method figures; never read evaluation labels or scores."""
from pathlib import Path
import argparse, hashlib, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Polygon
from shapely.geometry import shape, box

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo',type=Path,required=True)
parser.add_argument('--run',type=Path,default=Path('runs/revision_v2'))
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
REPO=args.repo.resolve()
RUN=(REPO/args.run).resolve() if not args.run.is_absolute() else args.run.resolve()
OUT=args.output.resolve()
assert RUN.is_relative_to(REPO/'runs'), 'Only a run inside the repository is admitted'
assert not OUT.exists(), 'Use a fresh figure directory; preserve earlier attempts'
OUT.mkdir(parents=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
identity=json.loads((RUN/'run_identity.json').read_text())
for module,h in identity['code'].items():
    assert sha(REPO/'caadapfl'/(module+'.py'))==h
schema=json.loads((RUN/'datasets/MAIN_FULL_schema.json').read_text())
assert len(schema['names'])==43 and len(schema['climate_names'])==7
assert all(s['role_counts']=={'train':299,'monitor':35,'validation':71,'evaluation':96} for s in schema['statistics']['clients'].values())
geo=REPO/'provenance/source_tables/india_country_ne_5_1_1.geojson'
assert sha(geo)=='4cdbe7cba953c55e4c6591b779128484c2e203ff0c3433b1826bafca95a4d090'
protocol_path=REPO/'provenance/source_tables/extraction_protocol.json'
assert sha(protocol_path)=='9f6a2f6ce237a7178e3100b7d73f2286a77d6a55d203fd2d8dc00c15fc09b1e2'
protocol=json.loads(protocol_path.read_text())
assert protocol['country_geometry']['sha256']==sha(geo)
hp_path=RUN/'development/hyperparameters_frozen.json'
hp=json.loads(hp_path.read_text())
assert hp['coefficients']==[1.0,0.0,0.0] and hp['ablation_coefficients']==[0.3,0.3,0.4]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,
 'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,'pdf.fonttype':42,
 'svg.fonttype':'none','savefig.facecolor':'white'})
captions={}
def save(fig,name,caption):
    fig.savefig(OUT/(name+'.png'),dpi=300,bbox_inches='tight')
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight')
    plt.close(fig);captions[name]=caption
def node(ax,x,y,w,h,text,color='#F1F4F6',fs=11):
    ax.add_patch(Rectangle((x,y),w,h,facecolor=color,edgecolor='#5E6973',lw=.9))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,linespacing=1.3)
def arrow(ax,start,end,text=None,xy=None,dashed=False):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=12,
                                color='#444B52',lw=1.1,linestyle='--' if dashed else '-'))
    if text:ax.text(*(xy or ((start[0]+end[0])/2,(start[1]+end[1])/2)),text,ha='center',va='center',fontsize=10,
                    bbox={'facecolor':'white','edgecolor':'none','pad':1.5})

# Portrait network at a readable print width.
fig,a=plt.subplots(figsize=(7.2,7.3));a.set(xlim=(0,7.2),ylim=(0,8.1));a.axis('off')
a.text(3.6,7.96,'FULL local network and training objective',ha='center',fontsize=13,fontweight='bold')
node(a,.15,6.7,3.05,.92,'12 historical rows ×43 channels\nCORE14 + climate7 + met22')
node(a,4,6.7,3.05,.92,'Climate context:6 indices + interaction\nLatest eligible period ends t−2',fs=10)
node(a,.15,5.15,3.05,1.08,'Bidirectional LSTM\n2 layers, 48 units per direction\n12 hidden states hⱼ of 96')
node(a,4,5.15,3.05,1.08,'Climate MLP\n7 →32 →16\nEmbedding e with tanh layers')
arrow(a,(1.68,6.7),(1.68,6.23));arrow(a,(5.52,6.7),(5.52,6.23))
node(a,1.12,3.57,4.96,1.00,'Joint attention: [hⱼ,e] →32 →1\ntanh then softmax over12 historical states\nWeighted hidden mean of96')
arrow(a,(1.68,5.15),(2.6,4.57));arrow(a,(5.52,5.15),(4.6,4.57))
node(a,1.12,2.36,4.96,.66,'Concatenate weighted mean96 + embedding16 =112')
arrow(a,(3.6,3.57),(3.6,3.02))
a.plot([7.05,7.15,7.15],[5.58,5.58,2.69],color='#444B52',lw=1.1)
arrow(a,(7.15,2.69),(6.08,2.69))
node(a,.15,.92,3.05,.9,'Rainfall head\n112 →64 →32 →1')
node(a,4,.92,3.05,.9,'Heavy-event head\n112 →32 →1 →sigmoid')
arrow(a,(2.6,2.36),(1.68,1.82));arrow(a,(4.6,2.36),(5.52,1.82))
a.text(3.6,.39,r'$L_i=\mathrm{MSE}(y_i^z,\hat y_i^z)+0.3\,\mathrm{BCE}(H_i,\hat p_i)+\frac{\mu_i}{2}\|w_i-w^r\|_2^2$',ha='center',fontsize=12)
a.text(3.6,-.03,'Shared TRAIN input scaling; local target scaling and q80.\nCAAA proximity: 0.3μ ≤ μᵢ ≤ μ.',ha='center',fontsize=10,linespacing=1.3)
fig.subplots_adjust(left=.025,right=.975,top=.98,bottom=.07)
save(fig,'01_network',
 'FULL ALCE architecture and the CAAA family training objective. Neural comparators share the two-layer bidirectional network; prespecified removal controls disable their designated components. Climate context enters nonlinear joint attention and is also concatenated directly with its weighted hidden mean. The heavy-event head produces logits during training with BCEWithLogitsLoss and probabilities through sigmoid for reporting. The event is observed rainfall strictly greater than each client’s TRAIN q80; the fixed probability cutoff is 0.5. Inputs use shared TRAIN scaling; target scaling and q80 are fitted locally. CAAA climate relevance uses fixed whole-TRAIN severity and two local sensitivity summaries; inner-monitor performance varies by round. The proximal coefficient uses eligible batch-example severity. Outer validation selects hyperparameters and global/personal checkpoints, never gradient updates. The selected primary aggregation coefficients are (1,0,0), so its performance and climate aggregation terms are inactive; the anchor removal experiments use (0.3,0.3,0.4). This is a one-process simulation with plain summaries and no secure aggregation or differential privacy. Graph controls require separate privileged access to aligned TRAIN rainfall; centralized and static-router controls use declared raw pooling.')

fig,b=plt.subplots(figsize=(7.2,6.6));b.set(xlim=(0,7.2),ylim=(0,6.8));b.axis('off')
b.text(3.6,6.6,'CAAA messages in a one-process simulation',ha='center',fontsize=13,fontweight='bold')
node(b,.15,5.02,3.05,1.15,'Five simulated clients\nLocal TRAIN labels and updates\nLocal monitor and outer validation\nLocal target scaling and q80',fs=10)
node(b,4,5.02,3.05,1.15,'Parameter server\nAdditive normalized CAAA weights\nSize + monitor score\n+ fixed TRAIN climate relevance',fs=10)
arrow(b,(3.2,5.65),(4,5.65))
arrow(b,(4,5.26),(3.2,5.26))
b.text(3.6,4.63,'Each round: states, scalar losses and fixed TRAIN strengths.\nBroadcast the global model at each training round.',ha='center',va='top',fontsize=11,linespacing=1.4)
node(b,.15,2.56,3.05,1.1,'One-time TRAIN summaries\nSample count; input sum, squared sum\nTwo local ENSO/IOD correlations',fs=10)
node(b,4,2.56,3.05,1.1,'Shared input mean and SD\nClimate correlation summaries\nPlain numerical payloads',fs=10)
arrow(b,(3.2,3.26),(4,3.26));arrow(b,(4,2.82),(3.2,2.82))
b.text(3.6,2.13,'Upload input statistics and climate summaries.\nBroadcast the input mean and SD once.',ha='center',va='top',fontsize=11,linespacing=1.4)
b.text(3.6,.65,'No secure aggregation or differential privacy is implemented.\nGraph controls separately require aligned TRAIN rainfall sharing.\nCentral and static-router controls use declared raw-data pooling.',ha='center',fontsize=10,linespacing=1.4)
fig.subplots_adjust(left=.025,right=.975,top=.98,bottom=.02)
save(fig,'02_messages',
 'CAAA family communication flow in the recorded one-process simulation. Model states and scalar monitoring/validation diagnostics are transferred each round, together with two fixed whole-TRAIN climate strengths. Counts and two local sensitivity correlations are one-time uploads. Shared input scaling uses plain TRAIN count/sum/squared-sum vectors, with mean and SD broadcast. Target mean/SD and q80 are fitted locally; two float64 target summaries are exported for inverse decoding (16 bytes/client), and q80 is exposed in schema and evaluation diagnostics. These are recorded reporting exports. Full local training rows are not required for CAAA parameter updates. The selected primary coefficients are (1,0,0), making performance and climate aggregation terms inactive; the anchor experiments use (0.3,0.3,0.4). This is no secure-aggregation or differential-privacy implementation, and scalar/summary sharing can reveal information. Graph controls separately use privileged aligned TRAIN rainfall; centralized and static-router controls use declared raw-data pooling. The network and training equation are shown in 01_network.')

fig,ax=plt.subplots(figsize=(9,5.9))
rows=[('CHIRPS rain for all12 rows',-24,-2,'#315B7E'),
      ('Six index period ends for all12 rows',-13,-2,'#71844A'),
      ('Underlying ONI three-month support',-15,-2,'#8B9E65'),
      ('Underlying MEIv2 two-month support',-14,-2,'#A4B185'),
      ('Meteorological months for all12 rows',-15,-4,'#B28735'),
      ('Historical row origins',-12,-1,'#5C6670')]
for i,(label,start,end,color) in enumerate(rows):
    y=5-i;ax.barh(y,end-start+1,left=start-.5,height=.52,color=color)
    ax.text((start+end)/2,y,f't{start:+d} … t{end:+d}',ha='center',va='center',fontsize=10,color='white' if i in (0,1,4,5) else '#202020')
ax.axvline(0,color='#333333',ls='--',lw=1.2)
ax.text(0,6.12,'Issue\nfirst day of t\n00:00 UTC',ha='center',va='bottom',fontsize=11,bbox={'facecolor':'white','edgecolor':'none','pad':2})
ax.set_yticks(range(5,-1,-1),[r[0] for r in rows]);ax.set_xticks([-24,-18,-12,-6,-4,-2,0],['t−24','t−18','t−12','t−6','t−4','t−2','t'])
ax.set(xlim=(-25,2.6),ylim=(-.8,6.95),xlabel='Calendar months relative to the forecast target t')
ax.spines[['top','right','left']].set_visible(False);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
fig.subplots_adjust(left=.4,right=.965,top=.96,bottom=.23)
fig.text(.035,.042,'Each sequence has12 rows; the availability sensitivity adds one month of delay.\nSupport rules are conservative assumptions; final archives do not establish operational vintages.',fontsize=10)
save(fig,'03_input_availability',
 'Calendar support for a target month t issued at00:00 UTC on its first day. Historical origins span t−12 through t−1. Rainfall features include lags1,2,5,11 relative to each origin and trailing3/6/12-month statistics ending one month before it, giving combined CHIRPS support t−24 through t−2. Climate period ends span t−13 through t−2; the underlying overlapping ONI and MEIv2 periods begin at t−15 and t−14. Meteorological inputs span t−15 through t−4. Known season harmonics do not use observed target rainfall. The prespecified availability sensitivity delays observed inputs by one further month. These support rules avoid using future observations but are archive-availability assumptions, not established real-time product vintages or optimized physical propagation lags.')

raw=json.loads(geo.read_text())
if raw.get('type')=='FeatureCollection':country=shape(raw['features'][0]['geometry'])
elif raw.get('type')=='Feature':country=shape(raw['geometry'])
else:country=shape(raw)
domain_names=[('Client1_NW_Arid','C1','NW Arid'),('Client2_Indo_Gangetic','C2','Indo-Gangetic'),
 ('Client3_NE_Himalaya','C3','NE Himalaya'),('Client4_South_Peninsular','C4','South Peninsular'),
 ('Client5_Central_India','C5','Central India')]
assert set(protocol['clients'])=={d[0] for d in domain_names}
domains=[(short,name,*protocol['clients'][client]) for client,short,name in domain_names]
colors=['#315B7E','#B28735','#71844A','#A6694A','#7E6897']
fig,(ax,notes)=plt.subplots(1,2,figsize=(8,6.4),gridspec_kw={'width_ratios':[1.05,1]})
def polygons(geometry):
    if geometry.is_empty:return []
    if geometry.geom_type=='Polygon':return [geometry]
    return [p for g in geometry.geoms for p in polygons(g)]
for poly in polygons(country):
    ax.add_patch(Polygon(list(poly.exterior.coords),facecolor='#F2F2F2',edgecolor='#777777',lw=.7))
for (short,name,south,north,west,east),color in zip(domains,colors):
    intersection=country.intersection(box(west,south,east,north))
    for poly in polygons(intersection):
        ax.add_patch(Polygon(list(poly.exterior.coords),facecolor=color,alpha=.22,edgecolor=color,lw=1.2))
    ax.add_patch(Rectangle((west,south),east-west,north-south,fill=False,edgecolor=color,lw=1.1))
    ax.text((west+east)/2,(south+north)/2,short,ha='center',va='center',fontsize=12,fontweight='bold',color=color)
ax.set(xlim=(66.5,99),ylim=(5.5,37.5),xlabel='Longitude °E',ylabel='Latitude °N',title='Fixed study domains')
ax.set_aspect('equal');ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
notes.axis('off');notes.set(xlim=(0,1),ylim=(0,1))
notes.text(0,.97,'Frozen geographic bounds',va='top',fontweight='bold',fontsize=12)
for i,((short,name,south,north,west,east),color) in enumerate(zip(domains,colors)):
    y=.85-i*.125
    notes.text(0,y,f'{short}  {name}',color=color,fontweight='bold',fontsize=11)
    notes.text(0,y-.045,f'{south:g}–{north:g}°N; {west:g}–{east:g}°E',fontsize=11)
notes.text(0,.175,'C1 and C5 overlap. These domains\ndo not partition or cover all India.\n\nNative centers select the mask.\nWhole-cell WGS84 areas supply weights.\nNo grid resampling is used.',va='top',fontsize=10,linespacing=1.4)
fig.tight_layout()
save(fig,'04_fixed_domains',
 'Frozen study domains intersected with Natural Earth5.1.1 country geometry. Bounding boxes are shown in the source coordinate system; each source mask selects native grid centers within the country and half-open box bounds and weights the whole selected cell area on WGS84. Native grids are not resampled. C1 and C5 overlap, and the five study domains do not partition or fully cover India. This is a study-domain map, not an April–June2025 rainfall map or a forecast. Geometry SHA256:4cdbe7cba953c55e4c6591b779128484c2e203ff0c3433b1826bafca95a4d090.')

(OUT/'captions.json').write_text(json.dumps(captions,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'captions.txt').write_text('\n\n'.join(name+'\n'+caption for name,caption in captions.items()),encoding='utf-8')
sources={'run_identity':sha(RUN/'run_identity.json'),
 'schema':sha(RUN/'datasets/MAIN_FULL_schema.json'),'country_geometry':sha(geo),
 'extraction_protocol':sha(protocol_path),'locked_hyperparameters':sha(hp_path),
 'scientific_modules':identity['code'],'generator':sha(__file__)}
manifest={'status':'RENDERED_AWAITING_VISUAL_QA','source_only':True,'figure_ids':list(captions),
 'evaluation_labels_or_scores_opened':False,'model_training_or_inference_performed':False,
 'run_sha256':identity['run_sha256'],'source_sha256':sources,
 'files':[{'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.iterdir()) if p.is_file() and (p.stem in captions or p.name in ['captions.json','captions.txt'])]}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'status':manifest['status'],'figures':len(captions),'source_only':True}))
