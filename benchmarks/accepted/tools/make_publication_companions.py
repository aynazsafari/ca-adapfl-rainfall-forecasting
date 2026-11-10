#!/usr/bin/env python
"""Nine manuscript-width companions from accepted, hash-verified figure CSVs.

Copies/reads accepted source exports only; no fitting, inference, evaluation,
source preparation, pickle loading, network access or subprocess execution.
Copy this file and layout_plan.json to repository tools/ before portable use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import uuid
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import pandas as pd

PLAN_SHA256 = '7c62f5c40d4f1d2a55c3f8d763dd6dae2b50c2cb3b9fcb61bc3e21884ff200a0'
ORIGINAL_IDS = ('01_global_and_classical', '02_personal', '03_paired_family',
                '04_anchor_ablations', '05_training_graphs', '06_validation_grid',
                '07_client_heavy', '08_client_PR_seed42')
COLORS = ('#315B7E', '#B28735', '#71844A', '#5C6670', '#A6694A')
MARKERS = ('o', 's', '^', 'D', 'v', 'P', 'X')


class IntegrityError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise IntegrityError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise IntegrityError(f'Cannot read required receipt {path}: {error}') from error


def safe_child(root, relative):
    require(isinstance(relative, str) and bool(relative), 'Invalid saved relative path')
    part = PurePosixPath(relative.replace('\\', '/'))
    require(not part.is_absolute() and not PureWindowsPath(relative).drive and '..' not in part.parts,
            f'Escaping/nonportable saved path: {relative}')
    child = (Path(root) / Path(*part.parts)).resolve()
    require(child.is_relative_to(Path(root).resolve()), f'Symlink escapes saved directory: {relative}')
    return child


def check_hash(path, expected):
    require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected) is not None,
            f'Invalid saved SHA256 for {path}')
    require(Path(path).is_file() and sha256(path) == expected, f'Missing/changed saved artifact: {path}')


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def load_plan(path=None):
    path = Path(path) if path else Path(__file__).with_name('layout_plan.json')
    check_hash(path, PLAN_SHA256)
    plan = read_json(path)
    require(len(plan['pairs']) == 31 and len(plan['plates']) == 9, 'Frozen plan counts differ')
    require(plan['forest_ranges'] == [[0, 8], [8, 16], [16, 24], [24, 31]], 'Forest partition differs')
    return plan


def accepted_plotter_source(repo, plan):
    """Admit only the disclosed arithmetic-type repair of the frozen plotter."""
    source = repo / 'tools' / 'make_figures.py'
    if sha256(source) == plan['original_script_sha256']:
        return plan['original_script_sha256'], []
    repair_path = repo / 'docs' / 'publication_repairs' / 'graph_precision_repair.json'
    repair = read_json(repair_path)
    require(repair.get('status') == 'PASS' and repair.get('scientific_modules_unchanged') is True,
            'Plotter repair is absent or unaccepted.')
    require(repair.get('original_plotter_sha256') == plan['original_script_sha256'] and
            repair.get('frozen_layout_plan_sha256') == PLAN_SHA256, 'Repair parent/unchanged layout differs.')
    parent = repo / 'docs' / 'publication_repairs' / 'pre_actual_qa_make_figures.py'
    check_hash(parent, plan['original_script_sha256'])
    old = b"    close(graphs['graphs']['enriched'], np.mean(np.asarray(components, float), axis=0), 'enriched equal-component mean')"
    new = (b"    require(np.array_equal(np.asarray(graphs['graphs']['enriched'], float),\n"
           b"                           np.mean(np.asarray(components, float), axis=0).astype(np.float32).astype(float)),\n"
           b"            'Enriched graph differs from the equal-component mean serialized as float32')")
    parent_bytes = parent.read_bytes()
    if b'\r\n' in parent_bytes:
        new = new.replace(b'\n', b'\r\n')
    require(parent_bytes.count(old) == 1 and parent_bytes.replace(old, new) == source.read_bytes(),
            'Plotter changes extend beyond the exact disclosed float32 verification repair.')
    check_hash(source, repair.get('repaired_plotter_sha256'))
    return repair['repaired_plotter_sha256'], [repair_path, parent]


def gates(repo, run, figures, qa_path, figure_qa_path, plan):
    """Complete every receipt/hash gate before the first accuracy CSV read."""
    complete_path = run / 'evaluation' / 'complete.json'
    require(complete_path.is_file(), 'Completed evaluation absent; this tool never starts evaluation or training.')
    complete, qa = read_json(complete_path), read_json(qa_path)
    run_sha = complete.get('run_sha256')
    require(isinstance(run_sha, str) and re.fullmatch('[0-9a-f]{64}', run_sha), 'Invalid completed-run identity')
    require(qa.get('status') == 'PASS' and qa.get('run_sha256') == run_sha,
            'Independent run QA missing, failed or from another run; no accuracy CSV read.')
    require(qa.get('evaluation_complete_sha256') == sha256(complete_path), 'Independent QA is not bound to evaluation completion.')
    require(isinstance(qa.get('scope'), str) and qa['scope'].strip(), 'Independent QA scope absent.')
    identity = read_json(run / 'run_identity.json')
    require(identity.get('run_sha256') == run_sha == canonical_digest({k:v for k,v in identity.items() if k != 'run_sha256'}),
            'Run identity digest differs.')
    require(isinstance(identity.get('code'),dict) and identity['code'], 'Run source-code hash ledger absent.')
    for name,saved_hash in identity['code'].items():
        require(re.fullmatch('[A-Za-z_][A-Za-z_0-9]*',name) is not None, 'Invalid saved source module key.')
        check_hash(repo/'caadapfl'/(name+'.py'),saved_hash)
    protocol = read_json(run / 'protocol.json')
    require(identity.get('protocol_hash') == canonical_digest(protocol), 'Protocol identity digest differs.')
    check_hash(run / 'protocol.json', plan['protocol_file_sha256'])
    require(protocol['comparisons'] == plan['pairs'] and protocol['seeds'] == plan['seeds'], 'Frozen comparison/seed order differs.')
    require(complete.get('seeds') == plan['seeds'] and complete.get('months') ==
            [str(x) for x in pd.period_range('2017-01', '2024-12', freq='M')], 'Incomplete evaluation calendar/seeds.')
    require(complete.get('evaluation_truth_opened_only_after_all_choices') is True, 'Evaluation choice-freeze gate differs.')
    check_hash(run / 'choices_frozen.json', complete.get('choices_frozen_sha256'))
    choices = read_json(run / 'choices_frozen.json')
    require(choices.get('run_sha256') == run_sha and choices.get('evaluation_labels_accessed') is False and
            choices.get('choices_including_checkpoints_and_router_weights_frozen') is True, 'Choices are not safely frozen.')
    require(isinstance(complete.get('files'), dict) and complete['files'], 'Evaluation completion hash map absent.')
    for relative, saved_hash in complete['files'].items():
        check_hash(safe_child(run / 'evaluation', relative), saved_hash)
    plotter_sha256, repair_sources = accepted_plotter_source(repo, plan)
    manifest_path = figures / 'figure_manifest.json'
    manifest, figure_qa = read_json(manifest_path), read_json(figure_qa_path)
    require(manifest.get('run_sha256') == run_sha and manifest.get('figure_count') == 8 and
            manifest.get('gates_passed_before_accuracy_reads') is True, 'Original figure run/gate/count differs.')
    require(manifest.get('script_sha256') == plotter_sha256, 'Original figure source differs from accepted source.')
    require(manifest.get('evaluation_complete', {}).get('sha256') == sha256(complete_path) and
            manifest.get('independent_qa', {}).get('sha256') == sha256(qa_path), 'Original figure receipt binding differs.')
    require(tuple(f['id'] for f in manifest.get('figures', [])) == ORIGINAL_IDS, 'Original eight-artifact order differs.')
    require(figure_qa.get('status') == 'PASS' and figure_qa.get('run_sha256') == run_sha and
            figure_qa.get('figure_manifest_sha256') == sha256(manifest_path), 'Original figure visual QA absent, failed or unbound.')
    require(isinstance(figure_qa.get('scope'), str) and figure_qa['scope'].strip(), 'Original figure QA scope absent.')
    outputs = manifest.get('output_files')
    require(isinstance(outputs, dict) and outputs, 'Original figure output hash map absent.')
    for relative, record in outputs.items():
        path = safe_child(figures, relative)
        check_hash(path, record.get('sha256'))
        require(record.get('bytes') == path.stat().st_size, f'Original output size differs: {relative}')
    check_hash(figures / 'figure_plan.json', manifest.get('figure_plan_sha256'))
    sources = [f"source_data/{p['source']}" for p in plan['plates']]
    sources += ['source_data/per_client_ALL.csv', 'source_data/paired_block6_sensitivity_ALL.csv',
                'source_data/paired_block24_sensitivity_ALL.csv', 'source_data/paired_per_seed_summary_ALL.csv',
                'source_data/05_training_graph_profiles.csv']
    for relative in set(sources):
        require(relative in outputs, f'Required CSV absent from verified original output ledger: {relative}')
    return dict(repo=repo,run_sha256=run_sha, completion=complete, independent_qa=qa,
                synthetic_fixture=complete.get('synthetic_fixture') is True,
                original_manifest=manifest, figure_qa=figure_qa,
                gate_sources=[complete_path, qa_path, run/'run_identity.json', run/'protocol.json',
                              run/'choices_frozen.json', manifest_path, figure_qa_path, figures/'figure_plan.json'] + repair_sources,
                source_paths=[safe_child(figures, relative) for relative in dict.fromkeys(sources)])


def numeric(frame, columns, name):
    for column in columns:
        require(column in frame and np.isfinite(pd.to_numeric(frame[column], errors='coerce')).all(),
                f'Nonfinite/missing {name}.{column}')


def validate_seed_table(frame, methods, plan, name):
    require(set(frame.model) == set(methods) and len(frame) == len(methods)*5, f'{name}: unexpected methods/count')
    require(not frame.duplicated(['model','seed']).any(), f'{name}: duplicate model-seed')
    require((frame.scope == 'macro_equal_client').all(), f'{name}: pooled/macro scope mixed')
    for model in methods:
        require(set(frame.loc[frame.model == model,'seed']) == set(plan['seeds']), f'{name}: seed set differs')
    numeric(frame, ['R2','RMSE_mm'], name)
    require((frame.RMSE_mm >= 0).all(), f'{name}: negative RMSE')


def validate_pairs(frame, pairs, name):
    require(len(frame) == len(pairs)*2 and set(frame.metric) == {'macro_R2','macro_RMSE_mm'}, f'{name}: paired metric count differs')
    require((frame.block_length_months == 12).all() and not frame.duplicated(['model_a','model_b','metric']).any(),
            f'{name}: block/duplicates differ')
    require(set(zip(frame.model_a,frame.model_b)) == set(map(tuple,pairs)), f'{name}: pair set differs')
    for i,(a,b) in enumerate(pairs):
        require((frame.loc[(frame.model_a == a)&(frame.model_b == b),'comparison_order'] == i).all(),
                f'{name}: frozen protocol order differs')
    numeric(frame, ['delta_a_minus_b','CI95_low','CI95_high'], name)
    require((frame.CI95_low <= frame.CI95_high).all(), f'{name}: reversed interval')


def load_tables(info, plan):
    # First and only accuracy CSV reads occur after gates() completes in load_inputs().
    frames = {p.name: pd.read_csv(p, float_precision='round_trip') for p in info['source_paths']}
    validate_seed_table(frames['01_global_and_classical.csv'], plan['global_models'], plan, 'global')
    validate_seed_table(frames['02_personal.csv'], plan['personal_models'], plan, 'personal')
    full = frames['03_paired_family.csv']; anchor = frames['04_anchor_ablations.csv']
    validate_pairs(full, plan['pairs'], 'family')
    anchor_pairs = [('CAAA_GLOBAL_FULL_ANCHOR','CAAA_GLOBAL_FULL_'+flag) for flag in plan['flags']]
    validate_pairs(anchor, anchor_pairs, 'anchor')
    for row in anchor.itertuples():
        original = full.loc[(full.model_a == row.model_a)&(full.model_b == row.model_b)&(full.metric == row.metric)].iloc[0]
        require(np.array_equal(np.array([row.delta_a_minus_b,row.CI95_low,row.CI95_high]),
                               original[['delta_a_minus_b','CI95_low','CI95_high']].to_numpy(float)), 'Anchor intervals differ from family.')
    graphs = frames['05_training_graphs.csv']
    require(len(graphs) == 75 and set(graphs.variant) == set(plan['graph_variants']) and
            not graphs.duplicated(['variant','client','reference_client']).any(), 'Graph cohorts differ.')
    numeric(graphs, ['input_affinity','trainer_row_normalized_weight','train_targets'], 'graphs')
    require(graphs.input_affinity.between(0,1).all() and (graphs.fit_role == 'gradient_train').all() and
            (graphs.target_start == '1983-01').all() and (graphs.target_end == '2007-11').all() and
            (graphs.train_targets == 299).all(), 'Graph values/training provenance differ.')
    for variant in plan['graph_variants']:
        part = graphs.loc[graphs.variant == variant]
        matrix = part.pivot(index='client',columns='reference_client',values='input_affinity').loc[plan['clients'],plan['clients']].to_numpy(float)
        weights = part.pivot(index='client',columns='reference_client',values='trainer_row_normalized_weight').loc[plan['clients'],plan['clients']].to_numpy(float)
        require(np.allclose(matrix,matrix.T,atol=1e-14) and np.array_equal(np.diag(matrix),np.ones(5)), 'Graph symmetry/diagonal differs.')
        require(np.allclose(weights,matrix/matrix.sum(axis=1,keepdims=True),atol=1e-14), 'Graph normalization differs.')
        if variant == 'uniform': require(np.array_equal(matrix,np.ones((5,5))), 'Uniform input graph differs.')
    pr = frames['08_client_PR_seed42.csv']; per = frames['per_client_ALL.csv']
    require(set(zip(pr.model,pr.seed,pr.client)) == {(m,42,c) for m in plan['pr_models'] for c in plan['clients']}, 'PR examples differ.')
    numeric(pr, ['precision','recall','point_order'], 'PR')
    require(pr.precision.between(0,1).all() and pr.recall.between(0,1).all(), 'PR coordinates outside [0,1].')
    for (model,seed,client),part in pr.groupby(['model','seed','client'],sort=False):
        require(part.point_order.tolist() == list(range(len(part))) and
                (part.score_kind == 'predicted heavy-event probability').all(), 'PR saved point order/probability kind differs.')
        truth = per.loc[(per.model == model)&(per.seed == seed)&(per.client == client)]
        require(len(truth) == 1 and 0 <= float(truth.actual_heavy_n.iloc[0]) <= 96, 'PR event-count lookup differs.')
    info['frames'] = frames
    return info


def load_inputs(repo, run, figures, qa_path, figure_qa_path, plan):
    info = gates(repo,run,figures,qa_path,figure_qa_path,plan)
    return load_tables(info,plan)


def common_limits(frame, metric):
    part = frame.loc[frame.metric == metric]
    all_values = np.r_[0., part[['delta_a_minus_b','CI95_low','CI95_high']].to_numpy(float).ravel()]
    low, high = float(all_values.min()), float(all_values.max())
    pad = max(high-low, .01)*.08
    return [low-pad,high+pad]


def style(ax, fixed_ticks=False):
    from matplotlib.ticker import MaxNLocator
    ax.spines[['top','right']].set_visible(False)
    ax.set_axisbelow(True); ax.grid(axis='x',color='#D8DCE0',linewidth=.6)
    if not fixed_ticks:
        low,high=ax.get_xlim()
        ticks=MaxNLocator(nbins=3).tick_values(low,high)
        ax.set_xticks(ticks[(ticks>=low)&(ticks<=high)])


class Writer:
    def __init__(self,directory,info,plan):
        self.directory,self.info,self.plan = directory,info,plan
        self.figures=[]
        (directory/'captions').mkdir(); (directory/'source_data').mkdir()
        for path in info['source_paths']: shutil.copyfile(path,directory/'source_data'/path.name)
        pd.DataFrame(plan['model_ledger']).to_csv(directory/'model_abbreviation_ledger.csv',index=False,lineterminator='\n')
        (directory/'layout_plan.json').write_text(json.dumps(plan,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

    def save(self,fig,plate,caption,tables,limits=None):
        import matplotlib.pyplot as plt
        from matplotlib.text import Text
        if self.info.get('synthetic_fixture'):
            fig.text(.5,.012,'SYNTHETIC layout fixture — no study scores',ha='center',fontsize=10,color='#5C6670')
            caption='SYNTHETIC LAYOUT FIXTURE. No research results were read. '+caption
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer(); canvas=fig.bbox
        texts=[a for a in fig.findobj(Text) if a.get_visible() and a.get_text().strip()]
        require(all(a.get_fontsize() >= 10 for a in texts), f"Font below 10pt: {plate['id']}")
        for text in texts:
            box=text.get_window_extent(renderer)
            require(box.x0 >= canvas.x0-.5 and box.y0 >= canvas.y0-.5 and box.x1 <= canvas.x1+.5 and box.y1 <= canvas.y1+.5,
                    f"Text leaves fixed physical canvas: {plate['id']}: {text.get_text()}")
        boxes=[a.get_window_extent(renderer) for a in texts]
        for i,first in enumerate(boxes):
            for j,second in enumerate(boxes[:i]):
                intersection_width=min(first.x1,second.x1)-max(first.x0,second.x0)
                intersection_height=min(first.y1,second.y1)-max(first.y0,second.y0)
                require(intersection_width<=1 or intersection_height<=1,
                        f"Text overlap on native canvas: {plate['id']}: {texts[i].get_text()} / {texts[j].get_text()}")
        require(np.allclose(fig.get_size_inches(),plate['size_inches']), 'Canvas dimensions differ.')
        files=[]
        for extension in ('png','pdf','svg'):
            relative=plate['id']+'.'+extension
            metadata={'Creator':'caadapfl Matplotlib publication companion'}
            if extension == 'pdf': metadata.update(CreationDate=None,ModDate=None)
            if extension == 'svg': metadata.update(Date=None)
            fig.savefig(self.directory/relative,dpi=300,bbox_inches=None,facecolor='white',metadata=metadata)
            files.append(relative)
        plt.close(fig)
        cap='captions/'+plate['id']+'.txt'
        common=('Revised-archive retrospective January2017–December2024 evaluation, 96 months and five fixed overlapping clients. '
                'All five paired seeds are42,123,2024,7,999. Raw rainfall forecasts include negatives. '
                'Conditional marginal intervals do not support simultaneous31-family significance claims. '
                'The model abbreviation ledger identifies every identity and scope. Original06 validation-grid and07 heavy-event figures remain repository/supplement artifacts. ')
        (self.directory/cap).write_text(caption+'\n\n'+common+'\nSource CSVs: '+', '.join(tables)+
             '\nRun SHA256: '+self.info['run_sha256']+'\n',encoding='utf-8')
        self.figures.append(dict(id=plate['id'],files=files,caption=cap,csv_lookups=['source_data/'+t for t in tables],
                                 size_inches=plate['size_inches'],minimum_font_pt=10,axis_limits=limits,
                                 text_canvas_check='PASS',text_overlap_check='PASS'))


def forest(writer,plate,pairs,labels,source,limits):
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=plate['size_inches'])
    bottom,height=(.28,.47) if plate['id']=='P07_anchor' else (.19,.64)
    axes=[fig.add_axes([.405,bottom,.235,height]),fig.add_axes([.715,bottom,.235,height])]
    for ax,metric in zip(axes,writer.plan['forest_metrics']):
        for y,(a,b) in enumerate(pairs):
            row=source.loc[(source.model_a == a)&(source.model_b == b)&(source.metric == metric)].iloc[0]
            ax.plot([row.CI95_low,row.CI95_high],[y,y],color=COLORS[0],linewidth=1.4)
            ax.vlines([row.CI95_low,row.CI95_high],y-.07,y+.07,color=COLORS[0],linewidth=1)
            ax.plot(row.delta_a_minus_b,y,'o',color='#26333E',markersize=4.5)
        ax.set_yticks(range(len(pairs)),labels if ax is axes[0] else ['']*len(pairs))
        ax.set_ylim(len(pairs)-.5,-.5); ax.set_xlim(limits[metric]); ax.axvline(0,color='#5C6670',linestyle='--',linewidth=.9)
        style(ax); ax.tick_params(axis='y',length=0,pad=6)
        ax.set_title('Δ macro R²' if metric == 'macro_R2' else 'Δ macro RMSE (mm)',fontsize=10,pad=10)
        ax.set_xlabel('Positive favors A' if metric == 'macro_R2' else 'Negative favors A',fontsize=10)
    title='Fixed anchor minus removed component' if plate['id']=='P07_anchor' else f"All 31 contrasts: {plate['rows'][0]+1}–{plate['rows'][1]}"
    fig.text(.05,.945,title,fontsize=11,weight='bold')
    fig.text(.05,.895,'A − B; marginal 95% paired block 12 intervals',fontsize=10)
    fig.text(.05,.065,'Full aliases and block 6/24 sensitivities accompany these plates.',fontsize=10)
    caption=('All contrast identities and their order were frozen before evaluation. Direct lines show percentile95% intervals, including when the point lies outside its interval. '
             'Both metrics share their all31-derived limits across the four family plates; every point, interval endpoint and zero is included without clipping. '
             'Intervals use2000 paired circular-block12 replicates and joint date blocks for all clients, with paired seed resampling. '
             'Positive A-minus-B R² favors A; negative A-minus-B RMSE favors A. No stars or rank ordering are used.')
    if plate['id']=='P07_anchor':
        caption=('Fixed(.3,.3,.4) anchor minus each removed component in climate-weight, performance-weight, proximal-loss, auxiliary-loss and embedding order. '
                 'These are component estimands at the nonzero anchor and differ from comparisons with the selected primary coefficient vector. '+caption)
    writer.save(fig,plate,caption,[plate['source'],'paired_block6_sensitivity_ALL.csv','paired_block24_sensitivity_ALL.csv','paired_per_seed_summary_ALL.csv'],limits)


def seeds(writer,plate,methods):
    import matplotlib.pyplot as plt
    source=writer.info['frames'][plate['source']]
    aliases={r['model']:r['alias'] for r in writer.plan['model_ledger']}
    fig=plt.figure(figsize=plate['size_inches'])
    axes=[fig.add_axes([.24,.21,.295,.61]),fig.add_axes([.655,.21,.295,.61])]
    for ax,metric in zip(axes,('R2','RMSE_mm')):
        extent=[]
        for i,model in enumerate(methods):
            values=source.loc[source.model == model].set_index('seed').loc[writer.plan['seeds'],metric].to_numpy(float)
            color,marker=COLORS[i%5],MARKERS[i%7]
            ax.scatter(values,i+np.linspace(-.15,.15,5),s=19,edgecolors=color,facecolors='none',marker=marker,linewidths=.85)
            mean,sd=values.mean(),values.std(ddof=1)
            ax.errorbar(mean,i,xerr=sd,fmt=marker,color=color,markersize=5,capsize=3,linewidth=1.1)
            extent.extend([*values,mean-sd,mean+sd])
        lo,hi=min(extent),max(extent)
        if metric=='R2': lo=min(lo,0); ax.axvline(0,color='#8B939B',linestyle=':',linewidth=.8)
        pad=max(hi-lo,.01)*.08
        # Even an SD whisker below zero remains visible; it is not a rainfall forecast.
        ax.set_xlim(lo-pad,hi+pad); ax.set_ylim(len(methods)-.5,-.5)
        ax.set_yticks(range(len(methods)),[aliases[m] for m in methods] if ax is axes[0] else ['']*len(methods))
        ax.tick_params(axis='y',length=0,pad=6); style(ax)
        ax.set_title('Macro R²' if metric=='R2' else 'Macro RMSE (mm)',fontsize=10,pad=10)
        ax.set_xlabel('Higher is better' if metric=='R2' else 'Lower is better',fontsize=10)
    fig.text(.05,.945,'FULL global and named classical controls' if plate['id']=='P05_global' else 'FULL personal neural and graph identities',fontsize=11,weight='bold')
    fig.text(.05,.895,'Fixed method order; all five paired seeds',fontsize=10)
    fig.text(.05,.065,'Open: individual seeds. Filled and whiskers: mean ± sample SD.',fontsize=10)
    caption=('FULL43 equal-client macro metrics, one value per seed; distinct from pooled metrics. Fixed identities/order preserve global versus personal scope. '
             'All five open seed markers and filled means±sampleSD are shown. SD is not a confidence interval. '
             'Named raw pooled, client-normalized pooled and local ExtraTrees controls are separate loss/data-access identities. No comparison across the separate global/personal plates is silently implied.')
    writer.save(fig,plate,caption,[plate['source']])


def graph(writer,plate):
    import matplotlib.pyplot as plt
    frame=writer.info['frames'][plate['source']]
    fig=plt.figure(figsize=plate['size_inches']); axes=[fig.add_axes([left,.27,.225,.53]) for left in (.07,.355,.64)]
    for ax,variant in zip(axes,writer.plan['graph_variants']):
        matrix=frame.loc[frame.variant == variant].pivot(index='client',columns='reference_client',values='input_affinity').loc[writer.plan['clients'],writer.plan['clients']].to_numpy(float)
        image=ax.imshow(matrix,cmap='cividis',vmin=0,vmax=1,interpolation='nearest')
        ax.set_xticks(range(5),[f'C{i+1}' for i in range(5)]); ax.set_yticks(range(5),[f'C{i+1}' for i in range(5)])
        ax.tick_params(length=0,pad=2); ax.set_title(variant.title(),fontsize=10,pad=8)
        for i in range(5):
            for j in range(5):
                ax.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',fontsize=10,color='white' if matrix[i,j]<.48 else '#20242A')
    colorbar=fig.colorbar(image,cax=fig.add_axes([.91,.29,.018,.48]),ticks=[0,.5,1])
    colorbar.ax.tick_params(labelsize=10,pad=2)
    fig.text(.05,.94,'Gradient-training graph input affinities',fontsize=11,weight='bold')
    fig.text(.05,.16,'Rows: client. Columns: reference client. Shared scale 0–1.',fontsize=10)
    fig.text(.05,.075,'1983-01–2007-11; 299 targets/client; privileged offline setup.',fontsize=10)
    caption=('Raw original, enriched and uniform input affinities in identical frozen client order and common 0–1 scale. Cell labels round to one decimal for native-width legibility; color and exported CSV values retain full precision. '
             'Training reference weights normalize each row and are exported separately. Original is shifted cosine of ENSO/IOD sensitivities; enriched equally weights four training-only components; uniform inputs are all one. '
             'Saved profiles establish gradient_train1983-01–2007-11,299targets/client. Enriched setup includes privileged cleartext aligned rainfall/profile access; no secure aggregation/privacy guarantee is claimed. C1/C5 overlap persists.')
    writer.save(fig,plate,caption,[plate['source'],'05_training_graph_profiles.csv'],{'affinity':[0,1]})


def pr(writer,plate):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    source=writer.info['frames'][plate['source']]; per=writer.info['frames']['per_client_ALL.csv']
    fig=plt.figure(figsize=plate['size_inches'])
    axes=[fig.add_axes([.12+column*.305,.565-row*.405,.23,.265]) for row in range(2) for column in range(3)]
    for ci,(client,ax) in enumerate(zip(writer.plan['clients'],axes)):
        for mi,model in enumerate(writer.plan['pr_models']):
            part=source.loc[(source.model == model)&(source.client == client)]
            ax.plot(part.recall,part.precision,color=COLORS[mi],linestyle=('-','--',':')[mi],marker=MARKERS[mi],
                    markevery=max(1,len(part)//7),markersize=3,linewidth=1.2,label=('CA-G/F','FA-G/F','FP-G/F')[mi])
        count=int(per.loc[(per.model == writer.plan['pr_models'][0])&(per.seed == 42)&(per.client == client),'actual_heavy_n'].iloc[0])
        ax.axhline(count/96,color='#8B939B',linestyle='-.',linewidth=.8,label='Observed prevalence')
        ax.set(xlim=(0,1),ylim=(0,1)); ax.set_xticks([0,.5,1]); ax.set_yticks([0,.5,1])
        ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0)); ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        style(ax,fixed_ticks=True); ax.set_title(f'C{ci+1}: {count}/96 heavy',fontsize=10,pad=7)
        ax.set_xlabel('Recall',fontsize=10,labelpad=2)
        if ci in (0,3): ax.set_ylabel('Precision',fontsize=10,labelpad=2)
    axes[-1].axis('off'); handles,labels=axes[0].get_legend_handles_labels()
    axes[-1].legend(handles,labels,loc='center',frameon=False,fontsize=10,handlelength=2)
    fig.text(.05,.95,'Per-client precision–recall: fixed illustration seed 42',fontsize=11,weight='bold')
    fig.text(.05,.895,'FULL global identities; no pooled-client curve',fontsize=10)
    fig.text(.05,.045,'Training-only q80 truth; all-seed metrics and PR coordinates exported.',fontsize=10)
    caption=('Five separate fixed-client PR examples for predeclared seed42 and three FULL global neural identities. '
             'Client events are observed rain strictly above that client’s gradient-training q80. Saved auxiliary probabilities define curves; observed prevalence is a per-client reference. '
             'Seed42 was fixed before scores and is not selected as best. All-seed metrics/PR coordinates remain in original full source exports. No pooled-client PR or spatial-population curve is constructed.')
    writer.save(fig,plate,caption,[plate['source'],'per_client_ALL.csv'],{'precision':[0,1],'recall':[0,1]})


def render(info,plan,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,'axes.titlesize':10,
        'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10,'svg.fonttype':'none','pdf.fonttype':42,
        'ps.fonttype':42,'svg.hashsalt':'caadapfl_native_companions_v1','text.color':'#26333E'})
    require(not output.exists(), f'Output exists: {output}. Choose a fresh output directory.')
    output.parent.mkdir(parents=True,exist_ok=True)
    partial=output.parent/(output.name+'.partial_'+uuid.uuid4().hex[:8]); partial.mkdir()
    writer=Writer(partial,info,plan); full=info['frames']['03_paired_family.csv']
    limits={metric:common_limits(full,metric) for metric in plan['forest_metrics']}
    for plate in plan['plates'][:4]:
        lo,hi=plate['rows']; forest(writer,plate,plan['pairs'][lo:hi],plan['pair_labels'][lo:hi],full,limits)
    seeds(writer,plan['plates'][4],plan['global_models']); seeds(writer,plan['plates'][5],plan['personal_models'])
    anchor_pairs=[('CAAA_GLOBAL_FULL_ANCHOR','CAAA_GLOBAL_FULL_'+f) for f in plan['flags']]
    ledger={r['model']:r['alias'] for r in plan['model_ledger']}
    labels=[f'{ledger[a]} − {ledger[b]}' for a,b in anchor_pairs]
    forest(writer,plan['plates'][6],anchor_pairs,labels,info['frames']['04_anchor_ablations.csv'],limits)
    graph(writer,plan['plates'][7]); pr(writer,plan['plates'][8])
    require(len(writer.figures)==9,'Companion figure count differs.')
    files={p.relative_to(partial).as_posix():{'sha256':sha256(p),'bytes':p.stat().st_size}
           for p in sorted(partial.rglob('*')) if p.is_file()}
    sources=[{'path':p.relative_to(info['repo']).as_posix() if p.is_relative_to(info['repo']) else str(p),
              'sha256':sha256(p),'bytes':p.stat().st_size} for p in info['gate_sources']+info['source_paths']]
    receipt=dict(status='RENDERED_AWAITING_VISUAL_QA',run_sha256=info['run_sha256'],script_sha256=sha256(__file__),
        frozen_plan_sha256=PLAN_SHA256,figure_count=9,figures=writer.figures,output_files=files,input_sources=sources,
        gates_passed_before_accuracy_reads=True,all31_forest_limits=limits,
        original_eight_artifacts_changed=False,model_execution=False,source_preparation=False,
        versions={'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__},
        verification='Exact hashed accepted CSVs; frozen identities/orders; 6.7-inch canvases; visible text>=10pt and contained. Visual overlap/legibility review still required.')
    (partial/'companion_manifest.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    partial.rename(output); plt.close('all')
    return receipt


def resolve(repo,value):
    path=Path(value)
    return path.resolve() if path.is_absolute() else (repo/path).resolve()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--run',default='runs/revision_v2')
    parser.add_argument('--figures',default='figures/revision_v2')
    parser.add_argument('--qa-receipt',default=None)
    parser.add_argument('--figure-qa',default=None)
    parser.add_argument('--output',default='figures/revision_v2_publication')
    parser.add_argument('--plan',action='store_true',help='Print frozen plan without reading any run/results.')
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args(argv)
    try:
        plan=load_plan()
        if args.plan: print(json.dumps(plan,indent=2,ensure_ascii=True,allow_nan=False)); return 0
        repo=args.repo.resolve(); run=resolve(repo,args.run); figures=resolve(repo,args.figures)
        qa=resolve(repo,args.qa_receipt) if args.qa_receipt else run/'independent_qa'/'complete.json'
        figure_qa=resolve(repo,args.figure_qa) if args.figure_qa else figures/'visual_qa.json'
        info=load_inputs(repo,run,figures,qa,figure_qa,plan)
        if args.verify_only:
            print(json.dumps(dict(status='PASS',run_sha256=info['run_sha256'],figure_count=9,model_execution=False))); return 0
        receipt=render(info,plan,resolve(repo,args.output))
        print(json.dumps({'status':receipt['status'],'figure_count':9,'run_sha256':info['run_sha256']})); return 0
    except IntegrityError as error:
        print(f'Refusing publication companion: {error}',file=sys.stderr); return 2


if __name__=='__main__':
    raise SystemExit(main())
