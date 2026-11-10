#!/usr/bin/env python
"""Render eight publication figures from a completed, independently accepted run.

Portable CLI (from a packaged repository):
    python tools/make_figures.py --repo . --run runs/revision_v2

No training, feature preparation, model loading, pickle, network or subprocess
is used. Accuracy CSVs are read only after the completion and independent-QA
receipts have matching run and evaluation hashes. SHA256 is a consistency
check, not a signature. Matplotlib is the sole plotting library.
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

SEEDS = (42, 123, 2024, 7, 999)
CLIENTS = ('Client1_NW_Arid', 'Client2_Indo_Gangetic', 'Client3_NE_Himalaya',
           'Client4_South_Peninsular', 'Client5_Central_India')
MONTHS = tuple(str(m) for m in pd.period_range('2017-01', '2024-12', freq='M'))
ROLES = {'train': ['1983-01', '2007-11'], 'monitor': ['2008-01', '2010-11'],
         'validation': ['2011-01', '2016-11'], 'evaluation': ['2017-01', '2024-12']}
GLOBAL = ('CAAA_GLOBAL_FULL', 'FEDAVG_GLOBAL_FULL', 'FEDPROX_GLOBAL_FULL',
          'CENTRAL_GLOBAL_FULL', 'raw_pooled_ET_FULL_latest',
          'local_ET_FULL_latest', 'client_normalized_pooled_ET_FULL_latest')
PERSONAL = ('CAAA_PERSONAL_FULL', 'DITTO_PERSONAL_FULL', 'LOCAL_PERSONAL_FULL',
            'GRAPH_ORIGINAL_PERSONAL_FULL', 'GRAPH_ENRICHED_PERSONAL_FULL',
            'GRAPH_UNIFORM_PERSONAL_FULL')
FLAGS = ('NO_CLIMATE_WEIGHT', 'NO_PERFORMANCE_WEIGHT', 'NO_PROX', 'NO_AUX', 'NO_EMBEDDING')
FLAG_NAMES = dict(zip(FLAGS, ('Remove climate weighting', 'Remove performance weighting',
                             'Remove proximal loss', 'Remove auxiliary loss', 'Remove climate embedding')))
PRIMARY_GLOBAL = GLOBAL[:3]
NAMES = {
    'CAAA_GLOBAL_FULL': 'CA-ALCE global', 'FEDAVG_GLOBAL_FULL': 'FedAvg global',
    'FEDPROX_GLOBAL_FULL': 'FedProx global', 'CENTRAL_GLOBAL_FULL': 'Central neural global',
    'raw_pooled_ET_FULL_latest': 'ExtraTrees raw pooled',
    'local_ET_FULL_latest': 'ExtraTrees local',
    'client_normalized_pooled_ET_FULL_latest': 'ExtraTrees client-normalized pooled',
    'CAAA_PERSONAL_FULL': 'CA-ALCE personal', 'DITTO_PERSONAL_FULL': 'Ditto personal',
    'LOCAL_PERSONAL_FULL': 'Local neural personal',
    'GRAPH_ORIGINAL_PERSONAL_FULL': 'ClimateGraph original',
    'GRAPH_ENRICHED_PERSONAL_FULL': 'ClimateGraph enriched',
    'GRAPH_UNIFORM_PERSONAL_FULL': 'ClimateGraph uniform',
}
COLORS = ('#315B7E', '#B28735', '#71844A', '#5C6670', '#A6694A')
MARKERS = ('o', 's', '^', 'D', 'v', 'P', 'X')
METHOD_STYLE = {
    m: (COLORS[i % len(COLORS)], MARKERS[i % len(MARKERS)])
    for methods in (GLOBAL, PERSONAL) for i, m in enumerate(methods)
}
CSV_NAMES = ('raw_predictions', 'per_client', 'aggregate_per_seed', 'seed_summary',
             'paired_block_bootstrap_summary', 'paired_per_seed_summary',
             'negative_forecast_diagnostics', 'fit_timing', 'fit_cost_ledger')
JSON_NAMES = ('model_identities', 'hardware', 'metrics_receipt', 'fit_cost_receipt',
              'shared_graph_setup_ledger', 'PR_curves')
FIGURE_PLAN = {
    'version': 1, 'default_run': 'runs/revision_v2', 'figure_count': 8,
    'gate': {'completion': 'evaluation/complete.json',
             'independent_qa': 'independent_qa/complete.json',
             'required_qa_fields': ['status=PASS', 'run_sha256', 'evaluation_complete_sha256', 'scope'],
             'accuracy_csv_reads_before_gate': False},
    'formats': ['PNG 300 dpi', 'PDF', 'SVG'], 'minimum_font_pt': 10,
    'selection': 'Fixed identities and protocol comparison order; no ranking by observed scores.',
    'figures': [
        {'id': '01_global_and_classical', 'question': 'FULL global neural and named classical controls',
         'models': list(GLOBAL), 'metrics': ['macro R2', 'macro RMSE mm'],
         'uncertainty': 'All five seed points and mean +/- sample SD; SD is not a confidence interval.'},
        {'id': '02_personal', 'question': 'FULL personal neural and graph identities',
         'models': list(PERSONAL), 'metrics': ['macro R2', 'macro RMSE mm'],
         'uncertainty': 'All five seed points and mean +/- sample SD; separate from global identities.'},
        {'id': '03_paired_family', 'question': 'All 31 predeclared paired contrasts in protocol order',
         'metrics': ['macro R2 A-minus-B', 'macro RMSE mm A-minus-B'],
         'uncertainty': 'Marginal 95% paired circular block12 intervals; block6/24 exported as sensitivity tables.'},
        {'id': '04_anchor_ablations', 'question': 'Fixed (.3,.3,.4) anchor minus each of five removed components',
         'flags': list(FLAGS), 'uncertainty': 'Same block12 marginal intervals as the full family; positive R2 favors anchor.'},
        {'id': '05_training_graphs', 'question': 'Original, enriched and uniform train-only input affinities',
         'scale': [0, 1], 'provenance': 'gradient_train; 1983-01 through 2007-11, 299 targets per client',
         'note': 'Raw input affinities shown; trainer row-normalizes. Normalized reference weights exported too.'},
        {'id': '06_validation_grid', 'question': 'Actual LR, method-specific mu, coefficient grid and selected round curves',
         'source': 'Hash-bound completed development candidates, seed42 outer validation only; every grid point exported.'},
        {'id': '07_client_heavy', 'question': 'Per-client heavy recall/AP/tail RMSE, global and personal panels',
         'models': {'global': list(PRIMARY_GLOBAL), 'personal': list(PERSONAL)},
         'uncertainty': 'All five seed points and mean +/- sample SD; no subgroup confidence intervals invented.'},
        {'id': '08_client_PR_seed42', 'question': 'Five separate per-client seed42 PR examples',
         'models': list(PRIMARY_GLOBAL), 'seed': 42, 'pooled_PR_curve': False,
         'note': 'Predeclared illustration seed, not best seed; all seeds, models and client metrics/curves exported.'},
    ],
    'all_source_data': 'All 96 evaluation months, every saved model/seed/client, full metric CSVs and PR coordinates retained.',
    'omitted_optional_figure': 'Geographic map omitted to keep scope at eight figures; no invented maps or 2025 forecasts.',
    'interpretation': 'Revised-archive retrospective 2017-2024 hindcast, five fixed overlapping regions; no prospective or spatial-generalization claim.',
}
FIGURE_TITLES = {
    '01_global_and_classical': 'FULL global neural models and named classical controls',
    '02_personal': 'FULL personal neural and graph models',
    '03_paired_family': 'All 31 predeclared paired contrasts',
    '04_anchor_ablations': 'Five component removals from the fixed coefficient anchor',
    '05_training_graphs': 'Three gradient-training graph inputs',
    '06_validation_grid': 'Saved development grids and validation-round curves',
    '07_client_heavy': 'Per-client heavy-event and rainfall-tail metrics',
    '08_client_PR_seed42': 'Per-client precision–recall examples for the fixed seed42',
}


class FigureIntegrityError(ValueError):
    """Incomplete or inconsistent results; do not publish plots."""


def require(condition, message):
    if not condition:
        raise FigureIntegrityError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise FigureIntegrityError(f'Cannot read required receipt {path}: {error}') from error


def safe_child(root, relative):
    require(isinstance(relative, str) and bool(relative), 'Invalid receipt path')
    normalized = relative.replace('\\', '/')
    part = PurePosixPath(normalized)
    require(not part.is_absolute() and not PureWindowsPath(relative).drive and '..' not in part.parts,
            f'Nonportable/escaping receipt path: {relative}')
    child = (Path(root) / Path(*part.parts)).resolve()
    require(child.is_relative_to(Path(root).resolve()), f'Symlink escapes receipt directory: {relative}')
    return child


def check_hash(path, expected):
    require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected) is not None,
            f'Invalid saved hash for {path}')
    require(Path(path).is_file() and sha256(path) == expected, f'Missing or changed saved artifact: {path}')


def close(actual, expected, label, tolerance=1e-8):
    require(np.allclose(actual, expected, rtol=tolerance, atol=tolerance, equal_nan=False),
            f'Numerical reconciliation failed: {label}')


def without_root(value):
    if isinstance(value, dict):
        return {k: without_root(v) for k, v in value.items() if k != 'root'}
    if isinstance(value, (tuple, list)):
        return [without_root(v) for v in value]
    return value


def gate_completed_run(run, qa_path):
    """Read only JSON gate receipts. Do not open any accuracy CSV here."""
    evaluation = run / 'evaluation'
    completion_path = evaluation / 'complete.json'
    require(completion_path.is_file(),
            f'Completed evaluation is absent at {completion_path}. This script never starts training or evaluation.')
    require(qa_path.is_file(), f'Independent QA receipt is absent at {qa_path}; no accuracy data were read.')
    completion = read_json(completion_path)
    qa = read_json(qa_path)
    require(qa.get('status') == 'PASS', 'Independent QA has not passed; no accuracy data were read.')
    run_sha = completion.get('run_sha256')
    require(isinstance(run_sha, str) and re.fullmatch('[0-9a-f]{64}', run_sha) is not None, 'Invalid completed-run identity')
    require(qa.get('run_sha256') == run_sha, 'Independent QA belongs to a different run')
    require(qa.get('evaluation_complete_sha256') == sha256(completion_path),
            'Independent QA is not bound to the exact completed evaluation receipt')
    require(isinstance(qa.get('scope'), str) and bool(qa['scope'].strip()), 'Independent QA scope is missing')
    return completion, qa


def load_accepted_run(repo, run, data, qa_path):
    completion, qa = gate_completed_run(run, qa_path)
    identity = read_json(run / 'run_identity.json')
    choices = read_json(run / 'choices_frozen.json')
    prepared = read_json(run / 'prepared.json')
    protocol = read_json(run / 'protocol.json')
    schedule = read_json(run / 'schedule.json')
    run_sha = completion['run_sha256']
    require(identity.get('run_sha256') == canonical_digest({k: v for k, v in identity.items() if k != 'run_sha256'}),
            'Run identity digest differs')
    require(all(r.get('run_sha256') == run_sha for r in (identity, choices, prepared)), 'Mixed run identities')
    require(identity.get('protocol_hash') == canonical_digest(protocol) and
            identity.get('schedule_hash') == canonical_digest(schedule), 'Protocol/schedule digest differs')
    require(protocol.get('role_bounds') == ROLES, 'Role calendar differs from this publication plan')
    require(tuple(protocol.get('seeds', [])) == SEEDS and tuple(completion.get('seeds', [])) == SEEDS,
            'Paired seed order differs')
    require(tuple(completion.get('months', [])) == MONTHS, 'Evaluation is not the complete 96-month calendar')
    check_hash(run / 'choices_frozen.json', completion.get('choices_frozen_sha256'))
    require(choices.get('choices_including_checkpoints_and_router_weights_frozen') is True and
            choices.get('evaluation_labels_accessed') is False and
            completion.get('evaluation_truth_opened_only_after_all_choices') is True, 'Evaluation/selection gate differs')
    for name, expected in identity.get('code', {}).items():
        require(re.fullmatch('[A-Za-z_][A-Za-z_0-9]*', name) is not None, 'Invalid source module key')
        check_hash(repo / 'caadapfl' / (name + '.py'), expected)
    for name, key in (('monthly.csv', 'monthly_sha256'), ('indices.csv', 'indices_sha256'), ('manifest.json', 'manifest_sha256')):
        check_hash(data / name, identity.get('source', {}).get(key))
    files = completion.get('files')
    require(isinstance(files, dict) and bool(files), 'Completion lacks its evaluation hash map')
    # Evaluation hashes are checked without interpreting CSV values; no tensor-state reads.
    for relative, expected in files.items():
        check_hash(safe_child(run / 'evaluation', relative), expected)
    for name in [*(s + '.csv' for s in CSV_NAMES), *(s + '.json' for s in JSON_NAMES)]:
        require(name in files, f'Figure input is not hash-bound in completion: {name}')
    graph_relative = 'datasets/MAIN_graphs.json'
    prepared_files = {k.replace('\\', '/'): v for k, v in prepared.get('files', {}).items()}
    check_hash(run / graph_relative, prepared_files.get(graph_relative))
    graphs = read_json(run / graph_relative)
    validate_graphs(graphs)
    frames = {name: pd.read_csv(run / 'evaluation' / (name + '.csv'),
                               dtype={'target_month': str, 'client': str, 'model': str}) for name in CSV_NAMES}
    metadata = {name: read_json(run / 'evaluation' / (name + '.json')) for name in JSON_NAMES}
    require(metadata['model_identities'] == choices['outputs'], 'Model export identities differ from frozen choices')
    run_info = dict(repo=repo, run=run, data=data, qa_path=qa_path, completion=completion, qa=qa,
                    identity=identity, choices=choices, prepared=prepared, protocol=protocol,
                    schedule=schedule, frames=frames, metadata=metadata, graphs=graphs)
    validate_metric_tables(run_info)
    grid, rounds, stage_sources = load_development(run_info)
    run_info.update(grid=grid, rounds=rounds, stage_sources=stage_sources)
    return run_info


def validate_graphs(graphs):
    require(tuple(graphs.get('client_ids', [])) == CLIENTS, 'Graph client order differs')
    meta = graphs.get('metadata', {})
    require(meta.get('fit_role') == 'gradient_train' and meta.get('fit_cutoff') == '2007-11',
            'Graph was not built only from gradient-training targets')
    for client in CLIENTS:
        profile = graphs['profiles'][client]
        require((profile.get('target_start'), profile.get('target_end'), profile.get('n')) == ('1983-01', '2007-11', 299),
                f'Graph training provenance differs for {client}')
        require(profile.get('index_support_month_supplied') is True, 'Graph lacks lag-eligible index support declarations')
    require(set(graphs['graphs']) == {'original', 'enriched', 'uniform'}, 'Expected three named graph variants')
    for variant, values in graphs['graphs'].items():
        matrix = np.asarray(values, float)
        require(matrix.shape == (5, 5) and np.isfinite(matrix).all() and (matrix >= 0).all() and (matrix <= 1).all(),
                f'Graph input outside declared [0,1] range: {variant}')
        close(matrix, matrix.T, variant + ' input symmetry')
        close(np.diag(matrix), np.ones(5), variant + ' self-affinity')
        require((matrix.sum(axis=1) > 0).all(), 'Invalid graph row normalization')
    close(graphs['graphs']['uniform'], np.ones((5, 5)), 'uniform graph')
    components = list(graphs['components'].values())
    require(len(components) == 4, 'Enriched graph must have four equal components')
    for component in components:
        matrix = np.asarray(component, float)
        require(matrix.shape == (5, 5) and np.isfinite(matrix).all() and (matrix >= 0).all() and (matrix <= 1).all(),
                'A graph component is outside the declared finite [0,1] scale')
    close(graphs['graphs']['enriched'], np.mean(np.asarray(components, float), axis=0), 'enriched equal-component mean')


def validate_metric_tables(info):
    frames, protocol = info['frames'], info['protocol']
    raw, client = frames['raw_predictions'], frames['per_client']
    require({'model', 'seed', 'client', 'target_month', 'y', 'prediction'}.issubset(raw), 'Prediction schema differs')
    require(not raw.duplicated(['model', 'seed', 'client', 'target_month']).any(), 'Duplicate prediction keys')
    require(np.isfinite(raw[['y', 'prediction']].to_numpy(float)).all(), 'Nonfinite raw predictions')
    require((raw.y >= 0).all(), 'Negative observed rainfall')
    require(not client.duplicated(['model', 'seed', 'client']).any(), 'Duplicate per-client metric keys')
    require(set(GLOBAL + PERSONAL).issubset(set(raw.model)), 'A fixed publication method is absent')
    require(raw.model.nunique() == info['completion'].get('models'), 'Completed model count differs')
    expected_keys = {(seed, c, m) for seed in SEEDS for c in CLIENTS for m in MONTHS}
    for model, group in raw.groupby('model', sort=False):
        require(set(zip(group.seed, group.client, group.target_month)) == expected_keys,
                f'Incomplete 5-seed/5-client/96-month prediction cohort: {model}')
    require(raw.groupby(['client', 'target_month']).y.nunique().eq(1).all(), 'Truth differs across models/seeds')
    checked = []
    for (model, seed, c), group in raw.groupby(['model', 'seed', 'client'], sort=False):
        y, p = group.y.to_numpy(float), group.prediction.to_numpy(float)
        denominator = np.sum((y - y.mean()) ** 2)
        require(denominator > 0, 'Constant observations; R2 is undefined')
        checked.append(dict(model=model, seed=int(seed), client=c,
                            R2=1 - np.sum((p - y) ** 2) / denominator,
                            RMSE_mm=np.sqrt(np.mean((p - y) ** 2))))
    recomputed = pd.DataFrame(checked).set_index(['model', 'seed', 'client']).sort_index()
    saved = client.set_index(['model', 'seed', 'client']).sort_index()
    require(recomputed.index.equals(saved.index), 'Metric/prediction cohorts differ')
    close(recomputed[['R2', 'RMSE_mm']], saved[['R2', 'RMSE_mm']], 'all per-client R2/RMSE')
    macro = recomputed.groupby(['model', 'seed'])[['R2', 'RMSE_mm']].mean()
    aggregate = frames['aggregate_per_seed']
    require(set(aggregate.scope) == {'macro_equal_client', 'pooled_all_client_months'}, 'Aggregate scopes differ')
    require(not aggregate.duplicated(['model', 'seed', 'scope']).any(), 'Duplicate aggregate metrics')
    macro_saved = aggregate.loc[aggregate.scope == 'macro_equal_client'].set_index(['model', 'seed']).sort_index()
    require(macro.index.equals(macro_saved.index), 'Macro aggregate cohort differs')
    close(macro[['R2', 'RMSE_mm']], macro_saved[['R2', 'RMSE_mm']], 'all macro R2/RMSE')
    pairs = [tuple(p) for p in protocol['comparisons']]
    require(len(pairs) == 31 and len(set(pairs)) == 31, 'Predeclared comparison family is not exactly 31 distinct pairs')
    intervals = frames['paired_block_bootstrap_summary']
    expected = {(a, b, metric, block) for a, b in pairs for metric in ('macro_R2', 'macro_RMSE_mm') for block in (12, 6, 24)}
    require(len(intervals) == 186 and set(zip(intervals.model_a, intervals.model_b, intervals.metric, intervals.block_length_months)) == expected,
            'Paired interval table omits or duplicates a protocol contrast/metric/block')
    require(np.isfinite(intervals[['delta_a_minus_b', 'CI95_low', 'CI95_high']].to_numpy(float)).all(), 'Nonfinite paired intervals')
    require((intervals.CI95_low <= intervals.CI95_high).all(), 'Reversed paired bounds')
    require((intervals.n_seeds == 5).all() and (intervals.n_clients == 5).all() and (intervals.n_months == 96).all()
            and (intervals.repetitions == 2000).all() and intervals.resample_seeds.eq(True).all(), 'Paired interval cohort/method differs')
    require(intervals.loc[intervals.block_length_months == 12, 'role'].eq('primary').all(), 'Primary interval role differs')
    means = macro.groupby('model').mean()
    for row in intervals.itertuples():
        column = 'R2' if row.metric == 'macro_R2' else 'RMSE_mm'
        close(row.delta_a_minus_b, means.loc[row.model_a, column] - means.loc[row.model_b, column], 'paired point ' + row.metric)
    require(set(FLAGS) == set(protocol['ablations']), 'Anchor ablation set differs')
    close(protocol['ablation_coefficient_reference']['coefficients'], [.3, .3, .4], 'anchor coefficients')
    for flag in FLAGS:
        require(('CAAA_GLOBAL_FULL_ANCHOR', 'CAAA_GLOBAL_FULL_' + flag) in pairs, 'Ablation parent is not the fixed anchor')
    info['macro_recomputed'] = macro.reset_index()


def bound_fit_file(path, receipt, relative):
    files = {k.replace('\\', '/'): v for k, v in receipt.get('files', {}).items()}
    check_hash(path / relative, files.get(relative))
    return path / relative


def load_development(info):
    """Read only completed JSON curves and receipts; never deserialize models."""
    run, choices = info['run'], info['choices']
    stages = ('LR', 'MU_fedprox', 'MU_ditto', 'MU_graph_original', 'MU_graph_enriched', 'MU_graph_uniform', 'MU_caaa', 'COEFFICIENTS')
    expected_counts = {'LR': 2, 'COEFFICIENTS': 16}
    records, round_rows, sources = [], [], []
    for stage in stages:
        path = run / 'development' / (stage + '.json')
        obj = read_json(path)
        require(obj.get('name') == stage and obj.get('evaluation_labels_accessed') is False, 'Development isolation differs')
        candidates = obj.get('candidates', [])
        require(len(candidates) == expected_counts.get(stage, 3), 'Development grid cardinality differs: ' + stage)
        identity = obj['selection_identity']
        require(identity == ('personal' if stage in ('MU_ditto', 'MU_graph_original', 'MU_graph_enriched', 'MU_graph_uniform') else 'global'),
                'Development selection identity differs: ' + stage)
        sources.append(path)
        candidate_keys = []
        for candidate in candidates:
            job = candidate['job']; key = job['job_sha256']; config = job['config']
            require(re.fullmatch('[0-9a-f]{64}', key) is not None and key in choices['fit_complete_hashes'], 'Unfrozen development fit')
            payload = {k: v for k, v in job.items() if k not in ('root', 'job_sha256')}
            require(canonical_digest({'run_sha256': info['identity']['run_sha256'], 'payload': payload}) == key, 'Development fit identity differs')
            fit_dir = run / 'fits' / key[:16]
            receipt_path = fit_dir / 'complete.json'
            check_hash(receipt_path, choices['fit_complete_hashes'][key])
            receipt = read_json(receipt_path)
            require(without_root(receipt['job']) == without_root(job) and receipt.get('evaluation_labels_accessed') is False,
                    'Development job/configuration differs')
            close(candidate['receipt']['validation_macro_R2'][identity], receipt['validation_macro_R2'][identity], 'development score binding')
            require(config['seed'] == 42 and config['rounds'] == 60, 'Development seed/round budget differs')
            curve_path = bound_fit_file(fit_dir, receipt, 'model/curves.json')
            manifest_path = bound_fit_file(fit_dir, receipt, 'model/manifest.json')
            curves, manifest = read_json(curve_path), read_json(manifest_path)
            selected_round = manifest['selected_rounds'][identity]
            relevant = [r for r in curves if r.get('kind') == 'evaluation' and r.get('identity') == identity]
            require(len(relevant) == 60 and {r['round'] for r in relevant} == set(range(1, 61)), 'Incomplete saved validation rounds')
            score = float(receipt['validation_macro_R2'][identity])
            close(max(r['validation_macro_r2'] for r in relevant), score, 'saved validation maximum', tolerance=1e-5)
            candidate_keys.append(key)
            records.append(dict(stage=stage, method=config['method'], graph_name=config.get('graph_name'), identity=identity,
                                seed=42, lr=config['lr'], mu=config['prox_mu'], alpha=config['alpha'], beta=config['beta'], gamma=config['gamma'],
                                validation_macro_R2=score, job_sha256=key, selected=key == obj['selected_job_sha256'], selected_round=selected_round))
            for curve in relevant:
                require(np.isfinite(curve['validation_macro_r2']), 'Nonfinite validation round')
                round_rows.append(dict(stage=stage, job_sha256=key, identity=identity, round=curve['round'],
                                       validation_macro_R2=curve['validation_macro_r2'],
                                       selected_candidate=key == obj['selected_job_sha256'], selected_checkpoint=curve['round'] == selected_round))
            sources.extend([receipt_path, curve_path, manifest_path])
        require(len(set(candidate_keys)) == len(candidate_keys), 'Repeated development candidate')
        require(obj['selected_job_sha256'] in candidate_keys, 'Selected candidate is absent')
        selected_config = next(c['job']['config'] for c in candidates if c['job']['job_sha256'] == obj['selected_job_sha256'])
        require(selected_config == obj['selected_config'], 'Selected configuration differs')
        if stage == 'LR':
            close(sorted(c['job']['config']['lr'] for c in candidates), info['protocol']['neural']['shared_lr_candidates'], 'LR grid')
            close(selected_config['lr'], choices['lr'], 'frozen LR')
        elif stage.startswith('MU_'):
            close(sorted(c['job']['config']['prox_mu'] for c in candidates), info['protocol']['neural']['mu_candidates'], 'mu grid')
            close(selected_config['prox_mu'], choices['mu'][stage[3:]], 'frozen mu')
        else:
            actual = {tuple(c['job']['config'][k] for k in ('alpha', 'beta', 'gamma')) for c in candidates}
            require(actual == {tuple(v) for v in info['protocol']['neural']['coefficient_candidates']}, 'Coefficient simplex grid differs')
            close([selected_config[k] for k in ('alpha', 'beta', 'gamma')], choices['coefficients'], 'frozen coefficients')
    return pd.DataFrame(records), pd.DataFrame(round_rows), sorted(set(sources))


def contrast_label(a, b):
    if a.endswith('_ANCHOR'):
        return 'FULL anchor − ' + FLAG_NAMES[b.removeprefix('CAAA_GLOBAL_FULL_')].removeprefix('Remove ')
    if a.startswith('ROUTER_'):
        group = a.split('_')[1]
        return f'{group} router: robust − ' + b.split('_', 2)[2].replace('_', ' ')
    if '_LATENCY_' in a:
        return a.split('_')[0] + ': extra lag − matched reference'
    if a.startswith('GRAPH_'):
        return 'FULL personal graph: enriched − ' + b.split('_')[1].lower()
    if b.endswith('_CLIMATE_FREE'):
        return 'CORE global: CA-ALCE − climate-free CA-ALCE'
    group = a.rsplit('_', 1)[1]
    scope = 'personal' if '_PERSONAL_' in a else 'global'
    other = b.split('_')[0].title().replace('Fedavg', 'FedAvg').replace('Fedprox', 'FedProx')
    return f'{group} {scope}: CA-ALCE − {other}'


def write_csv(frame, path):
    frame.to_csv(path, index=False, float_format='%.17g', lineterminator='\n')


class FigureWriter:
    def __init__(self, directory, info):
        self.directory, self.info, self.figures = directory, info, []
        (directory / 'source_data').mkdir(parents=True)
        (directory / 'captions').mkdir()

    def table(self, name, frame):
        path = self.directory / 'source_data' / (name + '.csv')
        write_csv(frame, path)
        return path.relative_to(self.directory).as_posix()

    def save(self, fig, name, caption, tables, sources):
        import matplotlib.pyplot as plt
        fig.suptitle(FIGURE_TITLES[name], fontsize=13)
        files = []
        for extension in ('png', 'pdf', 'svg'):
            relative = name + '.' + extension
            metadata = {'Creator': 'caadapfl standalone Matplotlib figures'}
            if extension == 'pdf': metadata.update(CreationDate=None, ModDate=None)
            if extension == 'svg': metadata.update(Date=None)
            fig.savefig(self.directory / relative, dpi=300, bbox_inches='tight', facecolor='white', metadata=metadata)
            files.append(relative)
        plt.close(fig)
        caption_relative = 'captions/' + name + '.txt'
        common = ('Study context: revised-archive retrospective evaluation, January 2017–December 2024; 96 months, five fixed study clients '
                  'and five paired seeds (42, 123, 2024, 7, 999). Regions overlap and are not independent spatial replicates. '
                  'Training/validation windows for their figures are stated above. All evaluation rainfall forecasts retain raw inverse-transformed values, including negatives. ')
        (self.directory / caption_relative).write_text(caption + '\n\n' + common +
            '\nSource CSVs: ' + ', '.join(tables) + '\nRun SHA256: ' + self.info['identity']['run_sha256'] + '\n', encoding='utf-8')
        self.figures.append(dict(id=name, files=files, caption=caption_relative, csv_lookups=tables,
                                 source_inputs=[source_record(p, self.info) for p in sorted(set(sources))]))


def source_record(path, info):
    path = Path(path).resolve()
    repo = info['repo']
    relative = path.relative_to(repo).as_posix() if path.is_relative_to(repo) else str(path)
    return dict(path=relative, sha256=sha256(path), bytes=path.stat().st_size)


def axes_style(ax, xgrid=False):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x' if xgrid else 'y', color='#D8DCE0', linewidth=.6)
    ax.set_axisbelow(True)


def plot_seed_comparison(writer, methods, name, title):
    import matplotlib.pyplot as plt
    source = writer.info['frames']['aggregate_per_seed']
    selected = source.loc[(source.scope == 'macro_equal_client') & source.model.isin(methods)].copy()
    selected['display_name'] = selected.model.map(NAMES)
    table = writer.table(name, selected)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout='constrained', sharey=True)
    for ax, metric, unit in zip(axes, ('R2', 'RMSE_mm'), ('Macro R² (higher is better)', 'Macro RMSE (mm; lower is better)')):
        for index, model in enumerate(methods):
            group = selected.loc[selected.model == model].set_index('seed').loc[list(SEEDS)]
            values = group[metric].to_numpy(float)
            color, marker = METHOD_STYLE[model]
            offsets = np.linspace(-.16, .16, 5)
            ax.scatter(values, index + offsets, s=27, color=color, marker=marker, facecolors='none', linewidths=1)
            ax.errorbar(values.mean(), index, xerr=values.std(ddof=1), fmt=marker, color=color,
                        markersize=7, capsize=4, linewidth=1.2)
        ax.set_yticks(range(len(methods)), [NAMES[m] for m in methods])
        ax.set_ylim(len(methods)-.5, -.5); ax.set_xlabel(unit); axes_style(ax, True)
        if metric == 'R2': ax.axvline(0, color='#8B939B', linestyle=':', linewidth=.8)
        else: ax.set_xlim(left=0)
    caption = (title + '. FULL uses 43 lag-eligible predictors. Macro metrics are the unweighted mean of five per-client metrics, '
               'computed separately for each seed; they are distinct from pooled metrics. Open markers show all five paired seed results; '
               'filled markers and whiskers show the mean ± sample SD across seeds, not a confidence interval. Method order is fixed before scoring. '
               'The local ExtraTrees control is explicitly local, and raw pooled and client-normalized pooled trees have different target-loss weighting. '
               'No global-versus-personal paired superiority claim is implied by separate figures.')
    writer.save(fig, name, caption, [table, 'source_data/aggregate_per_seed_ALL.csv'],
                [writer.info['run'] / 'evaluation' / 'aggregate_per_seed.csv'])


def plot_forests(writer):
    import matplotlib.pyplot as plt
    info = writer.info; source = info['frames']['paired_block_bootstrap_summary']
    pairs = [tuple(v) for v in info['protocol']['comparisons']]
    for name, subset in (('03_paired_family', pairs), ('04_anchor_ablations',
                         [('CAAA_GLOBAL_FULL_ANCHOR', 'CAAA_GLOBAL_FULL_' + flag) for flag in FLAGS])):
        rows = source.loc[source.block_length_months == 12].copy()
        order = {pair: i for i, pair in enumerate(subset)}
        rows = rows.loc[[pair in order for pair in zip(rows.model_a, rows.model_b)]].copy()
        rows['comparison_order'] = [order[pair] for pair in zip(rows.model_a, rows.model_b)]
        rows = rows.sort_values(['comparison_order', 'metric'])
        table = writer.table(name, rows)
        fig, axes = plt.subplots(1, 2, figsize=(19, max(5.5, len(subset) * .45 + 1.5)),
                                 sharey=True, layout='constrained', gridspec_kw={'width_ratios': [1, 1]})
        for ax, metric in zip(axes, ('macro_R2', 'macro_RMSE_mm')):
            frame = rows.loc[rows.metric == metric].sort_values('comparison_order')
            for y, row in enumerate(frame.itertuples()):
                # Direct interval line preserves percentile intervals even when the point is outside them.
                ax.plot([row.CI95_low, row.CI95_high], [y, y], color=COLORS[0], linewidth=1.5)
                ax.plot([row.CI95_low, row.CI95_low], [y - .07, y + .07], color=COLORS[0], linewidth=1)
                ax.plot([row.CI95_high, row.CI95_high], [y - .07, y + .07], color=COLORS[0], linewidth=1)
                ax.plot(row.delta_a_minus_b, y, 'o', color='#26333E', markersize=5)
            ax.axvline(0, color='#5C6670', linestyle='--', linewidth=1)
            ax.set_yticks(range(len(subset)), [contrast_label(a, b) for a, b in subset])
            ax.set_ylim(len(subset) - .5, -.5); axes_style(ax, True)
            ax.set_xlabel('Δ macro R²: A − B (positive favors A)' if metric == 'macro_R2'
                          else 'Δ macro RMSE: A − B (mm; negative favors A)')
        caption = ('Point differences and marginal 95% percentile confidence intervals for the full predeclared comparison family, '
                   'in its frozen protocol order. Each interval uses 2,000 paired circular bootstrap replicates, common 12-month date blocks '
                   'across all five clients and both models, and jointly resampled paired seeds. Intervals are conditional on these fixed overlapping '
                   'regions and are not simultaneous family-wide intervals. No star or significance declaration is inferred from marginal interval exclusion. '
                   'All block6 and block24 sensitivity intervals and all five-seed paired differences accompany the figure. '
                   'R² differences are A minus B; positive favors A. RMSE differences are A minus B; negative favors A.')
        if name == '04_anchor_ablations':
            caption = ('Five component removals compare the fixed nonzero (α,β,γ)=(.3,.3,.4) anchor parent with its matched removed-component model. '
                       'These estimate component changes at that anchor and do not compare removals with the separately selected primary coefficient vector. '
                       'Positive ΔR² favors the full anchor. ' + caption)
        writer.save(fig, name, caption, [table, 'source_data/paired_block6_sensitivity_ALL.csv',
                    'source_data/paired_block24_sensitivity_ALL.csv', 'source_data/paired_per_seed_summary_ALL.csv'],
                    [info['run'] / 'evaluation' / 'paired_block_bootstrap_summary.csv', info['run'] / 'protocol.json'])


def plot_graphs(writer):
    import matplotlib.pyplot as plt
    info = writer.info; graphs = info['graphs']; rows = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout='constrained')
    for ax, variant in zip(axes, ('original', 'enriched', 'uniform')):
        matrix = np.asarray(graphs['graphs'][variant], float)
        normalized = matrix / matrix.sum(axis=1, keepdims=True)
        image = ax.imshow(matrix, cmap='cividis', vmin=0, vmax=1, interpolation='nearest')
        ax.set_xticks(range(5), [f'C{i + 1}' for i in range(5)])
        ax.set_yticks(range(5), [f'C{i + 1}' for i in range(5)])
        ax.set_title(variant.title()); ax.set_xlabel('Reference client'); ax.set_ylabel('Client')
        for i, c in enumerate(CLIENTS):
            for j, other in enumerate(CLIENTS):
                ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center', fontsize=10,
                        color='white' if matrix[i,j] < .48 else '#20242A')
                rows.append(dict(variant=variant, client=c, reference_client=other,
                                 input_affinity=matrix[i,j], trainer_row_normalized_weight=normalized[i,j],
                                 fit_role='gradient_train', target_start='1983-01', target_end='2007-11', train_targets=299))
    fig.colorbar(image, ax=list(axes), shrink=.84, label='Input affinity (common 0–1 scale)')
    table = writer.table('05_training_graphs', pd.DataFrame(rows))
    profiles = writer.table('05_training_graph_profiles', pd.json_normalize(
        [dict(client=c, **graphs['profiles'][c]) for c in CLIENTS]))
    caption = ('Three input-affinity matrices, all displayed on the same 0–1 scale and in frozen client order. These are the saved pre-normalization '
               'inputs; the trainer divides each row by its sum to form reference weights, which are also exported. Original uses shifted cosine of '
               'ENSO/IOD sensitivities; enriched is an equal .25 mean of target-anomaly interclient correlation, standardized seasonal-climatology cosine, '
               'six-index sensitivity cosine and standardized rainfall-statistic distance; uniform has all input affinities one. '
               'The saved profile declarations and prepared artifact hash require only gradient-training targets, January 1983–November 2007, 299 per client, '
               'with lag-eligible climate support. These provenance checks supplement the independent run QA; they do not reconstruct a historical release vintage. '
               'Enriched graph setup is privileged offline cleartext access to aligned training rainfall and profiles; it is not secure aggregation or a privacy guarantee. '
               'C1–C5 refer to the full client identifiers in the source CSV, and C1/C5 spatial overlap remains part of the study.')
    writer.save(fig, '05_training_graphs', caption, [table, profiles],
                [info['run'] / 'datasets' / 'MAIN_graphs.json', info['run'] / 'prepared.json'])


def plot_development(writer):
    import matplotlib.pyplot as plt
    info = writer.info; grid, rounds = info['grid'], info['rounds']
    table = writer.table('06_validation_candidates_ALL', grid)
    curves_table = writer.table('06_validation_rounds_ALL', rounds)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), layout='constrained')
    candidate_lower, candidate_upper = float(grid.validation_macro_R2.min()), float(grid.validation_macro_R2.max())
    padding = max(candidate_upper-candidate_lower, .01)*.06
    lr = grid.loc[grid.stage == 'LR'].sort_values('lr')
    ax = axes[0, 0]
    ax.plot(lr.lr, lr.validation_macro_R2, '-o', color=COLORS[0], linewidth=1.3)
    chosen = lr.loc[lr.selected]; ax.scatter(chosen.lr, chosen.validation_macro_R2, s=110, facecolors='none', edgecolors='#26333E', linewidths=1.4)
    ax.set_xscale('log'); ax.set_xticks(lr.lr, [f'{v:g}' for v in lr.lr])
    ax.set_xlabel('Shared learning rate'); ax.set_ylabel('Validation macro R²'); ax.set_title('LR: FULL FedAvg global, seed42'); axes_style(ax)
    ax.set_ylim(candidate_lower-padding, candidate_upper+padding)
    ax = axes[0, 1]
    mu_names = ('fedprox', 'ditto', 'graph_original', 'graph_enriched', 'graph_uniform', 'caaa')
    for i, method in enumerate(mu_names):
        subset = grid.loc[grid.stage == 'MU_' + method].sort_values('mu')
        identity = subset.identity.iloc[0]
        color = COLORS[i % len(COLORS)]; marker = MARKERS[i]
        label = method.replace('_', ' ') + ' ' + identity
        ax.plot(subset.mu, subset.validation_macro_R2, marker=marker, color=color,
                linestyle='--' if i == 5 else '-', linewidth=1.1, label=label)
        selected = subset.loc[subset.selected]
        ax.scatter(selected.mu, selected.validation_macro_R2, s=100, facecolors='none', edgecolors=color, linewidths=1.3)
    values = info['protocol']['neural']['mu_candidates']
    ax.set_xscale('log'); ax.set_xticks(values, [f'{v:g}' for v in values]); ax.set_xlabel('Proximal μ')
    ax.set_ylabel('Validation macro R²'); ax.set_title('Method-specific μ, seed42'); axes_style(ax)
    ax.set_ylim(candidate_lower-padding, candidate_upper+padding)
    ax.legend(loc='best', fontsize=10, frameon=False)
    coeff = grid.loc[grid.stage == 'COEFFICIENTS']
    ax = axes[1, 0]; x = coeff.beta + coeff.gamma / 2; y = np.sqrt(3) * coeff.gamma / 2
    im = ax.scatter(x, y, c=coeff.validation_macro_R2, cmap='cividis', vmin=candidate_lower, vmax=candidate_upper,
                    s=130, edgecolors='#26333E', linewidths=.6)
    selected = coeff.selected.to_numpy(bool)
    ax.scatter(x[selected], y[selected], s=240, facecolors='none', edgecolors='#26333E', linewidths=1.8)
    ax.plot([0, 1, .5, 0], [0, 0, np.sqrt(3)/2, 0], color='#8B939B', linewidth=.8)
    ax.text(-.02, -.055, 'α=1', ha='center'); ax.text(1.02, -.055, 'β=1', ha='center')
    ax.text(.5, np.sqrt(3)/2 + .04, 'γ=1', ha='center')
    ax.set(xlim=(-.1, 1.1), ylim=(-.11, 1.01)); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('CA-ALCE global: 15 simplex points + anchor')
    fig.colorbar(im, ax=ax, shrink=.72, label='Validation macro R²')
    ax = axes[1, 1]
    for i, stage in enumerate(('LR', 'MU_caaa', 'COEFFICIENTS')):
        subset = rounds.loc[(rounds.stage == stage) & rounds.selected_candidate].sort_values('round')
        ax.plot(subset['round'], subset.validation_macro_R2, color=COLORS[i], linestyle=('-', '--', ':')[i],
                linewidth=1.4, label=stage + ' selected candidate')
        mark = subset.loc[subset.selected_checkpoint]
        ax.scatter(mark['round'], mark.validation_macro_R2, color=COLORS[i], marker=MARKERS[i], s=35)
    ax.set_xlabel('Actual federated round'); ax.set_ylabel('Validation macro R²')
    ax.set_xlim(1, 60); ax.set_xticks([1, 15, 30, 45, 60]); ax.set_title('Selected-stage saved validation curves')
    axes_style(ax); ax.legend(loc='best', frameon=False, fontsize=10)
    caption = ('Actual sequential development grids, selected only on January 2011–November 2016 outer validation, seed42. '
               'LR selection uses FULL FedAvg global; μ uses global FedProx/CA-ALCE and personal Ditto/each graph; the coefficient stage uses CA-ALCE global. '
               'Open rings mark the saved selected candidates, and the coefficient point color is its actual validation score. '
               'Lines merely join tested LR/μ values and do not evaluate intermediate values; coefficient points include all 15 quarter-simplex candidates '
               'and the (.3,.3,.4) anchor. The last panel shows all 60 saved rounds for the selected candidate of each indicated stage; '
               'markers denote each saved checkpoint. LR and μ panels use one common candidate-score range; the coefficient color scale uses that same range. '
               'The round-trajectory panel uses its separately labeled temporal range. All 36 stage-candidate rows and their validation-round curves are exported with fit hashes. '
               'Each curve/manifest is checked against a completed fit receipt already frozen in choices_frozen.json. No evaluation scores were used for these choices; '
               'outer validation is reused for tuning and checkpoint selection, so these scores are developmental rather than independent accuracy estimates.')
    writer.save(fig, '06_validation_grid', caption, [table, curves_table], info['stage_sources'])


def plot_heavy(writer):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    info = writer.info; source = info['frames']['per_client']; groups = (PRIMARY_GLOBAL, PERSONAL)
    selected = source.loc[source.model.isin(PRIMARY_GLOBAL + PERSONAL)].copy()
    selected['identity_panel'] = np.where(selected.model.isin(PRIMARY_GLOBAL), 'global', 'personal')
    table = writer.table('07_client_heavy', selected)
    metric_names = ('heavy_recall', 'heavy_average_precision', 'tail80_RMSE_mm')
    metric_limits = {}
    for metric in metric_names:
        stats = selected.groupby(['model', 'client'])[metric].agg(['mean', 'std'])
        lower, upper = float((stats['mean']-stats['std']).min()), float((stats['mean']+stats['std']).max())
        if metric != 'tail80_RMSE_mm':
            lower, upper = min(0, lower), max(1, upper)
        else:
            lower, upper = min(0, lower), max(float(selected[metric].max()), upper)
        padding = max(upper-lower, .01)*.03
        metric_limits[metric] = (lower-padding, upper+padding)
    fig, axes = plt.subplots(3, 2, figsize=(16, 13), layout='constrained', sharey='row')
    for column, methods in enumerate(groups):
        offsets = np.linspace(-.34, .34, len(methods))
        for row, metric in enumerate(metric_names):
            ax = axes[row, column]
            for mi, model in enumerate(methods):
                color, marker = METHOD_STYLE[model]
                for ci, client in enumerate(CLIENTS):
                    values = selected.loc[(selected.model == model) & (selected.client == client)].set_index('seed').loc[list(SEEDS), metric].to_numpy(float)
                    require(np.isfinite(values).all(), f'Undefined plotted heavy metric {model}/{client}/{metric}; inspect source instead of silently dropping it')
                    x = ci + offsets[mi]
                    ax.scatter(x + np.linspace(-.024, .024, 5), values, color=color, marker=marker, s=13, alpha=.4, linewidths=.7)
                    ax.errorbar(x, values.mean(), yerr=values.std(ddof=1), color=color, fmt=marker,
                                markersize=5.2, linewidth=1, capsize=2, label=NAMES[model] if ci == 0 else None)
            ax.set_xticks(range(5), [f'C{i+1}' for i in range(5)]); ax.set_xlim(-.65, 4.65); axes_style(ax)
            ax.set_ylabel(('Heavy-event recall', 'Average precision', 'Tail RMSE (mm)')[row])
            ax.set_ylim(*metric_limits[metric])
            if row < 2:
                ax.set_yticks([0, .25, .5, .75, 1]); ax.yaxis.set_major_formatter(PercentFormatter(1))
            if row == 0:
                ax.set_title(('FULL global identities', 'FULL personal identities')[column])
                ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.26), ncol=2, frameon=False, fontsize=10)
            if row == 2: ax.set_xlabel('Fixed study client')
    caption = ('Per-client heavy-event recall, average precision and tail RMSE, with global and personal identities kept in separate columns '
               'and identical metric scales across columns. Heavy truth is rainfall strictly above each client’s gradient-training q80; '
               'neural auxiliary probabilities use a .5 decision threshold for recall. Average precision uses continuous auxiliary probabilities. '
               'Tail RMSE uses the raw rainfall forecasts on observed heavy months, not classification scores. Small markers are all five seeds; '
               'large markers/whiskers are their mean ± sample SD, not subgroup confidence intervals. C1–C5 identifiers, training thresholds, '
               'event counts, confusion counts, score kinds, undefined reasons and all other methods are available in the source exports. '
               'These views condition on different client event prevalences; no pooled-client PR curve or spatial-population estimate is constructed.')
    writer.save(fig, '07_client_heavy', caption, [table, 'source_data/per_client_ALL.csv'],
                [info['run'] / 'evaluation' / 'per_client.csv', info['run'] / 'evaluation' / 'metrics_receipt.json'])


def plot_precision_recall(writer):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    info = writer.info; records = info['metadata']['PR_curves']; per_client = info['frames']['per_client']
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout='constrained', sharex=True, sharey=True)
    all_coordinates = []; example_coordinates = []
    keys = set()
    for record in records:
        key = (record['model'], record['seed'], record['client'])
        require(key not in keys, 'Repeated saved PR curve'); keys.add(key)
        precision, recall = record['precision'], record['recall']
        require(len(precision) == len(recall), 'PR coordinate lengths differ')
        thresholds = record.get('score_threshold', [])
        require(len(thresholds) == max(0, len(precision)-1), 'PR threshold alignment differs')
        for index, (p, r) in enumerate(zip(precision, recall)):
            require(np.isfinite([p, r]).all() and 0 <= p <= 1 and 0 <= r <= 1, 'PR value outside [0,1]')
            row = dict(model=key[0], seed=key[1], client=key[2], point_order=index,
                       precision=p, recall=r, score_threshold=thresholds[index] if index < len(thresholds) else np.nan,
                       score_kind=record.get('score_kind'))
            all_coordinates.append(row)
            if key[0] in PRIMARY_GLOBAL and key[1] == 42: example_coordinates.append(row)
    expected = set(zip(per_client.model, per_client.seed, per_client.client))
    require(keys == expected, 'PR coordinate/metric cohorts differ')
    lookup = {(r['model'], r['seed'], r['client']): r for r in records}
    for ci, (client, ax) in enumerate(zip(CLIENTS, axes.flat)):
        for mi, model in enumerate(PRIMARY_GLOBAL):
            curve = lookup[(model, 42, client)]
            require(curve.get('score_kind') == 'predicted heavy-event probability' and len(curve['precision']) > 0, 'Expected nonempty neural probability PR curve')
            color, marker = METHOD_STYLE[model]
            ax.plot(curve['recall'], curve['precision'], color=color, linestyle=('-', '--', ':')[mi],
                    marker=marker, markevery=max(1, len(curve['recall'])//9), markersize=3.4,
                    linewidth=1.4, label=NAMES[model])
        events = per_client.loc[(per_client.model == PRIMARY_GLOBAL[0]) & (per_client.seed == 42) & (per_client.client == client)].iloc[0]
        prevalence = float(events.actual_heavy_n) / 96
        ax.axhline(prevalence, color='#8B939B', linestyle='-.', linewidth=.8, label='Observed prevalence')
        ax.set_title(f'C{ci+1}: {int(events.actual_heavy_n)}/96 heavy months')
        ax.set(xlim=(0, 1), ylim=(0, 1)); axes_style(ax)
        ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    axes[1, 2].axis('off')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[1, 2].legend(handles, labels, loc='center', frameon=False, fontsize=11)
    examples = writer.table('08_client_PR_seed42', pd.DataFrame(example_coordinates))
    all_table = writer.table('PR_coordinates_ALL', pd.DataFrame(all_coordinates))
    caption = ('Separate per-client precision–recall curves for the predeclared illustration seed42, FULL global CA-ALCE, FedAvg and FedProx. '
               'This seed is not chosen by accuracy. Curves use each model’s saved auxiliary heavy-event probabilities and that client’s training-only q80 event threshold. '
               'The horizontal reference is the observed event prevalence for that client, and the title records its actual heavy-event count out of 96 months. '
               'No pooled-client curve is shown. Full per-client metrics for all five seeds and every model, all saved PR coordinates, score kinds and final endpoint '
               'with no finite score threshold are exported. Seed42 curves are illustrative; the all-seed metrics and paired regression intervals provide separate evidence.')
    writer.save(fig, '08_client_PR_seed42', caption, [examples, all_table, 'source_data/per_client_ALL.csv'],
                [info['run'] / 'evaluation' / 'PR_curves.json', info['run'] / 'evaluation' / 'per_client.csv'])


def render(info, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.labelsize': 11, 'axes.titlesize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 10, 'figure.titlesize': 13, 'savefig.dpi': 300,
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.hashsalt': 'caadapfl_publication_figures_v1', 'axes.edgecolor': '#5C6670',
        'text.color': '#26333E', 'axes.labelcolor': '#26333E', 'xtick.color': '#26333E', 'ytick.color': '#26333E'})
    require(not output.exists(), f'Output already exists: {output}. Use a fresh directory to preserve published figures.')
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.parent / (output.name + '.partial_' + uuid.uuid4().hex[:8])
    partial.mkdir()
    writer = FigureWriter(partial, info)
    # Every source table is exported, preserving exact saved CSV bytes and all 96 months.
    for name in CSV_NAMES:
        shutil.copyfile(info['run'] / 'evaluation' / (name + '.csv'), partial / 'source_data' / (name + '_ALL.csv'))
    intervals = info['frames']['paired_block_bootstrap_summary']
    for block in (6, 24):
        writer.table(f'paired_block{block}_sensitivity_ALL', intervals.loc[intervals.block_length_months == block])
    writer.table('macro_R2_RMSE_independent_recalculation_ALL', info['macro_recomputed'])
    plot_seed_comparison(writer, GLOBAL, '01_global_and_classical', 'FULL global neural models and explicitly named classical controls')
    plot_seed_comparison(writer, PERSONAL, '02_personal', 'FULL personal neural and graph models')
    plot_forests(writer); plot_graphs(writer); plot_development(writer); plot_heavy(writer); plot_precision_recall(writer)
    require(len(writer.figures) == 8, 'Figure count differs from the frozen plan')
    (partial / 'figure_plan.json').write_text(json.dumps(FIGURE_PLAN, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for name in ('hardware', 'metrics_receipt', 'fit_cost_receipt', 'shared_graph_setup_ledger'):
        shutil.copyfile(info['run'] / 'evaluation' / (name + '.json'), partial / 'source_data' / (name + '.json'))
    # Include all optional exact-block test results without turning them into plot stars.
    tests = info['run'] / 'evaluation' / 'block_sign_randomization_Holm.json'
    if tests.is_file(): shutil.copyfile(tests, partial / 'source_data' / tests.name)
    outputs = {p.relative_to(partial).as_posix(): {'sha256': sha256(p), 'bytes': p.stat().st_size}
               for p in sorted(partial.rglob('*')) if p.is_file()}
    receipt = dict(status='RENDERED_AWAITING_VISUAL_QA', script_sha256=sha256(__file__),
        run_sha256=info['identity']['run_sha256'],
        evaluation_complete=source_record(info['run'] / 'evaluation' / 'complete.json', info),
        independent_qa=source_record(info['qa_path'], info),
        independent_qa_scope=info['qa']['scope'],
        gates_passed_before_accuracy_reads=True, figure_count=8, figure_plan_sha256=sha256(partial / 'figure_plan.json'),
        figures=writer.figures, output_files=outputs,
        input_sources=[source_record(path, info) for path in sorted(set([
            *(info['run'] / 'evaluation' / name for name in info['completion']['files']),
            *(info['run'] / name for name in ('run_identity.json', 'choices_frozen.json', 'prepared.json', 'protocol.json', 'schedule.json')),
            info['run'] / 'datasets' / 'MAIN_graphs.json', info['qa_path'],
            *(info['data'] / name for name in ('monthly.csv', 'indices.csv', 'manifest.json')),
            *(info['repo'] / 'caadapfl' / (name+'.py') for name in info['identity']['code']),
            *info['stage_sources']]))],
        versions={'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__, 'matplotlib': matplotlib.__version__},
        interval_scope='Marginal 95%, conditional fixed-client paired-seed circular-block bootstrap, not simultaneous family intervals',
        significance_note='Optional exact 8-block sign tests have only 256 patterns; two-sided minimum nonzero p=.0078125. Holm across 31 cannot reject at .05 with this resolution. No stars used.',
        verification='Every prediction cohort and all saved per-client/macro R2/RMSE and paired point estimates reconciled; graph provenance and development curves hash-bound. Rendered visual acceptance must be recorded separately.',
        model_execution=False, source_preparation=False, prospective_accuracy_claim=False)
    (partial / 'figure_manifest.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    partial.rename(output)
    plt.close('all')
    return receipt


def resolve_argument(repo, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1], help='Packaged repository root')
    parser.add_argument('--run', default='runs/revision_v2', help='Run directory, relative to repo or absolute')
    parser.add_argument('--data', default='data', help='Canonical monthly.csv/indices.csv/manifest.json directory')
    parser.add_argument('--qa-receipt', default=None, help='Default: RUN/independent_qa/complete.json')
    parser.add_argument('--output', default='figures/revision_v2', help='Fresh output directory; existing paths refused')
    parser.add_argument('--verify-only', action='store_true', help='Run all gates/table checks without rendering or writing')
    parser.add_argument('--plan', action='store_true', help='Print predeclared figure plan without reading any run files')
    args = parser.parse_args(argv)
    if args.plan:
        print(json.dumps(FIGURE_PLAN, indent=2, allow_nan=False)); return 0
    repo = args.repo.resolve(); run = resolve_argument(repo, args.run); data = resolve_argument(repo, args.data)
    qa = resolve_argument(repo, args.qa_receipt) if args.qa_receipt else run / 'independent_qa' / 'complete.json'
    try:
        info = load_accepted_run(repo, run, data, qa)
        if args.verify_only:
            print(json.dumps({'status': 'PASS', 'run_sha256': info['identity']['run_sha256'],
                              'rows': len(info['frames']['raw_predictions']), 'figures_rendered': 0})); return 0
        result = render(info, resolve_argument(repo, args.output))
        print(json.dumps({'status': result['status'], 'figure_count': result['figure_count'],
                          'output': str(resolve_argument(repo, args.output)), 'run_sha256': result['run_sha256']})); return 0
    except (FigureIntegrityError, FileNotFoundError, KeyError) as error:
        print(f'Figure gate/check failed: {error}', file=sys.stderr); return 2


if __name__ == '__main__':
    raise SystemExit(main())
