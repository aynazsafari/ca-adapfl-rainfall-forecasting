"""Independent saved-table audit; no scientific helper imports or models.

Reads only accepted primary artifacts and admitted IMD/CHIRPS source tables.
Large forecast/key CSVs are streamed. NumPy formulas below independently
recalculate the nine descriptive scalars; no p-values or new scored choices.
"""
from pathlib import Path
from collections import Counter
import argparse
import datetime
import hashlib
import json
import math
import numpy as np
import pandas as pd

CLIENTS = ['Client1_NW_Arid', 'Client2_Indo_Gangetic', 'Client3_NE_Himalaya',
           'Client4_South_Peninsular', 'Client5_Central_India']
SEEDS = [7, 42, 123, 999, 2024]
MONTHS = list(pd.period_range('2017-01', '2024-12', freq='M').astype(str))
BINS = ['negative_ONI_value', 'neutral_ONI_value', 'positive_ONI_value']
ROLES = {'train': ('1983-01', '2007-11'), 'monitor': ('2008-01', '2010-11'),
         'validation': ('2011-01', '2016-11'), 'evaluation': ('2017-01', '2024-12')}
METRICS = ['R2', 'NSE', 'RMSE_mm', 'MAE_mm', 'PBIAS_signed_percent',
           'PBIAS_absolute_percent', 'forecast_bias_signed_percent', 'Pearson_r', 'KGE_2009']
TABLE_ROWS = {'oni_month_bins.csv': 96, 'oni_prediction_key_bins.csv': 230400,
              'oni_per_client_seed_bin.csv': 7200, 'oni_aggregate_per_seed_bin.csv': 2880,
              'oni_five_seed_summary.csv': 576, 'CHIRPS_IMD_same_month_pairs.csv': 2505,
              'CHIRPS_IMD_agreement_per_client_role.csv': 20,
              'CHIRPS_IMD_agreement_macro_pooled_role.csv': 8}
EXPECTED_IMD_SHA = '8c14f84a86e41cd3a847d26692a3f0bee2a2a4e2e33c6504202c01ac1b49af76'
EXPECTED_PRIMARY_QA_SHA = 'c379f9b16af248f6c34f709767ff0b66274789057c4a0bcf9e2d0e4b3cbe1d3a'
EXPECTED_DIAGNOSTIC_CODE_SHA = 'e5aaf654a5bd31ebc47fdfa02752402a5205af94180601ae9d02659244ca7fb0'
EXPECTED_METRIC_HELPER_SHA = '50629d02c571aa34cc71838df72fbd28c7d08da690dddff51dc03c1c8dd7ea26'


def require(value, message):
    if not value:
        raise AssertionError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def csv(path, **kwargs):
    return pd.read_csv(path, float_precision='round_trip', **kwargs)


def metrics(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    require(y.ndim == p.ndim == 1 and y.size == p.size and y.size > 0, 'Metric pair vectors differ')
    require(np.isfinite(y).all() and np.isfinite(p).all(), 'Nonfinite metric inputs')
    residual = y - p
    ym, pm = np.mean(y), np.mean(p)
    yc, pc = y - ym, p - pm
    ys, ps = np.dot(yc, yc), np.dot(pc, pc)
    r2 = 1.0 - np.dot(residual, residual) / ys if ys > 0 else np.nan
    pbias = 100.0 * np.sum(residual) / np.sum(y) if np.sum(y) != 0 else np.nan
    corr = np.dot(yc, pc) / np.sqrt(ys * ps) if y.size > 1 and ys > 0 and ps > 0 else np.nan
    alpha = np.sqrt(ps / ys) if y.size > 1 and ys > 0 else np.nan
    beta = pm / ym if ym != 0 else np.nan
    kge = 1.0 - np.linalg.norm(np.array([corr - 1, alpha - 1, beta - 1])) if np.isfinite([corr, alpha, beta]).all() else np.nan
    values = {'n': int(y.size), 'R2': r2, 'NSE': r2, 'RMSE_mm': np.sqrt(np.mean(residual ** 2)),
              'MAE_mm': np.mean(np.abs(residual)), 'PBIAS_signed_percent': pbias,
              'PBIAS_absolute_percent': abs(pbias), 'forecast_bias_signed_percent': -pbias,
              'Pearson_r': corr, 'KGE_2009': kge}
    values.update({name + '_undefined': not bool(np.isfinite(values[name])) for name in METRICS})
    values['any_metric_undefined'] = any(values[name + '_undefined'] for name in METRICS)
    return values


def macro(local):
    values = {'n': sum(row['n'] for row in local)}
    for name in METRICS:
        undefined = sum(row[name + '_undefined'] for row in local)
        values[name] = np.mean([row[name] for row in local]) if undefined == 0 else np.nan
        values[name + '_undefined'] = bool(undefined)
        values[name + '_undefined_clients'] = undefined
    values['any_metric_undefined'] = any(values[name + '_undefined'] for name in METRICS)
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--diagnostics', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run, diagnostic, output = args.run.resolve(), args.diagnostics.resolve(), args.output.resolve()
    require(not output.exists(), 'Use a fresh independent audit output directory')
    require((run / 'evaluation/complete.json').exists() and (run / 'execution_complete.json').exists(), 'Primary completion gates missing')
    execution = load(run / 'execution_complete.json')
    require(execution['exit_code'] == 0, 'Primary execution did not exit 0')
    qa_path = run / 'independent_qa/complete.json'
    require(qa_path.exists() and sha(qa_path) == EXPECTED_PRIMARY_QA_SHA, 'Original primary audit was not preserved')
    qa = load(qa_path)
    completed, identity, desc = load(run / 'evaluation/complete.json'), load(run / 'run_identity.json'), load(diagnostic / 'complete.json')
    require(qa['status'] == desc['status'] == 'PASS', 'Acceptance/diagnostic receipt is not PASS')
    require(qa['run_sha256'] == identity['run_sha256'] == completed['run_sha256'] == desc['gates']['run_sha256'], 'Run identities differ')
    require(sha(run / 'evaluation/complete.json') == qa['evaluation_complete_sha256'] == desc['gates']['evaluation_complete_sha256'], 'Evaluation completion hash differs')
    require(desc['gates']['independent_qa_sha256'] == EXPECTED_PRIMARY_QA_SHA, 'Diagnostic primary audit parent differs')
    require(desc['gates']['imd_manifest_sha256'] == EXPECTED_IMD_SHA, 'Admitted IMD parent differs')
    require(not desc['scientific_core_changed'] and not desc['new_fits_inference_or_pvalues'], 'Diagnostic scope differs')
    require(set(desc['files']) == set(TABLE_ROWS), 'Eight-table export set differs')
    package = run.parent.parent
    data = package / 'data'
    imd_dir = package / 'tools/derived/imd'
    chirps_path = package / 'provenance/source_tables/rainfall_panel.csv'
    script_dir = Path(__file__).resolve().parent
    paths = [run / 'execution_complete.json', run / 'evaluation/complete.json', qa_path,
             run / 'run_identity.json', run / 'choices_frozen.json', diagnostic / 'complete.json',
             data / 'monthly.csv', data / 'indices.csv', data / 'manifest.json', chirps_path,
             imd_dir / 'manifest.json', imd_dir / 'imd_supplement.csv', imd_dir / 'imd_supplement_quality.csv',
             run / 'evaluation/raw_predictions.csv', script_dir / 'descriptive_diagnostics.py',
             script_dir / 'metric_reference.py', script_dir / 'audit_completed_run.py']
    checked = {str(path): sha(path) for path in paths}
    for name in ['monthly', 'indices', 'manifest']:
        path = data / (name + ('.json' if name == 'manifest' else '.csv'))
        require(sha(path) == identity['source'][name + '_sha256'] == desc['gates']['source_sha256'][name + '_sha256'], 'Canonical source hash differs: ' + name)
    require(sha(run / 'evaluation/raw_predictions.csv') == completed['files']['raw_predictions.csv'] == desc['gates']['raw_predictions_sha256'], 'Prediction bytes differ')
    require(sha(run / 'choices_frozen.json') == completed['choices_frozen_sha256'], 'Frozen choices differ')
    require(sha(imd_dir / 'manifest.json') == EXPECTED_IMD_SHA, 'IMD manifest bytes differ')
    require(sha(chirps_path) == desc['gates']['CHIRPS_source_quality_sha256'] == load(data / 'manifest.json')['frozen_target_sha256'], 'Frozen CHIRPS bytes differ')
    imd_manifest = load(imd_dir / 'manifest.json')
    require(imd_manifest['status'] == 'PASS_WITH_EXPLICIT_INVALID_SOURCE' and imd_manifest['unit'] == 'mm month-1', 'IMD admission/units differ')
    for name in ['imd_supplement.csv', 'imd_supplement_quality.csv']:
        declared = imd_manifest['outputs'][name]
        declared = declared['sha256'] if isinstance(declared, dict) else declared
        require(sha(imd_dir / name) == declared == desc['gates']['imd_output_sha256'][name], 'IMD source bytes differ')
    require(sha(script_dir / 'descriptive_diagnostics.py') == desc['diagnostic_code_sha256'] == EXPECTED_DIAGNOSTIC_CODE_SHA, 'Diagnostic helper changed')
    require(sha(script_dir / 'metric_reference.py') == desc['metric_reference_sha256'] == EXPECTED_METRIC_HELPER_SHA, 'Metric helper changed')
    require(sha(script_dir / 'audit_completed_run.py') == '2bfa0e47cc656af7043057cd115e827786f6a478e7d207aefe8730027d5b80f0', 'Parent frozen audit helper changed')
    for name, count in TABLE_ROWS.items():
        require(desc['files'][name]['rows'] == count and sha(diagnostic / name) == desc['files'][name]['sha256'], 'Export count/hash differs: ' + name)
        checked[str(diagnostic / name)] = sha(diagnostic / name)
    # Scientific CSV reads occur only after authentication above.
    monthly = csv(data / 'monthly.csv').set_index(['client', 'month'])
    indices = csv(data / 'indices.csv').set_index('month')
    imd = csv(imd_dir / 'imd_supplement.csv').set_index(['client', 'month'])
    quality = csv(imd_dir / 'imd_supplement_quality.csv').set_index(['client', 'month'])
    chirps = csv(chirps_path).set_index(['client', 'month'])
    native_months = list(pd.period_range('1981-01', '2024-12', freq='M').astype(str))
    native_keys = pd.MultiIndex.from_product([CLIENTS, native_months], names=['client', 'month'])
    for frame in [monthly, imd, quality, chirps]:
        require(frame.index.is_unique and frame.sort_index().index.equals(native_keys), 'Native source calendar/key grain differs')
    require(indices.index.is_unique and indices.index.tolist() == native_months, 'Index calendar differs')
    require('oni_anomaly_3mo' in indices, 'Exact ONI source name missing')
    frozen = load(run / 'choices_frozen.json')
    models = sorted({row['label'] for row in frozen['outputs']} | {'training_seasonal_mean', 'available_rainfall_target_minus2', 'same_calendar_target_minus12'})
    require(len(models) == 96, 'Saved label scope differs')
    lookups = [{value: i for i, value in enumerate(values)} for values in [models, SEEDS, CLIENTS, MONTHS]]
    shape = (len(models), len(SEEDS), len(CLIENTS), len(MONTHS))
    observed = np.full(shape, np.nan)
    predicted = np.full(shape, np.nan)
    seen = np.zeros(shape, dtype=bool)
    for part in csv(run / 'evaluation/raw_predictions.csv', usecols=['model', 'seed', 'client', 'target_month', 'y', 'prediction'], chunksize=9600):
        coordinates = tuple(part[column].map(lookup).to_numpy() for column, lookup in zip(['model', 'seed', 'client', 'target_month'], lookups))
        require(all(np.isfinite(value).all() for value in coordinates), 'Unexpected raw forecast key')
        coordinates = tuple(value.astype(np.int64) for value in coordinates)
        require(len(np.unique(np.ravel_multi_index(coordinates, shape))) == len(part) and not seen[coordinates].any(), 'Duplicate raw forecast key')
        require(np.isfinite(part[['y', 'prediction']].to_numpy()).all(), 'Nonfinite raw forecast/observation')
        seen[coordinates] = True
        observed[coordinates] = part.y.to_numpy()
        predicted[coordinates] = part.prediction.to_numpy()
    require(seen.all(), 'Missing 96 x 5 x 5 x 96 forecast keys')
    canonical_y = np.array([[monthly.loc[(client, month), 'rainfall_mm'] for month in MONTHS] for client in CLIENTS])
    require(np.allclose(observed, canonical_y[None, None], rtol=0, atol=1e-10), 'Forecast observations differ from admitted CHIRPS')
    bin_labels, supports, oni_values = [], [], []
    for month in MONTHS:
        support = str(pd.Period(month, freq='M') - 2)
        value = float(indices.loc[support, 'oni_anomaly_3mo'])
        require(math.isfinite(value), 'Missing eligible ONI value')
        supports.append(support)
        oni_values.append(value)
        bin_labels.append(BINS[0] if value <= -0.5 else BINS[2] if value >= 0.5 else BINS[1])
    date_counts = Counter(bin_labels)
    require(dict(date_counts) == desc['ONI']['bin_date_counts'] and sum(date_counts.values()) == 96, 'ONI bin date counts differ')
    require(not desc['ONI']['event_classification'] and not desc['ONI']['NOAA_five_overlapping_season_persistence_applied'], 'ONI bins overstated as official events')
    maximum_errors = Counter()
    covered_columns = {}

    def compare(row, expected, table):
        require(set(row.index) == set(expected), 'Unverified/extra/missing columns in ' + table)
        covered_columns[table] = list(row.index)
        for column, wanted in expected.items():
            actual = row[column]
            if isinstance(wanted, str):
                actual = '' if pd.isna(actual) else actual
                require(actual == wanted, table + '/' + column + ' string differs')
            elif isinstance(wanted, (bool, np.bool_)):
                require(isinstance(actual, (bool, np.bool_)) and bool(actual) == bool(wanted), table + '/' + column + ' boolean differs')
            elif pd.isna(wanted):
                require(pd.isna(actual), table + '/' + column + ' undefined/NA was changed')
            else:
                require(np.isfinite(actual), table + '/' + column + ' unexpectedly undefined')
                difference = abs(float(actual) - float(wanted))
                require(np.isclose(actual, wanted, rtol=2e-12, atol=1e-10), table + '/' + column + ' numerical value differs')
                maximum_errors[column] = max(maximum_errors[column], difference)

    table = csv(diagnostic / 'oni_month_bins.csv')
    require(len(table) == 96 and table.target_month.tolist() == MONTHS, 'ONI mapping calendar differs')
    for position, row in table.iterrows():
        compare(row, {'target_month': MONTHS[position], 'eligible_ONI_support_month': supports[position],
                      'oni_anomaly_3mo': oni_values[position], 'ONI_value_bin': bin_labels[position]}, 'oni_month_bins.csv')
    key_seen = np.zeros(shape, dtype=bool)
    key_digest = hashlib.sha256()
    last_key = None
    key_rows = 0
    for part in csv(diagnostic / 'oni_prediction_key_bins.csv', chunksize=9600):
        require(list(part.columns) == ['model', 'seed', 'client', 'target_month', 'eligible_ONI_support_month', 'oni_anomaly_3mo', 'ONI_value_bin'], 'Prediction-key bin schema differs')
        for row in part.itertuples(index=False):
            key = (row.model, row.seed, row.client, row.target_month)
            require(last_key is None or last_key < key, 'Key bins are not uniquely canonically ordered')
            last_key = key
            coordinates = tuple(lookup[value] for lookup, value in zip(lookups, key))
            require(not key_seen[coordinates], 'Duplicate bin key')
            key_seen[coordinates] = True
            position = coordinates[-1]
            require(row.eligible_ONI_support_month == supports[position] and row.ONI_value_bin == bin_labels[position]
                    and row.oni_anomaly_3mo == oni_values[position], 'Key bin assignment/date/value differs')
            key_digest.update(('\t'.join(map(str, key)) + '\n').encode())
            key_rows += 1
    require(key_seen.all() and key_rows == 230400 and key_digest.hexdigest() == desc['ONI']['prediction_key_calendar_sha256'] == qa['prediction_keys']['actual_key_calendar_sha256'], 'Prediction-key calendar hash differs')
    covered_columns['oni_prediction_key_bins.csv'] = list(part.columns)
    client_table = csv(diagnostic / 'oni_per_client_seed_bin.csv').set_index(['model', 'seed', 'client', 'ONI_value_bin'], drop=False)
    aggregate_table = csv(diagnostic / 'oni_aggregate_per_seed_bin.csv').set_index(['model', 'seed', 'ONI_value_bin', 'scope'], drop=False)
    summary_table = csv(diagnostic / 'oni_five_seed_summary.csv').set_index(['model', 'ONI_value_bin', 'scope'], drop=False)
    require(client_table.index.is_unique and aggregate_table.index.is_unique and summary_table.index.is_unique, 'Duplicate ONI aggregate keys')
    require(len(client_table) == 7200 and len(aggregate_table) == 2880 and len(summary_table) == 576, 'ONI aggregate row counts differ')
    aggregates = {}
    for m, model in enumerate(models):
        for s, seed in enumerate(SEEDS):
            for label in BINS:
                mask = np.array(bin_labels) == label
                locals_ = []
                for c, client in enumerate(CLIENTS):
                    values = metrics(observed[m, s, c, mask], predicted[m, s, c, mask])
                    values['negative_forecast_n'] = int((predicted[m, s, c, mask] < 0).sum())
                    locals_.append(values)
                    expected = {'model': model, 'seed': seed, 'client': client, 'ONI_value_bin': label,
                                'unique_dates': date_counts[label], **values}
                    compare(client_table.loc[(model, seed, client, label)], expected, 'oni_per_client_seed_bin.csv')
                pooled = metrics(observed[m, s, :, mask].reshape(-1), predicted[m, s, :, mask].reshape(-1))
                pooled['negative_forecast_n'] = int((predicted[m, s, :, mask] < 0).sum())
                pooled.update({name + '_undefined_clients': np.nan for name in METRICS})
                macro_values = macro(locals_)
                macro_values['negative_forecast_n'] = sum(row['negative_forecast_n'] for row in locals_)
                for scope, values in [('macro_equal_five_clients', macro_values), ('pooled_all_client_months', pooled)]:
                    expected = {'model': model, 'seed': seed, 'ONI_value_bin': label, 'scope': scope,
                                'unique_dates': date_counts[label], 'client_count': 5, 'clients_with_pairs': 5, **values}
                    compare(aggregate_table.loc[(model, seed, label, scope)], expected, 'oni_aggregate_per_seed_bin.csv')
                    aggregates[(model, seed, label, scope)] = values
        for label in BINS:
            for scope in ['macro_equal_five_clients', 'pooled_all_client_months']:
                expected = {'model': model, 'ONI_value_bin': label, 'scope': scope, 'paired_seeds': 5,
                            'unique_dates': date_counts[label], 'client_count': 5}
                for name in METRICS:
                    values = np.array([aggregates[(model, seed, label, scope)][name] for seed in SEEDS])
                    defined = np.isfinite(values)
                    expected[name + '_undefined_seeds'] = int((~defined).sum())
                    expected[name + '_mean'] = np.mean(values) if defined.all() else np.nan
                    expected[name + '_std'] = np.std(values, ddof=1) if defined.all() else np.nan
                compare(summary_table.loc[(model, label, scope)], expected, 'oni_five_seed_summary.csv')
    print('All five ONI tables independently recalculated', flush=True)
    pair_table = csv(diagnostic / 'CHIRPS_IMD_same_month_pairs.csv')
    require(len(pair_table) == 2505 and not pair_table.duplicated(['role', 'client', 'target_month']).any(), 'Source pair keys/count differ')
    pair_table = pair_table.set_index(['role', 'client', 'target_month'], drop=False)
    source_client = csv(diagnostic / 'CHIRPS_IMD_agreement_per_client_role.csv').set_index(['role', 'client'], drop=False)
    source_aggregate = csv(diagnostic / 'CHIRPS_IMD_agreement_macro_pooled_role.csv').set_index(['role', 'scope'], drop=False)
    require(source_client.index.is_unique and source_aggregate.index.is_unique and len(source_client) == 20 and len(source_aggregate) == 8, 'Source agreement grain/count differs')
    excluded = []
    role_counts = {}
    source_key_digest = hashlib.sha256()
    for role, (start, end) in ROLES.items():
        dates = list(pd.period_range(start, end, freq='M').astype(str))
        local, role_y, role_p = [], [], []
        for client in CLIENTS:
            admitted_y, admitted_p = [], []
            for month in dates:
                key = (client, month)
                coverage = float(quality.loc[key, 'weighted_coverage'])
                admitted = int(quality.loc[key, 'feature_admitted']) == 1 and coverage >= 0.95 and coverage <= 1 + 1e-10
                require(admitted == np.isfinite(imd.loc[key, 'imd_rainfall_mm']), 'IMD admission/NA mismatch')
                require((not admitted) == (client == CLIENTS[0] and month == '2023-10'), 'Unexpected invalid source key')
                expected = {'role': role, 'client': client, 'target_month': month,
                            'CHIRPS_rainfall_mm': float(monthly.loc[key, 'rainfall_mm']),
                            'CHIRPS_weighted_coverage': float(chirps.loc[key, 'weighted_coverage']),
                            'CHIRPS_geometry_pixel_count': int(chirps.loc[key, 'geometry_pixel_count']),
                            'CHIRPS_valid_pixel_count': int(chirps.loc[key, 'valid_pixel_count']),
                            'CHIRPS_geometry_area_m2': float(chirps.loc[key, 'geometry_area_m2']),
                            'CHIRPS_valid_area_m2': float(chirps.loc[key, 'valid_area_m2']),
                            'IMD_admitted_rainfall_mm': float(imd.loc[key, 'imd_rainfall_mm']),
                            'IMD_physical_complete_cell_mean_mm': float(quality.loc[key, 'physical_complete_cell_mean_mm']),
                            'IMD_weighted_coverage': coverage, 'IMD_complete_cells': int(quality.loc[key, 'complete_cells']),
                            'IMD_feature_admitted': admitted, 'pair_included': admitted,
                            'exclusion_reason': str(quality.loc[key, 'quality_reason']) if not admitted else ''}
                require(np.isclose(monthly.loc[key, 'rainfall_mm'], chirps.loc[key, 'rainfall_mm'], atol=1e-10, rtol=0), 'CHIRPS canonical/quality rainfall differs')
                compare(pair_table.loc[(role, client, month)], expected, 'CHIRPS_IMD_same_month_pairs.csv')
                source_key_digest.update(('\t'.join([role, client, month]) + '\n').encode())
                if admitted:
                    require(imd.loc[key, 'imd_rainfall_mm'] == quality.loc[key, 'physical_complete_cell_mean_mm'], 'Admitted IMD rainfall was transformed')
                    admitted_y.append(expected['CHIRPS_rainfall_mm'])
                    admitted_p.append(expected['IMD_admitted_rainfall_mm'])
                else:
                    excluded.append({'role': role, 'client': client, 'target_month': month,
                                     'exclusion_reason': expected['exclusion_reason']})
            values = metrics(admitted_y, admitted_p)
            values['negative_IMD_source_n'] = int((np.array(admitted_p) < 0).sum())
            local.append(values)
            role_y.extend(admitted_y)
            role_p.extend(admitted_p)
            expected = {'role': role, 'client': client, 'raw_same_month_pair_keys': len(dates),
                        'valid_pair_n': len(admitted_y), 'excluded_invalid_IMD_n': len(dates) - len(admitted_y),
                        'missing_CHIRPS_n': 0, 'missing_admitted_IMD_n': len(dates) - len(admitted_y), **values}
            compare(source_client.loc[(role, client)], expected, 'CHIRPS_IMD_agreement_per_client_role.csv')
        pooled = metrics(role_y, role_p)
        pooled['negative_IMD_source_n'] = int((np.array(role_p) < 0).sum())
        pooled.update({name + '_undefined_clients': np.nan for name in METRICS})
        macro_values = macro(local)
        macro_values['negative_IMD_source_n'] = sum(row['negative_IMD_source_n'] for row in local)
        role_counts[role] = {'dates': len(dates), 'raw_pairs': len(dates) * 5, 'valid_pairs': len(role_y)}
        for scope, values in [('macro_equal_five_clients', macro_values), ('pooled_all_valid_client_months', pooled)]:
            expected = {'role': role, 'scope': scope, 'client_count': 5, 'raw_same_month_pair_keys': len(dates) * 5,
                        'valid_pair_n': len(role_y), 'excluded_invalid_IMD_n': len(dates) * 5 - len(role_y), **values}
            compare(source_aggregate.loc[(role, scope)], expected, 'CHIRPS_IMD_agreement_macro_pooled_role.csv')
    require(excluded == desc['CHIRPS_IMD']['excluded_pairs'] and len(excluded) == 1, 'Invalid-source exclusions differ')
    require(sum(value['raw_pairs'] for value in role_counts.values()) == 2505 and
            sum(value['valid_pairs'] for value in role_counts.values()) == 2504, 'Source pair totals differ')
    require(set(desc['CHIRPS_IMD']['boundary_December_labels_excluded_from_roles']) == {'2007-12', '2010-12', '2016-12'} and
            not desc['CHIRPS_IMD']['forecast_cohort_selection_used'] and not desc['CHIRPS_IMD']['invalid_IMD_filled'], 'Source/forecast calendar scope was conflated')
    source_late_2024 = pair_table.loc[pair_table.target_month.between('2024-02', '2024-12')]
    require(len(source_late_2024) == 55 and source_late_2024.pair_included.all(), 'Source agreement improperly adopts forecast warm-up exclusions')
    require('neither independent validation nor forecast evaluation' in desc['CHIRPS_IMD']['interpretation'] and
            'does not establish statistical independence' in desc['CHIRPS_IMD']['product_dependence'], 'Product-dependence caveat missing')
    require(set(covered_columns) == set(TABLE_ROWS), 'Not all eight tables were verified')
    for path, value in checked.items():
        require(sha(path) == value, 'Source/helper/parent/output changed during read-only audit: ' + path)
    record = {'status': 'PASS', 'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'scope': 'Independent full-column/full-row recalculation of the eight accepted primary post-completion descriptive tables; no supplementary evaluation reads.',
              'scope_details': {'tables': TABLE_ROWS, 'all_columns_verified': covered_columns,
                                'ONI': {'labels': 96, 'dates': 96, 'seeds': SEEDS, 'clients': CLIENTS,
                                        'saved_prediction_keys': key_rows, 'bin_date_counts': dict(date_counts),
                                        'eligible_support': 'oni_anomaly_3mo at target minus two calendar months',
                                        'bins': ['value <= -0.5', '-0.5 < value < +0.5', 'value >= +0.5'],
                                        'official_persistent_event_classification': False},
                                'CHIRPS_IMD': {'role_counts': role_counts, 'raw_pairs': 2505, 'valid_pairs': 2504,
                                               'excluded_pairs': excluded, 'retained_2024_Feb_Dec_source_pairs': 55,
                                               'forecast_IMD_evaluation_cohort_425_pairs_not_used': True}},
              'run_sha256': identity['run_sha256'], 'evaluation_complete_sha256': sha(run / 'evaluation/complete.json'),
              'parent_primary_audit_sha256': EXPECTED_PRIMARY_QA_SHA,
              'parent_diagnostic_complete_sha256': checked[str(diagnostic / 'complete.json')],
              'diagnostic_code_sha256': EXPECTED_DIAGNOSTIC_CODE_SHA, 'metric_helper_sha256': EXPECTED_METRIC_HELPER_SHA,
              'prediction_key_calendar_sha256': key_digest.hexdigest(),
              'source_role_key_calendar_sha256': source_key_digest.hexdigest(),
              'maximum_absolute_numerical_discrepancies': dict(maximum_errors),
              'formulas': {'independence': 'Separate NumPy dot/mean/std/norm implementation; neither diagnostic nor metric helper imported or executed.',
                           'R2_NSE': '1 - sum((CHIRPS-or-observed - IMD-or-predicted)^2) / sum((reference - mean(reference))^2)',
                           'RMSE_MAE': 'sqrt(mean(residual^2)); mean(abs(residual))',
                           'PBIAS': '100 * sum(reference - comparison) / sum(reference); forecast bias is its negative',
                           'correlation_KGE': 'Centered-vector normalized dot; KGE2009 = 1 - norm([r-1, sigma_comparison/sigma_reference-1, mean_comparison/mean_reference-1])',
                           'macro_pooled': 'Equal five-client metric mean versus one recomputation over all admitted within-bin/within-role pairs, separately.',
                           'seed_summary': 'Mean and sample standard deviation (ddof=1) of each five-seed metric; any undefined seed preserves NaN.',
                           'undefined': 'No denominator floors; per-metric flags and NaNs checked, macro/seed undefined propagation preserved.'},
              'checked_files': checked,
              'memory_scope': {'forecast_csv_chunk_rows': 9600,
                               'retained_forecast_and_key_array_bytes': int(observed.nbytes + predicted.nbytes + seen.nbytes + key_seen.nbytes),
                               'full_forecast_or_prediction_key_DataFrame_loaded': False},
              'findings': [],
              'limitations': ['Descriptive revised-archive ONI value bins are not official persistent NOAA ENSO event classifications or causal attributions.',
                              'Same-month station-blended CHIRPS versus gauge-grid IMD agreement is neither independent forecast validation nor a target-free verification.',
                              'Native grid aggregation/geometry was authenticated through frozen source receipts and earlier source QA, not rerun here.',
                              'No visual rendering was reviewed; this receipt covers the eight CSV tables and their stated interpretations.',
                              'No new fits, inference, p-values, retuning, tests, source mutations or supplementary evaluation reads were performed.']}
    output.mkdir(parents=True)
    result_path = output / 'complete.json'
    result_path.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'receipt': str(result_path), 'receipt_sha256': sha(result_path),
                      'tables': 8, 'ONI_date_counts': dict(date_counts), 'source_pairs': [2505, 2504],
                      'max_scalar_discrepancy': max(maximum_errors.values())}, indent=2))


if __name__ == '__main__':
    main()
