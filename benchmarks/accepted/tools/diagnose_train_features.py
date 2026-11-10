"""Read-only, latest-row TRAIN feature diagnostics; no label/member reads or fits."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import platform
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows, fields):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def physical_unit(name, units):
    if name in units:
        return units[name], 'canonical manifest units'
    if name.startswith('season_'):
        return 'dimensionless', 'known-calendar sine/cosine transform in authenticated features.py'
    if name.startswith('rain_'):
        return 'mm (monthly total)', 'CHIRPS monthly totals; lags, window means and population SD preserve the source unit'
    if name in ('nino34_anomaly', 'dmi_hadisst11_anomaly', 'oni_anomaly_3mo', 'amv_ersstv6_global_removed_1951_1980'):
        return 'degC anomaly', 'authenticated index name/support plus previously verified source definition; absent from canonical manifest units'
    if name in ('soi_cpc_standardized', 'meiv2_bimonthly'):
        return 'dimensionless standardized index', 'authenticated index name/support plus previously verified source definition; absent from canonical manifest units'
    if name == 'enso_x_iod':
        return 'degC2', 'product of Nino3.4 and DMI anomalies in authenticated features.py'
    raise ValueError('Unresolved feature unit: ' + name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package-root', required=True)
    parser.add_argument('--run', default='runs/revision_v2')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    package = Path(args.package_root).resolve()
    run = package / args.run
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    identity = read(run / 'run_identity.json')
    prepared = read(run / 'prepared.json')
    manifest = read(package / 'data/manifest.json')
    identity_payload = {k: v for k, v in identity.items() if k != 'run_sha256'}
    assert digest(identity_payload) == identity['run_sha256'] == prepared['run_sha256']
    assert digest(read(run / 'protocol.json')) == identity['protocol_hash']
    assert digest(read(run / 'schedule.json')) == identity['schedule_hash']
    authentication = {'run_sha256': identity['run_sha256'], 'code': {}, 'files': {}, 'source': {}}
    for name, expected in identity['code'].items():
        path = package / 'caadapfl' / (name + '.py')
        actual = sha(path)
        assert actual == expected, 'Frozen code changed: ' + name
        authentication['code'][name] = actual
    for name in ('monthly', 'indices', 'manifest'):
        path = package / 'data' / (name + ('.json' if name == 'manifest' else '.csv'))
        actual = sha(path)
        assert actual == identity['source'][name + '_sha256']
        authentication['source'][name] = actual
    assert manifest['source_gate']['passed'] is True
    claimed = {str(k).replace('\\', '/'): v for k, v in prepared['files'].items()}
    for name in ('MAIN_FULL_fit.npz', 'MAIN_FULL_schema.json', 'MAIN_CORE_schema.json', 'MAIN_CLIMATE_schema.json'):
        relative = 'datasets/' + name
        actual = sha(run / relative)
        assert actual == claimed[relative], 'Prepared payload changed: ' + name
        authentication['files'][relative] = actual
    for name in ('run_identity.json', 'prepared.json', 'protocol.json', 'schedule.json'):
        authentication['files'][name] = sha(run / name)
    schema = read(run / 'datasets/MAIN_FULL_schema.json')
    core = read(run / 'datasets/MAIN_CORE_schema.json')['names']
    climate = read(run / 'datasets/MAIN_CLIMATE_schema.json')['names']
    names = schema['names']
    assert len(names) == len(set(names)) == 43
    assert len(core) == 14 and len(climate) == 21
    assert names[:14] == core and names[:21] == climate
    assert schema['met_columns'] == manifest['feature_contract']['met_columns'] == names[21:]
    assert set(schema['support']) == set(names)
    assert schema['extra_lag'] == 0
    assert schema['statistics']['count'] == 1495
    dictionary = []
    for i, name in enumerate(names):
        unit, basis = physical_unit(name, manifest['units'])
        dictionary.append(dict(index=i, feature=name, incremental_group='CORE' if i < 14 else ('CLIMATE' if i < 21 else 'METEOROLOGY'),
                               core_member=i < 14, climate_member=i < 21, full_member=True,
                               raw_physical_unit=unit, analyzed_unit='dimensionless global TRAIN z score', unit_basis=basis,
                               support_json=json.dumps(schema['support'][name], sort_keys=True)))
    clients = list(schema['statistics']['clients'])
    data = {}
    cohort = []
    accessed_members = []
    with np.load(run / 'datasets/MAIN_FULL_fit.npz', allow_pickle=False) as archive:
        for client in clients:
            date_key, x_key = client + '::raw_target_month', client + '::scaled_x'
            dates = archive[date_key]
            train = dates <= '2007-11'  # Filter calendar BEFORE any feature arithmetic.
            assert train.sum() == 299
            assert dates[train].tolist() == [str(d) for d in np.arange(np.datetime64('1983-01'), np.datetime64('2007-12'), dtype='datetime64[M]')]
            x = archive[x_key][train, -1, :].astype(np.float64, copy=True)
            assert x.shape == (299, 43)
            data[client] = x
            accessed_members.extend([date_key, x_key])
            cohort.append(dict(scope=client, train_rows=len(x), first_target_month=str(dates[train][0]),
                               last_target_month=str(dates[train][-1]), latest_row_index=-1,
                               excluded_fit_archive_rows=int((~train).sum())))
    data['GLOBAL_POOLED'] = np.concatenate(list(data.values()), axis=0)
    cohort.append(dict(scope='GLOBAL_POOLED', train_rows=1495, first_target_month='1983-01', last_target_month='2007-11',
                       latest_row_index=-1, excluded_fit_archive_rows=530))
    profile, pairs, high = [], [], []
    summary = {}
    for scope, x in data.items():
        finite = np.isfinite(x)
        assert finite.all(), 'TRAIN standardized feature is missing or infinite: ' + scope
        mean, var = x.mean(axis=0), x.var(axis=0, ddof=0)
        ranges = np.ptp(x, axis=0)
        constants = ranges == 0
        for i, name in enumerate(names):
            profile.append(dict(scope=scope, feature=name, n_rows=len(x), finite_count=int(finite[:, i].sum()),
                                nan_count=int(np.isnan(x[:, i]).sum()), positive_inf_count=int(np.isposinf(x[:, i]).sum()),
                                negative_inf_count=int(np.isneginf(x[:, i]).sum()), standardized_mean=float(mean[i]),
                                standardized_population_variance=float(var[i]), standardized_population_std=float(np.sqrt(var[i])),
                                standardized_min=float(x[:, i].min()), standardized_max=float(x[:, i].max()),
                                exact_constant=bool(constants[i]), std_below_existing_scaler_floor=bool(np.sqrt(var[i]) < 1e-12)))
        centered = x - mean
        ss = (centered * centered).sum(axis=0)
        denom = np.sqrt(np.outer(ss, ss))
        correlation = np.divide(centered.T @ centered, denom, out=np.full((43, 43), np.nan), where=denom > 0)
        correlation = np.clip(correlation, -1, 1)
        matrix_rows = []
        for i, name in enumerate(names):
            matrix_rows.append({'feature': name, **{n: (float(correlation[i, j]) if np.isfinite(correlation[i, j]) else '') for j, n in enumerate(names)}})
            for j in range(i + 1, 43):
                r = correlation[i, j]
                row = dict(scope=scope, feature_a=name, feature_b=names[j], n_complete=len(x),
                           pearson_r=float(r) if np.isfinite(r) else '', abs_r=float(abs(r)) if np.isfinite(r) else '',
                           defined=bool(np.isfinite(r)), exact_constant_a=bool(constants[i]), exact_constant_b=bool(constants[j]))
                pairs.append(row)
                if np.isfinite(r) and abs(r) >= .95:
                    high.append(row)
        write_csv(out / ('correlation_matrix_' + scope + '.csv'), matrix_rows, ['feature'] + names)
        summary[scope] = dict(rows=len(x), cells=int(x.size), finite_cells=int(finite.sum()), nan_cells=int(np.isnan(x).sum()),
                              infinite_cells=int(np.isinf(x).sum()), exact_constant_features=[names[i] for i in range(43) if constants[i]],
                              std_below_scaler_floor_features=[names[i] for i in range(43) if np.sqrt(var[i]) < 1e-12],
                              pairs_total=903, defined_pairs=sum(v['defined'] for v in pairs if v['scope'] == scope),
                              high_abs_r_pairs=sum(v['scope'] == scope for v in high),
                              min_population_variance=float(var.min()), max_population_variance=float(var.max()))
    high.sort(key=lambda d: (d['scope'], -d['abs_r'], d['feature_a'], d['feature_b']))
    write_csv(out / 'feature_dictionary.csv', dictionary, list(dictionary[0]))
    write_csv(out / 'cohort.csv', cohort, list(cohort[0]))
    write_csv(out / 'profile_by_client_and_global.csv', profile, list(profile[0]))
    write_csv(out / 'pairwise_correlations.csv', pairs, list(pairs[0]))
    write_csv(out / 'high_correlation_pairs_abs_ge_0p95.csv', high, list(pairs[0]))
    authentication.update(status='PASS', group_counts={'CORE': 14, 'CLIMATE': 21, 'FULL': 43},
                          accessed_npz_members=accessed_members, label_array_members_read=[], evaluation_files_read=[],
                          source_csv_numeric_rows_read=0)
    write_json(out / 'authentication.json', authentication)
    pooled = data['GLOBAL_POOLED']
    pooled_var = pooled.var(axis=0, ddof=0)
    receipt = dict(status='PASS', created_utc=datetime.now(timezone.utc).isoformat(),
                   diagnostic='TRAIN latest-row feature quality and descriptive Pearson correlation only',
                   run_sha256=identity['run_sha256'], filter='raw_target_month <= 2007-11, applied before arithmetic',
                   representation='scaled_x[TRAIN mask, -1, :]; float32 prepared values promoted to float64 for diagnostics',
                   feature_count=43, group_counts={'CORE': 14, 'CLIMATE': 21, 'FULL': 43},
                   unique_months=299, clients=5, total_client_month_rows=1495, unique_feature_pairs_per_scope=903,
                   correlation_threshold={'abs_r_greater_equal': .95, 'fixed_descriptive_only': True},
                   variance_ddof=0, constant_flag='exact max-min == 0; separately report std < existing scaler floor 1e-12',
                   feature_selection_changed=False, fits_run=0, labels_read=False, evaluation_features_read=False,
                   privileged_offline_diagnostic=True, model_protocol_messages_added=0,
                   pooled_population_variance_max_abs_deviation_from_one=float(np.max(np.abs(pooled_var - 1))),
                   summary=summary, numpy_version=np.__version__, python_version=platform.python_version(),
                   caveats=[
                       'Pooled correlations mix regional distributions; the five clients share calendar/index observations and are not 1495 independent time observations.',
                       'Pearson correlation is descriptive and contemporaneous within eligible latest-row inputs, without p-values, causal claims or predictive selection.',
                       'This centralized offline TRAIN audit accesses client feature rows; it is not a core federated model message or included in training communication costs.',
                       'Raw units for the 22 meteorological channels are explicit in the canonical manifest; calendar/rain/climate units are derived from authenticated transformations and previously verified source definitions.',
                       'Prepared float32 encoding permits tiny deviations of global standardized variance from one.',
                       'No raw target, heavy-event label or evaluation payload member was loaded. Target-scaling statistics present in the authenticated schema were not used.'
                   ])
    write_json(out / 'diagnostic_receipt.json', receipt)
    global_high = [r for r in high if r['scope'] == 'GLOBAL_POOLED']
    lines = ['# TRAIN feature diagnostic', '',
             'Read-only diagnostic of active revision_v2; authentication PASS. No feature changes, fits, evaluation reads or label reads.', '',
             'Only the latest row of the 12-step sequence is analyzed after raw_target_month <= 2007-11: 299 rows per client, 1495 client-month rows.', '',
             'All analyzed channels are dimensionless global TRAIN z scores. The dictionary also records their original physical units and source-support basis.', '',
             '| Scope | Finite cells | Constant features | Pairs with abs(r) >= 0.95 |',
             '|---|---:|---:|---:|']
    lines += [f"| {scope} | {v['finite_cells']} | {len(v['exact_constant_features'])} | {v['high_abs_r_pairs']} |" for scope,v in summary.items()]
    lines += ['', '## Pooled high-correlation pairs', '', '| Feature A | Feature B | Pearson r |', '|---|---|---:|']
    lines += [f"| {r['feature_a']} | {r['feature_b']} | {r['pearson_r']:.8f} |" for r in global_high]
    lines += ['', '## All 43 channels and raw units', '', '| Index | Feature | Incremental group | Raw unit |', '|---:|---|---|---|']
    lines += [f"| {d['index']} | {d['feature']} | {d['incremental_group']} | {d['raw_physical_unit']} |" for d in dictionary]
    lines += ['', '## Interpretation limits', ''] + ['- ' + c for c in receipt['caveats']]
    lines += ['', 'The already frozen group-selection protocol is unchanged. These correlation descriptions do not measure forecasting utility.', '',
              'Reproduce using the supplied diagnose_train_features.py with --package-root, --run runs/revision_v2 and --output pointing to a separate owned audit directory.', '']
    (out / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    generated = {p.name: sha(p) for p in sorted(out.glob('*')) if p.is_file() and p.name != 'file_hashes.json'}
    write_json(out / 'file_hashes.json', generated)
    print(json.dumps({'status': 'PASS', 'run_sha256': identity['run_sha256'], 'summary': summary,
                      'pooled_high_pairs': global_high, 'output': str(out)}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
