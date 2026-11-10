"""Calendar-keyed features, source gates and training-only transformations.

Every sequence row is an historical forecast origin. For origin s, rainfall
and monthly indices end at s-1, reanalysis/POWER end at s-3. Issue time is the
first day of target month s+1. These are conservative availability scenarios
applied to revised archives, not proof of historically available vintages.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd

CLIENTS = ('Client1_NW_Arid', 'Client2_Indo_Gangetic', 'Client3_NE_Himalaya',
           'Client4_South_Peninsular', 'Client5_Central_India')
INDEX_NAMES = ('nino34_anomaly', 'dmi_hadisst11_anomaly', 'soi_cpc_standardized',
               'oni_anomaly_3mo', 'meiv2_bimonthly',
               'amv_ersstv6_global_removed_1951_1980')

def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()

def checked_panel(path: str | Path, expected_hash: str, required: list[str]) -> pd.DataFrame:
    if sha256(path) != expected_hash: raise ValueError('Input checksum changed')
    df = pd.read_csv(path)
    if df.duplicated(['client', 'month']).any(): raise ValueError('Duplicate monthly key')
    expected = pd.MultiIndex.from_product([CLIENTS, pd.period_range('1981-01', '2024-12', freq='M')], names=['client','month'])
    df['month'] = pd.PeriodIndex(df['month'], freq='M')
    df = df.set_index(['client','month']).sort_index()
    if set(df.index) != set(expected): raise ValueError('Incomplete client/month calendar')
    if not np.isfinite(df[required].to_numpy(dtype=float)).all():
        raise ValueError('Required upstream data missing; structural gaps cannot be imputed')
    if not (df['rainfall_mm'] >= 0).all(): raise ValueError('Invalid target rainfall')
    return df.reindex(expected)

@dataclass(frozen=True)
class FeatureSet:
    names: tuple[str, ...]
    climate_names: tuple[str, ...]
    # Every feature's support is expressed relative to its sequence-row origin.
    support: dict[str, dict]
    met_columns: tuple[str, ...] = ()
    extra_lag: int = 0

def _value(series: pd.Series, month: pd.Period) -> float:
    return float(series.get(month, np.nan))

def build_rows(monthly: pd.DataFrame, indices: pd.DataFrame, met_columns: tuple[str, ...] = (),
               with_climate: bool = True, extra_lag: int = 0) -> tuple[dict[str, pd.DataFrame], FeatureSet]:
    """Build unscaled historical rows without inspecting target labels to select inputs."""
    if extra_lag not in (0,1): raise ValueError('Only prespecified availability sensitivities')
    if list(indices.columns).count('month') != 1: raise ValueError('Index calendar column required')
    idx = indices.copy()
    idx['month'] = pd.PeriodIndex(idx['month'], freq='M')
    if idx['month'].duplicated().any(): raise ValueError('Duplicate index support month')
    idx = idx.set_index('month')
    rows, support = {}, {}
    climate_names = tuple(INDEX_NAMES) + ('enso_x_iod',) if with_climate else ()
    for client in CLIENTS:
        data = monthly.xs(client).sort_index()
        result = []
        for origin in data.index:
            record = {'origin_month': origin}
            target = origin + 1
            for h in (1,2):
                record[f'season_sin{h}'] = np.sin(2*np.pi*h*target.month/12)
                record[f'season_cos{h}'] = np.cos(2*np.pi*h*target.month/12)
                for trig in ('sin','cos'): support[f'season_{trig}{h}'] = {'kind':'known_calendar','target_month_offset':1}
            rain = data['rainfall_mm']
            for lag in (1,2,5,11):
                name = f'rain_origin_lag{lag}'
                record[name] = _value(rain, origin-lag-extra_lag)
                support[name] = {'kind':'observed_month','latest_origin_offset':-lag-extra_lag,'source':'CHIRPS-v2-final'}
            for window in (3,6,12):
                dates = pd.period_range(origin-window-extra_lag, origin-1-extra_lag, freq='M')
                values = rain.reindex(dates).to_numpy(dtype=float)
                for stat in ('mean','std'):
                    name = f'rain_roll{window}_{stat}'
                    record[name] = (np.mean(values) if stat=='mean' else np.std(values,ddof=0)) if np.isfinite(values).all() else np.nan
                    support[name] = {'kind':'observed_window','first_origin_offset':-window-extra_lag,'latest_origin_offset':-1-extra_lag,'source':'CHIRPS-v2-final'}
            if with_climate:
                for name in INDEX_NAMES:
                    record[name] = _value(idx[name], origin-1-extra_lag)
                    support[name] = {'kind':'index_period_end','latest_origin_offset':-1-extra_lag,'source':name}
                record['enso_x_iod'] = record['nino34_anomaly'] * record['dmi_hadisst11_anomaly']
                support['enso_x_iod'] = {'kind':'interaction','latest_origin_offset':-1-extra_lag,'parents':['nino34_anomaly','dmi_hadisst11_anomaly']}
            for name in met_columns:
                record[name] = _value(data[name], origin-3-extra_lag)
                support[name] = {'kind':'archived_regional_month','latest_origin_offset':-3-extra_lag,'source':name}
            # Raw anomaly drivers are carried separately and never standardized for severity.
            record['_driver_enso'] = _value(idx['nino34_anomaly'], origin-1-extra_lag)
            record['_driver_iod'] = _value(idx['dmi_hadisst11_anomaly'], origin-1-extra_lag)
            result.append(record)
        rows[client] = pd.DataFrame(result).set_index('origin_month')
    names = tuple(n for n in rows[CLIENTS[0]].columns if not n.startswith('_'))
    return rows, FeatureSet(names, climate_names, support, met_columns, extra_lag)

def samples(monthly: pd.DataFrame, rows: dict[str, pd.DataFrame], feature_set: FeatureSet,
            length: int = 12, start_target: str = '1983-01',
            target_months: tuple[str,...] | None = None,
            expected_evaluation_months: tuple[str,...] | None = None) -> dict[str, dict]:
    """One shared, finite cohort. No silent missing-prediction/label fallback."""
    if length != 12: raise ValueError('Sequence length was frozen at twelve')
    if target_months is None:
        targets = pd.period_range(start_target, '2024-12', freq='M')
        if expected_evaluation_months is not None:raise ValueError('A supplementary cohort requires explicit target keys')
    else:
        if expected_evaluation_months is None:raise ValueError('Explicit supplementary evaluation cohort required')
        targets=pd.PeriodIndex(target_months,freq='M')
        if targets.has_duplicates or not targets.is_monotonic_increasing:raise ValueError('Supplement target keys must be unique and ordered')
        if (targets<pd.Period(start_target,freq='M')).any() or (targets>pd.Period('2024-12',freq='M')).any():raise ValueError('Supplement target outside declared calendar')
    output = {}
    for client in CLIENTS:
        X, C, D, y, origins = [], [], [], [], []
        row = rows[client]
        target_series = monthly.xs(client)['rainfall_mm']
        for target in targets:
            origin = target-1
            window = pd.period_range(origin-length+1, origin, freq='M')
            frame = row.reindex(window)
            x = frame[list(feature_set.names)].to_numpy(dtype=np.float64)
            climate = frame.iloc[-1][list(feature_set.climate_names)].to_numpy(dtype=np.float64)
            drivers = frame[['_driver_enso','_driver_iod']].to_numpy(dtype=np.float64)
            value = _value(target_series, target)
            if not (np.isfinite(x).all() and np.isfinite(climate).all() and np.isfinite(drivers).all() and np.isfinite(value)):
                raise ValueError(f'{client}/{target}: unavailable observed feature/label; inspect source/support gate')
            X.append(x); C.append(climate); D.append(drivers); y.append(value); origins.append(str(origin))
        output[client] = {'x': np.stack(X), 'climate': np.stack(C), 'drivers': np.stack(D),
                          'y': np.array(y), 'target_month': np.array(targets.astype(str)),
                          'origin_month': np.array(origins)}
    expected_test = np.array(pd.period_range('2017-01','2024-12',freq='M').astype(str) if expected_evaluation_months is None else expected_evaluation_months)
    for data in output.values():
        test = data['target_month'] >= '2017-01'
        if not np.array_equal(data['target_month'][test], expected_test): raise ValueError('Evaluation cohort changed')
    return output

def split_roles(keys: np.ndarray) -> dict[str, np.ndarray]:
    # Final CHIRPS month labels are assumed published by day21 of the next
    # month. The last December label is unavailable at a January1 issue.
    # Embargo role-end December labels, including checkpoint/router selection.
    return {'train': keys <= '2007-11',
            'monitor': (keys >= '2008-01') & (keys <= '2010-11'),
            'validation': (keys >= '2011-01') & (keys <= '2016-11'),
            'evaluation': (keys >= '2017-01') & (keys <= '2024-12')}

def train_transform(data: dict[str, dict], feature_set: FeatureSet) -> tuple[dict[str, dict], dict]:
    """Global input scaling via count/sum/squared-sum; local training target scaling."""
    sums, squares, count = None, None, 0
    for d in data.values():
        roles = split_roles(d['target_month'])
        # One origin row/sample rather than counting replicated sequence cells unequally.
        x = d['x'][roles['train'], -1, :]
        if not np.isfinite(x).all(): raise ValueError('Nonfinite gradient-training covariates')
        sums = x.sum(0) if sums is None else sums + x.sum(0)
        squares = (x*x).sum(0) if squares is None else squares + (x*x).sum(0)
        count += len(x)
    mean = sums/count
    std = np.sqrt(np.maximum(squares/count-mean*mean,0))
    std[std < 1e-12] = 1.0
    c_idx = [feature_set.names.index(n) for n in feature_set.climate_names]
    transformed, stats = {}, {'global_input_mean':mean.tolist(), 'global_input_std':std.tolist(),
                              'count':count,'source_role':'gradient_train','protocol':'count/sum/squared-sum scalar vectors','clients':{}}
    feature_count = len(feature_set.names)
    stats['setup_communication'] = {
        'wire_format':'int64 count; float64 sum/squared-sum and broadcast mean/std',
        'messages':[{'sender':client,'receiver':'server','kind':'input_scaler_statistics',
                     'bytes':8+16*feature_count} for client in data]
                   + [{'sender':'server','receiver':client,'kind':'input_scaler_mean_std',
                       'bytes':16*feature_count} for client in data],
        'total_bytes':len(data)*(8+32*feature_count),
        'privacy_assumption':'Plain training summary sharing; no differential privacy or secure aggregation claimed. Local target scaling and heavy thresholds remain at clients; reported evaluation metrics reveal aggregated diagnostics.'}
    for client,d in data.items():
        roles = split_roles(d['target_month'])
        tr_y = d['y'][roles['train']]
        ym, ys = float(tr_y.mean()), float(tr_y.std(ddof=0))
        if ys < 1e-12: raise ValueError('Degenerate training target')
        threshold = float(np.quantile(tr_y,.8,method='linear'))
        result = dict(d)
        result['x'] = ((d['x']-mean)/std).astype(np.float32)
        result['climate'] = ((d['climate']-mean[c_idx])/std[c_idx]).astype(np.float32)
        result['y_z'] = ((d['y']-ym)/ys).astype(np.float32)
        result['heavy'] = (d['y'] > threshold).astype(np.float32)
        result['roles'] = roles
        result['target_mean'], result['target_std'] = ym,ys
        transformed[client] = result
        stats['clients'][client] = {'target_mean':ym,'target_std':ys,'heavy80_threshold_mm':threshold,
                                    'role_counts':{k:int(v.sum()) for k,v in roles.items()}}
    return transformed,stats
