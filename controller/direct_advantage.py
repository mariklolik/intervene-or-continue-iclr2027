import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'extension'))
from analysis import diagnostics
from policies import Features, LEAVES, MARGINS, SEED, decisions, digest, task_folds

ROW_KEYS = ['model', 'env', 'split', 'task_id']


def signed_targets(outcomes):
    y = np.asarray(outcomes, dtype=float)
    diagnostics(y)
    return (y[:, :, 1:] - y[:, :, :1]).mean(axis=1)


def choose(advantages, margin):
    advantages = np.asarray(advantages, dtype=float)
    mean = np.column_stack([np.zeros(len(advantages)), advantages])
    return decisions(mean, np.zeros_like(mean), 'DIRECT_ADVANTAGE', margin, None)


def select_candidate(candidates):
    return max(candidates, key=lambda c: (round(c['oof_utility'], 12), -c['oof_firing_rate'],
                                          c['mode'] == 'CONTINUE', c['leaf'] or 0, c['threshold'] or 0))


def forest(x, target, leaf):
    return RandomForestRegressor(n_estimators=200, min_samples_leaf=leaf,
                                 random_state=SEED, n_jobs=1).fit(x, target)


def fit_stratum(rows):
    y = np.asarray([row['Y'] for row in rows], dtype=float)
    target = signed_targets(y)
    prepared, folds = [], []
    for train, valid in task_folds(rows):
        train_rows, valid_rows = [rows[i] for i in train], [rows[i] for i in valid]
        transform = Features(text=True).fit(train_rows)
        prepared.append((train, valid, transform.transform(train_rows), transform.transform(valid_rows)))
        folds.append({'train_task_ids': sorted({r['task_id'] for r in train_rows}),
                      'validation_task_ids': sorted({r['task_id'] for r in valid_rows}), **transform.schema()})
    candidates = [{'mode': 'CONTINUE', 'leaf': None, 'threshold': None,
                   'oof_utility': float(y[:, :, 0].mean()), 'oof_firing_rate': 0.}]
    for leaf in LEAVES:
        predicted = np.zeros_like(target)
        for train, valid, train_x, valid_x in prepared:
            predicted[valid] = forest(train_x, target[train], leaf).predict(valid_x)
        for margin in MARGINS:
            p = choose(predicted, margin)
            candidates.append({'mode': 'DIRECT_ADVANTAGE', 'leaf': leaf, 'threshold': margin,
                               'oof_utility': float((p * y.mean(axis=1)).sum(axis=1).mean()),
                               'oof_firing_rate': float(1 - p[:, 0].mean())})
    selected = select_candidate(candidates)
    transform = Features(text=True).fit(rows)
    model = None if selected['mode'] == 'CONTINUE' else forest(transform.transform(rows), target, selected['leaf'])
    fitted = {'forest': model, 'transform': transform, 'contrasts': list(range(1, y.shape[2])), **selected}
    receipt = {'folds': folds, 'candidates': candidates, 'selected': selected,
               'target_sha256': digest(target.tolist()), 'feature_schema': transform.schema()}
    return fitted, receipt


def fit(rows):
    if not rows or any(row['split'] != 'dev' for row in rows):
        raise ValueError('Selection accepts dev rows only')
    keys = [(row['model'], row['env'], row['task_id']) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate development checkpoint rows')
    rows = sorted(rows, key=lambda r: (r['model'], r['env'], r['task_id'], digest(r['prefix_only'])))
    signed_targets([r['Y'] for r in rows])
    bundle = {'strata': {}, 'training_tasks': sorted({(r['env'], r['task_id']) for r in rows}),
              'training_data_sha256': digest(rows)}
    receipt = {'dev_records': len(rows), 'dev_data_sha256': digest(rows), 'strata': [],
               'text': True, 'seed': SEED, 'trees': 200, 'n_jobs': 1, 'folds': 4}
    for key in sorted({(r['model'], r['env']) for r in rows}):
        subset = [r for r in rows if (r['model'], r['env']) == key]
        if len(subset) < 8:
            raise ValueError('Each development stratum needs at least eight tasks')
        bundle['strata'][key], record = fit_stratum(subset)
        receipt['strata'].append({'model': key[0], 'env': key[1], 'n_tasks': len(subset),
                                   'task_ids': [r['task_id'] for r in subset], **record})
    return bundle, receipt


def predict(bundle, rows):
    safe = [{k: row[k] for k in ROW_KEYS + ['prefix_only']} for row in rows]
    keys = [(r['model'], r['env'], r['task_id']) for r in safe]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate prediction rows')
    training = set(map(tuple, bundle['training_tasks']))
    for row in safe:
        if row['split'] not in {'test', 'holdout'} or (row['env'], row['task_id']) in training:
            raise ValueError('Holdout/dev task overlap or invalid split')
        if (row['model'], row['env']) not in bundle['strata']:
            raise ValueError('Untrained actor/environment stratum')
    width = len(next(iter(bundle['strata'].values()))['contrasts']) + 1
    p, estimates = np.zeros((len(safe), width)), np.zeros((len(safe), width - 1))
    for key in sorted({(r['model'], r['env']) for r in safe}):
        indices = [i for i, r in enumerate(safe) if (r['model'], r['env']) == key]
        model = bundle['strata'][key]
        if model['mode'] == 'CONTINUE':
            p[indices, 0] = 1
        else:
            estimates[indices] = model['forest'].predict(model['transform'].transform([safe[i] for i in indices]))
            p[indices] = choose(estimates[indices], model['threshold'])
    return {'rows': [{k: r[k] for k in ROW_KEYS} for r in safe], 'probabilities': p.tolist(),
            'predicted_signed_advantages': estimates.tolist(), 'training_data_sha256': bundle['training_data_sha256']}
