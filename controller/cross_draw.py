import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'extension'), str(ROOT / 'controller')]
from direct_advantage import ROW_KEYS, forest, signed_targets
from policies import Features, LEAVES, MARGINS, SEED, digest, task_folds

DRAWS = (0, 1)
CANDIDATE_SETS = ('all', 'evidence')
EVIDENCE_WIDTH = 2


def draw_targets(outcomes):
    y = np.asarray(outcomes, dtype=float)
    signed_targets(y)
    return y[:, :, 1:] - y[:, :, :1]


def admissible(targets):
    ranked = np.argsort(-np.asarray(targets).mean(axis=(0, 1)))
    return sorted(int(index) for index in ranked[:EVIDENCE_WIDTH])


def choose(per_draw, margin, agreement=True, allowed=None):
    if allowed is not None:
        per_draw = np.where(np.asarray(allowed)[:, None, :], per_draw, -np.inf)
    selected = per_draw.argmax(axis=2)
    rows = np.arange(len(selected))
    crossed = np.stack([per_draw[rows, 1 - draw, selected[:, draw]] for draw in DRAWS], axis=1)
    winner = crossed.argmax(axis=1)
    index = selected[rows, winner]
    valued = np.where(selected[:, 0] == selected[:, 1], crossed.mean(axis=1), crossed[rows, winner])
    eligible = (selected[:, 0] == selected[:, 1]) if agreement else np.ones(len(rows), bool)
    return np.eye(per_draw.shape[2] + 1)[np.where(eligible & (valued > margin + 1e-12), index + 1, 0)]


def fold_predictions(rows, targets, leaf, splits):
    predicted = np.zeros((len(rows), len(DRAWS), targets.shape[2]))
    evidence = np.zeros((len(rows), targets.shape[2]), dtype=bool)
    for train, valid in splits:
        transform = Features(text=True).fit([rows[i] for i in train])
        train_x = transform.transform([rows[i] for i in train])
        valid_x = transform.transform([rows[i] for i in valid])
        for draw in DRAWS:
            predicted[valid, draw] = forest(train_x, targets[train, draw], leaf).predict(valid_x)
        evidence[np.ix_(valid, admissible(targets[train]))] = True
    return predicted, evidence


def grid_scores(rows, targets, utility, splits):
    scored = []
    for leaf in LEAVES:
        predicted, evidence = fold_predictions(rows, targets, leaf, splits)
        for margin in MARGINS:
            for agreement in (True, False):
                for candidates in CANDIDATE_SETS:
                    p = choose(predicted, margin, agreement, None if candidates == 'all' else evidence)
                    scored.append({'leaf': leaf, 'threshold': margin, 'agreement': agreement, 'candidates': candidates,
                                   'oof_utility': float((p * utility).sum(axis=1).mean()),
                                   'oof_firing_rate': float(1 - p[:, 0].mean())})
    return scored


def select_candidate(candidates):
    return max(candidates, key=lambda c: (round(c['oof_utility'], 12), -c['oof_firing_rate'], c['candidates'] == 'all', c['agreement'], c['leaf'], c['threshold']))


def fit_stratum(rows):
    y = np.asarray([row['Y'] for row in rows], dtype=float)
    targets, utility = draw_targets(y), y.mean(axis=1)
    splits = task_folds(rows)
    nested = []
    for train, valid in splits:
        inner_rows = [rows[i] for i in train]
        inner = select_candidate(grid_scores(inner_rows, draw_targets([r['Y'] for r in inner_rows]),
                                             np.asarray([r['Y'] for r in inner_rows], dtype=float).mean(axis=1),
                                             task_folds(inner_rows)))
        nested.append({'validation_task_ids': sorted({rows[i]['task_id'] for i in valid}), **inner})
    candidates = grid_scores(rows, targets, utility, splits)
    selected = select_candidate(candidates)
    transform = Features(text=True).fit(rows)
    matrix = transform.transform(rows)
    models = [forest(matrix, targets[:, draw], selected['leaf']) for draw in DRAWS]
    fitted = {'forests': models, 'transform': transform, 'contrasts': list(range(1, y.shape[2])),
              'admissible': admissible(targets), **selected}
    receipt = {'candidates': candidates, 'selected': selected, 'nested_selection': nested,
               'nested_honest_utility': float(np.mean([row['oof_utility'] for row in nested])),
               'target_sha256': digest(targets.tolist()), 'feature_schema': transform.schema()}
    return fitted, receipt


def fit(rows):
    if not rows or any(row['split'] != 'dev' for row in rows):
        raise ValueError('Selection accepts dev rows only')
    keys = [(row['model'], row['env'], row['task_id']) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate development checkpoint rows')
    rows = sorted(rows, key=lambda r: (r['model'], r['env'], r['task_id'], digest(r['prefix_only'])))
    bundle = {'strata': {}, 'training_tasks': sorted({(r['env'], r['task_id']) for r in rows}),
              'training_data_sha256': digest(rows)}
    receipt = {'dev_records': len(rows), 'dev_data_sha256': digest(rows), 'strata': [],
               'text': True, 'seed': SEED, 'trees': 200, 'draws': list(DRAWS), 'folds': 4}
    for key in sorted({(r['model'], r['env']) for r in rows}):
        subset = [r for r in rows if (r['model'], r['env']) == key]
        if len(subset) < 8:
            raise ValueError('Each development stratum needs at least eight tasks')
        bundle['strata'][key], record = fit_stratum(subset)
        receipt['strata'].append({'model': key[0], 'env': key[1], 'n_tasks': len(subset), **record})
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
    width = len(next(iter(bundle['strata'].values()))['contrasts'])
    p = np.zeros((len(safe), width + 1))
    estimates = np.zeros((len(safe), len(DRAWS), width))
    for key in sorted({(r['model'], r['env']) for r in safe}):
        indices = [i for i, r in enumerate(safe) if (r['model'], r['env']) == key]
        model = bundle['strata'][key]
        matrix = model['transform'].transform([safe[i] for i in indices])
        for draw in DRAWS:
            estimates[indices, draw] = model['forests'][draw].predict(matrix)
        allowed = None
        if model['candidates'] != 'all':
            allowed = np.zeros((len(indices), width), dtype=bool)
            allowed[:, model['admissible']] = True
        p[indices] = choose(estimates[indices], model['threshold'], model['agreement'], allowed)
    return {'rows': [{k: r[k] for k in ROW_KEYS} for r in safe], 'probabilities': p.tolist(),
            'predicted_draw_advantages': estimates.tolist(), 'training_data_sha256': bundle['training_data_sha256']}
