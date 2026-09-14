import csv
import json
import math
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import binom


def exact_power(n, delta, discordance, alpha):
    if not isinstance(n, int) or n < 1 or not 0 < discordance <= 1 or abs(delta) > discordance or not 0 < alpha < 1:
        raise ValueError('Invalid paired binary design parameters')
    d = np.arange(n + 1)
    lower = binom.ppf(alpha / 2, d, .5).astype(int)
    lower -= binom.cdf(lower, d, .5) > alpha / 2 + 1e-15
    favorable = (discordance + delta) / (2 * discordance)
    reject = binom.cdf(lower, d, favorable) + binom.sf(d - lower - 1, d, favorable)
    return float(binom.pmf(d, n, discordance) @ reject)


def sample_size(delta, discordance, alpha, target):
    low, high = 1, 100
    while exact_power(high, delta, discordance, alpha) < target:
        high *= 2
    while low < high:
        middle = (low + high) // 2
        if exact_power(middle, delta, discordance, alpha) >= target:
            high = middle
        else:
            low = middle + 1
    return {'n_analyzable_tasks': high, 'power_at_n': exact_power(high, delta, discordance, alpha),
            'power_at_n_minus_1': exact_power(high - 1, delta, discordance, alpha),
            'n_enroll_10pct_unusable': math.ceil(high / .9)}


def run(directory):
    results = []
    for comparisons, alpha in [(1, .05), (4, .0125)]:
        for delta in [.03, .05, .08]:
            for discordance in [.1, .2, .4, .6]:
                row = {'comparisons': comparisons, 'alpha_two_sided': alpha, 'delta': delta,
                       'discordance': discordance, 'favorable_discordance': (discordance + delta) / 2,
                       'harmful_discordance': (discordance - delta) / 2}
                for target in [.8, .9]:
                    row[f'target_{target}'] = sample_size(delta, discordance, alpha, target)
                row['power_curve'] = {str(n): exact_power(n, delta, discordance, alpha)
                                      for n in [50, 100, 200, 500, 1000]}
                results.append(row)
    output = {'method': 'Exact unconditional summation of the two-sided conditional McNemar/binomial rejection probability',
              'scipy': scipy.__version__, 'results': results,
              'assumptions': 'One paired binary outcome per independent fresh task; fixed candidate; q is total discordance; effects are scenarios, not observed estimates; no interim testing',
              'sample_size_search': 'Integer binary search under the smooth unconditional power curve; n and n-1 powers verified, no claim of exhaustive global minimum',
              'scope': 'Future design sensitivity; no GPU-hours or post-hoc observed power'}
    (directory / 'power.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    fields = ['comparisons', 'alpha', 'delta_pp', 'discordance', 'n80', 'n90', 'n80_enroll_10pct', 'n90_enroll_10pct']
    with (directory / 'power.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for row in results:
            a, b = row['target_0.8'], row['target_0.9']
            writer.writerow([row['comparisons'], row['alpha_two_sided'], row['delta'] * 100,
                             row['discordance'], a['n_analyzable_tasks'], b['n_analyzable_tasks'],
                             a['n_enroll_10pct_unusable'], b['n_enroll_10pct_unusable']])
    return output


if __name__ == '__main__':
    run(Path(__file__).resolve().parent)
