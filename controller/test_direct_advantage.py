import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'research/intervene-sota/extension'))
sys.path.insert(0, str(Path(__file__).resolve().parent))


class ProtectedRow(dict):
    def __getitem__(self, key):
        if key in {'Y', 'cells'}:
            raise AssertionError('Target or cost field was read')
        return super().__getitem__(key)


def synthetic_rows():
    return [{'model': 'actor', 'env': 'env', 'split': 'dev', 'task_id': str(i),
             'model_revision': 'revision', 'scaffold': 'scaffold',
             'prefix_only': {'features': {'step': i, f'unique_{i}': 1},
                             'recent_text': f'room common token{i}'},
             'Y': [[0, 1, 0, 0], [0, 1, 0, 0]]} for i in range(8)]


class DirectAdvantageTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('direct_advantage'),
                             'Signed advantage implementation must exist')
        import direct_advantage
        return direct_advantage

    def test_signed_targets_preserve_negative_and_fractional_differences(self):
        module = self.module()
        y = [[[1, 0, 1, 0], [0, 1, 0, 0]], [[1, 0, 0, 1], [1, 0, 1, 0]]]
        np.testing.assert_array_equal(module.signed_targets(y), [[0, 0, -.5], [-1, -.5, -.5]])

    def test_abstention_wins_threshold_and_near_ties(self):
        module = self.module()
        advantages = [[.10, .10, -.1], [.10 + 5e-13, 0, 0], [.2, .2, .1], [-.1, -.2, -.3]]
        np.testing.assert_array_equal(module.choose(advantages, .1).argmax(axis=1), [0, 0, 1, 0])

    def test_constant_continue_beats_harmful_grid_and_tied_intervention(self):
        module = self.module()
        candidates = [{'leaf': 10, 'threshold': .1, 'oof_utility': .2, 'oof_firing_rate': .5,
                       'mode': 'DIRECT_ADVANTAGE'},
                      {'leaf': None, 'threshold': None, 'oof_utility': .2, 'oof_firing_rate': 0.,
                       'mode': 'CONTINUE'}]
        self.assertEqual(module.select_candidate(candidates)['mode'], 'CONTINUE')

    def test_non_dev_rejected_before_targets_can_be_read(self):
        module = self.module()
        rows = [ProtectedRow({**row, 'split': 'test'}) for row in synthetic_rows()]
        with self.assertRaisesRegex(ValueError, 'dev'):
            module.fit(rows)

    def test_duplicate_checkpoint_rows_rejected(self):
        module = self.module()
        rows = synthetic_rows()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            module.fit(rows + [rows[0]])

    def test_folds_and_feature_fits_are_task_isolated(self):
        module = self.module()
        bundle, receipt = module.fit(synthetic_rows())
        folds = receipt['strata'][0]['folds']
        self.assertEqual(sorted(t for f in folds for t in f['validation_task_ids']), list('01234567'))
        for fold in folds:
            self.assertTrue(set(fold['train_task_ids']).isdisjoint(fold['validation_task_ids']))
            for task in fold['validation_task_ids']:
                self.assertNotIn(f'unique_{task}', fold['numeric_features'])
                self.assertNotIn(f'token{task}', fold['text_vocabulary'])
        self.assertEqual(receipt['strata'][0]['selected']['mode'], 'DIRECT_ADVANTAGE')
        rows = [ProtectedRow({**synthetic_rows()[0], 'split': 'test', 'task_id': 'unseen'})]
        np.testing.assert_array_equal(module.predict(bundle, rows)['probabilities'], [[0, 1, 0, 0]])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            module.predict(bundle, [ProtectedRow({**synthetic_rows()[0], 'split': 'test'})])

    def test_invalid_targets_fail(self):
        module = self.module()
        with self.assertRaises(ValueError):
            module.signed_targets([[[0, .5, 1, 1], [0, 1, 1, 1]]])


if __name__ == '__main__':
    unittest.main()
