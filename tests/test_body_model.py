"""엑셀 단위·ID, 자료 누출 방지, 구간 보정, 모델 저장/예측 계약을 검사한다."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.fspath(ROOT / 'app'))
sys.path.insert(0, os.fspath(ROOT / 'scripts'))

from body_model import (BodyDataset, FEATURES, HEADERS, TARGETS, conformal_radius,
                        fit_candidates, load_dataset, load_model, predict_body,
                        split_people, train_models)  # noqa: E402
from debug_body_model import analyze  # noqa: E402


def synthetic_dataset():
    rng = np.random.default_rng(10)
    x = np.column_stack((rng.uniform(155, 185, 120), rng.uniform(50, 95, 120)))
    y = {'shoulder_breadth_cm': 20 + 0.1 * x[:, 0],
         'body_length_cm': 0.3 * x[:, 0],
         'chest_circumference_cm': 60 + 0.5 * x[:, 1]}
    return BodyDataset(np.array([f'person-{index:03}' for index in range(120)]),
                       np.array(['M', 'F'] * 60), x, y,
                       {'accepted_rows': 120, 'dropped_rows': 0})


def fit_mean_only(x, y, seed=42, folds=5):
    from sklearn.dummy import DummyRegressor
    return {'mean': {'estimator': DummyRegressor().fit(x, y), 'cv_mae_cm': 1.0,
                     'cv_mae_std_cm': 0.0, 'parameters': {}}}, 'mean'


class TestDataset(unittest.TestCase):
    def write_workbook(self, directory, rows, headers=None):
        from openpyxl import Workbook
        path = Path(directory) / 'fixture.xlsx'
        book = Workbook()
        book.active.title = '전처리'
        book.active.append(['전처리 자료'])
        book.active.append(headers or [name + ' ' for name in HEADERS])
        for row in rows:
            book.active.append(row)
        book.save(path)
        book.close()
        return path

    def test_headers_cm_values_and_targets_are_preserved(self):
        # HEADERS 순서: ID, 성별, 키, 몸무게, 총장, 어깨사이너비, 가슴둘레.
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_workbook(directory, [['ID-1', 'M', 175, 70, 52, 41, 95]])
            dataset = load_dataset(path)
        self.assertEqual(dataset.x.tolist(), [[175, 70]])
        self.assertEqual(dataset.y['chest_circumference_cm'].tolist(), [95])
        self.assertEqual(dataset.y['body_length_cm'].tolist(), [52])
        self.assertEqual(dataset.x.shape[1], len(FEATURES))
        self.assertEqual(len(dataset.audit['source_sha256']), 64)

    def test_invalid_rows_are_audited_not_used_as_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_workbook(directory, [
                ['good', 'M', 175, 70, 52, 41, 95],
                ['missing', 'F', 165, 55, None, 37, 85],
                ['code', 'F', 165, 55, 50, 9999, 85],
            ])
            dataset = load_dataset(path)
        self.assertEqual(dataset.ids.tolist(), ['good'])
        self.assertEqual(dataset.audit['dropped_rows'], 2)

    def test_duplicate_people_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_workbook(directory, [
                ['same', 'M', 175, 70, 52, 41, 95],
                ['same', 'M', 175, 70, 52, 41, 95],
            ])
            with self.assertRaisesRegex(ValueError, '중복 HUMAN_ID'):
                load_dataset(path)

    def test_unconfirmed_units_are_rejected(self):
        headers = list(HEADERS)
        headers[2] = '키(mm)'
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_workbook(directory, [], headers)
            with self.assertRaisesRegex(ValueError, '헤더·단위'):
                load_dataset(path)


class TestModelProtocol(unittest.TestCase):
    def test_splits_are_repeatable_and_people_do_not_overlap(self):
        dataset = synthetic_dataset()
        first, second = split_people(dataset), split_people(dataset)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        sets = [set(dataset.ids[first[key]]) for key in first]
        self.assertFalse(sets[0] & sets[1] or sets[1] & sets[2] or sets[0] & sets[2])
        self.assertEqual(set.union(*sets), set(dataset.ids))

    def test_conformal_quantile_uses_finite_sample_correction(self):
        self.assertEqual(conformal_radius(np.arange(1, 10), np.zeros(9)), 9)
        with self.assertRaisesRegex(ValueError, '부족'):
            conformal_radius([1, 2], [0, 0])

    def test_regression_models_improve_on_mean_for_linear_signal(self):
        rng = np.random.default_rng(5)
        x = rng.uniform(0, 100, (100, 2))
        y = 2 * x[:, 0] + 0.5 * x[:, 1]
        candidates, selected = fit_candidates(x, y, folds=3)
        self.assertEqual(set(candidates), {'mean', 'ridge', 'gradient_boosting'})
        self.assertNotEqual(selected, 'mean')
        self.assertLess(candidates[selected]['cv_mae_cm'], candidates['mean']['cv_mae_cm'] / 2)

    def test_calibration_and_test_targets_cannot_change_fitted_model(self):
        dataset = synthetic_dataset()
        split = split_people(dataset)
        with patch('body_model.fit_candidates', side_effect=fit_mean_only):
            original, _, _ = train_models(dataset)
            for key in TARGETS:
                dataset.y[key][split['calibration']] += 100
                dataset.y[key][split['test']] += 200
            changed, _, _ = train_models(dataset)
        for sex in ('M', 'F'):
            for key in TARGETS:
                first = original['models'][sex][key]['candidates']['mean']
                second = changed['models'][sex][key]['candidates']['mean']
                np.testing.assert_array_equal(first['estimator'].constant_, second['estimator'].constant_)
                self.assertNotEqual(first['interval_radius_cm'], second['interval_radius_cm'])

    def test_saved_bundle_preserves_predictions_and_rejects_bad_input(self):
        import joblib
        with patch('body_model.fit_candidates', side_effect=fit_mean_only):
            bundle, _, _ = train_models(synthetic_dataset())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.joblib'
            joblib.dump(bundle, path)
            restored = load_model(path)
        self.assertEqual(predict_body(bundle, '남', 170, 70),
                         predict_body(restored, 'M', 170, 70))
        for sex, height, weight in ((None, 170, 70), ('M', float('nan'), 70),
                                    ('M', 170, 0), ('M', 250, 70)):
            with self.subTest(sex=sex, height=height, weight=weight):
                with self.assertRaises(ValueError):
                    predict_body(restored, sex, height, weight)
        self.assertIn('예측을 보류', analyze(restored, '남', 250, 70))
        report = analyze(restored, '남', 170, 70)
        self.assertIn('가슴둘레', report)
        self.assertIn('옷의 총장과 구분', report)


if __name__ == '__main__':
    unittest.main()
