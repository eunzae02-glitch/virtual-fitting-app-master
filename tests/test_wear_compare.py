"""실착 비교의 단위·정렬·확인 상태와 누락 치수 처리를 검사한다."""
import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.fspath(ROOT / 'app'))
sys.path.insert(0, os.fspath(ROOT / 'scripts'))

from wear_compare import (WearCase, confirm_marks, normalized_outline,
                          numeric_report, observation_report, outline_metrics,
                          put_point, ratio_hypothesis, read_cases)  # noqa: E402
from debug_wear_compare import export_comparison  # noqa: E402


def example_marks():
    return {'case': 'A', 'structure': '일반 어깨선', 'confirmed': False, 'source': 'test',
            'points': {'eye_left': [.40, .10], 'eye_right': [.50, .10], 'neck': [.45, .20],
                       'shoulder_left': [.25, .23], 'shoulder_right': [.65, .23],
                       'chest_left': [.28, .36], 'chest_right': [.62, .36],
                       'hem_left': [.29, .55], 'hem_right': [.61, .55]}}


def example_cases():
    return (WearCase('A', Path('a.jpg'), 'M', 173, 63, '반팔/L', 65, 51, 52),
            WearCase('B', Path('b.jpg'), 'M', 173, 63, '반팔/L', 63, 52, 56))


def example_profile():
    return {key: {'value_cm': value, 'low_cm': value-radius, 'high_cm': value+radius}
            for key, value, radius in [('shoulder_breadth_cm', 39.4, 2.5),
                                       ('body_length_cm', 53.5, 3.5),
                                       ('chest_circumference_cm', 95, 5)]}


class TestWearCompare(unittest.TestCase):
    def test_cm_comparison_keeps_body_reference_and_full_circumference_ease(self):
        a, b = example_cases()
        with patch('wear_compare.predict_body', return_value=example_profile()):
            report = numeric_report({}, a, b)
        self.assertIn('+11.5 (+8.0~+15.0)', report)
        self.assertIn('+9.0 (+4.0~+14.0)', report)
        self.assertIn('총장 -2cm', report)
        self.assertIn('가슴단면 +4cm', report)
        self.assertIn('굴곡·처짐 보정 전', report)

    def test_loader_preserves_missing_shoulder_and_product_units(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for key in ['A', 'B']:
                Image.new('RGB', (30, 50)).save(path / f'{key}.jpg')
            block = ('사진이름: {name}\n성별: 남\n키(cm): 173\n체중(kg): 63\n'
                     '옷 총장(cm): 65\n옷 어깨너비(cm): -\n옷 가슴단면(cm): 52\n')
            (path / 'case.txt').write_text(block.format(name='A') + block.format(name='B'),
                                          encoding='utf-8')
            cases = read_cases(path)
            self.assertIsNone(cases['A'].shoulder_cm)
            self.assertEqual(cases['A'].chest_flat_cm, 52)
            self.assertEqual(cases['A'].length_cm, 65)

    def test_translation_rotation_and_scale_do_not_change_observation(self):
        marks = example_marks()
        original = normalized_outline(marks, (1000, 1000))
        # 2배 확대하고 90도 회전하지 않는 소각도 회전 및 평행이동. 캔버스도 충분히 크게 둔다.
        import math
        cosine, sine = math.cos(.08), math.sin(.08)
        transformed = copy.deepcopy(marks)
        transformed['points'] = {
            key: [(2*(x*cosine-y*sine)+.10)/2,
                  (2*(x*sine+y*cosine)+.10)/2]
            for key, (x, y) in marks['points'].items()}
        actual = normalized_outline(transformed, (2000, 2000))
        for key in original:
            self.assertAlmostEqual(original[key][0], actual[key][0], places=8)
            self.assertAlmostEqual(original[key][1], actual[key][1], places=8)

    def test_ratio_hypothesis_keeps_neck_and_changes_only_requested_dimensions(self):
        a, b = example_cases()
        base = normalized_outline(example_marks(), (1000, 1000))
        result = ratio_hypothesis(base, a, b, True)
        original, changed = outline_metrics(base), outline_metrics(result)
        self.assertEqual(result['neck'], base['neck'])
        self.assertAlmostEqual(changed['length']/original['length'], 63/65)
        self.assertAlmostEqual(changed['chest']/original['chest'], 56/52)
        self.assertAlmostEqual(changed['shoulder']/original['shoulder'], 52/51)
        self.assertEqual(result['chest_left'][1], base['chest_left'][1])

    def test_raglan_skips_shoulder_without_inventing_missing_values(self):
        a, _ = example_cases()
        b = WearCase('B', Path('b.jpg'), 'M', 173, 63, '래글런/M', 68.5, None, 57)
        marks = example_marks()
        marks['structure'] = '래글런'
        del marks['points']['shoulder_left'], marks['points']['shoulder_right']
        confirmed = confirm_marks(marks, (1000, 1000))
        result = ratio_hypothesis(normalized_outline(confirmed, (1000, 1000)), a, b, False)
        self.assertNotIn('shoulder_left', result)
        report = observation_report(a, b, confirmed, confirmed, (1000, 1000), (1000, 1000), True)
        self.assertIn('비교를 보류', report)

    def test_edit_revokes_confirmation_and_drafts_are_not_observed(self):
        marks = confirm_marks(example_marks(), (1000, 1000))
        edited = put_point(marks, 'hem_left', 290, 560, (1000, 1000))
        self.assertFalse(edited['confirmed'])
        self.assertTrue(marks['confirmed'])
        a, b = example_cases()
        report = observation_report(a, b, marks, edited, (1000, 1000), (1000, 1000), True)
        self.assertIn('미확인 초안', report)
        self.assertNotIn('사진 관측 변화', report)

    def test_invalid_points_do_not_produce_physical_measurements(self):
        marks = example_marks()
        marks['points']['eye_right'] = marks['points']['eye_left']
        with self.assertRaises(ValueError):
            confirm_marks(marks, (1000, 1000))
        with self.assertRaises(ValueError):
            put_point(example_marks(), 'neck', 1000, 10, (1000, 1000))

    def test_export_preserves_draft_status_and_original_photos(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = {}
            for name in ['A', 'B']:
                image = root / f'{name}.jpg'
                Image.new('RGB', (100, 100)).save(image)
                cases[name] = WearCase(name, image, 'M', 173, 63, '반팔', 65, 51, 52)
            before = cases['A'].image.read_bytes()
            a, b = example_marks(), example_marks()
            b['case'] = 'B'
            with patch('wear_compare.predict_body', return_value=example_profile()):
                output = export_comparison(cases, {n: Image.new('RGB', (100, 100)) for n in cases},
                                           {}, root / 'model.joblib', 'A', 'B', a, b, False,
                                           output_root=root / 'exports')
            saved = json.loads((output / 'annotations.json').read_text(encoding='utf-8'))
            self.assertFalse(saved['annotations']['A']['confirmed'])
            self.assertFalse(saved['same_person_confirmed'])
            self.assertEqual(saved['length_definition'], '목뒤높이 - 위앞엉덩뼈가시높이')
            self.assertEqual(cases['A'].image.read_bytes(), before)
            self.assertTrue((output / 'comparison.html').is_file())


if __name__ == '__main__':
    unittest.main()
