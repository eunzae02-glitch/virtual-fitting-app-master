"""옷 단면·몸 둘레의 구분, 예측 범위 전파, 입력 변경과 결과 저장을 검증한다."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'app'))
sys.path.insert(0, str(ROOT / 'scripts'))

from garment_compare import (compare_body, compare_reference, parse_target_chart,
                             report_markdown)  # noqa: E402
from reference_garment import ReferenceGarment  # noqa: E402
from debug_size_compare import (MODES, clear_result, compare_inputs, save_result)  # noqa: E402


def body_predictions():
    return {key: {'label': label, 'value_cm': value, 'low_cm': low, 'high_cm': high,
                  'model': 'ridge', 'source': 'excel_body_model'}
            for key, label, value, low, high in [
                ('shoulder_breadth_cm', '어깨사이너비', 40, 38, 42),
                ('body_length_cm', '신체 기준 총장', 54, 50, 58),
                ('chest_circumference_cm', '가슴둘레', 96, 90, 102)]}


class TestGarmentCompare(unittest.TestCase):
    def test_sales_labels_order_and_flat_width_are_preserved(self):
        chart = parse_target_chart([['100', 70.5, 48, 54.25], ['95', 68, 46, 52],
                                    ['', None, None, None]], '일반 어깨선')
        self.assertEqual([size.size for size in chart.sizes], ['100', '95'])
        self.assertEqual(chart.sizes[0].chest_flat_cm, 54.25)
        self.assertEqual(chart.to_dict()['chest_measurement'], 'flat_width')

    def test_invalid_rows_are_rejected_without_silently_dropping_a_size(self):
        invalid = [[], [['S', None, 46, 52]], [[None, 68, 46, 52]],
                   [['S', 68, 46]], [['S', 68, 46, 52], [' s ', 70, 48, 54]],
                   [['S\nM', 68, 46, 52]]]
        invalid += [[['S', 68, 46, value]] for value in
                    (True, 0, -1, float('nan'), float('inf'), '52cm')]
        for rows in invalid:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                parse_target_chart(rows, '일반 어깨선')

    def test_raglan_missing_shoulder_does_not_create_a_number(self):
        chart = parse_target_chart([['M', 70, None, 54]], '래글런')
        self.assertIsNone(chart.sizes[0].shoulder_cm)
        result = compare_reference(chart, ReferenceGarment(70, 48, 54), '일반 어깨선')
        self.assertIsNone(result['rows'][0]['shoulder_difference_cm'])
        with self.assertRaises(ValueError):
            parse_target_chart([['M', 70, None, 54]], '일반 어깨선')

    def test_reference_differences_use_new_minus_reference_and_never_body_ease(self):
        chart = parse_target_chart([['S', 68, 46, 52], ['M', 70, 48, 54],
                                    ['L', 72, 50, 56]], '일반 어깨선')
        result = compare_reference(chart, ReferenceGarment(70, 48, 54), '일반 어깨선')
        self.assertEqual(result['rows'][0], {'size': 'S', 'length_difference_cm': -2,
                         'shoulder_difference_cm': -2, 'chest_flat_difference_cm': -2})
        self.assertEqual(result['rows'][1]['length_difference_cm'], 0)
        self.assertEqual(result['rows'][2]['chest_flat_difference_cm'], 2)
        self.assertNotIn('chest_circumference_ease_cm', result['rows'][0])
        self.assertNotIn('recommended_size', result)

    def test_shoulder_difference_is_withheld_for_drop_or_unknown_structure(self):
        for style in ('드롭숄더', '래글런', '확인 필요'):
            with self.subTest(style=style):
                chart = parse_target_chart([['M', 70, 50, 54]], style)
                result = compare_reference(chart, ReferenceGarment(70, 48, 54), '일반 어깨선')
                self.assertIsNone(result['rows'][0]['shoulder_difference_cm'])
                self.assertEqual(result['target_chart']['sizes'][0]['shoulder_cm'], 50)

    def test_body_ranges_reverse_subtraction_and_distinguish_half_from_full_ease(self):
        chart = parse_target_chart([['M', 70, 48, 54]], '일반 어깨선')
        result = compare_body(chart, body_predictions(), .9)
        row = result['rows'][0]
        self.assertEqual(row['length_difference_cm'], {'value_cm': 16, 'low_cm': 12, 'high_cm': 20})
        self.assertEqual(row['chest_circumference_ease_cm'],
                         {'value_cm': 12, 'low_cm': 6, 'high_cm': 18})
        self.assertEqual(row['chest_flat_ease_cm'], {'value_cm': 6, 'low_cm': 3, 'high_cm': 9})
        self.assertIsNone(row['shoulder_difference_cm'])
        report = report_markdown(result)
        self.assertIn('+16.0 (+12.0~+20.0)', report)
        self.assertIn('목뒤높이', report)
        self.assertIn('90%', report)

    def test_a_range_crossing_zero_is_kept_and_does_not_trigger_a_size_verdict(self):
        chart = parse_target_chart([['S', 54, 40, 48]])
        row = compare_body(chart, body_predictions(), .9)['rows'][0]
        self.assertEqual(row['chest_flat_ease_cm'], {'value_cm': 0, 'low_cm': -3, 'high_cm': 3})
        self.assertEqual(row['length_difference_cm'], {'value_cm': 0, 'low_cm': -4, 'high_cm': 4})

    def test_reference_mode_works_without_a_saved_body_model(self):
        with patch('debug_size_compare.load_model', side_effect=AssertionError('must not load model')):
            result, report = compare_inputs(MODES[0], 70, 48, 54, '일반 어깨선', None, None, None,
                                            '일반 어깨선', [['M', 70, 48, 54]])
        self.assertEqual(result['mode'], 'reference_garment')
        self.assertIn('직접 입력한 기존 옷', report)

    def test_body_mode_uses_ridge_and_preserves_input_provenance(self):
        bundle = {'report': {'coverage_target': .9}}
        with patch('debug_size_compare.load_model', return_value=bundle), \
                patch('debug_size_compare.predict_body', return_value=body_predictions()) as predictor:
            result, _ = compare_inputs(MODES[1], None, None, None, '확인 필요', '남', 173, 63,
                                       '일반 어깨선', [['M', 70, 48, 54]], ROOT / 'model.joblib')
        predictor.assert_called_once_with(bundle, '남', 173, 63, model='ridge')
        self.assertEqual(result['source']['inputs'], {'sex': 'M', 'height_cm': 173, 'weight_kg': 63})

    def test_invalid_or_out_of_training_range_input_clears_old_results(self):
        with patch('debug_size_compare.load_model', return_value={}), \
                patch('debug_size_compare.predict_body', side_effect=ValueError('학습 자료 범위 밖입니다.')):
            result, report = compare_inputs(MODES[1], 70, 48, 54, '확인 필요', '남', 250, 63,
                                            '일반 어깨선', [['M', 70, 48, 54]], ROOT / 'model.joblib')
        self.assertIsNone(result)
        self.assertIn('범위 밖', report)
        self.assertEqual(clear_result()[0], None)
        self.assertEqual(clear_result()[2:], ([], ''))

    def test_saved_result_preserves_sources_and_never_overwrites_a_previous_run(self):
        chart = parse_target_chart([['M', 70, 48, 54]])
        result = compare_reference(chart, ReferenceGarment(68, 46, 52))
        with tempfile.TemporaryDirectory() as directory:
            first, _ = save_result(result, directory)
            second, _ = save_result(result, directory)
            self.assertNotEqual(first, second)
            data = json.loads(Path(first[0]).read_text(encoding='utf-8'))
            self.assertEqual(data['source']['source'], 'manual_reference_garment')
            self.assertEqual(data['target_chart']['sizes'][0]['chest_flat_cm'], 54)
            self.assertTrue(Path(first[1]).exists())
        self.assertEqual(save_result(None)[0], [])


if __name__ == '__main__':
    unittest.main()
