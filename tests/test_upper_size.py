"""상의 추천 신규 경로의 사진/단위/보정/경계 테스트 (GPU 불필요)."""
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'app'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from chart_ocr import draft_from_text, image_to_draft  # noqa: E402
from upper_size import (RangeCM, UpperBody, estimate_upper, fit_chest_calibration,
                        parse_upper_chart, recommend_upper)  # noqa: E402
from debug_upper_size import analyze  # noqa: E402


TARGET = {'name': '새 반팔', 'unit': 'cm', 'sizes': {
    'S': {'가슴단면': 49, '어깨너비': 43},
    'M': {'가슴단면': 52, '어깨너비': 45},
    'L': {'가슴단면': 55, '어깨너비': 47},
}}


def fake_measure(chest=47, unsafe=False, segmentation=True):
    values = {'shoulder': 45}
    if chest is not None:
        values['chest'] = chest
    return SimpleNamespace(measurements_cm=values, unsafe=unsafe,
                           has_segmentation=segmentation, warnings=['사진 조건 확인'])


class TestPhotoPath(unittest.TestCase):
    def test_cropped_photo_requests_reupload(self):
        with self.assertRaisesRegex(ValueError, '다시 올려'):
            estimate_upper(object(), 173, measurement_fn=lambda *_: fake_measure(unsafe=True))

    def test_no_segmentation_requests_reupload(self):
        with self.assertRaisesRegex(ValueError, '다시 올려'):
            estimate_upper(object(), 173, measurement_fn=lambda *_: fake_measure(segmentation=False))

    def test_photo_chest_is_circumference_not_flat(self):
        body = estimate_upper(object(), 173, measurement_fn=lambda *_: fake_measure(47))
        self.assertEqual(body.chest_circ.middle, 94)

    def test_unclear_chest_without_calibration_requires_input(self):
        body = estimate_upper(object(), 173, basis='남', measurement_fn=lambda *_: fake_measure(None))
        self.assertIsNone(body.chest_circ)
        self.assertIn('평균 자료', ' '.join(body.notes))

    def test_manual_chest_is_used_if_photo_chest_missing(self):
        body = estimate_upper(object(), 173, manual_chest=91,
                              measurement_fn=lambda *_: fake_measure(None))
        self.assertEqual(body.chest_circ.middle, 91)

    def test_wrong_height_reports_input_error(self):
        with self.assertRaisesRegex(ValueError, '키 입력'):
            estimate_upper(object(), 17.3, measurement_fn=lambda *_: fake_measure())


class TestCharts(unittest.TestCase):
    def test_flat_and_confirmed_garment_circumference(self):
        chart = parse_upper_chart({'sizes': {'M': {'옷 가슴둘레': 104}}})
        self.assertEqual(chart.chart.sizes['M']['chest'], 52)

    def test_ambiguous_chest_circumference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, '완성 둘레'):
            parse_upper_chart({'sizes': {'M': {'가슴둘레': 104}}})

    def test_recommended_body_circumference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, '권장 신체'):
            parse_upper_chart({'sizes': {'M': {'권장 신체 가슴둘레': 96}}})

    def test_all_sizes_need_chest(self):
        with self.assertRaisesRegex(ValueError, '가슴단면'):
            parse_upper_chart({'sizes': {'S': {'가슴단면': 49}, 'M': {'어깨너비': 45}}})

    def test_reversed_size_numbers_are_rejected(self):
        with self.assertRaisesRegex(ValueError, '수정'):
            parse_upper_chart({'sizes': {'S': {'가슴단면': 52}, 'M': {'가슴단면': 49}}})

    def test_confirmed_oversize_measure_is_kept_with_warning(self):
        chart = parse_upper_chart({'sizes': {'M': {'가슴단면': 80}}})
        self.assertEqual(chart.chart.sizes['M']['chest'], 80)
        self.assertTrue(chart.warnings)


class TestRecommendation(unittest.TestCase):
    def test_photo_only_picks_middle_size(self):
        body = UpperBody(RangeCM(45, 45, '사진'), RangeCM(94, 94, '사진'))
        result = recommend_upper(body, parse_upper_chart(TARGET))
        self.assertEqual(result.candidates, ['M'])
        self.assertEqual(result.status, '잠정 추천')

    def test_interval_can_yield_multiple_candidates(self):
        body = UpperBody(RangeCM(45, 45, '사진'), RangeCM(86, 104, '보정'))
        result = recommend_upper(body, parse_upper_chart(TARGET))
        self.assertGreater(len(result.candidates), 1)

    def test_reference_shirt_comparison_when_chest_unknown(self):
        body = UpperBody(RangeCM(45, 45, '사진'), None)
        reference = parse_upper_chart({'sizes': {'M': {'가슴단면': 52}}})
        result = recommend_upper(body, parse_upper_chart(TARGET), reference, 'M', '적당함')
        self.assertEqual(result.candidates, ['M'])
        self.assertIn('착용 가능 여부는 미확인', ' '.join(result.notes))

    def test_bad_reference_fit_moves_up(self):
        body = UpperBody(RangeCM(45, 45, '사진'), RangeCM(94, 94, '직접 입력'))
        reference = parse_upper_chart({'sizes': {'M': {'가슴단면': 52}}})
        result = recommend_upper(body, parse_upper_chart(TARGET), reference, 'M', '작음')
        self.assertEqual(result.candidates, ['L'])

    def test_special_shoulder_is_excluded(self):
        body = UpperBody(RangeCM(45, 45, '사진'), RangeCM(94, 94, '직접 입력'))
        result = recommend_upper(body, parse_upper_chart(TARGET, False))
        self.assertTrue(all(row[3] is None for row in result.rows))


class TestCalibration(unittest.TestCase):
    def test_missing_csv_does_not_invent_average(self):
        self.assertIsNone(fit_chest_calibration(None, '남', 45))

    def test_small_sample_is_not_used(self):
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='',
                                         suffix='.csv', delete=False) as stream:
            path = stream.name
            stream.write('basis,shoulder_cm,chest_circ_cm\n남,45,94\n')
        try:
            self.assertIsNone(fit_chest_calibration(path, '남', 45))
        finally:
            os.unlink(path)

    def test_paired_data_creates_bounded_average(self):
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='',
                                         suffix='.csv', delete=False) as stream:
            path = stream.name
            stream.write('basis,shoulder_cm,chest_circ_cm\n')
            for i in range(12):
                stream.write(f'남,45,{90 + i}\n')
        try:
            estimate = fit_chest_calibration(path, '남', 45)
            self.assertLess(estimate.low, estimate.high)
            self.assertTrue(estimate.low < 96 < estimate.high)
            self.assertIsNone(fit_chest_calibration(path, '녀', 45))
        finally:
            os.unlink(path)


class TestOCRAndUI(unittest.TestCase):
    def test_row_table_draft(self):
        draft = draft_from_text('사이즈 가슴단면 어깨너비\nS 49 43\nM 52 45\nL 55 47')
        self.assertEqual(draft['sizes']['M']['가슴단면'], 52)

    def test_column_table_draft(self):
        draft = draft_from_text('구분 S M L\n가슴단면 49 52 55\n어깨너비 43 45 47')
        self.assertEqual(draft['sizes']['L']['가슴단면'], 55)

    def test_missing_ocr_engine_gives_manual_fallback(self):
        with patch('chart_ocr.shutil.which', return_value=None):
            draft, text, status = image_to_draft(object())
        self.assertEqual(draft, '')
        self.assertIn('직접 입력', status)

    def test_web_handler_requires_target_confirmation(self):
        with patch('debug_upper_size.estimate_upper', return_value=UpperBody(
                RangeCM(45, 45, '사진'), RangeCM(94, 94, '사진'))):
            report, table = analyze(object(), 173, '선택 안 함', None, None,
                                    '', False, '', '적당함', json.dumps(TARGET), False,
                                    False, None, None)
        self.assertIn('확인', report)
        self.assertEqual(table, '')


if __name__ == '__main__':
    unittest.main()
