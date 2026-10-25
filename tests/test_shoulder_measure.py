"""어깨 단독 경로의 계산과 재업로드 판정을 MediaPipe 실행 없이 검사한다."""
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'app'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from body_measure import POSE_LANDMARKS  # noqa: E402
from shoulder_measure import measure_shoulder  # noqa: E402
from debug_shoulder import analyze_shoulder  # noqa: E402


class Landmark:
    def __init__(self, x, y, visibility=1.0):
        self.x = x
        self.y = y
        self.visibility = visibility


def sample():
    photo = np.zeros((1200, 800, 3), dtype=np.uint8)
    mask = np.zeros((1200, 800), dtype=np.float32)
    mask[100:1100, 300:500] = 1.0
    points = [Landmark(0.5, 0.2) for _ in range(33)]
    coords = {
        'nose': (0.5, 0.13),
        'left_shoulder': (0.3875, 0.28),
        'right_shoulder': (0.6125, 0.28),
        'left_hip': (0.41, 0.55),
        'right_hip': (0.59, 0.55),
        'left_ankle': (0.47, 0.89),
        'right_ankle': (0.53, 0.89),
    }
    for name, (x, y) in coords.items():
        points[POSE_LANDMARKS[name]] = Landmark(x, y)
    return photo, points, mask


class TestShoulderMeasure(unittest.TestCase):
    def test_valid_measurement_matches_existing_formula(self):
        photo, points, mask = sample()
        _, result = measure_shoulder(photo, 175, points, mask)
        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.shoulder_landmark_px, 180, places=1)
        self.assertEqual(result.shoulder_cm, round(180 / (999 / 175) * 1.28, 1))
        self.assertEqual((result.head_y, result.foot_y), (100, 1099))

    def test_height_changes_scale_not_landmark_distance(self):
        photo, points, mask = sample()
        _, shorter = measure_shoulder(photo, 160, points, mask)
        _, taller = measure_shoulder(photo, 190, points, mask)
        self.assertTrue(shorter.valid)
        self.assertTrue(taller.valid)
        self.assertLess(shorter.shoulder_cm, taller.shoulder_cm)
        self.assertEqual(shorter.shoulder_landmark_px, taller.shoulder_landmark_px)

    def test_cropped_head_requests_new_photo(self):
        photo, points, mask = sample()
        mask[:100, 300:500] = 1.0
        _, result = measure_shoulder(photo, 175, points, mask)
        self.assertFalse(result.valid)
        self.assertIsNone(result.shoulder_cm)
        self.assertTrue(any('머리' in note for note in result.warnings))

    def test_missing_segmentation_does_not_use_ratio_fallback(self):
        photo, points, _ = sample()
        _, result = measure_shoulder(photo, 175, points, None)
        self.assertFalse(result.valid)
        self.assertIsNone(result.shoulder_cm)

    def test_low_landmark_visibility_rejects_measurement(self):
        photo, points, mask = sample()
        points[POSE_LANDMARKS['left_shoulder']].visibility = 0.2
        _, result = measure_shoulder(photo, 175, points, mask)
        self.assertFalse(result.valid)
        self.assertIsNone(result.shoulder_cm)

    def test_low_ankle_visibility_allows_silhouette_based_measurement(self):
        for names in (('left_ankle',), ('left_ankle', 'right_ankle'), ('nose',)):
            with self.subTest(names=names):
                photo, points, mask = sample()
                for name in names:
                    points[POSE_LANDMARKS[name]].visibility = 0.2
                _, result = measure_shoulder(photo, 175, points, mask)
                self.assertTrue(result.valid)
                self.assertTrue(result.notes)
                self.assertFalse(result.warnings)

    def test_low_ankle_visibility_does_not_allow_cropped_photo(self):
        photo, points, mask = sample()
        points[POSE_LANDMARKS['left_ankle']].visibility = 0.2
        mask[1100:, 300:500] = 1.0
        _, result = measure_shoulder(photo, 175, points, mask)
        self.assertFalse(result.valid)
        self.assertIsNone(result.shoulder_cm)
        self.assertTrue(any('발' in note for note in result.warnings))

    def test_report_shows_estimate_and_note_for_low_ankle_visibility(self):
        photo, points, mask = sample()
        points[POSE_LANDMARKS['left_ankle']].visibility = 0.2
        with patch('debug_shoulder.measure_shoulder',
                   return_value=measure_shoulder(photo, 175, points, mask)):
            _, report = analyze_shoulder(photo, 175, 42)
        self.assertIn('추정 어깨너비', report)
        self.assertIn('오차(사진 추정', report)
        self.assertIn('> 참고:', report)
        self.assertNotIn('측정 보류', report)

    def test_no_detected_person_requests_new_photo(self):
        photo, _, _ = sample()
        with patch('shoulder_measure._run_pose', return_value=(None, None)):
            _, result = measure_shoulder(photo, 175)
        self.assertFalse(result.valid)
        self.assertIn('사람', result.warnings[0])

    def test_invalid_height_is_not_used(self):
        photo, points, mask = sample()
        with self.assertRaises(ValueError):
            measure_shoulder(photo, 0, points, mask)

    def test_report_does_not_claim_measurement_on_bad_photo(self):
        photo, points, mask = sample()
        mask[:100, 300:500] = 1.0
        with patch('debug_shoulder.measure_shoulder',
                   return_value=measure_shoulder(photo, 175, points, mask)):
            _, report = analyze_shoulder(photo, 175, 42)
        self.assertIn('측정 보류', report)
        self.assertIn('오차 미계산', report)
        self.assertNotIn('오차(사진 추정', report)


if __name__ == '__main__':
    unittest.main()
