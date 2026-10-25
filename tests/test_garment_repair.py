"""복원에만 한정한 통제 실험. 실제 착용 신체나 바지의 복원 정확도 검증은 아니다."""
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/'app'))

from garment_repair import legacy_telea_plate, repair_exposed


class TestGarmentRepair(unittest.TestCase):
    def setUp(self):
        h, w = 260, 220
        yy, xx = np.mgrid[:h, :w]
        self.pants = np.stack((20+(xx//7 % 2)*24, 35+(xx//7 % 2)*28,
                               60+(xx//7 % 2)*36), axis=2).astype(np.uint8)
        self.rgb = np.full((h, w, 3), (190, 170, 110), dtype=np.uint8)
        self.rgb[181:] = self.pants[181:]
        self.rgb[60:181, 55:70] = (200, 145, 110)
        self.rgb[60:181, 151:166] = (200, 145, 110)
        self.source = (xx >= 70) & (xx <= 150) & (yy >= 40) & (yy <= 180)
        self.rgb[self.source] = (150, 150, 150)
        self.base = Image.fromarray(self.rgb)
        self.target = self.source & (yy <= 140)
        self.bottom = yy > 140
        self.frame = ([110, 40], [1, 0], [0, 1])
        self.points = {'neck': [110, 40], 'chest_left': [70, 95], 'chest_right': [150, 95],
                       'hem_left': [70, 180], 'hem_right': [150, 180]}

    def repair(self, person=None, points=None, target=None, bottom=None, reference_mask=None):
        points = points or self.points
        if reference_mask is None:
            yy = np.indices(self.source.shape)[0]
            reference_mask = self.source & (yy <= max(points['hem_left'][1], points['hem_right'][1]))
        return repair_exposed(person or self.base, self.base, self.source,
                              self.target if target is None else target, self.frame,
                              points, self.bottom if bottom is None else bottom, reference_mask)

    def test_lower_hole_uses_lower_texture_instead_of_diagonal_skin_and_background_smear(self):
        fixed, reused, estimated, report = self.repair()
        region = self.source & ~self.target
        old = legacy_telea_plate(self.rgb, self.source)
        old_mae = float(np.abs(old.astype(float)-self.pants)[region].mean())
        new_mae = float(np.abs(fixed.astype(float)-self.pants)[region].mean())
        self.assertGreater(old_mae, 5)
        self.assertEqual(new_mae, 0)
        self.assertFalse(reused.any())  # 원본에도 옷이 덮여 있던 부분이다.
        self.assertTrue(np.array_equal(estimated, region))
        self.assertFalse(report['is_ground_truth_reconstruction'])
        self.assertTrue(np.array_equal(fixed[~region], self.rgb[~region]))

    def test_visible_original_below_reference_hem_is_reused_exactly(self):
        original = self.rgb.copy()
        original[151:] = self.pants[151:]
        # 원본에 보이는 고유 무늬를 기본 합성이 가렸다고 가정한다.
        original[165:175, 90:120] = (70, 100, 150)
        points = dict(self.points, hem_left=[70, 150], hem_right=[150, 150])
        fixed, reused, estimated, report = self.repair(Image.fromarray(original), points)
        self.assertGreater(int(reused.sum()), 0)
        self.assertTrue(np.array_equal(fixed[reused], original[reused]))
        self.assertFalse(reused[:154].any())  # 원본 옷/밑단 경계는 복사하지 않는다.
        self.assertTrue(np.array_equal(reused | estimated, self.source & ~self.target))
        self.assertFalse((reused & estimated).any())
        self.assertEqual(report['reference_gate'], 'nearby_visible_region_agrees')

    def test_original_old_shirt_and_mismatched_visible_region_are_not_copied(self):
        original = self.rgb.copy()
        original[self.source] = (250, 0, 0)
        fixed, reused, _, _ = self.repair(Image.fromarray(original))
        self.assertFalse(reused.any())
        self.assertFalse(((fixed[:, :, 0] > 240) & (fixed[:, :, 1] < 10)).any())
        points = dict(self.points, hem_left=[70, 150], hem_right=[150, 150])
        original[151:] = (245, 5, 245)
        fixed, reused, estimated, report = self.repair(Image.fromarray(original), points)
        self.assertFalse(reused.any())
        self.assertTrue(estimated.any())
        self.assertEqual(report['reference_gate'], 'appearance_mismatch_do_not_copy')

    def test_curved_old_hem_below_anchor_points_is_excluded_and_missing_mask_disables_copy(self):
        original = self.rgb.copy()
        original[151:] = self.pants[151:]
        old_mask = self.source.copy()
        yy, xx = np.indices(self.source.shape)
        old_mask &= (yy <= 150) | ((abs(xx-110) < 15) & (yy <= 170))
        original[old_mask] = (250, 0, 0)
        points = dict(self.points, hem_left=[70, 150], hem_right=[150, 150])
        fixed, reused, _, _ = self.repair(Image.fromarray(original), points, reference_mask=old_mask)
        self.assertFalse((reused & old_mask).any())
        self.assertFalse(((fixed[:, :, 0] > 240) & (fixed[:, :, 1] < 10)).any())
        _, reused, _, report = repair_exposed(Image.fromarray(original), self.base, self.source,
                                               self.target, self.frame, points, self.bottom)
        self.assertFalse(reused.any())
        self.assertEqual(report['reference_gate'], 'reference_clothing_mask_unavailable')

    def test_side_gaps_sample_outward_and_do_not_change_known_skin(self):
        target = self.source.copy()
        target[:, :80] = False
        target[:, 141:] = False
        fixed, _, _, _ = self.repair(target=target, bottom=np.zeros(self.source.shape, bool))
        self.assertEqual(fixed[100, 75].tolist(), [200, 145, 110])
        self.assertEqual(fixed[100, 145].tolist(), [200, 145, 110])
        self.assertTrue(np.array_equal(fixed[~self.source], self.rgb[~self.source]))

    def test_no_repair_does_not_build_a_plate_or_touch_pixels(self):
        from unittest.mock import patch

        with patch('garment_repair._directional_plate') as plate:
            fixed, reused, estimated, report = self.repair(target=self.source)
        plate.assert_not_called()
        self.assertTrue(np.array_equal(fixed, self.rgb))
        self.assertFalse((reused | estimated).any())
        self.assertEqual(report['reference_gate'], 'not_needed')

    def test_rotation_keeps_non_repair_pixels_and_donor_colors(self):
        import cv2

        angle = 12
        matrix = cv2.getRotationMatrix2D((110, 130), angle, 1)
        base = cv2.warpAffine(self.rgb, matrix, (220, 260), borderValue=(190, 170, 110))
        mask = lambda value: cv2.warpAffine(value.astype(np.uint8), matrix, (220, 260),
                                           flags=cv2.INTER_NEAREST).astype(bool)
        point = lambda value: (matrix @ np.array([*value, 1])).tolist()
        radians = np.radians(angle)
        frame = (point(self.frame[0]), [np.cos(radians), -np.sin(radians)],
                 [np.sin(radians), np.cos(radians)])
        points = {key: point(value) for key, value in self.points.items()}
        source, target = mask(self.source), mask(self.target)
        fixed, reused, estimated, _ = repair_exposed(Image.fromarray(base), Image.fromarray(base),
                                                     source, target, frame, points, mask(self.bottom))
        self.assertTrue(np.array_equal(fixed[~(source & ~target)], base[~(source & ~target)]))
        self.assertTrue(np.array_equal(reused | estimated, source & ~target))
        self.assertGreater(float(fixed[source & ~target].mean()), 0)

    def test_full_mask_and_shape_mismatch_are_rejected_not_filled_black(self):
        with self.assertRaisesRegex(ValueError, '참고 영역'):
            repair_exposed(self.base, self.base, np.ones(self.source.shape, bool), self.target,
                            self.frame, self.points, self.bottom)
        with self.assertRaisesRegex(ValueError, '마스크'):
            repair_exposed(self.base, self.base, self.source[:10], self.target,
                            self.frame, self.points, self.bottom)
        with self.assertRaisesRegex(ValueError, '크기'):
            self.repair(self.base.resize((110, 130)))


if __name__ == '__main__':
    unittest.main()
