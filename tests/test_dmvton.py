"""GPU 가중치 다운로드 없이 correlation, 좌표 복원, 크기 변화와 캐시 연결을 검증한다."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'app'))
sys.path.insert(0, str(ROOT / 'scripts'))

from dmvton_core import correlation
from dmvton_fit import (apply_size, compare_inputs, geometry, make_base, padded_crop,
                        prepare_garment, restore_crop, source_key)
from debug_dmvton import run_comparison


class TestDMVTON(unittest.TestCase):
    def setUp(self):
        self.pixels = np.full((400, 300, 3), 80, np.uint8)
        self.mask = np.zeros((400, 300), bool)
        self.mask[60:241, 100:201] = True
        self.pixels[self.mask] = (240, 20, 10)
        self.person = Image.fromarray(self.pixels)
        self.garment = Image.new('RGB', (192, 256), 'blue')
        self.protected = np.zeros_like(self.mask)
        self.protected[60:70, 140:151] = True
        self.reference = geometry(self.mask)
        self.cache = {'base': self.person, 'mask': self.mask, 'protected': self.protected,
                      'reference_geometry': self.reference, 'metadata': {'inference_s': .12}}
        self.rows = [['S', 60, 48, 45], ['M', 65, 50, 50], ['L', 70, 52, 55]]

    def test_correlation_matches_original_kernel_offset_order_channel_mean_and_zero_padding(self):
        import torch

        first = torch.arange(1, 25, dtype=torch.float32).reshape(1, 2, 3, 4)
        second = first.flip(-1)
        actual = correlation(first, second)
        expected = torch.zeros(1, 49, 3, 4)
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                for y in range(3):
                    for x in range(4):
                        if 0 <= y+dy < 3 and 0 <= x+dx < 4:
                            expected[0, (dy+3)*7+dx+3, y, x] = (
                                first[0, :, y, x]*second[0, :, y+dy, x+dx]).mean()
        torch.testing.assert_close(actual, expected)
        with self.assertRaises(ValueError):
            correlation(first, second, 2)

    def test_length_and_width_follow_ratios_while_protected_and_far_pixels_stay_identical(self):
        for length_ratio, chest_ratio in [(1., 1.), (.8, .9), (1.2, 1.1)]:
            with self.subTest(length_ratio=length_ratio):
                result, report = apply_size(self.person, self.mask, self.protected, self.reference,
                                            length_ratio, chest_ratio)
                self.assertAlmostEqual(report['target_hem_y'], 60+180*length_ratio)
                self.assertAlmostEqual(report['target_width_px'], 100*chest_ratio)
                self.assertTrue(np.array_equal(np.asarray(result)[self.protected], self.pixels[self.protected]))
                self.assertTrue(np.array_equal(np.asarray(result)[:, :40], self.pixels[:, :40]))
                self.assertFalse(report['shoulder_applied'])
        result, _ = apply_size(self.person, self.mask, np.zeros_like(self.mask), self.reference, 1, 1)
        self.assertTrue(np.array_equal(np.asarray(result), self.pixels))
        with self.assertRaises(ValueError):
            apply_size(self.person, self.mask, self.protected, self.reference, 4, 1)

    def test_crop_roundtrip_does_not_stretch_full_body_coordinates(self):
        box = (50, 30, 250, 300)
        canvas, transform = padded_crop(self.person, box)
        restored = restore_crop(canvas, transform, self.person.size)
        self.assertEqual(canvas.size, (192, 256))
        self.assertEqual(restored.size, self.person.size)
        self.assertEqual(restored.getpixel((150, 150)), (240, 20, 10))
        self.assertEqual(restored.getpixel((0, 0)), (0, 0, 0))

    def test_transparent_product_mask_needs_no_extra_segmentation_and_stays_aligned(self):
        rgba = Image.new('RGBA', (100, 150))
        rgba.paste((240, 30, 20, 255), (20, 25, 80, 125))
        parser = Mock()
        clothes, edge = prepare_garment(rgba, parser)
        parser.predict.assert_not_called()
        self.assertEqual(clothes.size, edge.size)
        self.assertGreater(np.asarray(edge).sum(), 0)
        self.assertNotEqual(source_key(self.person, rgba), source_key(self.person, self.garment))

    def test_automatic_base_uses_real_network_output_mask_and_preserves_faces_and_arms(self):
        labels = np.zeros((400, 300), np.uint8)
        labels[self.mask] = 3
        labels[self.protected] = 13
        parser = Mock()
        parser.predict.return_value = labels
        rgba = Image.new('RGBA', (100, 150))
        rgba.paste((40, 50, 200, 255), (20, 25, 80, 125))
        generated = Image.new('RGB', (192, 256), 'green')
        network_mask = Image.new('L', (192, 256), 255)
        with patch('dmvton_fit.get_parser', return_value=parser), \
                patch('dmvton_fit.infer', return_value=(generated, network_mask, {'inference_s': .1})) as engine:
            cache = make_base(self.person, rgba)
        engine.assert_called_once()
        self.assertEqual(cache['base'].size, self.person.size)
        self.assertFalse(cache['mask'][self.protected].any())
        self.assertTrue(np.array_equal(np.asarray(cache['base'])[self.protected], self.pixels[self.protected]))

    def test_size_edits_reuse_base_and_photo_edits_invalidate_it(self):
        with patch('debug_dmvton.make_base', return_value=self.cache.copy()) as engine:
            initial = list(run_comparison(self.person, 65, 50, 50, self.rows, self.garment))[-1]
            engine.assert_called_once()
            updated = list(run_comparison(self.person, 66, 50, 50, self.rows, self.garment, initial[0]))[-1]
            engine.assert_called_once()
            self.assertIn('재사용', updated[2])
            self.assertEqual(len(updated[1]), 4)
            self.assertNotEqual(initial[1][1][0].tobytes(), updated[1][1][0].tobytes())
            changed = self.person.copy()
            changed.putpixel((0, 0), (1, 2, 3))
            list(run_comparison(changed, 65, 50, 50, self.rows, self.garment, initial[0]))
            self.assertEqual(engine.call_count, 2)

    def test_invalid_chart_and_missing_photos_stop_before_models(self):
        for rows, person in [([['S', 60, None, 45]], self.person),
                             ([['S', 60, 48, 45], ['s', 65, 50, 50]], self.person),
                             (self.rows, None)]:
            with self.subTest(rows=rows), patch('debug_dmvton.make_base') as engine, \
                    patch('debug_dmvton.traceback.print_exc'):
                cache, images, status = list(run_comparison(person, 65, 50, 50, rows, self.garment))[-1]
                engine.assert_not_called()
                self.assertIsNone(cache)
                self.assertEqual(images, [])
                self.assertIn('못했습니다', status)


if __name__ == '__main__':
    unittest.main()
