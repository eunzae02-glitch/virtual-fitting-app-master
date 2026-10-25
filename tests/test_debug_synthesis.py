"""샘플 사진·신체 모델 없이 기본 합성을 호출하고 오류를 표시하는지 확인한다."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

import debug_synthesis


class TestDebugSynthesis(unittest.TestCase):
    def setUp(self):
        self.photo = Image.new('RGB', (120, 160), 'blue')

    def test_missing_photos_and_invalid_settings_do_not_load_models(self):
        for person, garment, seed, steps, guidance in [
                (None, self.photo, 42, 50, 2.5), (self.photo, None, 42, 50, 2.5),
                (self.photo, self.photo, -1, 50, 2.5),
                (self.photo, self.photo, 42, .5, 2.5),
                (self.photo, self.photo, 42, 50, float('nan'))]:
            with self.subTest(seed=seed, steps=steps), \
                    patch.object(debug_synthesis, 'check_environment') as environment, \
                    patch.object(debug_synthesis, 'load_models') as load, \
                    patch.object(debug_synthesis, 'try_on') as engine, \
                    patch.object(debug_synthesis.traceback, 'print_exc'):
                result, status = list(debug_synthesis.run_synthesis(
                    person, garment, seed, steps, guidance))[-1]
                self.assertIsNone(result)
                self.assertIn('실패', status)
                environment.assert_not_called()
                load.assert_not_called()
                engine.assert_not_called()

    def test_direct_synthesis_needs_only_photos_and_reports_loading_and_inference(self):
        result = Image.new('RGB', (120, 160), 'red')
        progress = unittest.mock.Mock()
        with patch.object(debug_synthesis, 'check_environment', return_value='Test GPU'), \
                patch.object(debug_synthesis, 'load_models') as load, \
                patch.object(debug_synthesis, 'try_on',
                             return_value=(result, {'total_s': 1.5})) as engine:
            updates = list(debug_synthesis.run_synthesis(
                self.photo, self.photo, 17, 30, 2.5, progress))
        load.assert_called_once()
        engine.assert_called_once()
        self.assertEqual(engine.call_args.kwargs, {
            'cloth_type': 'upper', 'seed': 17, 'steps': 30, 'guidance_scale': 2.5,
            'garment_photo_type': 'flat-lay', 'return_timing': True})
        self.assertTrue(any('모델을 준비' in status for _, status in updates))
        self.assertTrue(any('합성 중' in status for _, status in updates))
        self.assertIs(updates[-1][0], result)
        self.assertIn('1.5초', updates[-1][1])
        self.assertIn('Test GPU', updates[-1][1])
        progress.assert_called_with(1, desc='합성 완료')

    def test_gpu_or_inference_failure_clears_result_and_reports_phase(self):
        for fail_loading in (True, False):
            with self.subTest(fail_loading=fail_loading), \
                    patch.object(debug_synthesis, 'check_environment', return_value='Test GPU'), \
                    patch.object(debug_synthesis, 'load_models',
                                 side_effect=RuntimeError('load failed') if fail_loading else None), \
                    patch.object(debug_synthesis, 'try_on', side_effect=RuntimeError('out of memory')), \
                    patch.object(debug_synthesis.traceback, 'print_exc'):
                result, status = list(debug_synthesis.run_synthesis(self.photo, self.photo))[-1]
                self.assertIsNone(result)
                self.assertIn('모델 준비 실패' if fail_loading else '상의 합성 실패', status)


if __name__ == '__main__':
    unittest.main()
