"""생성 사진 좌표 연결, 실제 마스크 기반 변형, 바깥 화소 보존과 재현 저장을 검증한다."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/'app'))
sys.path.insert(0, str(ROOT/'scripts'))

from debug_size_preview import CHART_SOURCES, compare_dimensions
from fit_plan import create_fit_plan
from garment_outline import confirm_anchors, create_preview, new_marks, photo_info
from size_synthesis import (adjust_size, bind_base, checked_preview, draft_mask, editor_mask,
                            editor_value, engine_settings, generate_base)
from size_synthesis_ui import mask_editor_payload, save_synthesis


class TestSizeSynthesis(unittest.TestCase):
    def setUp(self):
        self.person = Image.new('RGB', (300, 400), (80, 80, 80))
        self.photo = photo_info(self.person)
        self.marks = confirm_anchors(new_marks(self.photo, {
            'neck': [.5, .10], 'chest_left': [1/3, .25], 'chest_right': [2/3, .25],
            'hem_left': [1/3, .55], 'hem_right': [2/3, .55]}), self.photo)
        self.comparison = compare_dimensions(65, 51, 52, '확인 필요', '확인 필요',
                                            [['S', 55, 49, 46], ['M', 65, 51, 52], ['L', 75, 53, 58]],
                                            CHART_SOURCES[1])
        # 생성 모델이 목표 상품 치수와 무관한 폭/밑단을 만들었다고 가정한다.
        self.base = self.person.copy()
        ImageDraw.Draw(self.base).rectangle((110, 60, 190, 210), fill=(240, 40, 20))
        ImageDraw.Draw(self.base).rectangle((130, 20, 170, 35), fill=(20, 200, 80))
        self.mask = Image.new('L', self.base.size)
        ImageDraw.Draw(self.mask).rectangle((110, 60, 190, 210), fill=255)
        self.base, self.meta = bind_base(self.person, self.base, 'external_unverified')

    def preview(self, size='M', length=True, chest=True):
        return create_preview(create_fit_plan(self.comparison, size), self.photo, self.marks,
                              True, length, chest)

    def apply(self, size='M', **kwargs):
        return adjust_size(self.person, self.base, self.meta, self.preview(size, **kwargs), self.mask, True)

    def test_generated_geometry_is_read_from_mask_not_assumed_reference_ratio(self):
        result = self.apply()
        data = result['report']
        self.assertEqual(data['source_geometry_px']['hem_y'], 170)  # generated 210 - neck 40
        self.assertAlmostEqual(data['target_geometry_px']['hem_y'], 180)  # target 220 - neck 40
        yy, xx = np.nonzero(np.asarray(result['target_mask']))
        self.assertEqual(yy.max(), 220)
        self.assertLessEqual(abs(xx.min()-100), 1)
        self.assertLessEqual(abs(xx.max()-200), 1)
        # 선택 사이즈를 바꿔도 동일한 기본 합성을 다시 변형한다.
        large, small = self.apply('L'), self.apply('S')
        self.assertGreater(np.nonzero(np.asarray(large['target_mask']))[0].max(), yy.max())
        self.assertLess(np.nonzero(np.asarray(small['target_mask']))[0].max(), yy.max())
        self.assertEqual(large['report']['base'], small['report']['base'])

    def test_only_selected_dimension_changes_and_outside_pixels_are_identical(self):
        for length, chest in [(True, False), (False, True), (True, True)]:
            with self.subTest(length=length, chest=chest):
                result = self.apply('S', length=length, chest=chest)
                mask = (np.asarray(self.mask)>0) | (np.asarray(result['target_mask'])>0)
                output, before = np.asarray(result['image']), np.asarray(self.base)
                self.assertTrue(np.array_equal(output[~mask], before[~mask]))
                self.assertEqual(result['report']['outside_edit_max_pixel_difference'], 0)
                self.assertEqual(output[25, 150].tolist(), before[25, 150].tolist())
                if not length:
                    self.assertEqual(result['report']['target_geometry_px']['hem_y'], 170)
                if not chest:
                    self.assertEqual(result['report']['target_geometry_px']['chest_left'], -40)
        self.assertGreater(self.apply('S')['report']['repair_pixels'], 0)

    def test_generation_adapter_passes_fixed_settings_once_and_records_provenance(self):
        garment = Image.new('RGB', (120, 120), 'red')
        engine = Mock(return_value=(self.base, {'seed': 17, 'total_s': .1}))
        base, metadata = generate_base(self.person, garment, engine_settings(17, 30, 2.5), engine)
        engine.assert_called_once()
        self.assertEqual(engine.call_args.kwargs['seed'], 17)
        self.assertEqual(engine.call_args.kwargs['garment_photo_type'], 'flat-lay')
        self.assertTrue(engine.call_args.kwargs['return_timing'])
        self.assertEqual(metadata['source'], 'fashn')
        self.assertFalse(metadata['size_conditioned_by_engine'])
        self.assertEqual(metadata['garment_sha256'], photo_info(garment)['sha256'])
        self.assertEqual(metadata['base_sha256'], photo_info(base)['sha256'])

    def test_missing_gpu_environment_never_calls_inference(self):
        with patch('size_synthesis.fashn_environment', return_value=(False, 'GPU unavailable')):
            with self.assertRaisesRegex(ValueError, 'GPU unavailable'):
                generate_base(self.person, self.person)
        for settings in [(True, 50, 2.5), (-1, 50, 2.5), (42, .5, 2.5), (42, 50, float('nan'))]:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                engine_settings(*settings)

    def test_photo_bindings_mask_review_and_coordinates_reject_stale_inputs(self):
        other = self.person.copy()
        other.putpixel((0, 0), (1, 2, 3))
        for person, base, mask, confirmed in [
                (other, self.base, self.mask, True), (self.person, other, self.mask, True),
                (self.person, self.base, self.mask, False),
                (self.person, self.base, Image.new('L', self.base.size), True),
                (self.person, self.base, Image.new('L', (30, 40)), True)]:
            with self.subTest(confirmed=confirmed), self.assertRaises(ValueError):
                adjust_size(person, base, self.meta, self.preview(), mask, confirmed)
        with self.assertRaises(ValueError):
            bind_base(self.person, Image.new('RGB', (400, 300)), 'external_unverified')
        scaled, meta = bind_base(self.person, self.base.resize((150, 200)), 'external_unverified')
        self.assertEqual(scaled.size, self.person.size)
        self.assertEqual(meta['base_sha256'], photo_info(scaled)['sha256'])

    def test_recalculation_prevents_tampered_target_and_does_not_mutate_inputs(self):
        current = self.preview('L')
        original = copy.deepcopy(current)
        before = self.base.tobytes(), self.mask.tobytes()
        current['target_points_px']['hem_left'] = [0, 0]
        recovered = checked_preview(current, self.person)
        self.assertEqual(recovered['target_points_px'], original['target_points_px'])
        self.apply('L')
        self.assertEqual(before, (self.base.tobytes(), self.mask.tobytes()))

    def test_editor_keeps_alpha_and_hash_through_gradio_roundtrip(self):
        import gradio as gr

        editor = gr.ImageEditor(type='pil', format='png')
        payload = editor_value(self.base, self.mask)
        posted = editor.postprocess(payload)
        # 화면용 composite가 아직 업로드 중이어도 원본/레이어만으로 마스크를 확정할 수 있다.
        posted = posted.model_copy(update={'composite': posted.composite.model_copy(update={'path': 'missing_composite.png'})})
        restored = editor.preprocess(mask_editor_payload(editor, posted))
        self.assertTrue(np.array_equal(np.asarray(editor_mask(restored, self.base)), np.asarray(self.mask)))
        restored['layers'] = []
        self.assertFalse(np.asarray(editor_mask(restored, self.base)).any())
        restored['background'] = self.person
        with self.assertRaises(ValueError):
            editor_mask(restored, self.base)

    def test_mask_draft_uses_image_and_is_editable_before_review(self):
        draft = draft_mask(self.base, self.preview())
        self.assertEqual(draft.size, self.base.size)
        self.assertGreater(np.asarray(draft).sum(), 0)
        self.assertEqual(np.asarray(draft)[25, 150], 0)  # face is not a garment seed

    def test_export_preserves_inputs_and_uses_new_run_directory(self):
        result = self.apply('S')
        with tempfile.TemporaryDirectory() as directory:
            files, _ = save_synthesis(result, self.person, self.base, output_directory=directory)
            other, _ = save_synthesis(result, self.person, self.base, output_directory=directory)
            self.assertNotEqual(Path(files[0]).parent, Path(other[0]).parent)
            self.assertTrue(all(Path(file).is_file() for file in files))
            data = json.loads((Path(files[0]).parent/'synthesis.json').read_text(encoding='utf-8'))
            self.assertEqual(data['preview']['fit_plan']['selected_size'], 'S')
            self.assertEqual(data['base']['source'], 'external_unverified')
            with Image.open(Path(files[0]).parent/'adjusted.png') as image:
                self.assertTrue(np.array_equal(np.asarray(image), np.asarray(result['image'])))


if __name__ == '__main__':
    unittest.main()
