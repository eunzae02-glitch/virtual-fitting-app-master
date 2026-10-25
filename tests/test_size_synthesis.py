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

from debug_size_preview import CHART_SOURCES, build_ui, compare_dimensions
from fit_plan import create_fit_plan
from garment_outline import confirm_anchors, create_preview, new_marks, photo_info
from garment_repair import REPAIR_VERSION
from garment_mask_editor import GarmentMaskEditor, encode_png
from size_synthesis import (adjust_size, adjust_sizes, bind_base, checked_preview, draft_mask, editor_mask,
                            editor_value, engine_settings, generate_base)
from size_synthesis_ui import save_synthesis


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

    def test_shrink_repair_does_not_copy_remaining_shirt_color_outside_target(self):
        result = self.apply('S')
        image = np.asarray(result['image'])
        target = np.asarray(result['target_mask']) > 0
        red = (image[:, :, 0] > 160) & (image[:, :, 1] < 90) & (image[:, :, 2] < 90)
        self.assertTrue(red.any())
        self.assertEqual(int((red & ~target).sum()), 0)  # v0 leaked 708 red pixels below the target hem
        self.assertEqual(result['report']['repair_method'], REPAIR_VERSION)
        self.assertEqual(result['report']['changed_pixels'], int(np.any(image != np.asarray(self.base), axis=2).sum()))

    def test_all_sizes_share_base_but_have_independent_target_and_rgb_hashes(self):
        before = self.base.tobytes(), self.mask.tobytes()
        result, gallery = adjust_sizes(self.person, self.base, self.meta, self.preview(), self.mask, True)
        rows = result['report']['size_comparison']['rows']
        self.assertEqual([row['size'] for row in rows], ['S', 'M', 'L'])
        self.assertEqual(len(gallery), 3)
        self.assertEqual(len({row['adjusted_sha256'] for row in rows}), 3)
        self.assertEqual(len({row['target_mask_sha256'] for row in rows}), 3)
        self.assertEqual(result['report']['adjusted_sha256'], rows[1]['adjusted_sha256'])
        hems = [row['target_geometry_px']['hem_y'] for row in rows]
        self.assertTrue(hems[0] < hems[1] < hems[2])
        self.assertEqual(before, (self.base.tobytes(), self.mask.tobytes()))

    def test_all_sizes_keep_partial_review_and_withhold_out_of_frame_targets(self):
        result, gallery = adjust_sizes(self.person, self.base, self.meta,
                                       self.preview(chest=False), self.mask, True)
        rows = result['report']['size_comparison']['rows']
        self.assertTrue(all(not row['applied']['chest'] for row in rows))
        self.assertTrue(all('보류' in caption for _, caption in gallery))
        self.comparison['target_chart']['sizes'].append(
            {'size': 'huge', 'length_cm': 650, 'shoulder_cm': 51, 'chest_flat_cm': 520})
        result, gallery = adjust_sizes(self.person, self.base, self.meta, self.preview(), self.mask, True)
        self.assertEqual(len(gallery), 3)
        self.assertEqual(result['report']['size_comparison']['withheld'][0]['size'], 'huge')

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
        editor = GarmentMaskEditor()
        payload = editor_value(self.base, self.mask)
        self.assertIsNone(payload['composite'])
        posted = editor.postprocess(payload)
        restored = editor.preprocess(posted)
        self.assertTrue(np.array_equal(np.asarray(editor_mask(restored, self.base)), np.asarray(self.mask)))
        value = json.loads(posted)
        # 지우개는 초기 초안 자체에서도 영역을 지울 수 있다. 배경은 그대로여야 한다.
        layer = payload['layers'][0].copy()
        ImageDraw.Draw(layer).rectangle((140, 120, 160, 140), fill=(0, 0, 0, 0))
        value['layer'] = encode_png(layer)
        restored = editor.preprocess(json.dumps(value))
        mask = np.asarray(editor_mask(restored, self.base))
        self.assertFalse(mask[130, 150])
        self.assertTrue(mask[180, 150])
        self.assertEqual(restored['background'].convert('RGB').tobytes(), self.base.tobytes())
        value['background'] = encode_png(self.person)
        with self.assertRaisesRegex(ValueError, '사진이 바뀌었습니다'):
            editor_mask(editor.preprocess(json.dumps(value)), self.base)

    def test_mask_editor_rejects_invalid_or_resized_payloads(self):
        editor = GarmentMaskEditor()
        payload = json.loads(editor.postprocess(editor_value(self.base, self.mask)))
        for value in ('not json', '{}', '[]', json.dumps(dict(payload, layer='not png')),
                      json.dumps(dict(payload, layer=encode_png(Image.new('RGBA', (10, 10)))))):
            with self.subTest(value=value[:30]), self.assertRaises(ValueError):
                editor.preprocess(value)

    def test_mask_draft_uses_image_and_is_editable_before_review(self):
        draft = draft_mask(self.base, self.preview())
        self.assertEqual(draft.size, self.base.size)
        self.assertGreater(np.asarray(draft).sum(), 0)
        self.assertEqual(np.asarray(draft)[25, 150], 0)  # face is not a garment seed

    def test_export_preserves_inputs_and_uses_new_run_directory(self):
        result, _ = adjust_sizes(self.person, self.base, self.meta, self.preview('S'), self.mask, True)
        with tempfile.TemporaryDirectory() as directory:
            files, _ = save_synthesis(result, self.person, self.base, output_directory=directory)
            other, _ = save_synthesis(result, self.person, self.base, output_directory=directory)
            self.assertNotEqual(Path(files[0]).parent, Path(other[0]).parent)
            self.assertTrue(all(Path(file).is_file() for file in files))
            data = json.loads((Path(files[0]).parent/'synthesis.json').read_text(encoding='utf-8'))
            self.assertEqual(data['preview']['fit_plan']['selected_size'], 'S')
            self.assertEqual(data['base']['source'], 'external_unverified')
            self.assertEqual([row['size'] for row in data['size_comparison']['rows']], ['S', 'M', 'L'])
            self.assertEqual(data['adjusted_sha256'], data['size_comparison']['rows'][0]['adjusted_sha256'])
            with Image.open(Path(files[0]).parent/'adjusted.png') as image:
                self.assertTrue(np.array_equal(np.asarray(image), np.asarray(result['image'])))
            for name in ('reference_reused_mask.png', 'estimated_repair_mask.png', 'repair_overlay.png'):
                self.assertTrue((Path(files[0]).parent/name).is_file())
            self.assertEqual(data['repair']['reused_reference_pixels']+data['repair']['estimated_pixels'],
                             data['repair_pixels'])


class TestSynthesisUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo = build_ui()
        cls.handlers = {event.fn.__name__: event.fn for event in cls.demo.fns.values() if event.fn}

    @classmethod
    def tearDownClass(cls):
        cls.demo.close()

    def setUp(self):
        self.case = TestSizeSynthesis()
        self.case.setUp()

    def test_preview_events_apply_after_new_outline_instead_of_clearing_it(self):
        dependencies = self.demo.config['dependencies']
        for key, event in self.demo.fns.items():
            if event.fn and event.fn.__name__ == 'display':
                chained = [dep for dep in dependencies if dep.get('trigger_after') == key]
                self.assertEqual(len(chained), 1)
                self.assertEqual(self.demo.fns[chained[0]['id']].fn.__name__, 'refresh')

    def test_mask_editor_roundtrip_can_be_applied_without_a_flattened_composite(self):
        event = next(event for event in self.demo.fns.values()
                     if event.fn and event.fn.__name__ == 'refresh')
        editor = event.inputs[4]
        case = self.case
        edited = editor.preprocess(editor.postprocess(editor_value(case.base, case.mask)))
        result = event.fn(case.preview(), case.person, case.base, case.meta, edited, True, None)
        self.assertIsInstance(result[0], Image.Image)
        self.assertIn('선택 사이즈 M', result[3])

    def test_one_generation_then_mask_review_and_size_changes_use_same_base(self):
        case = self.case
        garment = Image.new('RGB', (100, 100), 'blue')
        engine = Mock(return_value=(case.base.copy(), {'seed': 42, 'total_s': .1}))
        with patch('size_synthesis_ui.draft_mask', return_value=case.mask), patch(
                'size_synthesis_ui.generate_base',
                side_effect=lambda person, clothes, settings: generate_base(person, clothes, settings, engine)):
            outputs = list(self.handlers['generate_with_status'](
                case.preview(), case.person, garment, 42, 50, 2.5, progress=Mock()))
        self.assertEqual(len(outputs), 2)
        self.assertTrue(all(update['interactive'] is False for update in outputs[0][14:]))
        self.assertTrue(all(isinstance(update.get('interactive'), bool) for update in outputs[-1][14:]))
        runner = next(event for event in self.demo.fns.values()
                      if event.fn and event.fn.__name__ == 'generate_with_status')
        self.assertFalse(any(component.__class__.__name__ == 'ImageEditor' for component in runner.outputs[14:]))
        metadata, base, edited, signature = outputs[-1][:4]
        refresh = self.handlers['refresh']
        unconfirmed = refresh(case.preview(), case.person, base, metadata, edited, False, garment)
        self.assertIsNone(unconfirmed[0])
        hashes = []
        for size in ('S', 'M', 'L'):
            applied = refresh(case.preview(size), case.person, base, metadata, edited, True, garment)
            self.assertIsInstance(applied[0], Image.Image)
            self.assertEqual(len(applied[2]), 3)
            self.assertIn(f'선택 사이즈 {size}', applied[3])
            hashes.append(photo_info(applied[0])['sha256'])
        self.assertEqual(len(set(hashes)), 3)
        engine.assert_called_once()
        # 편집기의 동일 마스크 재전송은 이미 완료된 자동 적용을 지우지 않는다.
        same_mask = self.handlers['mask_changed'](edited, signature, base)
        self.assertTrue(all(value == {'__type__': 'update'} for value in same_mask))

    def test_changed_photos_or_mask_review_do_not_keep_old_size_images(self):
        case = self.case
        edited = editor_value(case.base, case.mask)
        changed = case.person.copy()
        changed.putpixel((0, 0), (1, 1, 1))
        output = self.handlers['refresh'](case.preview(), changed, case.base, case.meta, edited, True, None)
        self.assertIsNone(output[0])
        self.assertEqual(output[2], [])
        output = self.handlers['refresh'](None, case.person, case.base, case.meta, edited, True, None)
        self.assertIsNone(output[0])
        self.assertEqual(output[2], [])
        output = self.handlers['mask_changed'](editor_value(case.base, Image.new('L', case.base.size)),
                                               'old_signature', case.base)
        self.assertFalse(output[1])
        self.assertIsNone(output[2])

    def test_generation_failure_clears_outputs_and_restores_input_controls(self):
        case = self.case
        with patch('size_synthesis_ui.generate_base', side_effect=RuntimeError('GPU unavailable')):
            outputs = list(self.handlers['generate_with_status'](
                case.preview(), case.person, case.base, 42, 50, 2.5, progress=Mock()))
        final = outputs[-1]
        self.assertIsNone(final[0])
        self.assertIn('GPU unavailable', final[5])
        self.assertIsNone(final[6])
        self.assertTrue(any(update.get('interactive') is True for update in final[14:]))


if __name__ == '__main__':
    unittest.main()
