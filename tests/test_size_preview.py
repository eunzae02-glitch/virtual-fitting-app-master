"""사진/치수 출처 연결, 부분 변형, 확인 해제와 저장 재현성을 검사한다."""
import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'app'))
sys.path.insert(0, str(ROOT / 'scripts'))

from fit_plan import create_fit_plan  # noqa: E402
from garment_compare import compare_body, parse_target_chart  # noqa: E402
from garment_outline import (confirm_anchors, create_preview, new_marks, photo_info,
                             put_anchor, render_outline)  # noqa: E402
from debug_size_preview import (CHART_SOURCES, build_preview, compare_dimensions,
                                load_sample, save_preview)  # noqa: E402


def comparison(style='일반 어깨선', rows=None):
    return compare_dimensions(65, 51, 52, '일반 어깨선', style,
                              rows or [['S', 63, 49, 50], ['M', 65, 51, 52], ['L', 67, 53, 54]],
                              CHART_SOURCES[1])


def anchors(info, shoulders=False):
    points = {'neck': [.5, .2], 'chest_left': [.3, .4], 'chest_right': [.7, .4],
              'hem_left': [.3, .7], 'hem_right': [.7, .7]}
    if shoulders:
        points.update(shoulder_left=[.25, .25], shoulder_right=[.75, .25])
    return new_marks(info, points, 'test_draft')


class TestSizePreview(unittest.TestCase):
    def setUp(self):
        self.photo = Image.new('RGB', (600, 800), 'gray')
        self.info = photo_info(self.photo, 'test')
        self.marks = confirm_anchors(anchors(self.info), self.info)

    def test_plan_recalculates_stale_differences_and_preserves_labels_and_provenance(self):
        result = comparison(rows=[['100', 67, 53, 54], ['95', 65, 51, 52]])
        result['rows'][0]['length_difference_cm'] = 999
        before = copy.deepcopy(result)
        plan = create_fit_plan(result, '100')
        self.assertEqual(plan.differences['length_difference_cm'], 2)
        self.assertAlmostEqual(plan.ratios['chest'], 54/52)
        self.assertTrue(plan.comparison['target_chart']['is_example'])
        self.assertEqual(plan.comparison['target_chart']['source'], 'debug_example_chart')
        self.assertEqual([row['size'] for row in plan.comparison['rows']], ['100', '95'])
        self.assertEqual(before, result)
        with self.assertRaises(ValueError):
            create_fit_plan(result, 'S')

    def test_identity_and_partial_changes_are_independent(self):
        identity = create_preview(create_fit_plan(comparison(), 'M'), self.info, self.marks, True, True, True)
        self.assertEqual(identity['reference_points_px'], identity['target_points_px'])
        for key, point in identity['reference_points_px'].items():
            for axis in (0, 1):
                self.assertAlmostEqual(point[axis], identity['target_points_px'][key][axis])
        plan = create_fit_plan(comparison(), 'L')
        longer = create_preview(plan, self.info, self.marks, True, True, False)
        wider = create_preview(plan, self.info, self.marks, True, False, True)
        base = identity['reference_points_px']
        self.assertEqual(longer['target_points_px']['chest_left'], base['chest_left'])
        self.assertEqual(longer['target_points_px']['hem_left'][0], base['hem_left'][0])
        self.assertAlmostEqual(longer['target_points_px']['hem_left'][1] - base['neck'][1],
                               (base['hem_left'][1] - base['neck'][1]) * 67/65)
        self.assertEqual(wider['target_points_px']['hem_left'][1], base['hem_left'][1])
        self.assertAlmostEqual(wider['target_points_px']['chest_right'][0] - base['neck'][0],
                               (base['chest_right'][0] - base['neck'][0]) * 54/52)

    def test_photographic_roll_is_kept_in_local_axis(self):
        rotated = anchors(self.info)
        theta = .08
        nx, ny = rotated['points']['neck']
        for key, (x, y) in list(rotated['points'].items()):
            dx, dy = (x-nx)*600, (y-ny)*800
            rotated['points'][key] = [nx+(dx*math.cos(theta)-dy*math.sin(theta))/600,
                                      ny+(dx*math.sin(theta)+dy*math.cos(theta))/800]
        marks = confirm_anchors(rotated, self.info)
        preview = create_preview(create_fit_plan(comparison(), 'L'), self.info, marks, True, True, False)
        neck = preview['reference_points_px']['neck']
        for axis in (0, 1):
            before = sum(preview['reference_points_px'][key][axis] for key in ('hem_left', 'hem_right'))/2
            after = sum(preview['target_points_px'][key][axis] for key in ('hem_left', 'hem_right'))/2
            self.assertAlmostEqual(after-neck[axis], (before-neck[axis])*67/65)

    def test_edit_or_photo_replacement_requires_review_and_blocks_stale_save(self):
        edited = put_anchor(self.marks, 'hem_left', 180, 570, self.info)
        self.assertFalse(edited['confirmed'])
        self.assertTrue(self.marks['confirmed'])
        plan = create_fit_plan(comparison(), 'M')
        with self.assertRaises(ValueError):
            create_preview(plan, self.info, edited, True, True, True)
        changed = self.photo.copy()
        changed.putpixel((0, 0), (255, 0, 0))
        with self.assertRaises(ValueError):
            build_preview(comparison(), 'M', changed, self.info, self.marks, True, True, True, False)
        preview = create_preview(plan, self.info, self.marks, True, True, True)
        with self.assertRaises(ValueError):
            save_preview(preview, changed)

    def test_five_points_suffice_but_shoulder_requires_two_confirmed_seams(self):
        plan = create_fit_plan(comparison(), 'L')
        with self.assertRaises(ValueError):
            create_preview(plan, self.info, self.marks, True, True, True, True)
        marks = confirm_anchors(anchors(self.info, True), self.info, True)
        preview = create_preview(plan, self.info, marks, True, True, True, True)
        self.assertTrue(preview['applied']['shoulder'])
        self.assertAlmostEqual(preview['target_points_px']['shoulder_right'][0]-300, 150*53/51)
        unknown = create_fit_plan(comparison('드롭숄더'), 'L')
        preview = create_preview(unknown, self.info, self.marks, True, True, True, True)
        self.assertFalse(preview['applied']['shoulder'])
        self.assertNotIn('shoulder_left', preview['target_points_px'])

    def test_ridge_keeps_uncertainty_and_refuses_photo_ratios(self):
        values = {'shoulder_breadth_cm': 40, 'body_length_cm': 54, 'chest_circumference_cm': 95}
        predictions = {key: {'value_cm': value, 'low_cm': value-3, 'high_cm': value+3,
                              'label': key, 'model': 'ridge', 'source': 'excel_body_model'}
                       for key, value in values.items()}
        plan = create_fit_plan(compare_body(parse_target_chart([['M', 65, 51, 52]]), predictions, .9), 'M')
        self.assertEqual(plan.status, 'numeric_only')
        self.assertTrue(all(value is None for value in plan.ratios.values()))
        self.assertEqual(plan.to_dict()['uncertainty']['body_prediction'], predictions)
        with self.assertRaises(ValueError):
            create_preview(plan, self.info, self.marks, True, True, True)

    def test_unsafe_geometry_unconfirmed_match_and_unsupported_dimensions_are_withheld(self):
        plan = create_fit_plan(comparison(), 'M')
        for same, length, chest in [(False, True, True), (True, False, False)]:
            with self.subTest(same=same), self.assertRaises(ValueError):
                create_preview(plan, self.info, self.marks, same, length, chest)
        for value in (float('nan'), float('inf'), -1, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                put_anchor(self.marks, 'neck', value, 10, self.info)
        swapped = anchors(self.info)
        swapped['points']['hem_left'], swapped['points']['hem_right'] = (
            swapped['points']['hem_right'], swapped['points']['hem_left'])
        with self.assertRaises(ValueError):
            confirm_anchors(swapped, self.info)
        huge = create_fit_plan(comparison(rows=[['XXL', 650, 51, 520]]), 'XXL')
        with self.assertRaises(ValueError):
            create_preview(huge, self.info, self.marks, True, True, True)
        short = create_fit_plan(comparison(rows=[['crop', 10, 51, 52]]), 'crop')
        with self.assertRaises(ValueError):
            create_preview(short, self.info, self.marks, True, True, True)

    def test_export_is_reproducible_and_does_not_overwrite(self):
        preview = create_preview(create_fit_plan(comparison(), 'L'), self.info, self.marks, True, True, True)
        original = self.photo.tobytes()
        with tempfile.TemporaryDirectory() as directory:
            files, _ = save_preview(preview, self.photo, directory)
            other, _ = save_preview(preview, self.photo, directory)
            self.assertEqual(len(files), 5)
            self.assertNotEqual(Path(files[0]).parent, Path(other[0]).parent)
            saved = json.loads(Path(files[0]).read_text(encoding='utf-8'))
            with Image.open(Path(files[0]).parent / 'reference.png') as photo:
                self.assertEqual(photo_info(photo)['sha256'], saved['photo']['sha256'])
                info = saved['photo']
                recreated = create_preview(create_fit_plan(saved['fit_plan']['comparison'], 'L'),
                                           info, saved['anchors'], **{
                                               'same_garment': True, 'length_compatible': True,
                                               'chest_compatible': True})
                self.assertEqual(recreated['target_points_px'], preview['target_points_px'])
            self.assertEqual(saved['fit_plan']['uncertainty']['photo_outline_error_px'], None)
            self.assertTrue(all(Path(file).exists() for file in files))
        self.assertEqual(self.photo.tobytes(), original)
        self.assertEqual(render_outline(self.photo, preview=preview).size, self.photo.size)

    @unittest.skipUnless((ROOT / 'debug').is_dir(), '개인 00_ME 사진/TXT가 없는 환경')
    def test_real_sample_remains_a_draft_until_reviewed(self):
        case, image, info, marks = load_sample()
        self.assertEqual((case.length_cm, case.shoulder_cm, case.chest_flat_cm), (65, 51, 52))
        self.assertFalse(marks['confirmed'])
        self.assertEqual(marks['source'], 'manual_visual_draft')
        self.assertEqual(info['sha256'], photo_info(image)['sha256'])
        self.assertTrue(confirm_anchors(marks, info)['confirmed'])

    def test_gradio_photo_roundtrip_preserves_hash(self):
        import gradio as gr

        photo, info = self.photo, self.info
        component = gr.Image(type='pil', format='png')
        restored = component.preprocess(component.postprocess(photo))
        self.assertEqual(photo_info(restored)['sha256'], info['sha256'])


if __name__ == '__main__':
    unittest.main()
