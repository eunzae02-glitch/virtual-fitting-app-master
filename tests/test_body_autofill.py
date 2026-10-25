"""사진 치수 추정값이 앱의 편집 가능한 입력칸으로 흐르는지 확인한다."""
import os
import sys
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app'))

try:
    import gradio_app
except ImportError:
    gradio_app = None

from body_measure import BodyMeasurement  # noqa: E402


@unittest.skipIf(gradio_app is None, 'Gradio가 설치되지 않은 환경')
class TestBodyAutofill(unittest.TestCase):
    def setUp(self):
        self.photo = Image.new('RGB', (800, 1200))

    def test_fills_circumferences_from_half_circumferences(self):
        estimate = BodyMeasurement(
            height_cm=175, px_per_cm=5.0, has_segmentation=True,
            measurements_cm={'shoulder': 44.0, 'chest': 48.0, 'waist': 40.0, 'hip': 49.0},
        )
        with patch.object(gradio_app, 'measure', return_value=estimate):
            result = gradio_app.auto_fill_measurements(
                self.photo, 175, None, None, None, None)
        self.assertEqual(result[:4], (44.0, 96.0, 80.0, 98.0))

    def test_ambiguous_part_keeps_manual_value(self):
        estimate = BodyMeasurement(
            height_cm=175, px_per_cm=5.0, has_segmentation=True,
            measurements_cm={'shoulder': 44.0, 'waist': 40.0, 'hip': 49.0},
            warnings=['가슴 경계가 팔/옷과 겹쳐 가슴둘레를 추정하지 않았습니다.'],
        )
        with patch.object(gradio_app, 'measure', return_value=estimate):
            result = gradio_app.auto_fill_measurements(
                self.photo, 175, None, 94.0, None, None)
        self.assertEqual(result[:4], (44.0, 94.0, 80.0, 98.0))
        self.assertIn('가슴 경계', result[4])

    def test_unsafe_photo_does_not_replace_manual_measurements(self):
        estimate = BodyMeasurement(
            height_cm=175, px_per_cm=5.0, unsafe=True, has_segmentation=True,
            measurements_cm={'shoulder': 44.0, 'chest': 48.0},
            warnings=['발 쪽이 사진 밖으로 잘린 것 같습니다.'],
        )
        with patch.object(gradio_app, 'measure', return_value=estimate):
            result = gradio_app.auto_fill_measurements(
                self.photo, 175, 45.0, 94.0, 82.0, 96.0)
        self.assertEqual(result[:4], (45.0, 94.0, 82.0, 96.0))


if __name__ == '__main__':
    unittest.main()
