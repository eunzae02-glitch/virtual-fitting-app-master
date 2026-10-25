"""기준 옷의 단면 보존, 입력 검증, 확인 상태 초기화를 검사한다."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'app'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from reference_garment import ReferenceGarment  # noqa: E402
from debug_reference_size import clear_confirmation, confirm_reference  # noqa: E402


class TestReferenceGarment(unittest.TestCase):
    def test_all_three_garment_dimensions_are_preserved(self):
        data, report = confirm_reference(70.5, 48, 54)
        self.assertEqual(data['dimensions'], {'length': 70.5, 'shoulder': 48, 'chest': 54})
        self.assertEqual(data['measurement_type'], 'garment')
        self.assertEqual(data['chest_measurement'], 'flat_width')
        self.assertIn('54 cm', report)

    def test_missing_field_is_identified(self):
        for index, label in enumerate(('총장', '어깨너비', '가슴단면')):
            with self.subTest(label=label):
                values = [70, 48, 54]
                values[index] = None
                data, report = confirm_reference(*values)
                self.assertIsNone(data)
                self.assertIn(label, report)
                self.assertNotIn('| 항목 |', report)

    def test_invalid_numbers_cannot_become_reference_data(self):
        for value in (0, -1, float('nan'), float('inf'), 'abc', '', True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ReferenceGarment(70, 48, value)

    def test_decimal_inputs_are_not_converted_to_body_measurements(self):
        garment = ReferenceGarment('70.25', '48.5', '54.75')
        self.assertEqual(garment.to_chart_dimensions(),
                         {'length': 70.25, 'shoulder': 48.5, 'chest': 54.75})

    def test_editing_clears_the_previously_confirmed_values(self):
        confirmed, _ = confirm_reference(70, 48, 54)
        self.assertIsNotNone(confirmed)
        cleared, report = clear_confirmation()
        self.assertIsNone(cleared)
        self.assertIn('입력 확인', report)


if __name__ == '__main__':
    unittest.main()
