"""기존 치수 비교에서 사용자가 선택한 판매 사이즈의 윤곽 계획을 만든다."""
import copy
import math
from dataclasses import dataclass

from garment_compare import compare_body, compare_reference, parse_target_chart
from reference_garment import ReferenceGarment


@dataclass(frozen=True)
class FitPlan:
    comparison: dict
    selected_size: str
    target: dict
    differences: dict
    ratios: dict
    status: str
    reasons: tuple[str, ...]

    def to_dict(self):
        return copy.deepcopy({
            'schema_version': 1, 'transform_version': 'ratio_v0',
            'selected_size': self.selected_size, 'comparison': self.comparison,
            'target': self.target, 'differences': self.differences,
            'ratios': self.ratios, 'status': self.status, 'reasons': list(self.reasons),
            'uncertainty': {'body_prediction': self.comparison['source'].get('predictions'),
                            'photo_outline_error_px': None},
        })


def create_fit_plan(comparison, selected_size):
    """원본 치수를 다시 검증한다. Ridge 경로는 수치 비교 상태로 보존한다."""
    if not isinstance(comparison, dict) or comparison.get('schema_version') != 1:
        raise ValueError('현재 형식의 치수 비교 결과가 필요합니다.')
    try:
        raw_chart = comparison['target_chart']
        chart = parse_target_chart([
            [row['size'], row['length_cm'], row.get('shoulder_cm'), row['chest_flat_cm']]
            for row in raw_chart['sizes']], raw_chart['shoulder_style'])
        source = comparison['source']
        if comparison['mode'] == 'reference_garment':
            dimensions = source['dimensions']
            reference = ReferenceGarment(dimensions['length'], dimensions['shoulder'],
                                         dimensions['chest'])
            checked = compare_reference(chart, reference, source['shoulder_style'])
        elif comparison['mode'] == 'ridge_body':
            checked = compare_body(chart, source['predictions'], source['coverage_target'])
        else:
            raise ValueError('지원하는 치수 비교 경로가 아닙니다.')
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('치수 비교 결과의 원본 치수를 확인해주세요.') from exc
    # 모델 경로·입력 출처·가상 표 표시를 보존하고 차이 값만 다시 계산한다.
    checked['source'] = copy.deepcopy(source)
    checked['target_chart'].update({key: copy.deepcopy(raw_chart[key]) for key in
                                   ('source', 'is_example') if key in raw_chart})
    matches = [(size, row) for size, row in zip(chart.sizes, checked['rows'])
               if size.size == selected_size]
    if not matches:
        raise ValueError('현재 치수표에 있는 판매 사이즈를 선택해주세요.')
    size, difference = matches[0]
    ratios = {'length': None, 'chest': None, 'shoulder': None}
    reasons = []
    if checked['mode'] == 'reference_garment':
        ratios.update(length=size.length_cm / reference.length_cm,
                      chest=size.chest_flat_cm / reference.chest_flat_cm)
        if difference['shoulder_difference_cm'] is not None:
            ratios['shoulder'] = size.shoulder_cm / reference.shoulder_cm
        else:
            reasons.append('어깨 변형 보류: 두 옷 모두 일반 어깨선으로 확인해야 합니다.')
        status = 'needs_photo_review'
    else:
        status = 'numeric_only'
        reasons.append('Ridge 신체 예측만으로 사진 윤곽의 변형 비율을 정할 수 없습니다.')
    if any(value is not None and (not math.isfinite(value) or value <= 0) for value in ratios.values()):
        raise ValueError('치수 비율이 계산 범위를 벗어났습니다. cm 단위 입력을 확인해주세요.')
    return FitPlan(checked, selected_size, size.to_dict(), difference, ratios,
                   status, tuple(reasons))
