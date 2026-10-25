"""판매 사이즈별 옷 실측을 기준 옷 또는 신체 예측 범위와 비교한다."""
import html
import math
from dataclasses import dataclass

from reference_garment import ReferenceGarment


SHOULDER_STYLES = ('일반 어깨선', '드롭숄더', '래글런', '확인 필요')
BODY_TARGETS = ('shoulder_breadth_cm', 'body_length_cm', 'chest_circumference_cm')


def _blank(value):
    return value is None or (isinstance(value, str) and not value.strip())


def _positive(value, label):
    if _blank(value):
        raise ValueError(f'{label}을 입력해주세요.')
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{label}은 cm 단위 숫자로 입력해주세요.') from exc
    if isinstance(value, bool) or not math.isfinite(number) or number <= 0:
        raise ValueError(f'{label}은 0보다 큰 유한한 숫자로 입력해주세요.')
    return number


def _style(value):
    if value not in SHOULDER_STYLES:
        raise ValueError('어깨 구조를 선택해주세요.')
    return value


@dataclass(frozen=True)
class TargetSize:
    size: str
    length_cm: float
    shoulder_cm: float | None
    chest_flat_cm: float

    def to_dict(self):
        return {'size': self.size, 'length_cm': self.length_cm,
                'shoulder_cm': self.shoulder_cm, 'chest_flat_cm': self.chest_flat_cm}


@dataclass(frozen=True)
class TargetChart:
    sizes: tuple[TargetSize, ...]
    shoulder_style: str

    def to_dict(self):
        return {'source': 'manual_target_chart', 'unit': 'cm', 'measurement_type': 'garment',
                'chest_measurement': 'flat_width', 'shoulder_style': self.shoulder_style,
                'sizes': [size.to_dict() for size in self.sizes]}


def parse_target_chart(rows, shoulder_style='확인 필요'):
    """행 순서·판매 라벨을 보존한다. 완전히 빈 행만 건너뛰고 일부 누락은 거부한다."""
    _style(shoulder_style)
    if not isinstance(rows, (list, tuple)):
        raise ValueError('새 옷의 사이즈별 치수표를 입력해주세요.')
    sizes, names = [], set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, (list, tuple)) or len(row) != 4:
            raise ValueError(f'{index}행은 사이즈·총장·어깨너비·가슴단면 네 칸이 필요합니다.')
        if all(_blank(value) for value in row):
            continue
        raw_name, length, shoulder, chest = row
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError(f'{index}행의 판매 사이즈 이름을 입력해주세요. 예: S, M, 100')
        name = raw_name.strip()
        if len(name) > 40 or any(ord(char) < 32 for char in name):
            raise ValueError(f'{index}행의 사이즈 이름을 확인해주세요.')
        key = name.casefold()
        if key in names:
            raise ValueError(f'판매 사이즈가 중복됐습니다: {name}')
        names.add(key)
        optional_shoulder = _blank(shoulder) or (isinstance(shoulder, str) and shoulder.strip() == '-')
        if optional_shoulder and shoulder_style != '일반 어깨선':
            shoulder_value = None
        else:
            shoulder_value = _positive(shoulder, f'{name} 어깨너비(cm)')
        sizes.append(TargetSize(name, _positive(length, f'{name} 총장(cm)'), shoulder_value,
                                _positive(chest, f'{name} 가슴단면(cm)')))
    if not sizes:
        raise ValueError('새 옷의 판매 사이즈를 한 개 이상 입력해주세요.')
    return TargetChart(tuple(sizes), shoulder_style)


def _difference(garment_value, body_range):
    return {'value_cm': garment_value - body_range['value_cm'],
            'low_cm': garment_value - body_range['high_cm'],
            'high_cm': garment_value - body_range['low_cm']}


def compare_reference(chart, reference, shoulder_style='확인 필요'):
    if not isinstance(reference, ReferenceGarment):
        raise ValueError('기준 옷의 총장·어깨너비·가슴단면을 입력해주세요.')
    _style(shoulder_style)
    shoulder_comparable = shoulder_style == chart.shoulder_style == '일반 어깨선'
    rows = []
    for size in chart.sizes:
        rows.append({'size': size.size,
                     'length_difference_cm': size.length_cm - reference.length_cm,
                     'shoulder_difference_cm': (size.shoulder_cm - reference.shoulder_cm
                                                if shoulder_comparable else None),
                     'chest_flat_difference_cm': size.chest_flat_cm - reference.chest_flat_cm})
    source = reference.to_dict()
    source['shoulder_style'] = shoulder_style
    notes = ['차이 = 새 옷 − 기준 옷. 양수면 새 옷 치수가 더 크고, 음수면 더 작습니다.',
             '가슴단면 차이는 펼친 옷의 폭 차이입니다. 몸 가슴둘레나 사진 폭으로 변환하지 않습니다.',
             '옷 총장의 측정 시작점이 같은지 확인해주세요. 소재·패턴·착용감 차이는 반영하지 않았습니다.']
    if not shoulder_comparable:
        notes.append('어깨 비교 보류: 두 옷 모두 일반 어깨선으로 확인한 경우에만 어깨 차이를 계산합니다. '
                     '드롭숄더·래글런·구조 미확인은 입력 치수만 보존합니다.')
    return {'schema_version': 1, 'mode': 'reference_garment', 'source': source,
            'target_chart': chart.to_dict(), 'rows': rows, 'notes': notes}


def compare_body(chart, predictions, coverage_target):
    """신체 범위의 상한을 빼서 여유 범위의 하한을 만든다. 어깨는 정의 차이로 보류한다."""
    body = {}
    for target in BODY_TARGETS:
        try:
            values = predictions[target]
            body[target] = {key: _positive(values[key], f'{target} {key}')
                            for key in ('value_cm', 'low_cm', 'high_cm')}
            if not body[target]['low_cm'] <= body[target]['value_cm'] <= body[target]['high_cm']:
                raise ValueError('신체 예측 범위의 순서를 확인해주세요.')
            if values['model'] != 'ridge':
                raise ValueError('현재 비교에는 Ridge 예측값이 필요합니다.')
            body[target].update({'label': values['label'], 'model': 'ridge',
                                 'source': values['source']})
        except (KeyError, TypeError) as exc:
            raise ValueError('신체 예측값 또는 예측 범위가 없습니다.') from exc
    if isinstance(coverage_target, bool) or not 0 < coverage_target < 1:
        raise ValueError('예측 범위 목표 포함률을 확인해주세요.')
    rows = []
    for size in chart.sizes:
        chest = _difference(2 * size.chest_flat_cm, body['chest_circumference_cm'])
        rows.append({'size': size.size,
                     'length_difference_cm': _difference(size.length_cm, body['body_length_cm']),
                     'shoulder_difference_cm': None,
                     'chest_circumference_ease_cm': chest,
                     'chest_flat_ease_cm': {key: value / 2 for key, value in chest.items()}})
    return {'schema_version': 1, 'mode': 'ridge_body',
            'source': {'source': 'excel_body_model', 'measurement_type': 'predicted_body', 'unit': 'cm',
                       'model': 'ridge', 'coverage_target': coverage_target, 'predictions': body},
            'target_chart': chart.to_dict(), 'rows': rows,
            'notes': ['기본 길이 차이 = 옷 총장 − 예측 신체 기준 총장. '
                      '신체 기준 총장 = 목뒤높이 − 위앞엉덩뼈가시높이입니다. '
                      '굴곡·처짐 보정 전 값이며 실제 밑단 위치를 측정한 결과가 아닙니다.',
                      '가슴 단면 여유 = 옷 가슴단면 − 예측 몸 가슴둘레/2. '
                      '가슴 둘레 여유 = 옷 가슴단면×2 − 예측 몸 가슴둘레입니다.',
                      '단면 여유는 사진 정면의 폭이나 한쪽 마스크 이동량이 아닙니다.',
                      '어깨 비교 보류: 신체 어깨사이너비와 옷 어깨너비는 측정 정의가 달라 '
                      '각 수치를 따로 표시하고 차이에 따른 핏 판정은 하지 않습니다.',
                      '괄호는 신체 예측 범위를 반영한 차이의 범위입니다. 옷 치수 오차·굴곡·처짐은 '
                      '포함하지 않으며, 개인별 또는 세 부위 동시 포함을 보장하지 않습니다.']}


def _text(value):
    return html.escape(str(value)).replace('|', '\\|')


def _number(value):
    return '미입력' if value is None else f'{value:g}'


def _range(value):
    return f"{value['value_cm']:+.1f} ({value['low_cm']:+.1f}~{value['high_cm']:+.1f})"


def report_markdown(result):
    source = result['source']
    lines = ['## 사이즈별 치수 비교', '', '단위: cm. 새 옷의 가슴 항목은 옷을 펼쳐 잰 가슴단면입니다.', '']
    if result['mode'] == 'reference_garment':
        dims = source['dimensions']
        lines += ['### 비교 기준: 직접 입력한 기존 옷', '',
                  f"총장 **{dims['length']:g}** · 어깨너비 **{dims['shoulder']:g}** · "
                  f"가슴단면 **{dims['chest']:g}** / {_text(source['shoulder_style'])}"]
    else:
        lines += ['### 비교 기준: Ridge 신체 예측', '',
                  f"| 신체 항목 | 예측값 | {source['coverage_target']:.0%} 목표 예측 범위 |",
                  '|---|---:|---|']
        for value in source['predictions'].values():
            lines.append(f"| {_text(value['label'])} | {value['value_cm']:.1f} | "
                         f"{value['low_cm']:.1f}~{value['high_cm']:.1f} |")
    lines += ['', '### 입력한 새 옷 치수표', '',
              f"어깨 구조: {_text(result['target_chart']['shoulder_style'])}", '',
              '| 판매 사이즈 | 총장 | 어깨너비 | 가슴단면 |', '|---|---:|---:|---:|']
    for size in result['target_chart']['sizes']:
        lines.append(f"| {_text(size['size'])} | {_number(size['length_cm'])} | "
                     f"{_number(size['shoulder_cm'])} | {_number(size['chest_flat_cm'])} |")
    lines += ['', '### 사이즈별 차이', '']
    if result['mode'] == 'reference_garment':
        lines += ['| 판매 사이즈 | 총장 차이 | 어깨너비 차이 | 가슴단면 차이 |', '|---|---:|---:|---:|']
        for row in result['rows']:
            shoulder = row['shoulder_difference_cm']
            lines.append(f"| {_text(row['size'])} | {row['length_difference_cm']:+.1f} | "
                         + ('비교 보류' if shoulder is None else f'{shoulder:+.1f}')
                         + f" | {row['chest_flat_difference_cm']:+.1f} |")
    else:
        lines += ['| 판매 사이즈 | 기본 길이 차이 | 가슴 단면 여유 | 가슴 둘레 여유 |', '|---|---|---|---|']
        for row in result['rows']:
            lines.append(f"| {_text(row['size'])} | {_range(row['length_difference_cm'])} | "
                         f"{_range(row['chest_flat_ease_cm'])} | {_range(row['chest_circumference_ease_cm'])} |")
    for note in result['notes']:
        lines += ['', note]
    lines += ['', '현재 결과는 치수 비교입니다. 적정 여유분·굴곡 보정을 검증하기 전에는 '
              '특정 판매 사이즈를 정답으로 확정하지 않습니다.']
    return '\n'.join(lines)
