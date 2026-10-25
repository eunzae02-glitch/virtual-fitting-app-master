"""상의 사진 추정 + 확인한 치수표로 사이즈를 비교하는 순수 계산층.

사진은 body_measure가 담당한다. 여기서는 옷/몸 단위를 섞지 않고,
가슴 경계가 안 보일 때 검증된 보정 자료가 없으면 숫자를 만들지 않는다.
"""
import csv
import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from chart_extract import (check_monotonic, check_ranges, normalize_size_name,
                           size_sort_key)
from size_fit import SizeChart, recommend


@dataclass(frozen=True)
class RangeCM:
    low: float
    high: float
    source: str

    def __post_init__(self):
        if not (math.isfinite(self.low) and math.isfinite(self.high)
                and 0 < self.low <= self.high):
            raise ValueError('치수 범위를 확인해주세요.')

    @property
    def middle(self) -> float:
        return (self.low + self.high) / 2


@dataclass
class UpperBody:
    shoulder: RangeCM
    chest_circ: Optional[RangeCM]
    notes: List[str] = field(default_factory=list)


@dataclass
class UpperChart:
    chart: SizeChart
    raw: dict
    shoulder_comparable: bool = False
    warnings: List[str] = field(default_factory=list)


@dataclass
class UpperResult:
    candidates: List[str]
    rows: List[Tuple[str, Optional[float], Optional[float], Optional[float], float]]
    notes: List[str]
    status: str


def _positive(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{label} 숫자를 확인해주세요.') from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f'{label}는 0보다 큰 유한한 값이어야 합니다.')
    return number


def parse_upper_chart(raw: dict, shoulder_comparable=False) -> UpperChart:
    """사용자가 확인한 원문 표를 옷 가슴단면 cm로 변환한다.

    허용하는 가슴 라벨: 가슴단면, 옷 가슴둘레(완성 둘레임을 명시).
    단순 '가슴둘레'는 옷 완성 둘레/권장 신체둘레가 섞일 수 있어 거부한다.
    """
    if not isinstance(raw, dict) or not isinstance(raw.get('sizes'), dict):
        raise ValueError('치수표 JSON에 sizes 객체가 필요합니다.')
    unit = str(raw.get('unit', 'cm')).strip().lower()
    if unit not in ('cm', '센티미터'):
        raise ValueError('치수표 단위가 cm인지 확인하고 환산해주세요.')
    sizes: Dict[str, Dict[str, float]] = {}
    for name, cells in raw['sizes'].items():
        if not isinstance(cells, dict):
            raise ValueError(f'{name} 행을 읽을 수 없습니다.')
        size = normalize_size_name(name)
        if not size or size in sizes:
            raise ValueError(f'{name} 사이즈가 중복되거나 비어 있습니다. 색상별 행은 하나로 정리해주세요.')
        chest = None
        shoulder = None
        for label, value in cells.items():
            key = str(label).replace(' ', '').replace('_', '').lower()
            if '신체' in key or '권장' in key:
                if '가슴' in key or '흉위' in key:
                    raise ValueError(f'{name}의 {label}은 옷 실측이 아닌 권장 신체 치수일 수 있습니다.')
                continue
            if key in ('가슴단면', '가슴너비', 'chestflat', 'chestwidth'):
                if chest is not None:
                    raise ValueError(f'{name} 가슴 항목이 중복됐습니다.')
                chest = _positive(value, f'{name} {label}')
            elif key in ('옷가슴둘레', '완성가슴둘레', 'garmentchestcircumference'):
                if chest is not None:
                    raise ValueError(f'{name} 가슴 항목이 중복됐습니다.')
                chest = _positive(value, f'{name} {label}') / 2
            elif key in ('가슴둘레', '가슴', '흉위', 'chest'):
                raise ValueError(f'{name}의 {label}이 옷 완성 둘레인지 권장 신체둘레인지 확인해주세요.')
            elif key in ('어깨너비', '어깨', '견폭', 'shoulder'):
                shoulder = _positive(value, f'{name} {label}')
        if chest is None:
            raise ValueError(f'{name} 사이즈에 옷 가슴단면이 없습니다.')
        sizes[size] = {'chest': chest}
        if shoulder is not None:
            sizes[size]['shoulder'] = shoulder
    if not sizes:
        raise ValueError('치수표에 사이즈가 없습니다.')
    problems = check_monotonic(sizes)
    if problems:
        raise ValueError('치수표를 수정해주세요: ' + ' '.join(problems))
    warnings = check_ranges(sizes)
    chart = SizeChart(name=str(raw.get('name') or '상의 치수표'), category='upper',
                      sizes=dict(sorted(sizes.items(), key=lambda item: size_sort_key(item[0]))),
                      source=str(raw.get('source') or '사용자 확인'))
    return UpperChart(chart, raw, bool(shoulder_comparable), warnings)


def fit_chest_calibration(csv_path: Optional[str], basis: str, shoulder_cm: float) -> Optional[RangeCM]:
    """짝지은 실측 CSV에서만 그룹별 평균 비율과 잔차 범위를 구한다.

    헤더: basis,shoulder_cm,chest_circ_cm. 그룹당 유효 표본 10개 미만이면 미사용.
    이 간단한 비율 모형의 오차는 해당 데이터 내부 잔차이므로 대외 정확도로 주장하지 않는다.
    """
    if not csv_path or basis not in ('남', '녀'):
        return None
    ratios = []
    try:
        with open(csv_path, newline='', encoding='utf-8-sig') as stream:
            for row in csv.DictReader(stream):
                if row.get('basis', '').strip() != basis:
                    continue
                shoulder = _positive(row.get('shoulder_cm'), '어깨너비')
                chest = _positive(row.get('chest_circ_cm'), '가슴둘레')
                if 25 <= shoulder <= 70 and 50 <= chest <= 180:
                    ratios.append(chest / shoulder)
    except (OSError, ValueError, KeyError):
        return None
    if len(ratios) < 10:
        return None
    center = statistics.median(ratios)
    errors = sorted(abs(r - center) for r in ratios)
    spread = errors[math.ceil(0.9 * len(errors)) - 1]
    if spread == 0:
        return None  # 중복·합성 데이터로 불확실성을 0이라고 주장하지 않음
    if center <= spread:
        return None  # 지나치게 이질적인 자료는 유효한 양수 범위를 만들 수 없음
    return RangeCM(round(shoulder_cm * (center - spread), 1),
                   round(shoulder_cm * (center + spread), 1),
                   f'{basis} 실측 {len(ratios)}명 평균 관계(내부 잔차, 외부 검증 필요)')


def estimate_upper(photo, height_cm, basis='선택 안 함', calibration_csv=None,
                   manual_chest=None, measurement_fn=None) -> UpperBody:
    """유효 사진만 사용한다. 직접 입력 가슴둘레는 사진 실패를 우회하지 않는다."""
    from body_measure import measure

    height = _positive(height_cm, '키')
    if not 90 <= height <= 230:
        raise ValueError('키 입력(cm)을 확인해주세요. 90~230cm 범위에서 입력할 수 있습니다.')
    if photo is None:
        raise ValueError('정면 전신 사진을 올려주세요.')
    estimate = (measurement_fn or measure)(photo, height)
    # 기존 measure.unsafe에는 이번 범위 밖인 골반 고정비율의 이상치도 포함된다.
    # 어깨/키 스케일 실패는 차단하되 골반 수치만으로 상체 분석을 막지 않는다.
    nonfatal_prefixes = ('가슴 경계', '허리 경계', '엉덩이둘레는', 'hip ', 'chest ')
    critical = [note for note in estimate.warnings
                if not note.startswith(nonfatal_prefixes)] if estimate.unsafe else []
    if (critical or not estimate.has_segmentation
            or 'shoulder' not in estimate.measurements_cm
            or getattr(estimate, 'px_per_cm', 1) <= 0):
        raise ValueError('사진을 다시 올려주세요. ' + ' '.join(estimate.warnings))
    shoulder = _positive(estimate.measurements_cm['shoulder'], '어깨너비')
    body = UpperBody(RangeCM(shoulder, shoulder, '사진 기반 추정(보정 오차 미검증)'), None)
    body.notes.append('사진 어깨너비는 실측이 아니며 보정 오차 범위가 아직 검증되지 않았습니다.')
    chest_implausible = any(note.startswith('chest ') for note in estimate.warnings)
    if 'chest' in estimate.measurements_cm and not chest_implausible:
        circumference = _positive(estimate.measurements_cm['chest'] * 2, '사진 가슴둘레')
        body.chest_circ = RangeCM(circumference, circumference, '사진 실루엣·깊이 가정 추정')
        body.notes.append('사진 가슴둘레는 깊이를 가정한 참고값이며 오차 범위 미검증입니다.')
    else:
        body.notes.append('가슴 경계가 팔/옷과 겹쳐 사진 추정을 사용하지 않았습니다.')
        body.chest_circ = fit_chest_calibration(calibration_csv, basis, shoulder)
        if body.chest_circ is None:
            body.notes.append('검증된 남/녀 평균 자료가 없어 가슴둘레를 자동 대체하지 않았습니다.')
    if manual_chest is not None:
        chest = _positive(manual_chest, '직접 입력 가슴둘레')
        body.chest_circ = RangeCM(chest, chest, '사용자 직접 입력(실측 여부 확인 필요)')
        body.notes.append('가슴둘레는 직접 입력값을 우선했습니다.')
    return body


def recommend_upper(body: UpperBody, target: UpperChart,
                    reference: Optional[UpperChart] = None,
                    worn_size: Optional[str] = None,
                    worn_fit: str = '적당함') -> UpperResult:
    """가슴 여유 범위와 기존 반팔의 상대적 핏을 비교한다."""
    if body.chest_circ is None and reference is None:
        return UpperResult([], [], body.notes + ['가슴둘레 직접 입력 또는 검증된 보정 자료가 필요합니다.'],
                           '추천 보류')
    if reference is not None and (not worn_size or normalize_size_name(worn_size) not in reference.chart.sizes):
        raise ValueError('평소 입는 반팔의 실제 사이즈를 치수표에서 선택해주세요.')
    if reference is not None and worn_fit not in ('작음', '적당함', '큼'):
        raise ValueError('평소 입는 반팔의 착용감을 선택해주세요.')

    notes = list(body.notes)
    notes.extend('대상 치수표 재확인: ' + warning for warning in target.warnings)
    if reference is not None:
        notes.extend('기존 반팔 치수표 재확인: ' + warning for warning in reference.warnings)
    preferred_flat = None
    if reference is not None:
        preferred_flat = reference.chart.sizes[normalize_size_name(worn_size)]['chest']
        notes.append('기존 반팔 치수는 몸 치수로 치환하지 않고 원하는 옷 가슴 폭의 비교 기준으로 사용했습니다.')
        if worn_fit != '적당함':
            notes.append(f'기존 반팔이 {worn_fit}이므로 새 옷에서 가슴 폭이 '
                         + ('더 큰' if worn_fit == '작음' else '더 작은') + ' 후보를 우선합니다.')
    else:
        notes.append('기존 반팔 없이 기존 핏 점수의 레귤러 기준을 임시로 사용합니다(실측 보정 전).')

    low_body = body.chest_circ.low / 2 if body.chest_circ else None
    high_body = body.chest_circ.high / 2 if body.chest_circ else None
    rows = []
    for size, dims in target.chart.sizes.items():
        chest_flat = dims['chest']
        shoulder_ease = (dims.get('shoulder', 0) - body.shoulder.middle
                         if target.shoulder_comparable and 'shoulder' in dims else None)
        rows.append((size, chest_flat - high_body if high_body is not None else None,
                     chest_flat - low_body if low_body is not None else None,
                     shoulder_ease, chest_flat))
    wearable = [r for r in rows if r[1] is None or r[1] >= 0]
    if not wearable:
        return UpperResult([], rows, notes + ['모든 사이즈가 추정 몸 가슴단면보다 작습니다.'], '추천 불가')
    if body.chest_circ is None:
        notes.append('몸 가슴둘레는 추정 불가입니다. 기존 반팔과 새 옷의 가슴 폭만 상대 비교하며 착용 가능 여부는 미확인입니다.')

    if preferred_flat is not None:
        if worn_fit == '적당함':
            distances = {r[0]: abs(r[4] - preferred_flat) for r in wearable}
        elif worn_fit == '작음':
            suitable = [r for r in wearable if r[4] > preferred_flat]
            if not suitable:
                return UpperResult([], rows, notes + ['기존 옷보다 가슴 폭이 넓은 사이즈가 없습니다.'], '추천 보류')
            distances = {r[0]: r[4] - preferred_flat for r in suitable}
        else:
            suitable = [r for r in wearable if r[4] < preferred_flat]
            if not suitable:
                return UpperResult([], rows, notes + ['기존 옷보다 가슴 폭이 좁으면서 몸보다 큰 사이즈가 없습니다.'], '추천 보류')
            distances = {r[0]: preferred_flat - r[4] for r in suitable}
        best = min(distances.values())
        candidates = [s for s, d in distances.items() if d == best]
    else:
        # 기존 size_fit의 휴리스틱을 범위 양 끝·중간에서 평가. 결과가 갈리면 복수 후보.
        candidates = []
        for circumference in (body.chest_circ.low, body.chest_circ.middle,
                              body.chest_circ.high):
            measure = {'chest': circumference / 2}
            if target.shoulder_comparable:
                measure['shoulder'] = body.shoulder.middle
            rec = recommend(target.chart, measure)
            if rec.best is not None and rec.best.wearable and rec.best.size not in candidates:
                candidates.append(rec.best.size)
        candidates = [s for s in candidates if s in {r[0] for r in wearable}]
    if not candidates:
        return UpperResult([], rows, notes + ['검증 가능한 후보가 없습니다. 직접 실측치를 확인해주세요.'], '추천 보류')
    if (body.chest_circ and body.chest_circ.low == body.chest_circ.high
            and '사진' in body.chest_circ.source):
        notes.append('사진 오차 범위가 검증되지 않아 단일 후보도 잠정 추천입니다.')
    status = '잠정 추천' if len(candidates) == 1 else '복수 후보'
    if len(candidates) > 1:
        notes.append('경계 사이즈입니다. 가슴둘레 직접 실측이나 기존 반팔 치수표로 좁혀주세요.')
    return UpperResult(candidates, rows, notes, status)
