"""알려진 옷 치수와 실착 사진의 수동 표시를 비교하는 실험 층.

cm 비교와 사진 정렬을 분리한다. 눈 사이거리는 같은 착용자 사진의 확대비 정렬에만
사용하며 몸 치수나 카메라 원근을 복원하지 않는다.
"""
import base64
import copy
import html
import io
import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from body_model import normalize_sex, positive_number, predict_body


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEBUG = ROOT / 'debug'
POINTS = {
    'eye_left': '1 눈 중심 · 사진 왼쪽',
    'eye_right': '2 눈 중심 · 사진 오른쪽',
    'neck': '3 뒤목 높이의 정면 보조점',
    'shoulder_left': '4 어깨 봉제선 끝 · 사진 왼쪽',
    'shoulder_right': '5 어깨 봉제선 끝 · 사진 오른쪽',
    'chest_left': '6 겨드랑이 아래 몸통 윤곽 · 사진 왼쪽',
    'chest_right': '7 같은 높이 몸통 윤곽 · 사진 오른쪽',
    'hem_left': '8 밑단 · 사진 왼쪽',
    'hem_right': '9 밑단 · 사진 오른쪽',
}
STRUCTURES = ['일반 어깨선', '래글런', '확인 필요']
# 사진을 보고 잡은 편의용 초안이다. 사용자가 위치를 확인하기 전에는 관측치로 쓰지 않는다.
DRAFTS = {
    '00': {'eye_left': [.501, .127], 'eye_right': [.550, .127], 'neck': [.519, .211],
           'shoulder_left': [.390, .229], 'shoulder_right': [.663, .229],
           'chest_left': [.393, .375], 'chest_right': [.643, .375],
           'hem_left': [.410, .516], 'hem_right': [.637, .516]},
    '01': {'eye_left': [.446, .131], 'eye_right': [.493, .132], 'neck': [.478, .215],
           'shoulder_left': [.333, .245], 'shoulder_right': [.637, .248],
           'chest_left': [.371, .374], 'chest_right': [.605, .374],
           'hem_left': [.376, .497], 'hem_right': [.594, .498]},
}


@dataclass(frozen=True)
class WearCase:
    name: str
    image: Path
    sex: str
    height_cm: float
    weight_kg: float
    garment: str
    length_cm: float
    shoulder_cm: float | None
    chest_flat_cm: float


def read_cases(directory=DEFAULT_DEBUG):
    """현재 debug TXT 형식을 읽고 없는 어깨 치수는 None으로 보존한다."""
    directory = Path(directory)
    blocks = []
    for path in sorted(directory.glob('*.txt')):
        current = None
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if ':' not in line:
                continue
            key, value = (piece.strip() for piece in line.split(':', 1))
            if key == '사진이름':
                if current:
                    blocks.append(current)
                current = {key: value}
            elif current is not None:
                current[key] = value
        if current:
            blocks.append(current)
    cases = {}
    for block in blocks:
        try:
            name = block['사진이름']
            if not name or Path(name).name != name or '/' in name or '\\' in name:
                raise ValueError('사진 이름은 폴더 경로를 포함할 수 없습니다.')
            if name in cases:
                raise ValueError(f'사진 이름이 중복됐습니다: {name}')
            matches = [p for p in directory.iterdir() if p.is_file() and p.stem == name
                       and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}]
            if len(matches) != 1:
                raise ValueError(f'{name}: 대응 사진이 없거나 여러 장입니다.')
            shoulder = block.get('옷 어깨너비(cm)', '-')
            cases[name] = WearCase(
                name, matches[0], normalize_sex(block['성별']),
                positive_number(block['키(cm)'], '키'),
                positive_number(block['체중(kg)'], '체중'),
                block.get('옷 종류 / 착용 사이즈', ''),
                positive_number(block['옷 총장(cm)'], '옷 총장'),
                None if shoulder in {'', '-', '미입력'} else positive_number(shoulder, '옷 어깨너비'),
                positive_number(block['옷 가슴단면(cm)'], '옷 가슴단면'))
        except KeyError as exc:
            raise ValueError(f'TXT 필수 항목이 없습니다: {exc.args[0]}') from exc
    if len(cases) < 2:
        raise ValueError('비교할 사진과 TXT 항목이 최소 두 개 필요합니다.')
    return cases


def open_photo(case):
    with Image.open(case.image) as source:
        photo = ImageOps.exif_transpose(source).convert('RGB')
    photo.thumbnail((1200, 1200))
    return photo


def initial_marks(case):
    return {'case': case.name, 'points': copy.deepcopy(DRAFTS.get(case.name, {})),
            'confirmed': False, 'structure': '래글런' if case.name == '03' else
            ('일반 어깨선' if case.name in {'00', '01'} else '확인 필요'),
            'source': 'manual_visual_draft' if case.name in DRAFTS else 'empty'}


def required_points(marks):
    return [key for key in POINTS if marks['structure'] == '일반 어깨선'
            or not key.startswith('shoulder_')]


def put_point(marks, key, x, y, image_size):
    if key not in POINTS:
        raise ValueError('표시할 점을 선택해주세요.')
    width, height = image_size
    if not all(math.isfinite(v) for v in (x, y)) or not (0 <= x < width and 0 <= y < height):
        raise ValueError('사진 안의 위치를 선택해주세요.')
    updated = copy.deepcopy(marks)
    updated['points'][key] = [x / width, y / height]
    updated['confirmed'] = False
    updated['source'] = 'manual_edited'
    return updated


def validate_marks(marks, size):
    missing = [POINTS[key] for key in required_points(marks) if key not in marks['points']]
    if missing:
        raise ValueError('표시가 필요합니다: ' + ', '.join(missing))
    for key, point in marks['points'].items():
        if key not in POINTS or len(point) != 2 or not all(
                math.isfinite(v) and 0 <= v < 1 for v in point):
            raise ValueError('표시 좌표가 사진 범위 밖입니다.')
    points = pixels(marks, size)
    for left, right in [('eye_left', 'eye_right'), ('chest_left', 'chest_right'),
                        ('hem_left', 'hem_right')]:
        if points[left][0] >= points[right][0]:
            raise ValueError('왼쪽·오른쪽은 사진을 보는 방향 기준으로 표시해주세요.')
    eye_mid = midpoint(points['eye_left'], points['eye_right'])
    hem_mid = midpoint(points['hem_left'], points['hem_right'])
    chest_mid = midpoint(points['chest_left'], points['chest_right'])
    if not eye_mid[1] < points['neck'][1] < chest_mid[1] < hem_mid[1]:
        raise ValueError('눈 → 목 보조점 → 가슴 → 밑단의 세로 순서를 확인해주세요.')
    if math.dist(points['eye_left'], points['eye_right']) < 2:
        raise ValueError('눈 두 점을 서로 다른 눈 중심에 표시해주세요.')
    if marks['structure'] == '일반 어깨선':
        if points['shoulder_left'][0] >= points['shoulder_right'][0]:
            raise ValueError('어깨 봉제선의 좌우 순서를 확인해주세요.')


def confirm_marks(marks, size):
    validate_marks(marks, size)
    updated = copy.deepcopy(marks)
    updated['confirmed'] = True
    updated['source'] = 'user_reviewed_annotation'
    return updated


def pixels(marks, size):
    return {key: (value[0] * size[0], value[1] * size[1])
            for key, value in marks['points'].items()}


def midpoint(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def normalized_outline(marks, size):
    """눈 사이거리=1, 목 보조점=(0,0)인 사진 평면 좌표. 단위는 cm가 아니다."""
    validate_marks(marks, size)
    points = pixels(marks, size)
    left, right = points['eye_left'], points['eye_right']
    dx, dy = right[0] - left[0], right[1] - left[1]
    span = math.hypot(dx, dy)
    cosine, sine = dx / span, dy / span
    neck = points['neck']
    return {key: (((p[0] - neck[0]) * cosine + (p[1] - neck[1]) * sine) / span,
                  (-(p[0] - neck[0]) * sine + (p[1] - neck[1]) * cosine) / span)
            for key, p in points.items()}


def outline_metrics(outline):
    metrics = {'length': midpoint(outline['hem_left'], outline['hem_right'])[1],
               'chest': outline['chest_right'][0] - outline['chest_left'][0]}
    if 'shoulder_left' in outline and 'shoulder_right' in outline:
        metrics['shoulder'] = outline['shoulder_right'][0] - outline['shoulder_left'][0]
    if any(value <= 0 for value in metrics.values()):
        raise ValueError('사진의 기울기와 표시 위치를 확인해주세요.')
    return metrics


def ratio_hypothesis(outline, case_a, case_b, shoulder_enabled):
    """상품 치수비를 적용하는 미검증 가설. 관측한 B 사진 좌표는 사용하지 않는다."""
    result = copy.deepcopy(outline)
    length_ratio = case_b.length_cm / case_a.length_cm
    chest_ratio = case_b.chest_flat_cm / case_a.chest_flat_cm
    for key in ['chest_left', 'chest_right', 'hem_left', 'hem_right']:
        x, y = result[key]
        result[key] = (x * chest_ratio, y * length_ratio if key.startswith('hem_') else y)
    if shoulder_enabled:
        ratio = case_b.shoulder_cm / case_a.shoulder_cm
        for key in ['shoulder_left', 'shoulder_right']:
            result[key] = (result[key][0] * ratio, result[key][1])
    else:
        result.pop('shoulder_left', None)
        result.pop('shoulder_right', None)
    return result


def numeric_report(bundle, case_a, case_b):
    lines = ['### 신체·옷의 cm 비교', '',
             '상품 총장은 사용자가 알려준 「뒤목 부근 옷 상단→밑단」을 비교 기준으로 사용합니다.', '',
             '| 사진 | 신체 어깨사이너비 예측 | 신체 기준 총장 예측 | 가슴둘레 예측 |', '|---|---|---|---|']
    predicted = {}
    for case in [case_a, case_b]:
        profile = predict_body(bundle, case.sex, case.height_cm, case.weight_kg, model='ridge')
        predicted[case.name] = profile
        values = [f"{profile[key]['value_cm']:.1f} ({profile[key]['low_cm']:.1f}~"
                  f"{profile[key]['high_cm']:.1f})" for key in
                  ['shoulder_breadth_cm', 'body_length_cm', 'chest_circumference_cm']]
        lines.append(f"| {case.name} | " + ' | '.join(values) + ' |')
    lines += ['', 'Ridge 예측, 단위 cm. 괄호는 90% 목표 예측 범위이며 신체 실측값이 아닙니다.', '',
              '| 사진 | 상품 총장 | 상품 어깨너비 | 가슴단면 | 기본 길이 차이 | 가슴 둘레 여유 |',
              '|---|---:|---:|---:|---|---|']
    for case in [case_a, case_b]:
        profile = predicted[case.name]
        length, chest = profile['body_length_cm'], profile['chest_circumference_cm']
        shoulder = '미입력' if case.shoulder_cm is None else f'{case.shoulder_cm:g}'
        lines.append(f'| {case.name} | {case.length_cm:g} | {shoulder} | {case.chest_flat_cm:g} | '
                     f"{case.length_cm-length['value_cm']:+.1f} "
                     f"({case.length_cm-length['high_cm']:+.1f}~{case.length_cm-length['low_cm']:+.1f}) | "
                     f"{2*case.chest_flat_cm-chest['value_cm']:+.1f} "
                     f"({2*case.chest_flat_cm-chest['high_cm']:+.1f}~{2*case.chest_flat_cm-chest['low_cm']:+.1f}) |")
    lines += ['', '**기본 길이 차이 = 상품 총장 − 예측 신체 기준 총장**. 목뒤높이를 공통 기준으로 '
              '위앞엉덩뼈가시 높이보다 옷이 내려오는 길이를 생각하는 출발값입니다. 굴곡·처짐 보정 전입니다.',
              '**가슴 둘레 여유 = 가슴단면×2 − 예측 가슴둘레**. 사진의 가로 폭이나 한쪽 이동량으로 직접 변환하지 않습니다.', '',
              f'**상품 {case_a.name} → {case_b.name}**: '
              f'총장 {case_b.length_cm-case_a.length_cm:+g}cm / '
              f'가슴단면 {case_b.chest_flat_cm-case_a.chest_flat_cm:+g}cm / '
              + ('어깨너비는 미입력으로 비교 보류.' if None in (case_a.shoulder_cm, case_b.shoulder_cm)
                 else f'어깨너비 {case_b.shoulder_cm-case_a.shoulder_cm:+g}cm.')]
    return '\n'.join(lines)


def observation_report(case_a, case_b, marks_a, marks_b, size_a, size_b, same_person):
    if not same_person:
        return '사진을 겹쳐 비교하려면 같은 착용자인지 확인해주세요.'
    if not marks_a['confirmed'] or not marks_b['confirmed']:
        return '현재 표시는 미확인 초안입니다. 두 사진의 점을 수정하고 「표시 확인」을 누르면 관측 비교를 계산합니다.'
    a = outline_metrics(normalized_outline(marks_a, size_a))
    b = outline_metrics(normalized_outline(marks_b, size_b))
    shoulder_enabled = (marks_a['structure'] == marks_b['structure'] == '일반 어깨선'
                        and None not in (case_a.shoulder_cm, case_b.shoulder_cm))
    lines = ['### 확인된 표시로 계산한 사진 비교', '',
             '| 항목 | 상품 치수 변화 | 사진 관측 변화 |', '|---|---:|---:|']
    for key, label, ratio in [('length', '목 보조점→밑단 낙차', case_b.length_cm/case_a.length_cm),
                              ('chest', '겨드랑이 아래 몸통 폭', case_b.chest_flat_cm/case_a.chest_flat_cm)]:
        lines.append(f'| {label} | {(ratio-1)*100:+.1f}% | {(b[key]/a[key]-1)*100:+.1f}% |')
    if shoulder_enabled:
        lines.append(f"| 어깨 봉제선 폭 | {(case_b.shoulder_cm/case_a.shoulder_cm-1)*100:+.1f}% | "
                     f"{(b['shoulder']/a['shoulder']-1)*100:+.1f}% |")
    else:
        lines += ['', '어깨는 래글런·구조 미확인 또는 상품 어깨 치수 누락으로 비교를 보류합니다.']
    lines += ['', '사진은 눈 사이거리와 눈선 각도로 확대비·회전을 맞추고 목 보조점을 겹쳤습니다. '
              '단위는 상대비이며 cm 환산·원근 복원이 아닙니다.',
              '실선은 수동 관측 윤곽, 점선은 상품 치수비만 적용한 미검증 가설입니다. '
              '둘의 차이를 굴곡 보정값으로 저장하거나 합성에 자동 적용하지 않습니다.']
    return '\n'.join(lines)


def _photo_data(photo):
    buffer = io.BytesIO()
    photo.save(buffer, format='JPEG', quality=83)
    return 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


def photo_svg(photo, marks, title):
    width, height = photo.size
    points = pixels(marks, photo.size)
    content = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
               f'role="img" aria-label="{html.escape(title)}" style="width:100%;max-height:540px">',
               f'<image width="{width}" height="{height}" href="{_photo_data(photo)}"/>']
    for i, (key, _) in enumerate(POINTS.items(), 1):
        if key not in points:
            continue
        x, y = points[key]
        content += [f'<circle cx="{x}" cy="{y}" r="7" fill="#ffbf36" stroke="#111" stroke-width="2"/>',
                    f'<text x="{x+10}" y="{y-8}" font-size="18" fill="white" '
                    f'stroke="#111" stroke-width=".4">{i}</text>']
    content.append('</svg>')
    return ''.join(content)


def overlay_svg(case_a, case_b, marks_a, marks_b, size_a, size_b):
    a, b = normalized_outline(marks_a, size_a), normalized_outline(marks_b, size_b)
    enabled = (marks_a['structure'] == marks_b['structure'] == '일반 어깨선'
               and None not in (case_a.shoulder_cm, case_b.shoulder_cm))
    if not enabled:
        for outline in [a, b]:
            outline.pop('shoulder_left', None)
            outline.pop('shoulder_right', None)
    hypothesis = ratio_hypothesis(a, case_a, case_b, enabled)
    all_points = [p for outline in [a, b, hypothesis] for key, p in outline.items()
                  if not key.startswith('eye_')]
    xmin, xmax = min(p[0] for p in all_points)-.6, max(p[0] for p in all_points)+.6
    ymin, ymax = -.6, max(p[1] for p in all_points)+.6
    pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{xmin} {ymin} '
              f'{xmax-xmin} {ymax-ymin}" style="width:100%;height:420px;background:#f3f6f8" '
              'role="img" aria-label="목 보조점과 눈 사이거리 기준 윤곽 비교">']
    for outline, color, dashed in [(a, '#087f8c', False), (b, '#de5b44', False),
                                  (hypothesis, '#7c4dba', True)]:
        keys = ['neck', 'shoulder_left', 'chest_left', 'hem_left', 'hem_right',
                'chest_right', 'shoulder_right', 'neck']
        coords = ' '.join(f'{outline[k][0]},{outline[k][1]}' for k in keys if k in outline)
        pieces.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width=".035" '
                      + ('stroke-dasharray=".12 .08"' if dashed else '') + '/>')
    pieces.append('<circle cx="0" cy="0" r=".07" fill="#222"/></svg>')
    return ''.join(pieces)


def render_comparison(case_a, case_b, marks_a, marks_b, photo_a, photo_b, same_person):
    panels = []
    for case, marks, photo in [(case_a, marks_a, photo_a), (case_b, marks_b, photo_b)]:
        status = '표시 확인됨' if marks['confirmed'] else '표시 초안 · 미확인'
        panels.append(f'<section style="flex:1;min-width:220px"><b>{html.escape(case.name)} · '
                      f'{status}</b>{photo_svg(photo, marks, case.name)}</section>')
    if same_person and marks_a['confirmed'] and marks_b['confirmed']:
        panels.append('<section style="flex:1;min-width:270px"><b>윤곽 정렬</b>'
                      '<p style="font-size:12px">청록=A · 주황=B · 보라 점선=치수비 가설</p>'
                      + overlay_svg(case_a, case_b, marks_a, marks_b, photo_a.size, photo_b.size)
                      + '</section>')
    return '<div style="display:flex;gap:18px;flex-wrap:wrap">' + ''.join(panels) + '</div>'
