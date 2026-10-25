"""확인한 옷 기준점으로 ratio_v0 목표선을 그린다. 신체/옷 분할 모델이 아니다."""
import copy
import hashlib
import html
import math

from PIL import Image, ImageDraw, ImageOps

from fit_plan import FitPlan


ANCHORS = {
    'neck': '1 목 부근 옷 상단의 정면 보조점',
    'chest_left': '2 몸통 윤곽 · 사진 왼쪽 (소매 제외)',
    'chest_right': '3 같은 높이 몸통 윤곽 · 사진 오른쪽',
    'hem_left': '4 밑단 · 사진 왼쪽',
    'hem_right': '5 밑단 · 사진 오른쪽',
    'shoulder_left': '6 어깨 봉제선 끝 · 사진 왼쪽 (선택)',
    'shoulder_right': '7 어깨 봉제선 끝 · 사진 오른쪽 (선택)',
}
REQUIRED = tuple(ANCHORS)[:5]
COLORS = {'reference': '#40dfff', 'target': '#ff9c39'}


def prepare_photo(image):
    if image is None:
        raise ValueError('기준 옷을 입은 사진을 업로드해주세요.')
    photo = ImageOps.exif_transpose(image).convert('RGB')
    photo.thumbnail((1200, 1200))
    if min(photo.size) < 20:
        raise ValueError('사진 해상도를 확인해주세요.')
    return photo


def photo_info(image, name='upload'):
    photo = prepare_photo(image)
    digest = hashlib.sha256(f'{photo.width}x{photo.height}:RGB:'.encode() + photo.tobytes()).hexdigest()
    return {'name': name, 'width': photo.width, 'height': photo.height,
            'sha256': digest, 'hash_definition': 'exif_transposed_rgb_thumbnail_1200_pixels'}


def new_marks(info, points=None, source='empty'):
    return {'photo_sha256': info['sha256'], 'points': copy.deepcopy(points or {}),
            'source': source, 'point_sources': {key: source for key in (points or {})},
            'confirmed': False, 'shoulders_confirmed': False}


def put_anchor(marks, key, x, y, info):
    if key not in ANCHORS:
        raise ValueError('표시할 기준점을 선택해주세요.')
    if marks.get('photo_sha256') != info['sha256']:
        raise ValueError('사진이 바뀌었습니다. 기준점을 다시 확인해주세요.')
    if (any(isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) for value in (x, y))
            or not (0 <= x < info['width'] and 0 <= y < info['height'])):
        raise ValueError('사진 안의 위치를 선택해주세요.')
    edited = copy.deepcopy(marks)
    edited['points'][key] = [x / info['width'], y / info['height']]
    edited['point_sources'][key] = 'manual_edited'
    edited.update(confirmed=False, shoulders_confirmed=False, source='manual_edited')
    return edited


def _points(marks, info, shoulders=False):
    if marks.get('photo_sha256') != info['sha256']:
        raise ValueError('사진과 기준점의 연결을 다시 확인해주세요.')
    required = tuple(ANCHORS) if shoulders else REQUIRED
    raw = marks.get('points', {})
    missing = [ANCHORS[key] for key in required if key not in raw]
    if missing:
        raise ValueError('위치 확인이 필요합니다: ' + ', '.join(missing))
    for key, point in raw.items():
        if (key not in ANCHORS or not isinstance(point, (list, tuple)) or len(point) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) or not 0 <= v < 1 for v in point)):
            raise ValueError('기준점 좌표가 사진 범위 밖이거나 올바르지 않습니다.')
    points = {key: (raw[key][0] * info['width'], raw[key][1] * info['height']) for key in required}
    pairs = [('chest_left', 'chest_right'), ('hem_left', 'hem_right')]
    if shoulders:
        pairs.append(('shoulder_left', 'shoulder_right'))
    for left, right in pairs:
        if points[right][0] - points[left][0] < 2:
            raise ValueError('좌우는 사진을 보는 방향으로 서로 다른 윤곽에 표시해주세요.')
    if not points['neck'][1] < min(points[key][1] for key in ('chest_left', 'chest_right')):
        raise ValueError('목 보조점은 몸통 기준점보다 위에 표시해주세요.')
    if not max(points[key][1] for key in ('chest_left', 'chest_right')) < min(
            points[key][1] for key in ('hem_left', 'hem_right')):
        raise ValueError('몸통 두 점은 밑단 두 점보다 위에 표시해주세요.')
    if not points['chest_left'][0] < points['neck'][0] < points['chest_right'][0]:
        raise ValueError('목 보조점은 몸통 좌우 사이에 표시해주세요.')
    if shoulders and any(not points['neck'][1] <= points[key][1] < min(
            points['chest_left'][1], points['chest_right'][1])
            for key in ('shoulder_left', 'shoulder_right')):
        raise ValueError('어깨 봉제선 끝은 목 보조점과 몸통 기준점 사이에 표시해주세요.')
    return points


def confirm_anchors(marks, info, shoulders=False):
    _points(marks, info, shoulders)
    confirmed = copy.deepcopy(marks)
    confirmed.update(confirmed=True, shoulders_confirmed=bool(shoulders))
    return confirmed


def create_preview(plan, info, marks, same_garment=False, length_compatible=False,
                   chest_compatible=False, shoulders=False):
    if not isinstance(plan, FitPlan):
        raise ValueError('현재 입력으로 치수 비교를 먼저 실행해주세요.')
    if plan.status == 'numeric_only':
        raise ValueError(plan.reasons[0])
    if not same_garment:
        raise ValueError('입력한 기준 옷과 사진 속 옷이 같은 옷인지 확인해주세요.')
    if not marks.get('confirmed'):
        raise ValueError('사진의 기준점 위치를 먼저 확인해주세요.')
    shoulder_enabled = bool(shoulders and plan.ratios['shoulder'] is not None)
    if shoulder_enabled and not marks.get('shoulders_confirmed'):
        raise ValueError('어깨 봉제선 두 점을 추가하고 기준점 확인을 눌러주세요.')
    points = _points(marks, info, shoulder_enabled)
    applied = {'length': bool(length_compatible), 'chest': bool(chest_compatible),
               'shoulder': shoulder_enabled}
    reasons = list(plan.reasons)
    if not length_compatible:
        reasons.append('총장 변형 보류: 두 옷의 총장 측정 시작점·방향이 같은지 확인해주세요.')
    if not chest_compatible:
        reasons.append('몸통 폭 변형 보류: 두 옷의 가슴단면 위치·구조가 비교 가능한지 확인해주세요.')
    if not shoulders and plan.ratios['shoulder'] is not None:
        reasons.append('어깨 변형은 선택하지 않았습니다.')
    if not any(applied.values()):
        raise ValueError('적용할 부위의 치수 기준을 한 개 이상 확인해주세요.')

    # 목→밑단 중심을 세로축으로 삼아 사진 평면의 기울기를 유지한다.
    neck = points['neck']
    hem_mid = tuple((points['hem_left'][i] + points['hem_right'][i]) / 2 for i in (0, 1))
    length = math.dist(neck, hem_mid)
    vy = ((hem_mid[0] - neck[0]) / length, (hem_mid[1] - neck[1]) / length)
    vx = (vy[1], -vy[0])

    def local(p):
        return (sum((p[i]-neck[i]) * vx[i] for i in (0, 1)),
                sum((p[i]-neck[i]) * vy[i] for i in (0, 1)))

    target = copy.deepcopy(points)
    for key, p in points.items():
        x, y = local(p)
        original_local = (x, y)
        if key.startswith(('chest_', 'hem_')) and applied['chest']:
            x *= plan.ratios['chest']
        if key.startswith('hem_') and applied['length']:
            y *= plan.ratios['length']
        if key.startswith('shoulder_') and applied['shoulder']:
            x *= plan.ratios['shoulder']
        if (x, y) != original_local:
            target[key] = tuple(neck[i] + x * vx[i] + y * vy[i] for i in (0, 1))
    # 뷰어에서 잘라 보여주거나 뒤집힌 윤곽을 정상 결과로 저장하지 않는다.
    if any(not 0 <= p[0] < info['width'] or not 0 <= p[1] < info['height'] for p in target.values()):
        raise ValueError('목표 윤곽이 사진 밖으로 나갑니다. 여백이 있는 사진이나 치수를 확인해주세요.')
    if min(local(target[key])[1] for key in ('hem_left', 'hem_right')) <= max(
            local(target[key])[1] for key in ('chest_left', 'chest_right')):
        raise ValueError('목표 밑단이 몸통 기준점 위로 올라갑니다. 기준점·총장 호환성을 확인해주세요.')
    return {'schema_version': 1, 'fit_plan': plan.to_dict(), 'photo': copy.deepcopy(info),
            'anchors': copy.deepcopy(marks), 'reference_points_px': points,
            'target_points_px': target, 'applied': applied, 'reasons': reasons,
            'review': {'same_garment': True, 'length_compatible': bool(length_compatible),
                       'chest_compatible': bool(chest_compatible), 'shoulders': bool(shoulders)},
            'status': 'preview_ready', 'transform_version': 'ratio_v0',
            'notes': ['상품 치수비를 적용한 윤곽 가설입니다. 굴곡·처짐·원근 보정 전입니다.',
                      '기준점 기반 몸통·길이 보조선입니다. 전체 옷 분할·소매·합성은 후속 단계입니다.',
                      '신체 예측 범위와 사진 윤곽의 오차는 별개이며 윤곽 오차는 아직 검증하지 않았습니다.']}


def render_outline(image, marks=None, preview=None):
    photo = prepare_photo(image)
    draw = ImageDraw.Draw(photo)
    width = max(2, round(photo.width / 240))

    def outline(points, color):
        # 어깨가 미확인일 때 목과 겨드랑이를 잇는 가짜 삼각형 윤곽을 만들지 않는다.
        keys = ['chest_left', 'hem_left', 'hem_right', 'chest_right', 'chest_left']
        draw.line([tuple(points[key]) for key in keys], fill=color, width=width)
        neck = tuple(points['neck'])
        hem = tuple((points['hem_left'][i]+points['hem_right'][i])/2 for i in (0, 1))
        draw.line([neck, hem], fill=color, width=max(1, width//2))
        radius = width+1
        draw.ellipse((neck[0]-radius, neck[1]-radius, neck[0]+radius, neck[1]+radius), fill=color)
        if 'shoulder_left' in points and 'shoulder_right' in points:
            draw.line([tuple(points['shoulder_left']), tuple(points['shoulder_right'])], fill=color, width=width)
            for side in ('left', 'right'):
                draw.line([tuple(points['shoulder_'+side]), tuple(points['chest_'+side])], fill=color, width=width)

    if preview:
        outline(preview['reference_points_px'], COLORS['reference'])
        outline(preview['target_points_px'], COLORS['target'])
    elif marks:
        raw = marks.get('points', {})
        points = {key: (p[0] * photo.width, p[1] * photo.height) for key, p in raw.items()}
        if all(key in points for key in REQUIRED):
            outline(points, COLORS['reference'])
        radius = max(4, round(photo.width / 120))
        for key, (x, y) in points.items():
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=COLORS['reference'], outline='black')
            draw.text((x+radius+2, y-radius), str(tuple(ANCHORS).index(key)+1), fill='black',
                      stroke_width=2, stroke_fill='white')
    return photo


def preview_report(preview):
    plan = preview['fit_plan']
    label = html.escape(plan['selected_size']).replace('|', '\\|')
    lines = [f'### 선택 사이즈: {label}', '',
             '하늘색: 기준 옷 윤곽 / 주황색: 선택 사이즈 목표 윤곽', '',
             '| 부위 | 적용 비율 | 상태 |', '|---|---:|---|']
    for key, label in [('length', '총장'), ('chest', '몸통 폭'), ('shoulder', '어깨')]:
        ratio = plan['ratios'][key]
        enabled = preview['applied'][key]
        lines.append(f"| {label} | {f'{ratio:.4f}' if enabled else '-'} | {'적용' if enabled else '보류'} |")
    lines += ['', *[f'- {reason}' for reason in preview['reasons']], '',
              *[f'> {note}' for note in preview['notes']]]
    return '\n'.join(lines)
