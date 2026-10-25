"""FASHN 기본 합성과 확인한 옷 마스크를 목표 윤곽에 연결하는 CPU 후처리 실험."""
import copy
from collections import OrderedDict
import hashlib
import html
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

import numpy as np
from PIL import Image, ImageDraw

from fit_plan import create_fit_plan
from garment_outline import create_preview, photo_info, prepare_photo
from garment_repair import repair_exposed, repair_overlay


_repair_reference_masks = OrderedDict()
_repair_reference_lock = threading.Lock()


def checked_preview(preview, person):
    """저장/웹 상태의 목표선을 원래 치수·확인 좌표에서 다시 만든다."""
    if not preview or preview.get('status') != 'preview_ready':
        raise ValueError('3단계에서 목표 윤곽을 먼저 만들어주세요.')
    if photo_info(person)['sha256'] != preview['photo']['sha256']:
        raise ValueError('목표 윤곽과 인물 사진이 다릅니다. 다시 계산해주세요.')
    review = preview['review']
    plan = create_fit_plan(preview['fit_plan']['comparison'], preview['fit_plan']['selected_size'])
    return create_preview(plan, preview['photo'], preview['anchors'],
                          review['same_garment'], review['length_compatible'],
                          review['chest_compatible'], False)


def _digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode('utf-8')).hexdigest()


def engine_settings(seed=42, steps=50, guidance=2.5):
    try:
        if any(isinstance(v, bool) for v in (seed, steps, guidance)):
            raise ValueError
        if not float(seed).is_integer() or not float(steps).is_integer():
            raise ValueError
        seed, steps, guidance = int(seed), int(steps), float(guidance)
        if not 0 <= seed < 2**32 or not 1 <= steps <= 200 or not math.isfinite(guidance) or guidance <= 0:
            raise ValueError
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('시드는 0~4294967295 정수, 스텝은 1~200 정수, guidance는 양수여야 합니다.') from exc
    return {'seed': seed, 'steps': steps, 'guidance_scale': guidance,
            'cloth_type': 'upper', 'garment_photo_type': 'flat-lay'}


def fashn_environment():
    """다운로드/모델 로딩 없이 현재 설치와 가중치 경로만 확인한다."""
    from paths import FASHN_REPO, FASHN_WEIGHTS

    if not (importlib.util.find_spec('fashn_vton') or (Path(FASHN_REPO) / 'src' / 'fashn_vton').is_dir()):
        return False, 'FASHN이 설치되어 있지 않습니다. GPU 환경에서 기본 합성을 생성해 불러올 수 있습니다.'
    required = ('model.safetensors', 'dwpose/yolox_l.onnx', 'dwpose/dw-ll_ucoco_384.onnx')
    if any(not (Path(FASHN_WEIGHTS)/name).is_file() for name in required):
        return False, 'FASHN_WEIGHTS 경로에 생성·포즈 가중치가 없습니다. 이 도구는 모델을 자동 다운로드하지 않습니다.'
    if not importlib.util.find_spec('torch'):
        return False, 'GPU용 PyTorch가 설치된 환경에서 기본 합성을 생성해주세요.'
    import torch

    if not torch.cuda.is_available():
        return False, 'FASHN 기본 합성에는 CUDA GPU 환경이 필요합니다. 마스크·윤곽 조정은 CPU에서 가능합니다.'
    return True, '설치된 FASHN과 CUDA GPU로 기본 합성을 실행합니다.'


def bind_base(person, result, source, garment=None, settings=None, timing=None):
    person, result = prepare_photo(person), prepare_photo(result)
    # FASHN은 원본 비율의 근사 크기를 반환한다. 구도는 보존하고 단순 크기만 맞춘다.
    if abs(result.width / result.height / (person.width / person.height) - 1) > .01:
        raise ValueError('기본 합성 사진은 원본과 같은 구도·가로세로 비율이어야 합니다. 크롭하지 않은 결과를 사용해주세요.')
    result = result.resize(person.size, Image.Resampling.LANCZOS) if result.size != person.size else result.copy()
    metadata = {'schema_version': 1, 'source': source,
                'person_sha256': photo_info(person)['sha256'],
                'garment_sha256': photo_info(garment)['sha256'] if garment is not None else None,
                'base_sha256': photo_info(result)['sha256'], 'settings': copy.deepcopy(settings),
                'timing': copy.deepcopy(timing), 'size_conditioned_by_engine': False}
    return result, metadata


def generate_base(person, garment, settings=None, engine=None):
    if garment is None:
        raise ValueError('합성할 새 옷의 상품 사진을 업로드해주세요.')
    settings = engine_settings(**{'seed': (settings or {}).get('seed', 42),
                                 'steps': (settings or {}).get('steps', 50),
                                 'guidance': (settings or {}).get('guidance_scale', 2.5)})
    if engine is None:
        ready, message = fashn_environment()
        if not ready:
            raise ValueError(message)
        # 파서도 가중치를 자동 다운로드할 수 있어 별도 오프라인 프로세스에서 실행한다.
        with tempfile.TemporaryDirectory(prefix='fitcheck_base_') as directory:
            directory = Path(directory)
            prepare_photo(person).save(directory/'person.png')
            prepare_photo(garment).save(directory/'garment.png')
            command = [sys.executable, str(Path(__file__).resolve().parent.parent/'scripts'/'run_size_synthesis.py'),
                       '--person', str(directory/'person.png'), '--garment', str(directory/'garment.png'),
                       '--output', str(directory/'result'), '--seed', str(settings['seed']),
                       '--steps', str(settings['steps']), '--guidance', str(settings['guidance_scale'])]
            try:
                run = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace',
                                     env=dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUTF8='1'),
                                     timeout=900, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            except subprocess.TimeoutExpired as exc:
                raise ValueError('FASHN 실행이 15분을 초과해 중단했습니다. GPU 환경을 확인해주세요.') from exc
            if run.returncode:
                raise ValueError('FASHN 실행에 실패했습니다. 파서 가중치까지 준비된 GPU 환경인지 확인해주세요. '
                                 + (run.stderr or run.stdout)[-1200:])
            with Image.open(directory/'result'/'base.png') as image:
                result = image.convert('RGB')
            data = json.loads((directory/'result'/'base.json').read_text(encoding='utf-8'))
            return bind_base(person, result, 'fashn', garment, settings, data['timing'])
    person, garment = prepare_photo(person), prepare_photo(garment)
    result, timing = engine(person, garment, **settings, return_timing=True)
    return bind_base(person, result, 'fashn', garment, settings, timing)


def _frame(preview):
    p = preview['reference_points_px']
    neck = np.asarray(p['neck'], dtype=float)
    hem = (np.asarray(p['hem_left']) + p['hem_right']) / 2
    vertical = (hem - neck) / np.linalg.norm(hem - neck)
    horizontal = np.array([vertical[1], -vertical[0]])
    return neck, horizontal, vertical


def draft_mask(base, preview):
    """기준점 부근 GrabCut 초안. 사람/소매 분할의 정답으로 취급하지 않는다."""
    import cv2

    image = np.asarray(base.convert('RGB'))
    height, width = image.shape[:2]
    neck, vx, vy = _frame(preview)
    grid_y, grid_x = np.mgrid[:height, :width]
    u = (grid_x - neck[0])*vx[0] + (grid_y - neck[1])*vx[1]
    v = (grid_x - neck[0])*vy[0] + (grid_y - neck[1])*vy[1]
    p = preview['reference_points_px']
    half = np.linalg.norm(np.subtract(p['chest_right'], p['chest_left'])) / 2
    length = np.linalg.norm((np.asarray(p['hem_left']) + p['hem_right']) / 2 - neck)
    labels = np.zeros((height, width), np.uint8)
    labels[(abs(u) < half*1.65) & (v > -length*.06) & (v < length*1.30)] = cv2.GC_PR_BGD
    polygon = Image.new('L', base.size)
    ImageDraw.Draw(polygon).polygon([tuple(p[key]) for key in
                                    ('neck', 'chest_left', 'hem_left', 'hem_right', 'chest_right')], fill=1)
    labels[np.asarray(polygon) > 0] = cv2.GC_PR_FGD
    labels[(abs(u) < half*.35) & (v > length*.30) & (v < length*.70)] = cv2.GC_FGD
    try:
        # 다른 웹 요청의 OpenCV RNG/스레드 설정을 변경하지 않는다.
        cv2.grabCut(image, labels, None, np.zeros((1, 65)), np.zeros((1, 65)), 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error as exc:
        raise ValueError('옷 영역 초안을 찾지 못했습니다. 편집기에서 옷 영역을 칠해주세요.') from exc
    mask = np.isin(labels, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count > 1:
        mask = (components == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    return Image.fromarray(mask * 255)


def editor_value(base, mask):
    """원본 배경과 편집 마스크만 전달한다. 합쳐진 composite를 배경으로 재사용하지 않는다."""
    layer = Image.new('RGBA', base.size, (32, 200, 170, 0))
    layer.putalpha(mask.point(lambda value: 130 if value else 0))
    return {'background': base.convert('RGBA'), 'layers': [layer],
            'composite': None}


def _reference_repair_mask(person, preview):
    """복원 전용 기존 옷 제외 초안. 자동 기준점은 만들지 않는다. 최근 4개 마스크만 메모리에 둔다."""
    key = (preview['photo']['sha256'], _digest(preview['reference_points_px']))
    with _repair_reference_lock:
        if key not in _repair_reference_masks:
            try:
                mask = np.asarray(draft_mask(person, preview)) > 127
                mask.setflags(write=False)
            except ValueError:
                mask = None
            _repair_reference_masks[key] = mask
            if len(_repair_reference_masks) > 4:
                _repair_reference_masks.popitem(last=False)
        _repair_reference_masks.move_to_end(key)
        return _repair_reference_masks[key]


def editor_mask(value, base):
    if not value or value.get('background') is None:
        raise ValueError('기본 합성 사진과 옷 영역을 먼저 불러와주세요.')
    if photo_info(value['background'])['sha256'] != photo_info(base)['sha256']:
        raise ValueError('옷 영역 편집기의 사진이 바뀌었습니다. 기본 합성을 다시 불러와주세요.')
    output = np.zeros((base.height, base.width), np.uint8)
    for layer in value.get('layers', []):
        if layer.size != base.size or layer.mode != 'RGBA':
            raise ValueError('옷 영역의 크기와 투명 채널을 확인해주세요.')
        output |= (np.asarray(layer.getchannel('A')) > 0).astype(np.uint8)
    return Image.fromarray(output * 255)


def _geometry(mask, preview):
    """생성 결과의 확인된 마스크에서 원래 밑단과 폭을 찾는다. 상품 치수를 가정하지 않는다."""
    height, width = mask.shape
    neck, vx, vy = _frame(preview)
    ys, xs = np.nonzero(mask)
    if len(xs) < 25:
        raise ValueError('옷 영역이 비어 있거나 너무 작습니다. 칠한 영역을 확인해주세요.')
    u = (xs-neck[0])*vx[0] + (ys-neck[1])*vx[1]
    v = (xs-neck[0])*vy[0] + (ys-neck[1])*vy[1]
    p = preview['reference_points_px']
    chest = (np.asarray(p['chest_left']) + p['chest_right']) / 2
    chest_y = float(np.dot(chest-neck, vy))
    band = abs(v-chest_y) <= max(2, height*.004)
    if band.sum() < 4:
        raise ValueError('몸통 기준 높이에 옷 영역이 없습니다. 기준점과 마스크를 확인해주세요.')
    left, right = float(u[band].min()), float(u[band].max())
    center = (left + right) / 2
    # 소매/소수 잡점이 밑단을 결정하지 않도록 몸통 중앙 60%의 마지막 행을 사용한다.
    central = abs(u-center) < (right-left)*.30
    if not central.any():
        raise ValueError('몸통 중앙의 밑단을 찾을 수 없습니다.')
    hem_y = float(v[central].max())
    if hem_y <= chest_y + 4:
        raise ValueError('몸통 기준점 아래까지 옷 영역을 칠해주세요.')
    band = (v > hem_y-max(3, height*.006)) & (v <= hem_y)
    hem_left, hem_right = float(u[band].min()), float(u[band].max())
    if right-left < 4 or hem_right-hem_left < 4:
        raise ValueError('몸통과 밑단의 옷 영역 폭을 확인해주세요.')
    return {'chest_y': chest_y, 'hem_y': hem_y, 'chest_left': left, 'chest_right': right,
            'hem_left': hem_left, 'hem_right': hem_right}, (neck, vx, vy)


def adjust_size(person, base, metadata, preview, mask, confirmed=False):
    import cv2

    preview = checked_preview(preview, person)
    base = prepare_photo(base)
    if (not metadata or metadata['person_sha256'] != preview['photo']['sha256']
            or photo_info(base)['sha256'] != metadata['base_sha256'] or base.size != prepare_photo(person).size):
        raise ValueError('기본 합성과 현재 인물 사진의 연결이 달라졌습니다. 기본 합성을 다시 불러와주세요.')
    if not confirmed:
        raise ValueError('칠한 영역이 옷에만 해당하는지 확인해주세요. 팔·손·얼굴·바지는 지워주세요.')
    if mask.size != base.size:
        raise ValueError('옷 영역과 기본 합성 사진의 크기가 다릅니다.')
    source_mask = np.asarray(mask.convert('L')) > 127
    geometry, (neck, vx, vy) = _geometry(source_mask, preview)
    target = preview['target_points_px']

    def local(key):
        point = np.asarray(target[key]) - neck
        return np.array([np.dot(point, vx), np.dot(point, vy)])

    chest_l, chest_r, hem_l, hem_r = [local(key) for key in
                                    ('chest_left', 'chest_right', 'hem_left', 'hem_right')]
    target_chest_y = float((chest_l[1]+chest_r[1])/2)
    target_hem_y = float((hem_l[1]+hem_r[1])/2)
    if target_hem_y <= target_chest_y + 4:
        raise ValueError('목표 총장이 몸통 높이보다 짧습니다. 기준점을 확인해주세요.')
    # 확인하지 않은 부위는 기본 합성 결과의 위치를 유지한다.
    if not preview['applied']['length']:
        target_hem_y = geometry['hem_y']
    if not preview['applied']['chest']:
        chest_l[0], chest_r[0] = geometry['chest_left'], geometry['chest_right']
        hem_l[0], hem_r[0] = geometry['hem_left'], geometry['hem_right']
    yy, xx = np.mgrid[:base.height, :base.width].astype(np.float32)
    u, v = (xx-neck[0])*vx[0]+(yy-neck[1])*vx[1], (xx-neck[0])*vy[0]+(yy-neck[1])*vy[1]
    progress = np.clip((v-target_chest_y)/(target_hem_y-target_chest_y), 0, 1)
    source_v = np.where(v >= target_chest_y,
                        geometry['chest_y'] + progress*(geometry['hem_y']-geometry['chest_y']), v)
    source_l = geometry['chest_left']*(1-progress) + geometry['hem_left']*progress
    source_r = geometry['chest_right']*(1-progress) + geometry['hem_right']*progress
    target_l = chest_l[0]*(1-progress) + hem_l[0]*progress
    target_r = chest_r[0]*(1-progress) + hem_r[0]*progress
    # 목~어깨 부근은 고정하고 몸통 기준 높이까지 폭 변형을 부드럽게 늘린다.
    blend = np.clip((v-target_chest_y*.5)/max(target_chest_y*.5, 1), 0, 1)
    blend = blend*blend*(3-2*blend)
    mapped_u = source_l + (u-target_l)/np.maximum(target_r-target_l, 1)*(source_r-source_l)
    source_u = u*(1-blend) + mapped_u*blend
    below = v > target_hem_y
    mx = (neck[0]+source_u*vx[0]+source_v*vy[0]).astype(np.float32)
    my = (neck[1]+source_u*vx[1]+source_v*vy[1]).astype(np.float32)
    original = np.asarray(base)
    warped = cv2.remap(original, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    warped_mask = cv2.remap(source_mask.astype(np.uint8), mx, my, cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT).astype(bool)
    warped_mask[below] = False
    if not warped_mask.any():
        raise ValueError('변형 후 옷 영역이 없습니다. 마스크와 목표선을 확인해주세요.')
    repair = source_mask & ~warped_mask
    reference_mask = _reference_repair_mask(prepare_photo(person), preview) if repair.any() else None
    canvas, reused, estimated, repair_info = repair_exposed(
        prepare_photo(person), base, source_mask, warped_mask, (neck, vx, vy),
        preview['reference_points_px'], below, reference_mask)
    # 옷 안쪽에서만 경계를 섞는다. 편집 영역 밖은 기본 합성과 화소 단위로 동일하다.
    alpha = np.clip(cv2.distanceTransform(warped_mask.astype(np.uint8), cv2.DIST_L2, 3)/2, 0, 1)[..., None]
    output = np.rint(warped*alpha + canvas*(1-alpha)).astype(np.uint8)
    edit = source_mask | warped_mask
    output[~edit] = original[~edit]
    changed = np.any(output != original, axis=2)
    mask_y, mask_x = np.nonzero(warped_mask)
    report = {'schema_version': 1, 'transform_version': 'masked_torso_warp_v1',
              'preview': preview, 'base': copy.deepcopy(metadata), 'source_geometry_px': geometry,
              'target_geometry_px': {'chest_y': target_chest_y, 'hem_y': target_hem_y,
                                     'chest_left': float(chest_l[0]), 'chest_right': float(chest_r[0]),
                                     'hem_left': float(hem_l[0]), 'hem_right': float(hem_r[0])},
              'mask_sha256': hashlib.sha256(source_mask.tobytes()).hexdigest(),
              'preview_sha256': _digest(preview), 'mask_confirmed': True,
              'repair_pixels': int(repair.sum()), 'edited_pixels': int(edit.sum()),
              'changed_pixels': int(changed.sum()),
              'adjusted_sha256': photo_info(Image.fromarray(output))['sha256'],
              'target_mask_sha256': hashlib.sha256(warped_mask.tobytes()).hexdigest(),
              'target_mask_bbox_px': [int(mask_x.min()), int(mask_y.min()), int(mask_x.max()), int(mask_y.max())],
              'repair_method': repair_info['method'], 'repair': repair_info,
              'outside_edit_max_pixel_difference': int(np.abs(output.astype(int)-original.astype(int))[~edit].max())
              if (~edit).any() else 0,
              'notes': ['기본 합성 결과에서 확인한 옷 영역을 목표선에 맞춘 2D 후처리 실험입니다.',
                        '생성된 옷의 길이와 폭은 마스크에서 읽습니다. 기본 합성에 특정 상품 사이즈를 가정하지 않습니다.',
                        '드러난 영역 중 원본에서 이미 보이는 밑단 아래 부분만 보수적으로 재사용합니다. 원본의 옷 추출 초안과 밑단 주변은 복사 대상에서 제외합니다.',
                        '원본 미관측 또는 정합 불확실로 재사용하지 못한 부분은 밑단 아래/옆 패치를 방향별로 연장한 추정입니다. 피부·바지의 실제 모습이나 생성형 복원을 보장하지 않습니다.',
                        '어깨·소매 구조 보정과 실제 착용 핏 정확도는 검증하지 않았습니다.',
                        '편집 영역 밖은 기본 합성 사진을 유지합니다. FASHN 자체가 바꾼 얼굴·배경은 이 단계의 보존 대상과 별개입니다.']}
    return {'image': Image.fromarray(output), 'mask': Image.fromarray(source_mask.astype(np.uint8)*255),
            'target_mask': Image.fromarray(warped_mask.astype(np.uint8)*255),
            'repair_mask': Image.fromarray(repair.astype(np.uint8)*255),
            'reference_reused_mask': Image.fromarray(reused.astype(np.uint8)*255),
            'estimated_repair_mask': Image.fromarray(estimated.astype(np.uint8)*255),
            'repair_overlay': repair_overlay(base, reused, estimated), 'report': report}


def adjust_sizes(person, base, metadata, preview, mask, confirmed=False):
    """한 기본 합성·한 확인 마스크에 판매 사이즈별 목표선을 독립적으로 적용한다."""
    preview = checked_preview(preview, person)
    selected = preview['fit_plan']['selected_size']
    result = adjust_size(person, base, metadata, preview, mask, confirmed)
    results, summaries, withheld = [], [], []
    for row in preview['fit_plan']['comparison']['target_chart']['sizes']:
        label = row['size']
        try:
            if label == selected:
                current = result
            else:
                plan = create_fit_plan(preview['fit_plan']['comparison'], label)
                review = preview['review']
                target = create_preview(plan, preview['photo'], preview['anchors'],
                                        review['same_garment'], review['length_compatible'],
                                        review['chest_compatible'], False)
                current = adjust_size(person, base, metadata, target, mask, confirmed)
            data = current['report']
            ratios = data['preview']['fit_plan']['ratios']
            length_label = f'×{ratios["length"]:.3f}' if data['preview']['applied']['length'] else '보류'
            chest_label = f'×{ratios["chest"]:.3f}' if data['preview']['applied']['chest'] else '보류'
            caption = f'{label} · 총장 {length_label} · 가슴단면 {chest_label}'
            results.append((current['image'], caption))
            summaries.append({'size': label, 'applied': data['preview']['applied'], 'ratios': ratios,
                              'target_geometry_px': data['target_geometry_px'],
                              'adjusted_sha256': data['adjusted_sha256'],
                              'target_mask_sha256': data['target_mask_sha256'],
                              'changed_pixels': data['changed_pixels'], 'repair': data['repair']})
        except ValueError as exc:
            withheld.append({'size': label, 'reason': str(exc)})
    result['report']['size_comparison'] = {'rows': summaries, 'withheld': withheld,
                                         'base_reused': True}
    return result, results


def synthesis_report(result):
    data = result['report']
    source = data['base']['source']
    label = {'fashn': 'FASHN 생성', 'external_unverified': '외부 합성 결과 (엔진·설정 미확인)',
             'reference_photo_dry_run': '기준 사진 변형 테스트 · 새 옷 합성 아님'}[source]
    selected = html.escape(data['preview']['fit_plan']['selected_size']).replace('|', '\\|')
    text = (f"### 선택 사이즈 {selected} · 치수 적용 완료\n\n"
            f"기본 사진 출처: **{label}** · 기본 합성 재사용 / 길이·몸통 폭 후처리\n\n"
            f"실제 변경 화소: **{data['changed_pixels']:,} px**\n\n"
            f"편집 영역 밖 최대 화소 차이: **{data['outside_edit_max_pixel_difference']}** / "
            f"복원 영역: **{data['repair_pixels']:,} px**\n\n"
            + '\n\n'.join(data['notes']))
    repaired = data['repair']
    text += (f"\n\n복원 구분: 원본 재사용 **{repaired['reused_reference_pixels']:,} px** / "
             f"방향성 패치 추정 **{repaired['estimated_pixels']:,} px**. "
             '초록색은 원본 재사용, 주황색은 원본을 재사용하지 못해 패치로 추정한 영역입니다.')
    comparison = data.get('size_comparison')
    if comparison:
        lines = ['| 사이즈 | 목 기준 밑단(px) | 목표 몸통 폭(px) | 변경 화소 |',
                 '|---|---:|---:|---:|']
        for row in comparison['rows']:
            geometry = row['target_geometry_px']
            size = html.escape(row['size']).replace('|', '\\|')
            lines.append(f"| {size} | {geometry['hem_y']:.1f} | "
                         f"{geometry['chest_right']-geometry['chest_left']:.1f} | {row['changed_pixels']:,} |")
        for item in comparison['withheld']:
            lines += ['', f"변형 보류: {html.escape(item['size'])} — {html.escape(item['reason'])}"]
        text += '\n\n' + '\n'.join(lines)
    return text
