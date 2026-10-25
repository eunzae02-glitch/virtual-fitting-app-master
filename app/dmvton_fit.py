"""자동 옷 영역을 이용한 저해상도 사이즈 비교 실험. 실제 착용 핏을 측정하지 않는다."""
import hashlib
import time

import numpy as np
from PIL import Image, ImageOps

from dmvton_core import MODEL_SIZE, get_parser, infer
from garment_compare import parse_target_chart
from reference_garment import ReferenceGarment


TRANSFORM_VERSION = 'automatic_ratio_v0'
PROTECTED_LABELS = (1, 2, 12, 13, 14, 15)  # face, hair, arms, hands, legs, feet


def photo(image):
    if image is None:
        raise ValueError('전신 사진과 새 옷 상품 사진을 올려주세요.')
    result = ImageOps.exif_transpose(image).convert('RGBA')
    result.thumbnail((1200, 1200))
    if min(result.size) < 20:
        raise ValueError('사진 해상도가 너무 작습니다.')
    return result


def source_key(person, garment):
    digest = hashlib.sha256()
    for image in (person, garment):
        image = photo(image)
        digest.update(str(image.size).encode() + image.tobytes())
    return digest.hexdigest()


def compare_inputs(length, shoulder, chest, rows):
    reference = ReferenceGarment(length, shoulder, chest)
    chart = parse_target_chart(rows)
    if any(size.shoulder_cm is None for size in chart.sizes):
        raise ValueError('새 옷의 어깨너비도 cm로 입력해주세요.')
    return reference, chart


def largest_mask(mask):
    import cv2

    mask = np.asarray(mask, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        raise ValueError('상의 영역을 찾지 못했습니다. 옷이 잘 보이는 사진을 사용해주세요.')
    result = labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    if result.sum() < 25:
        raise ValueError('상의 영역이 너무 작습니다. 더 가까이 찍은 사진을 사용해주세요.')
    return result


def geometry(mask, chest_y=None):
    """분할 실루엣의 정면 보조 좌표. 가려진 목뒤점이나 신체 실측값이 아니다."""
    mask = largest_mask(mask)
    yy, xx = np.nonzero(mask)
    top, bottom = int(yy.min()), int(yy.max())
    if bottom - top < 8:
        raise ValueError('상의 길이를 구분하지 못했습니다.')
    chest_y = top + (bottom - top) * .55 if chest_y is None else chest_y
    if not top < chest_y < bottom:
        raise ValueError('기준 몸통 높이에 생성된 옷이 없습니다.')
    rows = range(max(top, int(chest_y)-2), min(bottom, int(chest_y)+2)+1)
    spans = [(np.flatnonzero(mask[y])[0], np.flatnonzero(mask[y])[-1])
             for y in rows if mask[y].sum() >= 4]
    if not spans:
        raise ValueError('상의 몸통 폭을 구분하지 못했습니다.')
    left, right = np.median(spans, axis=0)
    center = (left + right) / 2
    central = (xx >= center-(right-left)*.25) & (xx <= center+(right-left)*.25)
    if not central.any() or right-left < 4:
        raise ValueError('상의 몸통 중앙을 구분하지 못했습니다.')
    neck_y, hem_y = float(yy[central].min()), float(yy[central].max())
    if hem_y <= chest_y + 3:
        raise ValueError('상의 밑단을 구분하지 못했습니다.')
    return {'neck_x': float(center), 'neck_y': neck_y, 'chest_y': float(chest_y),
            'width': float(right-left), 'hem_y': hem_y}


def padded_crop(image, box):
    crop = image.crop(box)
    scale = min(MODEL_SIZE[0] / crop.width, MODEL_SIZE[1] / crop.height)
    size = (max(1, round(crop.width*scale)), max(1, round(crop.height*scale)))
    offset = ((MODEL_SIZE[0]-size[0])//2, (MODEL_SIZE[1]-size[1])//2)
    canvas = Image.new('RGB', MODEL_SIZE, 'white')
    canvas.paste(crop.resize(size, Image.Resampling.LANCZOS), offset)
    return canvas, {'box': box, 'size': size, 'offset': offset}


def restore_crop(image, transform, full_size, is_mask=False):
    x, y = transform['offset']
    width, height = transform['size']
    left, top, right, bottom = transform['box']
    mode = 'L' if is_mask else 'RGB'
    canvas = Image.new(mode, full_size)
    resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.LANCZOS
    region = image.convert(mode).crop((x, y, x+width, y+height)).resize(
        (right-left, bottom-top), resample)
    canvas.paste(region, (left, top))
    return canvas


def prepare_person(person, labels):
    mask = largest_mask(labels == 3)
    reference_geometry = geometry(mask)
    yy, xx = np.nonzero(np.isin(labels, (1, 2, 3, 12, 13, 16)))
    shirt_y = np.nonzero(mask)[0]
    margin = max(8, int((shirt_y.max()-shirt_y.min())*.15))
    box = (max(0, int(xx.min())-margin), max(0, int(yy.min())-margin),
           min(person.width, int(xx.max())+margin+1),
           min(person.height, int(shirt_y.max())+margin*2+1))
    crop, transform = padded_crop(person, box)
    return crop, transform, mask, reference_geometry


def prepare_garment(garment, parser):
    alpha = np.asarray(garment.getchannel('A'))
    if (alpha < 128).any() and (alpha > 127).any():
        mask = largest_mask(alpha > 127)
    else:
        mask = largest_mask(parser.predict(garment.convert('RGB')) == 3)
    yy, xx = np.nonzero(mask)
    box = (int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1)
    rgb = Image.new('RGB', garment.size, 'white')
    rgb.paste(garment.convert('RGB'), mask=Image.fromarray(mask.astype(np.uint8)*255))
    canvas, transform = padded_crop(rgb, box)
    # 같은 resize/pad를 마스크에도 적용하고 nearest 보간으로 경계를 유지한다.
    size, offset = transform['size'], transform['offset']
    edge = Image.new('L', MODEL_SIZE)
    edge.paste(Image.fromarray(mask.astype(np.uint8)*255).crop(box).resize(
        size, Image.Resampling.NEAREST), offset)
    return canvas, edge


def make_base(person, garment):
    started = time.perf_counter()
    person, garment = photo(person).convert('RGB'), photo(garment)
    parser = get_parser()
    labels = np.asarray(parser.predict(person))
    if labels.shape != (person.height, person.width):
        raise ValueError('자동 분할과 사진의 좌표 크기가 다릅니다.')
    crop, transform, reference_mask, reference_geometry = prepare_person(person, labels)
    clothes, edge = prepare_garment(garment, parser)
    result, mask, metadata = infer(crop, clothes, edge)
    restored = np.asarray(restore_crop(result, transform, person.size))
    generated_mask = largest_mask(np.asarray(restore_crop(mask, transform, person.size, True)) > 127)
    protected = np.isin(labels, PROTECTED_LABELS)
    edit = (reference_mask | generated_mask) & ~protected
    base = np.asarray(person).copy()
    base[edit] = restored[edit]
    generated_mask &= ~protected
    metadata.update(total_prepare_and_base_s=time.perf_counter()-started,
                    geometry_source='automatic_human_parser_unreviewed', crop=transform)
    return {'base': Image.fromarray(base), 'mask': generated_mask, 'protected': protected,
            'reference_geometry': reference_geometry, 'metadata': metadata}


def apply_size(base, mask, protected, reference_geometry, length_ratio, chest_ratio):
    """기본 합성의 옷만 기준 사진 치수비로 변형한다. 어깨는 변형하지 않는다."""
    import cv2

    ref = reference_geometry
    source = geometry(mask, ref['chest_y'])
    tx, ty = ref['neck_x'], ref['neck_y']
    target_hem = ty + (ref['hem_y']-ty)*length_ratio
    target_width = ref['width']*chest_ratio
    if (target_hem >= base.height or target_hem <= ref['chest_y']+3 or
            tx-target_width/2 < 0 or tx+target_width/2 >= base.width):
        raise ValueError('목표 옷 윤곽이 사진 범위를 벗어납니다. 주변 여백이 있는 사진을 사용해주세요.')
    yy, xx = np.mgrid[:base.height, :base.width].astype(np.float32)
    fraction = np.clip((yy-ref['chest_y'])/max(target_hem-ref['chest_y'], 1), 0, 1)
    my = np.where(yy >= ref['chest_y'],
                  ref['chest_y'] + fraction*(source['hem_y']-ref['chest_y']), yy)
    # 목·어깨 부근은 고정하고 몸통 높이로 갈수록 폭 변형을 적용한다.
    start = ty + (ref['chest_y']-ty)*.5
    blend = np.clip((yy-start)/max(ref['chest_y']-start, 1), 0, 1)
    blend = blend*blend*(3-2*blend)
    mapped_x = source['neck_x'] + (xx-tx)*source['width']/target_width
    mx = xx*(1-blend) + mapped_x*blend
    warped = cv2.remap(np.asarray(base), mx.astype(np.float32), my.astype(np.float32),
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    warped_mask = cv2.remap(mask.astype(np.uint8), mx.astype(np.float32), my.astype(np.float32),
                            cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT).astype(bool)
    warped_mask[(yy < ty) | (yy > target_hem) | protected] = False
    if not warped_mask.any():
        raise ValueError('변형된 옷 영역을 찾지 못했습니다.')
    repair = mask & ~warped_mask & ~protected
    original = np.asarray(base)
    canvas = cv2.inpaint(original, repair.astype(np.uint8)*255, 3, cv2.INPAINT_TELEA) if repair.any() else original.copy()
    canvas[warped_mask] = warped[warped_mask]
    canvas[protected] = original[protected]
    return Image.fromarray(canvas), {'length_ratio': length_ratio, 'chest_ratio': chest_ratio,
                                    'target_hem_y': float(target_hem), 'target_width_px': float(target_width),
                                    'transform_version': TRANSFORM_VERSION, 'shoulder_applied': False}


def render_sizes(cache, reference, chart):
    gallery = [(cache['base'], 'DM-VTON 기본 합성 · 치수 조정 전')]
    reports, errors = [], []
    for size in chart.sizes:
        try:
            image, report = apply_size(cache['base'], cache['mask'], cache['protected'],
                                       cache['reference_geometry'], size.length_cm/reference.length_cm,
                                       size.chest_flat_cm/reference.chest_flat_cm)
            report.update(size=size.size, length_difference_cm=size.length_cm-reference.length_cm,
                          chest_flat_difference_cm=size.chest_flat_cm-reference.chest_flat_cm,
                          shoulder_cm=size.shoulder_cm, reference_shoulder_cm=reference.shoulder_cm)
            reports.append(report)
            gallery.append((image, f'{size.size} · 총장 {report["length_difference_cm"]:+g}cm · '
                                   f'가슴단면 {report["chest_flat_difference_cm"]:+g}cm'))
        except ValueError as exc:
            errors.append(f'{size.size}: {exc}')
    return gallery, reports, errors
