"""옷 축소로 드러난 영역만 복원한다. 가려졌던 신체의 실제 모습을 추정하는 모델은 아니다."""
import numpy as np


REPAIR_VERSION = 'reference_directional_patch_v2'


def legacy_telea_plate(image, garment_mask):
    """이전 v1 실험 재현용. 기본 복원 경로에서는 사용하지 않는다."""
    import cv2

    excluded = cv2.dilate(garment_mask.astype(np.uint8), np.ones((3, 3), np.uint8))
    if excluded.all():
        raise ValueError('옷 밖 복원 참고 영역이 없습니다. 옷 마스크를 확인해주세요.')
    return cv2.inpaint(image, excluded * 255, 3, cv2.INPAINT_TELEA)


def _directional_plate(image, garment_mask, bottom_cut, frame):
    """밑단은 아래, 옆 틈은 바깥쪽의 실제 패치를 반사 연장한다. 옷 화소는 공여부에서 제외한다."""
    import cv2

    neck, vx, vy = [np.asarray(value, dtype=float) for value in frame]
    height, width = garment_mask.shape
    corners = np.array([[0, 0], [width-1, 0], [0, height-1], [width-1, height-1]]) - neck
    local = np.column_stack((corners @ vx, corners @ vy))
    origin = np.floor(local.min(axis=0))
    size = np.ceil(local.max(axis=0) - origin + 1).astype(int)
    matrix = np.array([[*vx, -neck @ vx - origin[0]],
                       [*vy, -neck @ vy - origin[1]]], dtype=np.float32)
    shape = tuple(int(value) for value in size)
    plate = cv2.warpAffine(image, matrix, shape, flags=cv2.INTER_LINEAR)
    inside = cv2.warpAffine(np.ones((height, width), np.uint8), matrix, shape,
                            flags=cv2.INTER_NEAREST).astype(bool)
    garment = cv2.warpAffine(garment_mask.astype(np.uint8), matrix, shape,
                             flags=cv2.INTER_NEAREST)
    excluded = cv2.dilate(garment, np.ones((5, 5), np.uint8)).astype(bool)
    bottom = cv2.warpAffine(bottom_cut.astype(np.uint8), matrix, shape,
                            flags=cv2.INTER_NEAREST).astype(bool)
    safe_inside = cv2.erode(inside.astype(np.uint8), np.ones((3, 3), np.uint8), borderValue=0).astype(bool)
    donor = safe_inside & ~excluded
    if not donor.any():
        raise ValueError('옷 밖 복원 참고 영역이 없습니다. 여백이 있는 사진을 사용해주세요.')
    h, w = donor.shape
    yy, xx = np.mgrid[:h, :w]
    down = np.minimum.accumulate(np.where(donor, yy, h)[::-1], axis=0)[::-1]
    left = np.maximum.accumulate(np.where(donor, xx, -1), axis=1)
    right = np.minimum.accumulate(np.where(donor, xx, w)[:, ::-1], axis=1)[:, ::-1]
    left_ok, right_ok, down_ok = left >= 0, right < w, down < h
    # 밑단에서 가장 가까운 옆 피부/배경을 끌어오지 않는다. 아래 공여부가 있을 때 우선한다.
    use_down = bottom & down_ok
    use_left = ~use_down & left_ok & (~right_ok | (xx-left <= right-xx))
    use_right = ~use_down & ~use_left & right_ok
    use_down |= ~use_down & ~use_left & ~use_right & down_ok
    missing = excluded & inside & ~(use_down | use_left | use_right)
    if missing.any():
        raise ValueError('드러난 영역의 복원 참고 화소를 찾지 못했습니다. 잘리지 않은 사진을 사용해주세요.')

    sample_y = np.where(use_down, np.minimum(2*down-yy-1, h-1), yy)
    sample_x = np.where(use_left, np.maximum(2*left-xx+1, 0),
                        np.where(use_right, np.minimum(2*right-xx-1, w-1), xx))
    sample_y, sample_x = sample_y.clip(0, h-1), sample_x.clip(0, w-1)
    # 반사한 패치가 다른 옷·사진 밖을 만나면 처음 확인된 공여 화소까지만 사용한다.
    valid_sample = donor[sample_y, sample_x]
    boundary_y = np.where(use_down, down.clip(0, h-1), yy)
    boundary_x = np.where(use_left, left.clip(0, w-1),
                          np.where(use_right, right.clip(0, w-1), xx))
    sample_y = np.where(valid_sample, sample_y, boundary_y)
    sample_x = np.where(valid_sample, sample_x, boundary_x)
    fill = plate[sample_y, sample_x]
    # 역회전 보간에서 남은 옷 색이 섞이지 않도록 복원용 판에서는 옷 전체를 제외한다.
    plate[excluded & inside] = fill[excluded & inside]
    restored = cv2.warpAffine(plate, cv2.invertAffineTransform(matrix), (width, height),
                              flags=cv2.INTER_LINEAR)
    return restored


def repair_exposed(person, base, source_mask, target_mask, frame, reference_points, bottom_cut,
                   reference_garment_mask=None):
    """원본에서 확인 가능한 밑단 아래 영역을 재사용하고 나머지는 방향성 패치로 추정한다."""
    original, image = np.asarray(person.convert('RGB')), np.asarray(base.convert('RGB'))
    if original.shape != image.shape:
        raise ValueError('복원용 원본과 기본 합성 사진의 크기가 다릅니다.')
    shape = image.shape[:2]
    masks = [np.asarray(value, dtype=bool) for value in (source_mask, target_mask, bottom_cut)]
    if any(mask.shape != shape for mask in masks):
        raise ValueError('옷 복원 마스크와 사진의 크기가 다릅니다.')
    if reference_garment_mask is not None and np.asarray(reference_garment_mask).shape != shape:
        raise ValueError('원본 옷 제외 마스크와 사진의 크기가 다릅니다.')
    source, target, bottom = masks
    repair = source & ~target
    empty = np.zeros(shape, dtype=bool)
    canvas = image.copy()
    info = {'method': REPAIR_VERSION, 'reused_reference_pixels': 0, 'estimated_pixels': 0,
            'reference_gate': 'not_needed', 'reference_support_pixels': 0,
            'reference_clothing_mask_available': reference_garment_mask is not None,
            'reference_gate_definition': 'lower_hem_torso_support_rgb_mae_24_p90_48_v1',
            'is_ground_truth_reconstruction': False}
    if not repair.any():
        return canvas, empty, empty.copy(), info

    # 상의가 가리고 있던 원본 화소를 복사하면 예전 옷이 다시 나타난다.
    # 확인된 두 밑단보다 아래이면서 몸통 가로 범위 안인 보수적인 부분만 원본 후보로 쓴다.
    neck, vx, vy = [np.asarray(value, dtype=float) for value in frame]
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    u = (xx-neck[0])*vx[0] + (yy-neck[1])*vx[1]
    v = (xx-neck[0])*vy[0] + (yy-neck[1])*vy[1]
    points = {key: np.asarray(value, dtype=float)-neck for key, value in reference_points.items()}
    hem_v = max(points[key] @ vy for key in ('hem_left', 'hem_right'))
    left = min(points[key] @ vx for key in ('hem_left', 'chest_left'))
    right = max(points[key] @ vx for key in ('hem_right', 'chest_right'))
    margin = max(3, round(min(shape) * .005))
    visible = (v > hem_v+margin) & (u > left+margin) & (u < right-margin)
    if reference_garment_mask is not None:
        import cv2

        # 두 끝점보다 내려오는 곡선 밑단 등은 원본 옷 초안에서도 제외한다.
        excluded_reference = cv2.dilate(np.asarray(reference_garment_mask, dtype=np.uint8),
                                         np.ones((margin*2+1, margin*2+1), np.uint8)) > 0
        visible &= ~excluded_reference
    else:
        # 옷 경계를 구분하지 못한 원본을 그대로 붙이지 않는다.
        visible[:] = False
    # 기존 연결 검사는 같은 크기/구도를 요구한다. 추가로 가까운, 이미 보이는 부분의 색 일치를 확인한다.
    # 이 공학적 보수 조건은 인물 정합이나 참 신체 복원을 보장하는 검사가 아니다.
    support = visible & ~source & (v < hem_v+max(16, (right-left)*.7))
    info['reference_support_pixels'] = int(support.sum())
    reused = empty.copy()
    if support.sum() >= 25:
        differences = np.abs(original.astype(np.int16)-image.astype(np.int16)).mean(axis=2)[support]
        mae, p90 = float(np.median(differences)), float(np.quantile(differences, .90))
        info.update(reference_support_median_mae=mae, reference_support_p90_mae=p90)
        if mae <= 24 and p90 <= 48:
            reused = repair & visible
            info['reference_gate'] = 'nearby_visible_region_agrees'
        else:
            info['reference_gate'] = 'appearance_mismatch_do_not_copy'
    else:
        info['reference_gate'] = ('insufficient_visible_support' if reference_garment_mask is not None
                                  else 'reference_clothing_mask_unavailable')
    estimated = repair & ~reused
    if estimated.any():
        plate = _directional_plate(image, source, bottom, frame)
        canvas[estimated] = plate[estimated]
    canvas[reused] = original[reused]
    info.update(reused_reference_pixels=int(reused.sum()), estimated_pixels=int(estimated.sum()))
    return canvas, reused, estimated, info


def repair_overlay(base, reused, estimated):
    from PIL import Image

    image = np.asarray(base.convert('RGB')).copy()
    for mask, color in [(reused, (30, 210, 130)), (estimated, (255, 150, 30))]:
        image[mask] = np.rint(image[mask] * .4 + np.array(color) * .6).astype(np.uint8)
    return Image.fromarray(image)
