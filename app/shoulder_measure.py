"""정면 전신 사진의 어깨 관절 간 거리만 추정하는 1단계 디버깅 경로.

기존 body_measure의 MediaPipe/전신 높이 계산을 공유한다. 1.28 보정치는
기존 실험값이며 Size Korea의 '어깨사이길이'와 일치한다고 검증된 값은 아니다.
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PIL import ImageOps

from body_measure import (MIN_VISIBILITY, PLAUSIBLE_RATIO_TO_HEIGHT,
                          POSE_LANDMARKS, SHOULDER_LANDMARK_TO_BREADTH,
                          _pixel_height, _run_pose)


@dataclass
class ShoulderEstimate:
    height_cm: float
    shoulder_cm: Optional[float] = None
    shoulder_landmark_px: Optional[float] = None
    body_height_px: Optional[float] = None
    px_per_cm: Optional[float] = None
    factor: float = SHOULDER_LANDMARK_TO_BREADTH
    head_y: Optional[int] = None
    foot_y: Optional[int] = None
    left_shoulder: Optional[Tuple[float, float]] = None
    right_shoulder: Optional[Tuple[float, float]] = None
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return self.shoulder_cm is not None and not self.warnings


def measure_shoulder(image, height_cm, pose_landmarks=None, seg_mask=None):
    """(EXIF 보정·축소된 이미지, 진단값)를 반환한다.

    pose_landmarks와 seg_mask는 테스트/이미 추출한 포즈 재사용용이다.
    사진이 무효이면 어깨 수치를 반환하지 않고 재업로드 사유를 남긴다.
    """
    if image is None:
        raise ValueError('정면 전신 사진을 올려주세요.')
    try:
        height = float(height_cm)
    except (TypeError, ValueError) as exc:
        raise ValueError('키(cm)를 입력해주세요.') from exc
    if not math.isfinite(height) or not 90 <= height <= 230:
        raise ValueError('키는 90~230cm 범위의 숫자로 입력해주세요.')

    if hasattr(image, 'convert'):
        photo = ImageOps.exif_transpose(image).convert('RGB')
        if pose_landmarks is None:
            photo.thumbnail((1280, 1280))
        pixels = np.asarray(photo)
    else:
        from PIL import Image
        pixels = np.asarray(image)
        if pixels.ndim != 3 or pixels.shape[2] != 3:
            raise ValueError('RGB 사진을 사용해주세요.')
        photo = Image.fromarray(pixels.astype('uint8'), 'RGB')
    h, w = pixels.shape[:2]
    result = ShoulderEstimate(height_cm=height)
    if w >= h:
        result.warnings.append('세로 방향의 정면 전신 사진으로 다시 올려주세요.')

    if pose_landmarks is None:
        pose_landmarks, seg_mask = _run_pose(pixels)
    if pose_landmarks is None:
        result.warnings.append('사람을 찾지 못했습니다. 머리와 발이 모두 보이는 사진을 올려주세요.')
        return photo, result

    names = ('nose', 'left_shoulder', 'right_shoulder', 'left_hip',
             'right_hip', 'left_ankle', 'right_ankle')
    # 어깨 위치와 정면 자세에 쓰는 관절만 측정 보류 조건으로 검사한다.
    # 키 척도는 실루엣에서 얻으므로 얼굴/발목 신뢰도는 참고 안내로 처리한다.
    required_names = ('left_shoulder', 'right_shoulder', 'left_hip', 'right_hip')
    bad_visibility = [name for name in required_names
                      if pose_landmarks[POSE_LANDMARKS[name]].visibility < MIN_VISIBILITY]
    if bad_visibility:
        result.warnings.append('랜드마크가 흐리거나 가려졌습니다: ' + ', '.join(bad_visibility))
    out_of_frame = [name for name in names
                    if not (0 <= pose_landmarks[POSE_LANDMARKS[name]].x <= 1
                            and 0 <= pose_landmarks[POSE_LANDMARKS[name]].y <= 1)]
    if out_of_frame:
        result.warnings.append('몸 일부가 사진 밖에 있습니다: ' + ', '.join(out_of_frame))

    def point(name):
        landmark = pose_landmarks[POSE_LANDMARKS[name]]
        return landmark.x * w, landmark.y * h

    left, right = point('left_shoulder'), point('right_shoulder')
    result.left_shoulder, result.right_shoulder = left, right
    shoulder_px = math.dist(left, right)
    result.shoulder_landmark_px = round(shoulder_px, 1)

    mask = np.squeeze(np.asarray(seg_mask)) if seg_mask is not None else None
    if mask is None or mask.ndim != 2 or mask.shape != (h, w):
        result.warnings.append('전신 실루엣 마스크가 없어 키 기준 척도를 확인할 수 없습니다.')
        return photo, result
    rows = np.where(np.any(mask > 0.5, axis=1))[0]
    if not rows.size:
        result.warnings.append('전신 실루엣을 찾지 못했습니다.')
        return photo, result
    result.head_y, result.foot_y = int(rows.min()), int(rows.max())
    body_px, scale_note = _pixel_height(pixels, pose_landmarks, mask, w, h)
    result.body_height_px = round(body_px, 1)
    if scale_note:
        result.warnings.append(scale_note)
    if body_px < 600:
        result.warnings.append('사진에서 전신 높이가 600px 미만입니다. 더 가까이서 찍어주세요.')
    if body_px <= 0:
        return photo, result
    low_aux_visibility = [name for name in ('nose', 'left_ankle', 'right_ankle')
                          if pose_landmarks[POSE_LANDMARKS[name]].visibility < MIN_VISIBILITY]
    if low_aux_visibility:
        result.notes.append(
            '얼굴/발목 랜드마크 신뢰도가 낮습니다: ' + ', '.join(low_aux_visibility)
            + '. 키 기준 척도는 전신 실루엣으로 계산했습니다. 초록 기준선이 머리끝과 발끝에 맞는지 확인해주세요.'
        )
    px_per_cm = body_px / height
    result.px_per_cm = round(px_per_cm, 3)

    hip_left, hip_right = point('left_hip'), point('right_hip')
    shoulder_mid = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)
    hip_mid = ((hip_left[0] + hip_right[0]) / 2,
               (hip_left[1] + hip_right[1]) / 2)
    torso_px = math.dist(shoulder_mid, hip_mid)
    if torso_px <= 0 or shoulder_px / torso_px < 0.45:
        result.warnings.append('어깨가 정면으로 보이지 않습니다. 정면 자세로 다시 찍어주세요.')
    shoulder_cm = shoulder_px / px_per_cm * result.factor
    low, high = PLAUSIBLE_RATIO_TO_HEIGHT['shoulder']
    if not low <= shoulder_cm / height <= high:
        result.warnings.append('어깨 추정치가 키 대비 상식 범위를 벗어납니다. 사진/키를 확인해주세요.')
    if not result.warnings:
        result.shoulder_cm = round(shoulder_cm, 1)
    return photo, result
