"""사진 한 장 + 키(cm)로 몸 치수를 추정한다 (MediaPipe Pose).

**정확도에 대한 정직한 전제**

단일 정면 사진에서 실측 치수를 얻는 것은 불가능하다. 이 모듈이 하는 일은
`추정`이고, 아래 가정 위에서만 성립한다.

1. 정면·전신·정자세, 카메라가 대략 몸 높이에서 수평으로 촬영
2. 입력한 키(cm)로 픽셀→cm 스케일을 잡는다 (머리끝~발끝 픽셀 = 키)
3. MediaPipe 랜드마크는 관절 중심이라 옷 치수의 기준점(어깨 솔기 등)과 다르다.
   보정계수(SHOULDER_LANDMARK_TO_BREADTH 등)로 메우며, 이 값들은 경험적 상수다.
4. 정면 사진에서는 두께(깊이)를 볼 수 없다. 가슴/허리 둘레는 타원 가정으로 환산한다.

따라서 이 값들은 **절대 치수보다 상대 비교(어떤 사이즈가 더 맞는가)** 에 쓰는 것이 맞다.
계획서의 "정밀 치수보다 상대적 사이즈 추천으로 스코프를 잡는다"와 같은 맥락이다.

출력은 전부 cm이고, 옷 치수표와 맞추기 위해 **단면(폭) 기준**으로 낸다.
(가슴단면 = 가슴둘레 / 2)
"""
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# --- 경험적 보정 상수 -------------------------------------------------------
# MediaPipe 어깨 랜드마크는 어깨관절 중심이라 실제 어깨너비(견봉~견봉)보다 좁다.
SHOULDER_LANDMARK_TO_BREADTH = 1.28
# 가슴/허리/엉덩이 단면은 정면 폭의 절반이 아니라, 타원 둘레의 절반이다.
# 몸통 단면을 타원으로 보고 깊이/폭 비를 아래 값으로 가정한다.
DEPTH_TO_WIDTH = {'chest': 0.72, 'waist': 0.78, 'hip': 0.80}
# 엉덩이 랜드마크는 고관절 중심이라 실제 엉덩이 폭보다 훨씬 좁다.
HIP_LANDMARK_TO_WIDTH = 1.38
# 랜드마크 신뢰도가 이보다 낮으면 경고를 붙인다.
MIN_VISIBILITY = 0.6

# 키 대비 인체 비율의 상식적인 범위. 추정치가 여기를 벗어나면 스케일이 틀어진 것으로 본다.
# (성인 기준 대략치이며, 정확한 값을 주장하려는 게 아니라 '말이 되는가'만 거른다)
PLAUSIBLE_RATIO_TO_HEIGHT = {
    'shoulder': (0.20, 0.30),
    'chest': (0.21, 0.35),
    'hip': (0.21, 0.35),
    'torso_length': (0.24, 0.36),
}

POSE_LANDMARKS = {
    'nose': 0, 'left_shoulder': 11, 'right_shoulder': 12,
    'left_elbow': 13, 'right_elbow': 14, 'left_wrist': 15, 'right_wrist': 16,
    'left_hip': 23, 'right_hip': 24, 'left_knee': 25, 'right_knee': 26,
    'left_ankle': 27, 'right_ankle': 28, 'left_heel': 29, 'right_heel': 30,
}


@dataclass
class BodyMeasurement:
    height_cm: float
    px_per_cm: float
    measurements_cm: Dict[str, float] = field(default_factory=dict)  # 단면 기준
    raw_widths_cm: Dict[str, float] = field(default_factory=dict)    # 정면 폭 (참고용)
    warnings: List[str] = field(default_factory=list)
    landmark_visibility: Dict[str, float] = field(default_factory=dict)
    unsafe: bool = False  # 사진/스케일이 나빠 자동 입력에 쓰면 안 되는 경우
    has_segmentation: bool = False

    @property
    def reliable(self) -> bool:
        return not self.warnings

    def describe(self) -> str:
        lines = [f'키 {self.height_cm:.0f}cm 기준 추정 (단면):']
        for k, v in sorted(self.measurements_cm.items()):
            lines.append(f'  {k:9s} {v:5.1f} cm')
        for w in self.warnings:
            lines.append(f'  [주의] {w}')
        return '\n'.join(lines)


def _ellipse_perimeter(width: float, depth: float) -> float:
    """라마누잔 근사. 타원 둘레."""
    a, b = width / 2.0, depth / 2.0
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))


def _circumference_half_from_width(front_width_cm: float, part: str) -> float:
    """정면 폭 -> 둘레의 절반(=단면). 깊이는 가정값으로 채운다."""
    depth = front_width_cm * DEPTH_TO_WIDTH[part]
    return _ellipse_perimeter(front_width_cm, depth) / 2.0


def _dist(p, q) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def _torso_width_from_mask(mask, y, center_x, shoulder_px):
    """중심과 이어진 몸통 폭만 읽는다. 팔/두꺼운 옷과 합쳐지면 추정하지 않는다."""
    import numpy as np

    mask = np.squeeze(np.asarray(mask))
    if mask.ndim != 2:
        return None
    h, w = mask.shape
    cy, cx = int(round(y)), int(round(center_x))
    if not (0 <= cy < h and 0 <= cx < w):
        return None
    row = mask[cy] > 0.35
    if not row[cx]:
        return None

    limit = shoulder_px * 0.52
    left = cx
    while left > 0 and row[left] and cx - left < limit:
        left -= 1
    right = cx
    while right < w - 1 and row[right] and right - cx < limit:
        right += 1
    # 경계가 실제 배경으로 끝나지 않으면 팔과 몸통을 구별할 수 없다.
    if row[left] or row[right] or right - left < shoulder_px * 0.55:
        return None
    return float(right - left)


def measure(image, height_cm: float, pose_landmarks=None, seg_mask=None,
            shoulder_factor: float = SHOULDER_LANDMARK_TO_BREADTH) -> BodyMeasurement:
    """이미지에서 몸 치수를 추정한다.

    image: PIL.Image 또는 numpy 배열(RGB)
    height_cm: 사용자가 입력한 키
    pose_landmarks/seg_mask: 미리 뽑아둔 포즈와 마스크가 있으면 재사용
    """
    import numpy as np

    if not math.isfinite(height_cm) or height_cm <= 0:
        raise ValueError('키는 0보다 큰 cm 값이어야 합니다.')
    if hasattr(image, 'convert'):  # PIL
        from PIL import ImageOps
        image = ImageOps.exif_transpose(image).convert('RGB')
        if pose_landmarks is None:
            # 폰 원본을 그대로 Pose에 넣으면 메모리 오류가 날 수 있다.
            image.thumbnail((1280, 1280))
        image = np.array(image)
    h, w = image.shape[:2]

    if pose_landmarks is None:
        pose_landmarks, seg_mask = _run_pose(image)

    if pose_landmarks is None:
        return BodyMeasurement(height_cm=height_cm, px_per_cm=0.0,
                               warnings=['사람을 찾지 못했습니다. 전신이 나온 정면 사진인지 확인하세요.'],
                               unsafe=True)

    def pt(name):
        lm = pose_landmarks[POSE_LANDMARKS[name]]
        return (lm.x * w, lm.y * h)

    visibility = {name: float(pose_landmarks[idx].visibility)
                  for name, idx in POSE_LANDMARKS.items()}
    warnings: List[str] = []

    low_vis = [n for n in ('nose', 'left_shoulder', 'right_shoulder',
                           'left_hip', 'right_hip', 'left_ankle', 'right_ankle')
               if visibility[n] < MIN_VISIBILITY]
    unsafe = bool(low_vis) or w > h
    if w > h:
        warnings.append('세로 방향의 정면 전신 사진을 사용해주세요.')
    if low_vis:
        warnings.append(f"가려졌거나 흐릿한 부위가 있습니다: {', '.join(low_vis)}. "
                        '정면 전신 사진이면 정확도가 올라갑니다.')
    if any(not (0 <= pose_landmarks[POSE_LANDMARKS[n]].x <= 1
                    and 0 <= pose_landmarks[POSE_LANDMARKS[n]].y <= 1)
           for n in ('nose', 'left_ankle', 'right_ankle')):
        warnings.append('머리나 발이 사진 밖에 있습니다. 전신 사진을 사용해주세요.')
        unsafe = True

    px_height, scale_note = _pixel_height(image, pose_landmarks, seg_mask, w, h)
    if px_height <= 0:
        return BodyMeasurement(height_cm=height_cm, px_per_cm=0.0,
                               warnings=['몸 전체 높이를 잡지 못했습니다. 발끝까지 나오게 찍어주세요.'],
                               unsafe=True)
    if scale_note:
        warnings.append(scale_note)
        if '잘린' in scale_note or seg_mask is None:
            unsafe = True
    if px_height < 600:
        warnings.append('사진에서 전신 높이가 600px 미만입니다. 더 가까이서 선명하게 찍어주세요.')
        unsafe = True

    px_per_cm = px_height / height_cm

    shoulder_px = _dist(pt('left_shoulder'), pt('right_shoulder'))
    hip_px = _dist(pt('left_hip'), pt('right_hip'))
    shoulder_mid = ((pt('left_shoulder')[0] + pt('right_shoulder')[0]) / 2,
                    (pt('left_shoulder')[1] + pt('right_shoulder')[1]) / 2)
    hip_mid = ((pt('left_hip')[0] + pt('right_hip')[0]) / 2,
               (pt('left_hip')[1] + pt('right_hip')[1]) / 2)
    torso_px = _dist(shoulder_mid, hip_mid)
    if torso_px <= 0 or shoulder_px / torso_px < 0.45:
        warnings.append('어깨가 정면에서 보이지 않습니다. 정면 자세로 다시 찍어주세요.')
        unsafe = True

    shoulder_cm = shoulder_px / px_per_cm * shoulder_factor
    hip_width_cm = hip_px / px_per_cm * HIP_LANDMARK_TO_WIDTH
    torso_cm = torso_px / px_per_cm

    shoulder_y = shoulder_mid[1]
    hip_y = hip_mid[1]
    torso_height = hip_y - shoulder_y
    center_x = (pt('left_shoulder')[0] + pt('right_shoulder')[0]
                + pt('left_hip')[0] + pt('right_hip')[0]) / 4.0
    chest_width_px = waist_width_px = None
    if seg_mask is not None and torso_height > 0:
        chest_width_px = _torso_width_from_mask(
            seg_mask, shoulder_y + torso_height * 0.28, center_x, shoulder_px)
        waist_width_px = _torso_width_from_mask(
            seg_mask, shoulder_y + torso_height * 0.65, center_x, shoulder_px)
        if chest_width_px is None:
            warnings.append('가슴 경계가 팔/옷과 겹쳐 가슴둘레를 추정하지 않았습니다.')
        if waist_width_px is None:
            warnings.append('허리 경계가 팔/옷과 겹쳐 허리둘레를 추정하지 않았습니다.')
    else:
        # 기존 랜드마크 경로: 세그멘테이션이 없는 경우에도 가슴은 참고값으로 남긴다.
        chest_width_px = shoulder_px * 0.92
        warnings.append('몸통 마스크가 없어 가슴은 어깨 비율로 근사했고 허리는 추정하지 않았습니다.')

    measurements = {
        'shoulder': round(shoulder_cm, 1),
        'hip': round(_circumference_half_from_width(hip_width_cm, 'hip'), 1),
        'torso_length': round(torso_cm, 1),
    }
    raw_widths = {
        'hip_front_width': round(hip_width_cm, 1),
    }
    if chest_width_px is not None:
        chest_width_cm = chest_width_px / px_per_cm
        measurements['chest'] = round(_circumference_half_from_width(chest_width_cm, 'chest'), 1)
        raw_widths['chest_front_width'] = round(chest_width_cm, 1)
    if waist_width_px is not None:
        waist_width_cm = waist_width_px / px_per_cm
        measurements['waist'] = round(_circumference_half_from_width(waist_width_cm, 'waist'), 1)
        raw_widths['waist_front_width'] = round(waist_width_cm, 1)
    warnings.append('엉덩이둘레는 골반 관절폭과 고정 비율로 계산한 참고값입니다.')

    implausible = _implausible_notes(measurements, height_cm)
    warnings.extend(implausible)
    unsafe = unsafe or bool(implausible)

    return BodyMeasurement(
        height_cm=height_cm,
        px_per_cm=round(px_per_cm, 3),
        measurements_cm=measurements,
        raw_widths_cm=raw_widths,
        warnings=warnings,
        landmark_visibility={k: round(v, 2) for k, v in visibility.items()},
        unsafe=unsafe,
        has_segmentation=seg_mask is not None,
    )


def _implausible_notes(measurements: Dict[str, float], height_cm: float) -> List[str]:
    """추정치가 인체 비율 범위를 벗어나면 이유를 남긴다.

    대부분 픽셀->cm 스케일이 틀어졌을 때 발생한다. 사진이 잘려 있으면
    '머리끝~발끝 = 키' 전제가 깨져서 모든 치수가 같은 비율로 부풀거나 줄어든다.
    """
    notes = []
    for name, (low, high) in PLAUSIBLE_RATIO_TO_HEIGHT.items():
        if name not in measurements:
            continue
        ratio = measurements[name] / height_cm
        if not (low <= ratio <= high):
            notes.append(
                f'{name} {measurements[name]:.1f}cm 는 키 {height_cm:.0f}cm 대비 비율 '
                f'{ratio:.2f} 로 상식 범위({low}~{high})를 벗어납니다. '
                '사진이 잘렸거나 전신이 아니어서 스케일이 틀어졌을 가능성이 큽니다.'
            )
    return notes


def _pixel_height(image, landmarks, seg_mask, w: int, h: int):
    """머리끝~발끝 픽셀 높이. 세그멘테이션이 있으면 그걸 쓰고, 없으면 랜드마크로 추정."""
    import numpy as np

    if seg_mask is not None:
        # Tasks API는 (H, W, 1)로 주고 레거시는 (H, W)로 준다. 어느 쪽이든 2차원으로 만든다.
        mask2d = np.squeeze(np.asarray(seg_mask))
        if mask2d.ndim == 2:
            rows = np.where(np.any(mask2d > 0.5, axis=1))[0]
            if rows.size > 0:
                top, bottom = int(rows.min()), int(rows.max())
                # 몸이 위/아래 가장자리에 닿아 있으면 잘린 사진이다.
                # 이때 '머리끝~발끝 = 키' 전제가 깨져 모든 치수가 함께 틀어진다.
                cropped = []
                if top <= 1:
                    cropped.append('머리')
                if bottom >= mask2d.shape[0] - 2:
                    cropped.append('발')
                note = ''
                if cropped:
                    note = (f"{'/'.join(cropped)} 쪽이 사진 밖으로 잘린 것 같습니다. "
                            '전신이 다 나와야 키 기준 스케일이 맞습니다.')
                return float(bottom - top), note

    # 폴백: 코~발목 거리에 머리 윗부분과 발 높이를 보정해서 더한다.
    nose_y = landmarks[POSE_LANDMARKS['nose']].y * h
    ankles = [landmarks[POSE_LANDMARKS[n]].y * h for n in ('left_ankle', 'right_ankle')]
    ankle_y = max(ankles)
    if ankle_y <= nose_y:
        return 0.0, ''
    # 코는 정수리에서 약 12% 아래, 발목은 바닥에서 약 4% 위 (인체 비율 통계 근사)
    span = (ankle_y - nose_y) / (1.0 - 0.12 - 0.04)
    return float(span), '세그멘테이션을 못 써서 키 스케일을 인체 비율로 근사했습니다(오차 큼).'


# MediaPipe 1.x 에서 레거시 mp.solutions 가 제거되고 Tasks API 로 바뀌었다.
# 두 경로를 모두 지원한다. 랜드마크 33점 토폴로지는 동일해서 POSE_LANDMARKS 는 그대로 쓴다.
POSE_TASK_URL = ('https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
                 'pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task')


def _pose_task_model_path() -> str:
    """Tasks API용 모델 파일을 받아 캐시한다."""
    import urllib.request

    cache_dir = os.environ.get('VFA_SCRATCH') or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.scratch')
    cache_dir = os.path.join(cache_dir, 'mediapipe')
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, 'pose_landmarker_heavy.task')
    if not os.path.exists(path):
        urllib.request.urlretrieve(POSE_TASK_URL, path)
    return path


def _run_pose(image):
    """MediaPipe Pose 실행. (landmarks, segmentation_mask) 반환."""
    try:
        import mediapipe as mp
    except ImportError as e:
        raise ImportError(
            'mediapipe가 필요합니다. venv에 설치하세요:\n'
            '    <venv>/bin/pip install mediapipe'
        ) from e

    if hasattr(mp, 'solutions'):  # mediapipe 0.10.x 이하
        with mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5,
        ) as pose:
            result = pose.process(image)
        if not result.pose_landmarks:
            return None, None
        return result.pose_landmarks.landmark, result.segmentation_mask

    # mediapipe 1.x — Tasks API
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    # 한글 경로의 모델 파일을 Tasks가 열지 못하는 환경이 있어 버퍼로 전달한다.
    with open(_pose_task_model_path(), 'rb') as model_file:
        model_buffer = model_file.read()
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=model_buffer),
        running_mode=vision.RunningMode.IMAGE,
        output_segmentation_masks=True,
    )
    import numpy as np
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        result = landmarker.detect(mp_image)
        # numpy_view()는 mediapipe 내부 버퍼를 가리킨다. landmarker가 닫힌 뒤에 읽으면
        # 해제된 메모리라 segfault가 난다(photo_check에서 실제로 났다). 안에서 복사한다.
        mask = None
        if result.pose_landmarks and getattr(result, 'segmentation_masks', None):
            mask = np.squeeze(result.segmentation_masks[0].numpy_view()).copy()

    if not result.pose_landmarks:
        return None, None
    return result.pose_landmarks[0], mask
