"""신체 치수 측정 및 시각화 디버깅 도구 (scripts/debug_body_measure.py).

[목적]
사진에서 MediaPipe Pose를 통해 추출한 인체 랜드마크와 타원 기하학 공식을 이용하여
어깨너비, 가슴둘레, 허리둘레 등을 추정하고, 실제 치수와 비교하여 오차를 디버깅합니다.

몸통 마스크가 팔과 합쳐지면 해당 부위의 추정을 보류합니다.
계산은 앱과 같은 app/body_measure.py를 사용합니다.
"""
import argparse
import os
import sys
import urllib.request
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
APP_DIR = os.path.join(REPO_ROOT, 'app')
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from body_measure import POSE_LANDMARKS, measure

# --- 튜닝 가능한 인체공학 보정 계수 ---
DEFAULT_SHOULDER_CORRECTION = 1.28

POSE_TASK_URL = (
    'https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
    'pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task'
)

_cached_model_buffer = None


def _get_model_buffer() -> bytes:
    """한글 경로 에러를 피하기 위해 모델 파일을 메모리 버퍼로 캐싱."""
    global _cached_model_buffer
    if _cached_model_buffer is not None:
        return _cached_model_buffer

    cache_dir = os.path.join(REPO_ROOT, '.scratch', 'mediapipe')
    os.makedirs(cache_dir, exist_ok=True)
    model_path = os.path.join(cache_dir, 'pose_landmarker_heavy.task')

    if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000000:
        print("[MediaPipe] 모델 가중치 준비 중 (약 30MB)...", flush=True)
        urllib.request.urlretrieve(POSE_TASK_URL, model_path)
        print("[MediaPipe] 준비 완료!", flush=True)

    with open(model_path, 'rb') as f:
        _cached_model_buffer = f.read()

    return _cached_model_buffer


def _run_pose_safe(img_np: np.ndarray):
    """MediaPipe Tasks API를 안전하게 실행 (한글 경로 버그 방지)."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model_buf = _get_model_buffer()
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=model_buf),
        running_mode=vision.RunningMode.IMAGE,
        output_segmentation_masks=True,
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
        result = landmarker.detect(mp_image)

        mask = None
        if result.pose_landmarks and getattr(result, 'segmentation_masks', None):
            mask = np.squeeze(result.segmentation_masks[0].numpy_view()).copy()

    if not result.pose_landmarks:
        return None, None

    return result.pose_landmarks[0], mask


def analyze_body(
    image: Image.Image,
    height_cm: float,
    shoulder_factor: float = DEFAULT_SHOULDER_CORRECTION,
    real_shoulder: Optional[float] = None,
    real_chest: Optional[float] = None,
    real_waist: Optional[float] = None,
    real_hip: Optional[float] = None,
) -> Tuple[Image.Image, str, Dict[str, float]]:
    """사진에서 신체 치수를 측정하고 시각화 이미지와 분석 리포트를 생성합니다."""
    image = ImageOps.exif_transpose(image).convert('RGB')
    image.thumbnail((1280, 1280))
    img_np = np.array(image)
    h_img, w_img, _ = img_np.shape

    landmarks, mask = _run_pose_safe(img_np)
    if landmarks is None:
        return image, "[알림] 사람의 전신 자세를 인식하지 못했습니다. 머리부터 발끝까지 화면 가득 나온 정면 사진인지 확인해주세요.", {}

    result = measure(image, height_cm, pose_landmarks=landmarks, seg_mask=mask,
                     shoulder_factor=shoulder_factor)
    if not result.measurements_cm:
        return image, '\n'.join(result.warnings), {}

    def get_pt(name: str) -> Tuple[float, float]:
        idx = POSE_LANDMARKS[name]
        lm = landmarks[idx]
        return lm.x * w_img, lm.y * h_img

    nose = get_pt('nose')
    l_sh, r_sh = get_pt('left_shoulder'), get_pt('right_shoulder')
    l_hip, r_hip = get_pt('left_hip'), get_pt('right_hip')
    l_ankle, r_ankle = get_pt('left_ankle'), get_pt('right_ankle')

    sh_y_mid = (l_sh[1] + r_sh[1]) / 2.0
    head_top_y = max(0.0, nose[1] - abs(sh_y_mid - nose[1]) * 1.1)
    feet_y = max(l_ankle[1], r_ankle[1])
    feet_y = min(float(h_img), feet_y + (feet_y - head_top_y) * 0.035)
    if mask is not None:
        rows = np.where(np.any(np.squeeze(mask) > 0.5, axis=1))[0]
        if rows.size:
            head_top_y, feet_y = float(rows.min()), float(rows.max())
    px_per_cm = result.px_per_cm
    total_px_h = px_per_cm * height_cm
    torso_h = ((l_hip[1] + r_hip[1]) / 2.0) - sh_y_mid
    chest_y = sh_y_mid + torso_h * 0.28
    waist_y = sh_y_mid + torso_h * 0.65
    hip_y = (l_hip[1] + r_hip[1]) / 2.0
    center_x = (l_sh[0] + r_sh[0] + l_hip[0] + r_hip[0]) / 4.0
    raw = result.raw_widths_cm
    measurements = {
        'shoulder': result.measurements_cm['shoulder'],
        'hip_circ': round(result.measurements_cm['hip'] * 2, 1),
    }
    if 'chest' in result.measurements_cm:
        measurements['chest_flat'] = result.measurements_cm['chest']
        measurements['chest_circ'] = round(result.measurements_cm['chest'] * 2, 1)
    if 'waist' in result.measurements_cm:
        measurements['waist_circ'] = round(result.measurements_cm['waist'] * 2, 1)

    # 5. 시각화 이미지 그리기
    vis_img = image.copy()
    draw = ImageDraw.Draw(vis_img)

    # 가이드선
    draw.line([(0, head_top_y), (w_img, head_top_y)], fill=(0, 230, 115), width=3)
    draw.line([(0, feet_y), (w_img, feet_y)], fill=(0, 230, 115), width=3)
    draw.text((15, max(5, int(head_top_y) - 25)), f"정수리 기준선 (키: {height_cm}cm)", fill=(0, 230, 115))
    draw.text((15, int(feet_y) + 8), f"발바닥 기준선 (총 {int(total_px_h)}px)", fill=(0, 230, 115))

    # 어깨선
    draw.line([(r_sh[0], r_sh[1]), (l_sh[0], l_sh[1])], fill=(255, 60, 60), width=4)
    r_val = 6
    draw.ellipse([(r_sh[0] - r_val, r_sh[1] - r_val), (r_sh[0] + r_val, r_sh[1] + r_val)], fill=(255, 255, 0))
    draw.ellipse([(l_sh[0] - r_val, l_sh[1] - r_val), (l_sh[0] + r_val, l_sh[1] + r_val)], fill=(255, 255, 0))
    draw.text((int(l_sh[0]) + 15, int(l_sh[1]) - 10), f"어깨너비: {measurements['shoulder']:.1f}cm", fill=(255, 60, 60))

    # 같은 추정 결과로 가이드선을 그린다. 측정 불가 부위에는 선을 그리지 않는다.
    for key, y, label, color in (
        ('chest', chest_y, '가슴둘레', (0, 180, 255)),
        ('waist', waist_y, '허리둘레', (255, 170, 0)),
        ('hip', hip_y, '엉덩이둘레', (200, 100, 255)),
    ):
        width_cm = raw.get(f'{key}_front_width')
        if width_cm is None:
            continue
        half_px = width_cm * px_per_cm / 2.0
        left, right = center_x - half_px, center_x + half_px
        draw.line([(left, y), (right, y)], fill=color, width=4)
        draw.text((int(right) + 10, int(y) - 8),
                  f"{label}: {result.measurements_cm[key] * 2:.1f}cm", fill=color)

    # 6. 리포트 테이블 마크다운 생성
    report_lines = [
        "### [신체 치수 추정 및 오차 분석 리포트]",
        f"- 입력 신체 키: {height_cm:.1f} cm (기준 척도: {px_per_cm:.2f} px/cm)",
        f"- 어깨 보정 계수: {shoulder_factor:.2f}",
        "",
        "| 신체 부위 | AI 사진 추정치 | 사용자 실제 실측치 | 오차 (추정 - 실제) | 정확도 판정 |",
        "|---|---|---|---|---|",
    ]

    items = [
        ('어깨너비', measurements['shoulder'], real_shoulder, 'cm'),
        ('가슴둘레', measurements.get('chest_circ'), real_chest, 'cm'),
        ('가슴단면', measurements.get('chest_flat'), (real_chest / 2.0 if real_chest else None), 'cm'),
        ('허리둘레', measurements.get('waist_circ'), real_waist, 'cm'),
        ('엉덩이둘레', measurements['hip_circ'], real_hip, 'cm'),
    ]

    tune_suggestions = []

    for name, est, real, unit in items:
        if est is None:
            report_lines.append(f'| **{name}** | 추정 불가 | {f"{real:.1f} {unit}" if real else "-"} | - | 팔/옷 경계 확인 필요 |')
            continue
        if real is not None and real > 0:
            diff = est - real
            err_pct = abs(diff) / real * 100.0
            status = "[O] 매우 정확 (≤2cm)" if abs(diff) <= 2.0 else ("[!] 양호 (≤4cm)" if abs(diff) <= 4.0 else "[X] 보정 필요 (>4cm)")
            report_lines.append(f"| **{name}** | {est:.1f} {unit} | {real:.1f} {unit} | **{diff:+.1f} {unit}** ({err_pct:.1f}%) | {status} |")

            if name == '어깨너비' and abs(diff) > 1.5:
                ideal_factor = round(shoulder_factor * (real / est), 2)
                tune_suggestions.append(f"- 어깨너비 튜닝 제안: 현재 보정 계수 {shoulder_factor} 대신 **{ideal_factor}** 로 설정하면 실제 어깨 치수와 일치합니다.")
        else:
            report_lines.append(f"| **{name}** | {est:.1f} {unit} | (실측치 미입력) | - | - |")

    if tune_suggestions:
        report_lines.append("\n#### [보정 계수 최적화 제안]")
        report_lines.extend(tune_suggestions)

    report_lines.append("\n> 사진 한 장의 실루엣과 가정한 몸 두께 비율로 추정했습니다. 두꺼운 옷·박시한 옷은 실제 몸보다 크게 보일 수 있습니다.")
    report_lines.extend(f'\n> 주의: {warning}' for warning in result.warnings)

    return vis_img, "\n".join(report_lines), measurements


def launch_ui():
    """Gradio 기반 시각적 디버깅 웹 도구 실행."""
    import gradio as gr

    with gr.Blocks(title="신체 치수 측정 & 오차 디버깅 툴") as demo:
        gr.Markdown(
            "# [신체 치수 추정 & 오차 디버깅 도구]\n"
            "몸에 붙는 옷을 입은 정면 전신 사진과 키를 입력하면 치수를 추정합니다. 팔/옷 경계가 불분명하면 일부 항목은 비워 둡니다."
        )

        with gr.Row():
            with gr.Column(scale=1):
                img_in = gr.Image(label="전신 사진 (정면/반팔)", type="pil", height=420)
                height_in = gr.Number(value=175.0, label="내 키 (cm, 필수)")

                with gr.Accordion("내 실제 실측 치수 (오차 디버깅용 / 아는 것만 입력)", open=True):
                    real_sh_in = gr.Number(value=None, label="실제 어깨너비 (cm)")
                    real_ch_in = gr.Number(value=None, label="실제 가슴둘레 (cm)")
                    real_w_in = gr.Number(value=None, label="실제 허리둘레 (cm)")
                    real_hip_in = gr.Number(value=None, label="실제 엉덩이둘레 (cm)")

                with gr.Accordion("고급 설정 (보정 계수 튜닝)", open=False):
                    sh_factor_in = gr.Slider(1.0, 1.4, value=DEFAULT_SHOULDER_CORRECTION, step=0.01,
                                             label="어깨 관절 보정 계수 (기본 1.28)")

                run_btn = gr.Button("측정 및 오차 분석 실행", variant="primary")

            with gr.Column(scale=1):
                img_out = gr.Image(label="랜드마크 측정 시각화 결과", height=480)
                report_out = gr.Markdown(label="분석 리포트")

        def _on_run(img, h, sh_f, r_sh, r_ch, r_w, r_hip):
            if img is None:
                raise gr.Error("사진을 업로드해주세요.")
            if not h or h <= 0:
                raise gr.Error("키(cm)를 입력해주세요.")
            vis, rep, _ = analyze_body(img, h, sh_f, r_sh, r_ch, r_w, r_hip)
            return vis, rep

        run_btn.click(
            fn=_on_run,
            inputs=[img_in, height_in, sh_factor_in, real_sh_in, real_ch_in, real_w_in, real_hip_in],
            outputs=[img_out, report_out],
        )

    print("=" * 60)
    print("치수 측정 디버깅 웹 도구를 실행합니다.")
    print("브라우저에서 아래 로컬 주소로 접속하세요:")
    print("=" * 60)
    demo.launch(share=False)


def main():
    parser = argparse.ArgumentParser(description="신체 치수 측정 & 디버깅 도구")
    parser.add_argument("--ui", action="store_true", help="Gradio 웹 UI 모드로 실행")
    parser.add_argument("--image", type=str, help="분석할 인물 사진 경로")
    parser.add_argument("--height", type=float, default=175.0, help="사용자 키 (cm)")
    parser.add_argument("--shoulder", type=float, default=None, help="실제 어깨너비 (cm)")
    parser.add_argument("--chest", type=float, default=None, help="실제 가슴둘레 (cm)")
    parser.add_argument("--waist", type=float, default=None, help="실제 허리둘레 (cm)")
    parser.add_argument("--hip", type=float, default=None, help="실제 엉덩이둘레 (cm)")
    parser.add_argument("--output", type=str, default="debug_measurement_result.jpg", help="결과 이미지 저장 경로")
    args = parser.parse_args()

    if args.ui or len(sys.argv) == 1:
        launch_ui()
        return

    if not args.image or not os.path.exists(args.image):
        print(f"오류: 이미지 파일을 찾을 수 없습니다: {args.image}")
        return

    img = Image.open(args.image)
    vis_img, report, meas = analyze_body(
        img, args.height, DEFAULT_SHOULDER_CORRECTION,
        args.shoulder, args.chest, args.waist, args.hip
    )

    vis_img.save(args.output)
    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)
    print(f"시각화 결과 이미지가 저장되었습니다: {os.path.abspath(args.output)}")


if __name__ == '__main__':
    main()
