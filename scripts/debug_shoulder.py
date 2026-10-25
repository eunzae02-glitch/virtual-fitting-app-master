"""정면 전신 사진에서 어깨 추정치만 확인하는 독립 웹 디버거.

    python -X utf8 scripts/debug_shoulder.py --ui

키 입력은 픽셀을 cm로 바꾸는 데만 사용한다. 성별·체중 모델과 옷 치수표는
이 단계에서 사용하지 않는다.
"""
import argparse
import os
import sys

from PIL import ImageDraw

APP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app')
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from shoulder_measure import measure_shoulder  # noqa: E402


def analyze_shoulder(image, height_cm, actual_shoulder_cm=None):
    """주석 달린 사진과 어깨 추정·오차 마크다운을 반환한다. 파일은 저장하지 않는다."""
    if actual_shoulder_cm is not None:
        try:
            actual_shoulder_cm = float(actual_shoulder_cm)
        except (TypeError, ValueError) as exc:
            raise ValueError('실측 어깨사이길이는 cm 단위 숫자로 입력해주세요.') from exc
        if not 15 <= actual_shoulder_cm <= 80:
            raise ValueError('실측 어깨사이길이는 15~80cm 범위로 입력해주세요.')

    photo, result = measure_shoulder(image, height_cm)
    visual = photo.copy()
    draw = ImageDraw.Draw(visual)
    line_width = max(3, visual.width // 250)
    if result.head_y is not None:
        draw.line([(0, result.head_y), (visual.width, result.head_y)],
                  fill=(40, 220, 120), width=line_width)
    if result.foot_y is not None:
        draw.line([(0, result.foot_y), (visual.width, result.foot_y)],
                  fill=(40, 220, 120), width=line_width)
    if result.left_shoulder and result.right_shoulder:
        draw.line([result.left_shoulder, result.right_shoulder],
                  fill=(255, 80, 80), width=line_width)
        radius = max(5, line_width * 2)
        for x, y in (result.left_shoulder, result.right_shoulder):
            draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                         fill=(255, 220, 50), outline=(30, 30, 30), width=2)

    lines = ['### 어깨 추정 결과']
    if result.valid:
        lines.append(f'- 추정 어깨너비: **{result.shoulder_cm:.1f} cm**')
    else:
        lines.append('- **측정 보류 — 사진을 확인하고 재업로드해주세요.**')
    if result.shoulder_landmark_px is not None:
        lines.append(f'- 어깨 관절점 간 거리: {result.shoulder_landmark_px:.1f} px')
    if result.body_height_px is not None:
        lines.append(f'- 머리끝~발끝 높이: {result.body_height_px:.1f} px')
    if result.px_per_cm is not None:
        lines.append(f'- 키 {result.height_cm:g} cm 기준 척도: {result.px_per_cm:.3f} px/cm')
    lines.append(f'- 기존 실험 보정계수: ×{result.factor:g} (고정, 검증 전)')

    if result.valid and actual_shoulder_cm is not None:
        error = result.shoulder_cm - actual_shoulder_cm
        lines.append(f'- 입력한 실측 어깨사이길이: {actual_shoulder_cm:.1f} cm')
        lines.append(f'- 오차(사진 추정 − 실측): **{error:+.1f} cm**')
    elif actual_shoulder_cm is not None:
        lines.append(f'- 입력한 실측 어깨사이길이: {actual_shoulder_cm:.1f} cm (사진 측정 보류로 오차 미계산)')

    lines.extend(f'\n> 재촬영/확인: {warning}' for warning in result.warnings)
    lines.extend(f'\n> 참고: {note}' for note in result.notes)
    lines.append('\n빨간 선=MediaPipe 어깨 관절점, 초록 선=높이 기준. '
                 '사진상의 관절점 간 직선거리×1.28은 Size Korea의 `어깨사이길이`와 '
                 '측정 정의가 같다고 검증되지 않았습니다. 위 값은 실측이 아닌 실험용 추정치입니다.')
    return visual, '\n'.join(lines)


def launch_ui(port):
    import gradio as gr

    with gr.Blocks(title='FitCheck 어깨사이길이 디버거') as demo:
        gr.Markdown('# 전신 사진 어깨너비 디버깅\n'
                    '정면 전신 사진과 키를 넣어 **어깨만** 확인합니다. '
                    '가슴·허리·엉덩이·성별·체중·옷 치수표는 이 단계에서 사용하지 않습니다.')
        with gr.Row():
            with gr.Column():
                photo_in = gr.Image(label='정면 전신 사진 (머리부터 발끝까지)', type='pil', height=450)
                height_in = gr.Number(label='키 (cm, 필수)', value=175)
                actual_in = gr.Number(label='실제 어깨사이길이 (cm, 선택: 오차 비교용)', value=None)
                run_btn = gr.Button('어깨 측정하기', variant='primary')
            with gr.Column():
                photo_out = gr.Image(label='기준선과 어깨 랜드마크', height=450)
                report_out = gr.Markdown()

        def on_run(photo, height, actual):
            try:
                return analyze_shoulder(photo, height, actual)
            except (ValueError, ImportError, OSError) as exc:
                raise gr.Error(str(exc)) from exc

        run_btn.click(on_run, [photo_in, height_in, actual_in], [photo_out, report_out])

    demo.launch(server_name='127.0.0.1', server_port=port, share=False)


def main():
    parser = argparse.ArgumentParser(description='어깨너비만 측정하는 로컬 웹 디버거')
    parser.add_argument('--ui', action='store_true', help='웹 UI 실행 (생략해도 실행)')
    parser.add_argument('--port', type=int, default=7862, help='로컬 웹 포트 (기본 7862)')
    args = parser.parse_args()
    launch_ui(args.port)


if __name__ == '__main__':
    main()
