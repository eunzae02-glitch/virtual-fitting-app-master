"""상의 치수표 추천 웹 디버거. GPU/합성 모델 불필요.

프로젝트 루트에서: python -X utf8 scripts/debug_upper_size.py --ui
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app'))

from chart_ocr import image_to_draft
from upper_size import estimate_upper, parse_upper_chart, recommend_upper


def _display_number(value):
    return '—' if value is None else f'{value:+.1f} cm'


def _clear_chart(_image):
    return '', '', '', False


def _unconfirm(_text):
    return False


def analyze(person, height, basis, calibration_csv, manual_chest,
            reference_json, reference_confirmed, worn_size, worn_fit,
            target_json, target_confirmed, target_shoulder_standard,
            actual_shoulder, actual_chest):
    """Gradio와 독립적으로 테스트 가능한 단일 실행 경로."""
    try:
        body = estimate_upper(person, height, basis, calibration_csv, manual_chest)
        lines = ['### 사진 기반 상체 추정', '',
                 f'- 어깨너비: {body.shoulder.middle:.1f} cm — {body.shoulder.source}']
        if body.chest_circ:
            chest = body.chest_circ
            value = (f'{chest.low:.1f}~{chest.high:.1f}' if chest.low != chest.high
                     else f'{chest.middle:.1f}')
            lines.append(f'- 가슴둘레: {value} cm — {chest.source}')
        else:
            lines.append('- 가슴둘레: 추정 불가')
        if actual_shoulder is not None:
            lines.append(f'- 어깨 실측 비교 오차: {body.shoulder.middle - float(actual_shoulder):+.1f} cm')
        if actual_chest is not None and body.chest_circ:
            lines.append(f'- 가슴 실측 비교 오차(중앙값): {body.chest_circ.middle - float(actual_chest):+.1f} cm')

        if not target_json or not target_json.strip():
            lines.append('\n대상 상품 치수표가 없어 S/M/L 추천은 보류합니다.')
            lines += [f'\n> {note}' for note in body.notes]
            return '\n'.join(lines), ''
        if not target_confirmed:
            raise ValueError('대상 치수표의 OCR/직접 입력 숫자와 라벨을 확인하고 체크해주세요.')
        target = parse_upper_chart(json.loads(target_json), target_shoulder_standard)
        reference = None
        if reference_json and reference_json.strip():
            if not reference_confirmed:
                raise ValueError('기존 반팔 치수표의 OCR/직접 입력 숫자와 라벨을 확인하고 체크해주세요.')
            reference = parse_upper_chart(json.loads(reference_json))
        result = recommend_upper(body, target, reference, worn_size, worn_fit)
        lines += ['', f'### {result.status}: {", ".join(result.candidates) or "없음"}',
                  f'대상 상품: {target.chart.name}']
        lines.extend(f'\n> {note}' for note in result.notes)
        table = ['| 사이즈 | 옷 가슴단면 | 가슴 여유 범위 | 어깨 여유 |',
                 '|---|---:|---:|---:|']
        for size, low, high, shoulder_ease, chest_flat in result.rows:
            ease = '— (몸 가슴 추정 불가)' if low is None else (
                f'{low:+.1f}~{high:+.1f} cm' if low != high else f'{low:+.1f} cm')
            table.append(f'| {size} | {chest_flat:.1f} cm | {ease} | '
                         f'{_display_number(shoulder_ease)} |')
        return '\n'.join(lines), '\n'.join(table)
    except (ValueError, json.JSONDecodeError, ImportError, OSError, RuntimeError) as exc:
        return f'### 분석/추천 보류\n\n{exc}', ''


def launch_ui():
    import gradio as gr

    with gr.Blocks(title='FitCheck 상의 사이즈 디버거') as demo:
        gr.Markdown('# FitCheck 상의 사이즈 추천 디버거\n'
                    '정면 전신 사진과 키를 입력하세요. 합성 모델이나 GPU는 사용하지 않습니다. '
                    '치수표 OCR은 로컬 Tesseract가 있을 때 실행되며, 결과는 반드시 직접 확인해야 합니다.\n\n'
                    '수동 JSON도 가능합니다: `{"unit":"cm","sizes":{"S":{"가슴단면":49},'
                    '"M":{"가슴단면":52}}}`. 옷 완성 둘레라면 `옷 가슴둘레`, '
                    '권장 신체둘레라면 추천에 사용하지 않습니다.')
        with gr.Row():
            with gr.Column():
                person = gr.Image(label='정면 전신 사진', type='pil', height=350)
                height = gr.Number(label='키 (cm)', value=None)
                basis = gr.Radio(['선택 안 함', '남', '녀'], value='선택 안 함',
                                 label='가슴 경계 불명확 시 평균 기준(사진에서 판별하지 않음)')
                calibration = gr.File(label='실측 보정 CSV (선택, basis/shoulder_cm/chest_circ_cm)',
                                      type='filepath')
                manual_chest = gr.Number(label='가슴둘레 직접 입력 (cm, 선택)', value=None)
                with gr.Accordion('실측 오차 대조 (디버그용)', open=False):
                    actual_shoulder = gr.Number(label='실제 어깨너비 (cm)', value=None)
                    actual_chest = gr.Number(label='실제 가슴둘레 (cm)', value=None)
            with gr.Column():
                with gr.Accordion('평소 잘 입는 레귤러핏 반팔 치수표 (선택)', open=False):
                    reference_image = gr.Image(label='기존 반팔 치수표 이미지', type='pil')
                    reference_ocr_btn = gr.Button('기존 반팔 치수표 OCR')
                    reference_text = gr.Textbox(label='OCR 원문', lines=3, interactive=False)
                    reference_json = gr.Textbox(label='확인/수정할 치수표 JSON', lines=8,
                                                placeholder='{"name":"평소 반팔","unit":"cm","sizes":{"M":{"가슴단면":52,"어깨너비":45}}}')
                    reference_ocr_status = gr.Markdown()
                    reference_confirmed = gr.Checkbox(label='원본과 대조하고 숫자·라벨 확인', value=False)
                    worn_size = gr.Textbox(label='실제로 입는 사이즈 (예: M)', value='')
                    worn_fit = gr.Radio(['작음', '적당함', '큼'], value='적당함', label='그 반팔의 착용감')
                with gr.Accordion('새 옷 치수표 (상품별 추천에 필수)', open=True):
                    target_image = gr.Image(label='새 옷 치수표 이미지', type='pil')
                    target_ocr_btn = gr.Button('새 옷 치수표 OCR')
                    target_text = gr.Textbox(label='OCR 원문', lines=3, interactive=False)
                    target_json = gr.Textbox(label='확인/수정할 치수표 JSON', lines=10,
                                             placeholder='{"name":"새 반팔","unit":"cm","sizes":{"S":{"가슴단면":49},"M":{"가슴단면":52},"L":{"가슴단면":55}}}')
                    target_ocr_status = gr.Markdown()
                    target_confirmed = gr.Checkbox(label='원본과 대조하고 숫자·라벨 확인', value=False)
                    target_shoulder_standard = gr.Checkbox(
                        label='새 옷이 일반 어깨선이며 어깨너비를 몸 어깨와 비교 가능', value=False)
        run_btn = gr.Button('상체 추정 + 사이즈 비교', variant='primary')
        report = gr.Markdown()
        table = gr.Markdown()
        reference_ocr_btn.click(fn=image_to_draft, inputs=[reference_image],
                                outputs=[reference_json, reference_text, reference_ocr_status])
        target_ocr_btn.click(fn=image_to_draft, inputs=[target_image],
                             outputs=[target_json, target_text, target_ocr_status])
        reference_image.change(fn=_clear_chart, inputs=[reference_image],
                               outputs=[reference_json, reference_text,
                                        reference_ocr_status, reference_confirmed])
        target_image.change(fn=_clear_chart, inputs=[target_image],
                            outputs=[target_json, target_text, target_ocr_status, target_confirmed])
        reference_json.change(fn=_unconfirm, inputs=[reference_json], outputs=[reference_confirmed])
        target_json.change(fn=_unconfirm, inputs=[target_json], outputs=[target_confirmed])
        run_btn.click(
            fn=analyze,
            inputs=[person, height, basis, calibration, manual_chest,
                    reference_json, reference_confirmed, worn_size, worn_fit,
                    target_json, target_confirmed, target_shoulder_standard,
                    actual_shoulder, actual_chest],
            outputs=[report, table],
        )
    print('로컬 주소에서 상의 사이즈 디버거를 여세요.', flush=True)
    demo.launch(share=False)


def main():
    parser = argparse.ArgumentParser(description='FitCheck 상의 사이즈 웹 디버거')
    parser.add_argument('--ui', action='store_true', help='Gradio 로컬 웹 UI 실행')
    args = parser.parse_args()
    if args.ui:
        launch_ui()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
