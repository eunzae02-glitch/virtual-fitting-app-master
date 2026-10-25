"""키·체중·성별로 신체 치수 모델을 비교하는 GPU 없는 웹 디버거.

    python -X utf8 scripts/debug_body_model.py --ui
"""
import argparse
import os
import sys

APP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app')
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from body_model import (MODEL_NAMES, TARGETS, load_model, positive_number, predict_body,
                        report_markdown)  # noqa: E402


MODEL_CHOICES = {'자동 선택 (CV 기준)': 'selected',
                 **{label: name for name, label in MODEL_NAMES.items()}}


def analyze(bundle, sex, height, weight, model_choice='자동 선택 (CV 기준)',
            actual_shoulder=None, actual_length=None, actual_chest=None):
    try:
        model = MODEL_CHOICES.get(model_choice)
        if model is None:
            raise ValueError('비교할 모델을 선택해주세요.')
        predictions = predict_body(bundle, sex, height, weight, model)
        actuals = dict(zip(TARGETS, (actual_shoulder, actual_length, actual_chest)))
        for target, value in actuals.items():
            if value is not None:
                actuals[target] = positive_number(value, f'실측 {TARGETS[target]}')
        coverage = bundle['report']['coverage_target']
        lines = ['### 예상 신체 치수', '',
                 f'| 항목 | 예측값 | {coverage:.0%} 목표 예측 범위 | 모델 | 입력한 실측 | 오차(예측−실측) |',
                 '|---|---:|---|---|---:|---:|']
        for target, result in predictions.items():
            actual = actuals[target]
            actual_text = '—' if actual is None else f'{actual:g} cm'
            error = '—' if actual is None else f"{result['value_cm'] - actual:+.1f} cm"
            lines.append(f"| {result['label']} | {result['value_cm']:.1f} cm | "
                         f"{result['low_cm']:.1f}~{result['high_cm']:.1f} cm | "
                         f"{MODEL_NAMES[result['model']]} | {actual_text} | {error} |")
        lines += ['', '신체 기준 총장 = 목뒤높이 − 위앞엉덩뼈가시높이. 옷의 총장과 구분해주세요.',
                  '가슴둘레는 몸의 둘레이며 옷 가슴단면이나 사진 정면의 폭이 아닙니다.',
                  '', '별도 자료로 보정한 성별·부위별 예측 범위입니다. 개인별 정확도나 세 항목의 '
                  '동시 포함을 보장하지 않으며, 실제 최종 평가 포함률은 아래 성능표에서 확인할 수 있습니다.']
        return '\n'.join(lines)
    except ValueError as exc:
        return f'예측을 보류했습니다. {exc}'


def clear_prediction():
    return '입력을 확인한 뒤 **신체 치수 예측**을 눌러주세요.'


def build_ui(model_path=None):
    import gradio as gr

    bundle = load_model(model_path)
    with gr.Blocks(title='FitCheck 엑셀 기반 신체 치수 예측') as demo:
        gr.Markdown('# 엑셀 기반 신체 치수 예측\n\n'
                    '성별·키·체중을 입력해 어깨사이너비, 신체 기준 총장, 가슴둘레를 예측합니다. '
                    '모델을 바꿔 결과를 비교할 수 있습니다.')
        with gr.Row():
            with gr.Column():
                sex = gr.Radio(['남', '여'], value=None, label='성별')
                height = gr.Number(label='키 (cm)', value=None)
                weight = gr.Number(label='체중 (kg)', value=None)
                model = gr.Dropdown(list(MODEL_CHOICES), value='Ridge', label='비교 모델')
                with gr.Accordion('실측과 오차 비교 (아는 값만 입력)', open=False):
                    shoulder = gr.Number(label='실제 신체 어깨사이너비 (cm)', value=None)
                    length = gr.Number(label='실제 신체 기준 총장 (cm, 옷 총장 아님)', value=None)
                    chest = gr.Number(label='실제 몸 가슴둘레 (cm)', value=None)
                run = gr.Button('신체 치수 예측', variant='primary')
            with gr.Column():
                output = gr.Markdown(clear_prediction())
        with gr.Accordion('모델 성능 비교와 학습 자료', open=False):
            gr.Markdown(report_markdown(bundle['report']))

        def on_predict(sex_value, height_value, weight_value, model_value, real_shoulder,
                       real_length, real_chest):
            return analyze(bundle, sex_value, height_value, weight_value, model_value,
                           real_shoulder, real_length, real_chest)

        inputs = [sex, height, weight, model, shoulder, length, chest]
        for component in inputs:
            component.change(clear_prediction, inputs=[], outputs=[output], queue=False)
        run.click(on_predict, inputs=inputs, outputs=[output], api_name='predict_body')
    return demo


def main():
    parser = argparse.ArgumentParser(description='엑셀 기반 신체 치수 예측 웹 디버거')
    parser.add_argument('--ui', action='store_true', help='웹 UI 실행 (생략해도 실행)')
    parser.add_argument('--model', default=None, help='모델 파일. 생략하면 가장 최근 학습 결과 사용')
    parser.add_argument('--port', type=int, default=7864)
    args = parser.parse_args()
    build_ui(args.model).launch(server_name='127.0.0.1', server_port=args.port, share=False)


if __name__ == '__main__':
    main()
