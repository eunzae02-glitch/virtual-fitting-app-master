"""새 옷 판매 사이즈를 기준 옷 또는 Ridge 신체 예측과 비교하는 웹 디버거.

프로젝트 루트에서: python -X utf8 scripts/debug_size_compare.py --ui
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'app'))

from body_model import (latest_model_path, load_model, normalize_sex, positive_number,
                        predict_body)  # noqa: E402
from garment_compare import (SHOULDER_STYLES, compare_body, compare_reference,
                             parse_target_chart, report_markdown)  # noqa: E402
from reference_garment import ReferenceGarment  # noqa: E402


MODES = ('기준 옷과 비교', '신체 예측과 비교 (기준 옷 없음)')
PROMPT = '비교 기준과 새 옷 치수표를 입력한 뒤 **치수 비교**를 눌러주세요.'


def compare_inputs(mode, length, shoulder, chest_flat, reference_style,
                   sex, height, weight, target_style, table, model_path=None):
    """한 번의 클릭으로 입력을 검증하고 현재 입력에 대한 결과만 반환한다."""
    try:
        if mode not in MODES:
            raise ValueError('비교 기준을 선택해주세요.')
        chart = parse_target_chart(table, target_style)
        if mode == MODES[0]:
            reference = ReferenceGarment(length, shoulder, chest_flat)
            result = compare_reference(chart, reference, reference_style)
        else:
            path = Path(model_path) if model_path else latest_model_path()
            bundle = load_model(path)
            predictions = predict_body(bundle, sex, height, weight, model='ridge')
            result = compare_body(chart, predictions, bundle['report']['coverage_target'])
            result['source'].update({
                'model_path': str(path.resolve()),
                'inputs': {'sex': normalize_sex(sex), 'height_cm': positive_number(height, '키(cm)'),
                           'weight_kg': positive_number(weight, '체중(kg)')},
            })
        return result, report_markdown(result)
    except (ValueError, FileNotFoundError) as exc:
        return None, f'비교를 보류했습니다. {exc}'


def clear_result():
    return None, PROMPT, [], ''


def save_result(result, output_directory=None):
    if not result:
        return [], '현재 입력으로 치수 비교를 먼저 실행해주세요.'
    base = Path(output_directory) if output_directory else ROOT / 'outputs' / 'size_compare'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    destination = base / stamp
    destination.mkdir(parents=True, exist_ok=False)
    data_path, report_path = destination / 'comparison.json', destination / 'report.md'
    data_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    report_path.write_text(report_markdown(result), encoding='utf-8')
    return [str(data_path), str(report_path)], f'비교 결과를 저장했습니다: `{destination}`'


def build_ui(model_path=None):
    import gradio as gr

    with gr.Blocks(title='FitCheck 새 옷 사이즈별 치수 비교') as demo:
        gr.Markdown('# 새 옷 사이즈별 치수 비교\n\n'
                    '잘 맞는 기준 옷이 있으면 옷끼리 비교합니다. 기준 옷이 없으면 '
                    '성별·키·체중으로 신체 치수를 예측해 새 옷과 비교합니다. 모든 옷 치수는 **cm**입니다.')
        result = gr.State(None)
        mode = gr.Radio(MODES, value=MODES[0], label='비교 기준')
        with gr.Row():
            with gr.Column():
                with gr.Group(visible=True) as reference_group:
                    gr.Markdown('### 평소 잘 맞는 기준 옷')
                    length = gr.Number(label='기준 옷 총장 (cm)', value=None)
                    shoulder = gr.Number(label='기준 옷 어깨너비 (cm)', value=None)
                    chest_flat = gr.Number(label='기준 옷 가슴단면 (cm)', value=None)
                    reference_style = gr.Radio(SHOULDER_STYLES, value='확인 필요', label='기준 옷 어깨 구조')
                with gr.Group(visible=False) as body_group:
                    gr.Markdown('### 착용자 정보 · Ridge 예측')
                    sex = gr.Radio(['남', '여'], value=None, label='사용자가 선택한 성별')
                    height = gr.Number(label='키 (cm)', value=None)
                    weight = gr.Number(label='체중 (kg)', value=None)
            with gr.Column(scale=2):
                gr.Markdown('### 새로 구매할 옷의 판매 사이즈별 치수\n\n'
                            '가슴은 **옷 가슴단면**을 입력해주세요. 총장은 뒤목 부근 옷 상단부터 밑단까지의 '
                            '치수인지 확인해주세요. 사이즈 이름은 S/M/L, 95/100 등 실제 표기를 사용합니다.')
                target_style = gr.Radio(SHOULDER_STYLES, value='확인 필요', label='새 옷 어깨 구조')
                table = gr.Dataframe(headers=['판매 사이즈', '총장 (cm)', '어깨너비 (cm)', '가슴단면 (cm)'],
                                     datatype=['str', 'number', 'number', 'number'],
                                     value=[['S', None, None, None], ['M', None, None, None],
                                            ['L', None, None, None]],
                                     type='array', row_count=(3, 'dynamic'), col_count=(4, 'fixed'),
                                     interactive=True, label='새 옷 치수표')
                gr.Markdown('필요한 행만 남기거나 행을 추가할 수 있습니다. 일반 어깨선은 어깨너비까지 '
                            '입력하고, 래글런 등 어깨 치수가 없으면 해당 칸을 비워두세요.')
                example = gr.Button('가상 예시 채우기')
        compare = gr.Button('치수 비교', variant='primary')
        report = gr.Markdown(PROMPT)
        with gr.Accordion('비교 결과 저장', open=False):
            save = gr.Button('현재 결과 저장')
            downloads = gr.File(label='치수 비교 JSON · 보고서', file_count='multiple')
            saved = gr.Markdown()

        def change_mode(value):
            return (gr.update(visible=value == MODES[0]), gr.update(visible=value == MODES[1]),
                    *clear_result())

        def compare_now(*values):
            data, text = compare_inputs(*values, model_path=model_path)
            return data, text, [], ''

        def fill_example():
            # 실제 상품 치수와 구별되는 수동 디버깅용 가상 표다.
            return (70, 48, 54, '일반 어깨선', '남', 173, 63, '일반 어깨선',
                    [['S', 68, 46, 52], ['M', 70, 48, 54], ['L', 72, 50, 56]], *clear_result())

        controls = [mode, length, shoulder, chest_flat, reference_style,
                    sex, height, weight, target_style, table]
        result_outputs = [result, report, downloads, saved]
        mode.input(change_mode, inputs=[mode], outputs=[reference_group, body_group, *result_outputs],
                   queue=False)
        for component in controls[1:]:
            component.input(clear_result, inputs=[], outputs=result_outputs, queue=False)
        example.click(fill_example, inputs=[],
                      outputs=[length, shoulder, chest_flat, reference_style, sex, height, weight,
                               target_style, table,
                               *result_outputs], queue=False)
        compare.click(compare_now, inputs=controls, outputs=result_outputs, api_name='compare_sizes',
                      concurrency_limit=1)
        save.click(save_result, inputs=[result], outputs=[downloads, saved], api_name='save_comparison')
    return demo


def main():
    parser = argparse.ArgumentParser(description='기준 옷 또는 신체 예측과 새 옷 사이즈별 치수 비교')
    parser.add_argument('--ui', action='store_true', help='웹 UI 실행 (생략해도 실행)')
    parser.add_argument('--model', default=None, help='신체 예측 모델 파일. 생략하면 가장 최근 모델')
    parser.add_argument('--port', type=int, default=7866)
    args = parser.parse_args()
    build_ui(args.model).launch(server_name='127.0.0.1', server_port=args.port, share=False)


if __name__ == '__main__':
    main()
