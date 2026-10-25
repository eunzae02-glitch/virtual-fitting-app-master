"""기준 상의 치수 직접 입력을 확인하는 1단계 웹 디버거.

프로젝트 루트에서: python -X utf8 scripts/debug_reference_size.py --ui
"""
import argparse
import os
import sys

APP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app')
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from reference_garment import ReferenceGarment  # noqa: E402


def confirm_reference(length, shoulder, chest_flat):
    """확인된 옷 데이터와 표시 문구를 반환한다. 오류 시 이전 확인값도 비운다."""
    try:
        garment = ReferenceGarment(length, shoulder, chest_flat)
    except ValueError as exc:
        return None, f'입력을 확인해주세요: {exc}'

    report = '\n'.join([
        '### 기준 옷 치수 확인', '',
        '| 항목 | 옷 실측값 |',
        '|---|---:|',
        f'| 총장 | {garment.length_cm:g} cm |',
        f'| 어깨너비 | {garment.shoulder_cm:g} cm |',
        f'| 가슴단면 | {garment.chest_flat_cm:g} cm |',
        '', '이 옷을 새 옷과 비교할 기준으로 사용합니다. 가슴단면은 옷을 펼쳐 잰 폭입니다.',
        '', '확인값은 현재 화면에서 유지됩니다. 새로고침하면 다시 입력해주세요.',
    ])
    return garment.to_dict(), report


def clear_confirmation():
    return None, '치수를 입력하거나 수정한 뒤 **입력 확인**을 눌러주세요.'


def build_ui():
    import gradio as gr

    with gr.Blocks(title='FitCheck 기준 옷 치수 입력') as demo:
        gr.Markdown('# 기준 옷 치수 입력\n\n'
                    '평소 잘 맞는 상의의 치수표 또는 직접 잰 값을 **cm** 단위로 입력해주세요. '
                    '몸 치수가 아닌 **옷의 실측값**을 입력합니다.')
        confirmed_reference = gr.State(value=None)
        with gr.Row():
            with gr.Column():
                length = gr.Number(label='총장 (cm)', value=None,
                                   info='옷의 치수표에 적힌 총장. 예: 70')
                shoulder = gr.Number(label='어깨너비 (cm)', value=None,
                                     info='옷의 치수표에 적힌 어깨너비. 예: 48')
                chest_flat = gr.Number(label='가슴단면 (cm)', value=None,
                                       info='옷을 펼쳤을 때 좌우 겨드랑이 사이의 폭. 예: 54')
                confirm = gr.Button('입력 확인', variant='primary')
            with gr.Column():
                report = gr.Markdown(clear_confirmation()[1])

        for component in (length, shoulder, chest_flat):
            component.change(clear_confirmation, inputs=[],
                             outputs=[confirmed_reference, report], queue=False)
        confirm.click(confirm_reference, inputs=[length, shoulder, chest_flat],
                      outputs=[confirmed_reference, report])
    return demo


def main():
    parser = argparse.ArgumentParser(description='기준 상의 치수 직접 입력 웹 디버거')
    parser.add_argument('--ui', action='store_true', help='웹 UI 실행 (생략해도 실행)')
    parser.add_argument('--port', type=int, default=7863, help='로컬 포트 (기본 7863)')
    args = parser.parse_args()
    build_ui().launch(server_name='127.0.0.1', server_port=args.port, share=False)


if __name__ == '__main__':
    main()
