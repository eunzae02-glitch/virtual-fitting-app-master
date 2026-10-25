"""기준 옷 치수 연결 → 사진 기준점 확인 → 판매 사이즈 목표 윤곽 미리보기.

프로젝트 루트에서: python -X utf8 scripts/debug_size_preview.py --ui
"""
import argparse
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'app'))

from fit_plan import create_fit_plan  # noqa: E402
from garment_compare import (compare_reference, parse_target_chart,
                             report_markdown)  # noqa: E402
from garment_outline import (ANCHORS, REQUIRED, confirm_anchors, create_preview, new_marks,
                             photo_info, prepare_photo, preview_report, put_anchor,
                             render_outline)  # noqa: E402
from reference_garment import ReferenceGarment  # noqa: E402
from wear_compare import read_cases  # noqa: E402
from size_synthesis_ui import add_synthesis_ui  # noqa: E402


PROMPT = '기준 옷과 새 옷의 치수를 입력하고 **치수 연결**을 눌러주세요.'
PREVIEW_PROMPT = '사진의 기준점을 확인한 뒤 **목표 윤곽 보기**를 눌러주세요.'
CHART_SOURCES = ('상품 치수표 직접 입력', '가상 테스트 치수표')


def compare_dimensions(length, shoulder, chest, reference_style, target_style, table, chart_source):
    if chart_source not in CHART_SOURCES:
        raise ValueError('새 옷 치수 출처를 선택해주세요.')
    chart = parse_target_chart(table, target_style)
    result = compare_reference(chart, ReferenceGarment(length, shoulder, chest), reference_style)
    result['target_chart']['is_example'] = chart_source == CHART_SOURCES[1]
    if result['target_chart']['is_example']:
        result['target_chart']['source'] = 'debug_example_chart'
        result['notes'].append('새 옷 S/M/L 치수는 가상 테스트 값입니다. 실제 상품 치수로 교체해주세요.')
    return result


def mark_status(marks):
    if not marks:
        return '치수를 입력한 기준 옷을 입은 사진을 업로드해주세요.'
    done = sum(key in marks['points'] for key in REQUIRED)
    status = '확인 완료' if marks['confirmed'] else '위치 확인 필요'
    return f'필수 기준점 {done}/5 · {status}. 좌우는 사진을 보는 방향입니다.'


def load_sample(directory=None):
    case = read_cases(directory or ROOT / 'debug')['00_ME']
    with Image.open(case.image) as image:
        photo = prepare_photo(image)
    info = photo_info(photo, '00_ME')
    # 이 사진을 눈으로 보고 잡은 초안이다. 자동 분할 결과나 확인된 관측값이 아니다.
    marks = new_marks(info, {
        'neck': [.520, .207], 'chest_left': [.409, .389], 'chest_right': [.644, .389],
        'hem_left': [.414, .515], 'hem_right': [.640, .515],
    }, 'manual_visual_draft')
    return case, photo, info, marks


def build_preview(comparison, size, image, info, marks, same, length_ok, chest_ok, shoulders):
    if image is None or not info or not marks:
        raise ValueError('기준 옷 착용 사진을 먼저 불러와주세요.')
    if photo_info(image)['sha256'] != info['sha256']:
        raise ValueError('사진이 바뀌었습니다. 기준점을 다시 확인해주세요.')
    plan = create_fit_plan(comparison, size)
    return create_preview(plan, info, marks, same, length_ok, chest_ok, shoulders)


def save_preview(preview, image, output_directory=None):
    if not preview or preview.get('status') != 'preview_ready':
        raise ValueError('저장할 목표 윤곽을 먼저 만들어주세요.')
    if photo_info(image)['sha256'] != preview['photo']['sha256']:
        raise ValueError('미리보기와 사진이 다릅니다. 다시 계산해주세요.')
    destination = Path(output_directory) if output_directory else ROOT / 'outputs' / 'size_preview'
    destination = destination / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    destination.mkdir(parents=True, exist_ok=False)
    payload = dict(preview, artifacts={'reference_photo': 'reference.png', 'overlay': 'overlay.png',
                                     'comparison': 'comparison.json', 'report': 'report.md'})
    paths = []
    for name, data in [('fit_plan.json', payload), ('comparison.json', preview['fit_plan']['comparison'])]:
        path = destination / name
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        paths.append(str(path))
    photo = prepare_photo(image)
    photo.save(destination / 'reference.png')
    render_outline(photo, preview=preview).save(destination / 'overlay.png')
    (destination / 'report.md').write_text(
        '# 사이즈별 목표 윤곽 미리보기\n\n'
        + report_markdown(preview['fit_plan']['comparison']) + '\n\n' + preview_report(preview),
        encoding='utf-8')
    paths.extend(str(destination / name) for name in ('reference.png', 'overlay.png', 'report.md'))
    return paths, f'저장 완료: `{destination}`'


def build_ui():
    import gradio as gr

    with gr.Blocks(title='FitCheck 사이즈별 목표 윤곽') as demo:
        gr.Markdown('# 사이즈별 목표 윤곽 미리보기\n\n'
                    '기준 옷과 새 옷의 치수를 연결하고 사진 위에서 길이·몸통 폭 변화를 확인합니다. '
                    '**상품 치수비를 적용한 실험용 목표선**을 만든 뒤, 4단계에서 옷 영역을 조정해봅니다.')
        has_sample = (ROOT / 'debug').is_dir()
        sample = gr.Button('00_ME 사진·기준 옷 + 가상 S/M/L 불러오기', visible=has_sample)
        gr.Markdown('직접 올린 기준 옷 착용 사진과 치수로 진행할 수 있습니다. '
                    '00_ME 예시가 있는 환경에서만 예시 불러오기 버튼이 표시됩니다.')
        comparison = gr.State(None)
        info = gr.State(None)
        marks = gr.State(None)
        preview = gr.State(None)

        gr.Markdown('## 1. 기준 옷과 새 옷 치수 연결')
        with gr.Row():
            with gr.Column():
                length = gr.Number(label='기준 옷 총장 (cm)', value=None)
                shoulder = gr.Number(label='기준 옷 어깨너비 (cm)', value=None)
                chest = gr.Number(label='기준 옷 가슴단면 (cm)', value=None)
                reference_style = gr.State('확인 필요')
            with gr.Column(scale=2):
                chart_source = gr.Radio(CHART_SOURCES, value=CHART_SOURCES[0], label='새 옷 치수 출처')
                target_style = gr.State('확인 필요')
                counts = ({'row_count': 3, 'column_count': 4}
                          if 'column_count' in inspect.signature(gr.Dataframe).parameters
                          else {'row_count': (3, 'dynamic'), 'col_count': (4, 'fixed')})
                table = gr.Dataframe(headers=['판매 사이즈', '총장 (cm)', '어깨너비 (cm)', '가슴단면 (cm)'],
                                     datatype=['str', 'number', 'number', 'number'], type='array',
                                     value=[['S', None, None, None], ['M', None, None, None],
                                            ['L', None, None, None]],
                                     **counts, interactive=True, label='새 옷 치수표')
        compare = gr.Button('치수 연결', variant='primary')
        comparison_report = gr.Markdown(PROMPT)

        gr.Markdown('## 2. 기준 사진의 위치 확인\n\n'
                    '사진 속 옷의 목 부근 보조점, 소매를 제외한 몸통 양끝, 밑단 양끝 **5점**을 확인합니다. '
                    '수정할 점을 선택하고 오른쪽 사진을 클릭하세요. 목 보조점은 보이지 않는 목뒤점을 '
                    '실측한 위치가 아닙니다. 팔·다른 옷·배경이 경계와 겹치면 잘 보이는 사진을 사용해주세요.')
        with gr.Row():
            with gr.Column():
                source_photo = gr.Image(type='pil', format='png', label='기준 옷 착용 사진', sources=['upload'], height=410)
                same = gr.Checkbox(value=False, label='사진 속 옷과 위에 치수를 입력한 기준 옷이 같습니다')
                point = gr.Dropdown([(ANCHORS[key], key) for key in REQUIRED], value='neck',
                                    label='수정할 기준점')
                shoulders = gr.State(False)
                with gr.Row():
                    confirm = gr.Button('기준점 위치 확인', variant='primary')
                    reset = gr.Button('기준점 모두 지우기')
                status = gr.Markdown(mark_status(None))
            with gr.Column():
                editor = gr.Image(type='pil', format='png', interactive=False, label='점 확인·수정 (사진 클릭)', height=580)

        gr.Markdown('## 3. 판매 사이즈별 목표 윤곽\n\n'
                    '길이는 목 보조점부터 밑단까지의 거리, 몸통 폭은 사진 윤곽을 기준으로 '
                    '새 옷/기준 옷 치수비를 적용합니다. 하늘색은 기준선, 주황색은 목표선입니다.')
        selected = gr.Dropdown(choices=[], value=None, label='미리 볼 판매 사이즈')
        length_ok = gr.Checkbox(value=False, label='두 옷의 총장 측정 시작점·방향이 같아 길이를 비교할 수 있습니다')
        chest_ok = gr.Checkbox(value=False, label='두 옷의 가슴단면 측정 위치·구조가 비교 가능해 몸통 폭을 비교할 수 있습니다')
        show = gr.Button('목표 윤곽 보기', variant='primary')
        with gr.Row():
            overlay = gr.Image(type='pil', format='png', interactive=False, label='기준 윤곽 + 선택 사이즈 목표선', height=600)
            preview_text = gr.Markdown(PREVIEW_PROMPT)
        save = gr.Button('현재 치수·기준점·목표 윤곽 저장')
        downloads = gr.File(file_count='multiple', label='계획 JSON · 치수 비교 · 기준 사진 · 목표선 · 보고서')
        saved = gr.Markdown()

        preview_outputs = [preview, overlay, preview_text, downloads, saved]

        def clear_preview():
            return None, None, PREVIEW_PROMPT, [], ''

        def clear_dimensions():
            return None, PROMPT, gr.update(choices=[], value=None), *clear_preview(), False

        def connect(*values):
            try:
                result = compare_dimensions(*values)
                names = [row['size'] for row in result['target_chart']['sizes']]
                return result, report_markdown(result), gr.update(choices=names, value=names[0]), *clear_preview()
            except ValueError as exc:
                return None, f'치수 연결을 보류했습니다. {exc}', gr.update(choices=[], value=None), *clear_preview()

        def upload(image):
            if image is None:
                return None, None, None, mark_status(None), 'neck', False, *clear_preview()
            try:
                metadata = photo_info(image)
                annotation = new_marks(metadata)
                return metadata, annotation, render_outline(image, annotation), mark_status(annotation), 'neck', False, *clear_preview()
            except ValueError as exc:
                return None, None, None, str(exc), 'neck', False, *clear_preview()

        def sample_values():
            case, image, metadata, annotation = load_sample()
            return (case.length_cm, case.shoulder_cm, case.chest_flat_cm, '확인 필요', '확인 필요',
                    [['S', 63, 49, 50], ['M', 65, 51, 52], ['L', 67, 53, 54]], CHART_SOURCES[1],
                    image, metadata, annotation, render_outline(image, annotation), mark_status(annotation),
                    'neck', False, False, False, False, None, PROMPT,
                    gr.update(choices=[], value=None), *clear_preview())

        def click(image, metadata, annotation, key, use_shoulders, evt: gr.SelectData):
            try:
                edited = put_anchor(annotation, key, *evt.index, metadata)
                keys = tuple(ANCHORS) if use_shoulders else REQUIRED
                next_key = keys[min(keys.index(key)+1, len(keys)-1)] if key in keys else key
                return edited, render_outline(image, edited), next_key, mark_status(edited), *clear_preview()
            except (ValueError, TypeError, AttributeError) as exc:
                return annotation, render_outline(image, annotation) if image else None, key, str(exc), *clear_preview()

        def review(metadata, annotation, use_shoulders, ref_style, new_style):
            try:
                if not metadata or not annotation:
                    raise ValueError('사진을 먼저 불러와주세요.')
                checked = confirm_anchors(annotation, metadata,
                                          use_shoulders and ref_style == new_style == '일반 어깨선')
                return checked, mark_status(checked), *clear_preview()
            except ValueError as exc:
                return annotation, f'확인을 보류했습니다. {exc}', *clear_preview()

        def reset_points(image, metadata):
            annotation = new_marks(metadata) if metadata else None
            return (annotation, render_outline(image, annotation) if image else None, 'neck',
                    mark_status(annotation), *clear_preview())

        def display(*values):
            try:
                result = build_preview(*values)
                return result, render_outline(values[2], preview=result), preview_report(result), [], ''
            except ValueError as exc:
                return None, None, f'미리보기를 보류했습니다. {exc}', [], ''

        def export(*values):
            try:
                # 저장 클릭 시 현재 입력으로 재계산해 이전 사이즈/사진의 저장을 막는다.
                return save_preview(build_preview(*values), values[2])
            except (ValueError, OSError) as exc:
                return [], f'저장을 보류했습니다. {exc}'

        dimensions = [length, shoulder, chest, reference_style, target_style, table, chart_source]
        comparison_outputs = [comparison, comparison_report, selected, *preview_outputs]
        for control in dimensions:
            if not isinstance(control, gr.State):
                control.input(clear_dimensions, inputs=[], outputs=[*comparison_outputs, same], queue=False)
        compare.click(connect, inputs=dimensions, outputs=comparison_outputs, api_name='connect_dimensions', queue=False)
        source_photo.input(upload, inputs=[source_photo],
                           outputs=[info, marks, editor, status, point, same, *preview_outputs], queue=False)
        sample.click(sample_values, inputs=[], outputs=[*dimensions, source_photo, info, marks, editor,
                                                       status, point, same, length_ok, chest_ok, shoulders,
                                                       *comparison_outputs], queue=False)
        editor.select(click, inputs=[source_photo, info, marks, point, shoulders],
                      outputs=[marks, editor, point, status, *preview_outputs], queue=False)
        confirm.click(review, inputs=[info, marks, shoulders, reference_style, target_style],
                      outputs=[marks, status, *preview_outputs], api_name='confirm_anchors', queue=False)
        reset.click(reset_points, inputs=[source_photo, info],
                    outputs=[marks, editor, point, status, *preview_outputs], queue=False)
        preview_inputs = [comparison, selected, source_photo, info, marks, same, length_ok, chest_ok, shoulders]
        preview_events = [show.click(display, inputs=preview_inputs, outputs=preview_outputs,
                                     api_name='preview_outline', queue=False)]
        for control in (selected, same, length_ok, chest_ok):
            preview_events.append(control.input(display, inputs=preview_inputs,
                                                 outputs=preview_outputs, queue=False))
        save.click(export, inputs=preview_inputs, outputs=[downloads, saved], api_name='save_preview', queue=False)
        add_synthesis_ui(gr, preview, source_photo,
                         reset_controls=[length, shoulder, chest, table, chart_source],
                         reset_buttons=[compare, confirm, reset], point_editor=editor, sample=sample,
                         preview_events=preview_events,
                         busy_controls=[length, shoulder, chest, table, chart_source, selected,
                                        same, length_ok, chest_ok, compare, confirm, reset, show, save,
                                        editor, point, sample])
    return demo.queue(max_size=4, default_concurrency_limit=1)


def main():
    parser = argparse.ArgumentParser(description='기준 옷과 새 옷 사이즈별 목표 윤곽 미리보기')
    parser.add_argument('--ui', action='store_true', help='웹 UI 실행 (생략해도 실행)')
    parser.add_argument('--share', action='store_true', help='코랩용 공개 Gradio 링크 생성')
    parser.add_argument('--port', type=int, default=None, help='생략하면 비어 있는 포트 자동 선택')
    args = parser.parse_args()
    build_ui().launch(server_name='0.0.0.0' if args.share else '127.0.0.1',
                      server_port=args.port, share=args.share,
                      allowed_paths=[str(ROOT / 'outputs' / 'size_preview'),
                                     str(ROOT / 'outputs' / 'size_synthesis')])


if __name__ == '__main__':
    main()
