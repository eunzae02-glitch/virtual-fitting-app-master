"""알려진 옷 치수와 실착 사진을 비교하는 GPU 없는 웹 디버거.

    python -X utf8 scripts/debug_wear_compare.py --ui
    python -X utf8 scripts/debug_wear_compare.py --report
"""
import argparse
import copy
import hashlib
import html
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.fspath(ROOT / 'app'))

from body_model import latest_model_path, load_model  # noqa: E402
from wear_compare import (DEFAULT_DEBUG, POINTS, STRUCTURES, confirm_marks, initial_marks,
                          numeric_report, observation_report, open_photo, put_point,
                          read_cases, render_comparison, required_points)  # noqa: E402


def mark_status(marks):
    present = sum(key in marks['points'] for key in required_points(marks))
    status = '표시 확인됨' if marks['confirmed'] else '위치 확인 필요'
    return f"**{status}** · {present}/{len(required_points(marks))}개 점 · {marks['structure']}"


def export_comparison(cases, photos, bundle, model_path, a_name, b_name, a_marks, b_marks,
                      same_person, output_root=None):
    if a_name == b_name:
        raise ValueError('서로 다른 두 사례를 선택해주세요.')
    if a_marks['case'] != a_name or b_marks['case'] != b_name:
        raise ValueError('선택한 사진과 표시 좌표가 다릅니다. 사진을 다시 불러와주세요.')
    case_a, case_b = cases[a_name], cases[b_name]
    numeric = numeric_report(bundle, case_a, case_b)
    observations = observation_report(case_a, case_b, a_marks, b_marks,
                                      photos[a_name].size, photos[b_name].size, same_person)
    visual = render_comparison(case_a, case_b, a_marks, b_marks, photos[a_name],
                               photos[b_name], same_person)
    output_root = Path(output_root) if output_root else ROOT / 'outputs' / 'wear_compare'
    output = output_root / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output.mkdir(parents=True, exist_ok=False)
    snapshot = {'schema_version': 1, 'model_path': str(model_path), 'model': 'ridge',
                'same_person_confirmed': same_person,
                'length_definition': '목뒤높이 - 위앞엉덩뼈가시높이',
                'garment_length_definition': '사용자 설명: 뒤목 부근 옷 상단부터 밑단',
                'comparison_units': 'cm 수치 비교 / 사진 눈 사이거리 정렬 상대비',
                'annotations': {}, 'cases': {}}
    for case, marks in [(case_a, a_marks), (case_b, b_marks)]:
        data = asdict(case)
        data['image'] = str(case.image)
        data['image_sha256'] = hashlib.sha256(case.image.read_bytes()).hexdigest()
        data['display_size_after_exif'] = photos[case.name].size
        snapshot['cases'][case.name] = data
        snapshot['annotations'][case.name] = marks
    (output / 'annotations.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2),
                                            encoding='utf-8')
    (output / 'report.md').write_text('# 실착 치수 비교 실험\n\n' + numeric + '\n\n'
                                    + observations + '\n\n시각 자료: comparison.html\n', encoding='utf-8')
    (output / 'comparison.html').write_text(
        '<!doctype html><html lang="ko"><meta charset="utf-8"><title>FitCheck 실착 비교</title>'
        '<body style="font-family:system-ui;max-width:1400px;margin:32px auto;padding:20px">'
        '<h1>FitCheck 실착 치수 비교</h1>' + visual + '<pre style="white-space:pre-wrap">'
        + html.escape(numeric + '\n\n' + observations) + '</pre></body></html>', encoding='utf-8')
    return output


def build_ui(directory=DEFAULT_DEBUG, model_path=None):
    import gradio as gr

    cases = read_cases(directory)
    photos = {key: open_photo(case) for key, case in cases.items()}
    model_path = Path(model_path) if model_path else latest_model_path()
    bundle = load_model(model_path)
    names = list(cases)
    default_a, default_b = names[:2]

    def refresh(a, b, marks_a, marks_b, same):
        if marks_a['case'] != a or marks_b['case'] != b:
            return '', '', '사진 정보를 불러오는 중입니다.'
        if a == b:
            return '', '', '서로 다른 두 사례를 선택해주세요.'
        try:
            return (numeric_report(bundle, cases[a], cases[b]),
                    render_comparison(cases[a], cases[b], marks_a, marks_b, photos[a], photos[b], same),
                    observation_report(cases[a], cases[b], marks_a, marks_b,
                                       photos[a].size, photos[b].size, same))
        except ValueError as exc:
            return numeric_report(bundle, cases[a], cases[b]), '', f'비교를 보류했습니다. {exc}'

    def load_case(name):
        marks = initial_marks(cases[name])
        return photos[name], marks, marks['structure'], mark_status(marks), next(iter(POINTS))

    def click_point(name, marks, key, evt: gr.SelectData):
        if marks['case'] != name:
            return marks, key, '사진을 다시 선택해주세요.'
        try:
            updated = put_point(marks, key, float(evt.index[0]), float(evt.index[1]), photos[name].size)
            keys = required_points(updated)
            next_key = keys[(keys.index(key) + 1) % len(keys)] if key in keys else keys[0]
            return updated, next_key, mark_status(updated)
        except (ValueError, TypeError, IndexError) as exc:
            return marks, key, f'표시 위치를 확인해주세요: {exc}'

    def review(name, marks):
        try:
            updated = confirm_marks(marks, photos[name].size)
            return updated, mark_status(updated)
        except ValueError as exc:
            return marks, f'**확인 필요:** {exc}'

    def change_structure(marks, value):
        updated = copy.deepcopy(marks)
        updated['structure'] = value
        updated['confirmed'] = False
        return updated, mark_status(updated)

    def clear_marks(name):
        marks = initial_marks(cases[name])
        marks['points'] = {}
        marks['source'] = 'empty'
        return marks, mark_status(marks), next(iter(POINTS))

    def export(a, b, marks_a, marks_b, same):
        try:
            if a == b:
                raise ValueError('서로 다른 두 사례를 선택해주세요.')
            output = export_comparison(cases, photos, bundle, model_path, a, b, marks_a, marks_b, same)
            return str(output / 'annotations.json'), f'저장 완료: `{output.relative_to(ROOT)}`'
        except ValueError as exc:
            return None, f'저장 보류: {exc}'

    with gr.Blocks(title='FitCheck 실착 치수 비교') as demo:
        gr.Markdown('# 실착 사진과 옷 치수 비교\n\n'
                    '뒤목 기준 신체 총장과 상품 총장의 차이를 확인하고, 사진에 표시한 어깨선·몸통·밑단을 비교합니다. '
                    '**현재 00·01의 점은 수동으로 잡은 초안입니다. 위치를 확인·수정해주세요.**')
        numeric = gr.Markdown(numeric_report(bundle, cases[default_a], cases[default_b]))
        gr.Markdown('### 사진에 기준점 표시\n\n'
                    '표시할 점을 고르고 **사진을 클릭**하면 다음 점으로 넘어갑니다. '
                    '숫자가 붙은 위치는 아래 미리보기에서 확인하세요. '
                    '3번은 앞목 중앙의 파인 곳이 아닌 **목 양옆 옷 상단 높이의 중앙 보조점**입니다. '
                    '가려진 목뒤점이나 골반점의 실제 위치를 측정한 것으로 취급하지 않습니다.')
        with gr.Row():
            columns = []
            for title, default in [('기준 A', default_a), ('관측 B', default_b)]:
                with gr.Column():
                    name = gr.Dropdown(names, value=default, label=title + ' 사진')
                    marks = gr.State(initial_marks(cases[default]))
                    structure = gr.Radio(STRUCTURES, value=initial_marks(cases[default])['structure'],
                                         label='옷의 어깨 구조')
                    photo = gr.Image(value=photos[default], type='pil', interactive=False,
                                     height=430, label='점 위치를 클릭할 사진', buttons=['fullscreen'])
                    point = gr.Dropdown([(value, key) for key, value in POINTS.items()],
                                        value=next(iter(POINTS)), label='표시할 점')
                    status = gr.Markdown(mark_status(initial_marks(cases[default])))
                    with gr.Row():
                        confirm = gr.Button('표시 확인', variant='primary')
                        reset = gr.Button('점 모두 지우기')
                    columns.append((name, marks, structure, photo, point, status, confirm, reset))
        same = gr.Checkbox(value=False, label='두 사진은 같은 착용자의 사진입니다')
        visual = gr.HTML(render_comparison(cases[default_a], cases[default_b],
                                          initial_marks(cases[default_a]), initial_marks(cases[default_b]),
                                          photos[default_a], photos[default_b], False))
        observed = gr.Markdown('두 사진의 점을 확인하고 같은 착용자인지 선택해주세요.')
        with gr.Accordion('표시와 비교 결과 보관', open=False):
            gr.Markdown('현재 표시와 확인 상태, 상품 치수, 보고서, 사진을 포함한 HTML을 새 실행 폴더에 저장합니다. '
                        '원본 TXT·사진과 학습 모델은 수정하지 않습니다.')
            save = gr.Button('현재 비교 저장')
            download = gr.File(label='표시 좌표 JSON')
            saved = gr.Markdown()

        a, ma = columns[0][:2]
        b, mb = columns[1][:2]
        inputs = [a, b, ma, mb, same]
        outputs = [numeric, visual, observed]
        for name, marks, structure, photo, point, status, confirm, reset in columns:
            name.change(load_case, inputs=[name], outputs=[photo, marks, structure, status, point],
                        queue=False).then(refresh, inputs=inputs, outputs=outputs, queue=False)
            photo.select(click_point, inputs=[name, marks, point], outputs=[marks, point, status],
                         queue=False).then(refresh, inputs=inputs, outputs=outputs, queue=False)
            confirm.click(review, inputs=[name, marks], outputs=[marks, status],
                          queue=False).then(refresh, inputs=inputs, outputs=outputs, queue=False)
            reset.click(clear_marks, inputs=[name], outputs=[marks, status, point],
                        queue=False).then(refresh, inputs=inputs, outputs=outputs, queue=False)
            structure.input(change_structure, inputs=[marks, structure], outputs=[marks, status],
                            queue=False).then(refresh, inputs=inputs, outputs=outputs, queue=False)
        same.change(refresh, inputs=inputs, outputs=outputs, queue=False)
        save.click(export, inputs=inputs, outputs=[download, saved], api_name='export_comparison')
    return demo


def main():
    parser = argparse.ArgumentParser(description='알려진 옷의 실착 비교 웹 디버거')
    parser.add_argument('--ui', action='store_true')
    parser.add_argument('--report', action='store_true', help='미확인 초안 보고서만 저장')
    parser.add_argument('--directory', default=str(DEFAULT_DEBUG))
    parser.add_argument('--model', default=None)
    parser.add_argument('--port', type=int, default=7865)
    args = parser.parse_args()
    if args.report:
        cases = read_cases(args.directory)
        names = list(cases)[:2]
        photos = {name: open_photo(case) for name, case in cases.items()}
        model_path = Path(args.model) if args.model else latest_model_path()
        output = export_comparison(cases, photos, load_model(model_path), model_path, *names,
                                   initial_marks(cases[names[0]]), initial_marks(cases[names[1]]), False)
        print(f'미확인 초안 저장: {output}')
        return
    build_ui(args.directory, args.model).launch(server_name='127.0.0.1', server_port=args.port, share=False)


if __name__ == '__main__':
    main()
