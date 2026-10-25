"""선택적 실제 브라우저 검사. FASHN만 모의 엔진으로 바꾸며 GPU/개인 사진은 사용하지 않는다.

python -X utf8 tests/browser_size_synthesis_smoke.py
Playwright와 설치된 Chrome(또는 PLAYWRIGHT_CHROMIUM_EXECUTABLE 환경변수)이 필요하다.
"""
import json
import os
from pathlib import Path
import sys
from unittest.mock import Mock, patch

import gradio as gr
import numpy as np
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
for folder in ('app', 'scripts', 'tests'):
    sys.path.insert(0, str(ROOT / folder))

from debug_size_preview import CHART_SOURCES, build_ui  # noqa: E402
from size_synthesis import editor_mask, generate_base  # noqa: E402
from test_size_synthesis import TestSizeSynthesis  # noqa: E402


def main():
    case = TestSizeSynthesis()
    case.setUp()
    demo = build_ui()
    handlers = {event.fn.__name__: event for event in demo.fns.values() if event.fn}
    display, connect, upload = (handlers[name] for name in ('display', 'connect', 'upload'))
    refresh, runner = handlers['refresh'], handlers['generate_with_status']
    selected, person, preview = display.inputs[1], display.inputs[2], display.outputs[0]
    generate = next(block for block in demo.blocks.values()
                    if isinstance(block, gr.Button) and block.value == 'FASHN 기본 합성 생성')
    report, adjusted, gallery = refresh.outputs[3], refresh.outputs[0], refresh.outputs[2]
    approved, garment = refresh.inputs[5], runner.inputs[2]
    for block, elem_id in [(selected, 'smoke_size'), (generate, 'smoke_generate'),
                           (approved, 'smoke_approved'), (report, 'smoke_report'),
                           (adjusted, 'smoke_result'), (gallery, 'smoke_gallery'),
                           (runner.outputs[5], 'smoke_base_status'),
                           (refresh.inputs[4], 'smoke_editor')]:
        block.elem_id = elem_id
    style = case.comparison['source']['shoulder_style']
    rows = [['S', 55, 49, 46], ['M', 65, 51, 52], ['L', 75, 53, 58]]
    initial = [65, 51, 52, style, style, rows, CHART_SOURCES[1], case.comparison,
               case.person, case.photo, case.marks, case.preview(),
               gr.update(choices=['S', 'M', 'L'], value='M'), True, True, True, case.person]
    load_outputs = [*connect.inputs, connect.outputs[0], person, upload.outputs[0],
                    upload.outputs[1], preview, selected, *display.inputs[5:8], garment]
    with demo:
        demo.load(lambda: tuple(initial), inputs=[], outputs=load_outputs)
    engine = Mock(return_value=(case.base.copy(), {'seed': 42, 'total_s': .01}))
    observed_masks = []

    def read_mask(value, base):
        mask = editor_mask(value, base)
        observed_masks.append(np.asarray(mask).copy())
        return mask

    def fake_generate(photo, clothes, settings):
        return generate_base(photo, clothes, settings, engine)

    candidates = [os.environ.get('PLAYWRIGHT_CHROMIUM_EXECUTABLE'),
                  r'C:\Program Files\Google\Chrome\Application\chrome.exe',
                  r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe']
    executable = next((path for path in candidates if path and Path(path).is_file()), None)
    browser = None
    try:
        with patch('size_synthesis_ui.generate_base', side_effect=fake_generate), patch(
                'size_synthesis_ui.draft_mask', return_value=case.mask), patch(
                'size_synthesis_ui.editor_mask', side_effect=read_mask), sync_playwright() as p:
            _, url, _ = demo.launch(server_name='127.0.0.1', server_port=None,
                                    prevent_thread_lock=True, quiet=True)
            print('Server ready:', url, flush=True)
            browser = p.chromium.launch(executable_path=executable, headless=True)
            page = browser.new_page(viewport={'width': 1440, 'height': 1200})
            errors, requests, failed_responses = [], [], []
            page.on('pageerror', lambda error: errors.append(error.stack))
            page.on('response', lambda response: failed_responses.append(response.url)
                    if response.url.startswith(url) and response.status >= 400 else None)

            def request_log(request):
                if '/queue/join' in request.url:
                    requests.append(request.post_data_json.get('fn_index'))

            page.on('request', request_log)
            try:
                # Gradio의 SSE 연결은 열린 채 유지되므로 networkidle은 기다리지 않는다.
                page.goto(url, wait_until='domcontentloaded', timeout=30000)
                expect(page.locator('#smoke_size input')).to_have_value('M', timeout=30000)
                page.locator('#smoke_generate').click()
                expect(page.locator('#smoke_base_status')).to_contain_text(
                    'FASHN 기본 합성을 생성했습니다', timeout=30000)
                print('Base created, engine calls:', engine.call_count, flush=True)
                page.locator('#smoke_approved input').check(timeout=15000)
                expect(page.locator('#smoke_report')).to_contain_text('선택 사이즈 M', timeout=30000)
                assert np.array_equal(observed_masks[-1], np.asarray(case.mask))
                editor = page.locator('#smoke_editor')
                editor.scroll_into_view_if_needed()
                canvas = editor.locator('canvas').first
                expect(canvas).to_be_visible(timeout=15000)
                # 초기 초안 위를 직접 지웠다 칠한다. 확인 체크만 하는 시험으로는 이 오류를 못 잡는다.
                counts = [int(np.count_nonzero(observed_masks[-1]))]
                for tool in ('지우개', '브러시', '지우개'):
                    editor.get_by_role('button', name=tool, exact=True).click()
                    bounds = canvas.bounding_box()
                    x, y = bounds['x']+bounds['width']*.5, bounds['y']+bounds['height']*.375
                    page.mouse.move(x, y)
                    page.mouse.down()
                    page.mouse.move(x+6, y+3, steps=6)
                    page.mouse.up()
                    expect(page.locator('#smoke_approved input')).not_to_be_checked(timeout=15000)
                    expect(page.locator('#smoke_report')).not_to_contain_text('선택 사이즈 M', timeout=15000)
                    page.locator('#smoke_approved input').check()
                    expect(page.locator('#smoke_report')).to_contain_text('선택 사이즈 M', timeout=15000)
                    counts.append(int(np.count_nonzero(observed_masks[-1])))
                assert 0 < counts[1] < counts[0], counts
                assert counts[2] > counts[1], counts
                assert counts[3] < counts[2], counts
                editor.get_by_role('button', name='초안으로 되돌리기', exact=True).click()
                expect(page.locator('#smoke_approved input')).not_to_be_checked(timeout=15000)
                page.locator('#smoke_approved input').check()
                expect(page.locator('#smoke_report')).to_contain_text('선택 사이즈 M', timeout=15000)
                assert np.array_equal(observed_masks[-1], np.asarray(case.mask))
                sources = []
                for size in ('S', 'M', 'L'):
                    page.locator('#smoke_size input').click()
                    page.get_by_role('option', name=size, exact=True).click()
                    expect(page.locator('#smoke_report')).to_contain_text('선택 사이즈 '+size, timeout=30000)
                    image = page.locator('#smoke_result img').first
                    expect(image).to_be_visible(timeout=30000)
                    sources.append(image.get_attribute('src'))
                    print('Size applied:', size, flush=True)
                assert len(set(sources)) == 3, sources
                assert engine.call_count == 1, engine.call_count
                assert not errors, errors
                assert not failed_responses, failed_responses
                print(json.dumps({'base_generations': engine.call_count,
                                  'mask_pixels_initial_erased_painted': counts[:3],
                                  'draft_reset_verified': True,
                                  'distinct_size_images': len(set(sources)), 'browser_errors': errors,
                                  'failed_responses': failed_responses}), flush=True)
            except Exception:
                print('Engine calls:', engine.call_count, 'Browser errors:', errors, flush=True)
                print('Observed mask pixels:', [int(np.count_nonzero(mask)) for mask in observed_masks], flush=True)
                print('Queued handlers:', [demo.fns[index].fn.__name__ for index in requests], flush=True)
                print('Active jobs:', repr(demo._queue.active_jobs), flush=True)
                print('Statuses:', page.locator('[data-testid="status-tracker"]').all_text_contents(), flush=True)
                print('Report:', page.locator('#smoke_report').inner_text(), flush=True)
                raise
            finally:
                browser.close()
    finally:
        demo.close()


if __name__ == '__main__':
    main()
