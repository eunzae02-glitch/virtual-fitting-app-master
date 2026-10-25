"""전신 사진·기준 옷·새 옷 사이즈표와 DM-VTON 합성만 표시하는 화면."""
import argparse
import inspect
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'app'))

from dmvton_fit import compare_inputs, make_base, render_sizes, source_key  # noqa: E402


PROMPT = '사진과 치수를 입력한 뒤 **사이즈별 합성 비교**를 눌러주세요.'


def run_comparison(person, length, shoulder, chest, rows, garment, cache=None, progress=None):
    started = time.perf_counter()
    try:
        reference, chart = compare_inputs(length, shoulder, chest, rows)
        key = source_key(person, garment)
        cached = isinstance(cache, dict) and cache.get('source_key') == key
        if not cached:
            cache = None
            yield cache, [], '옷 영역과 상반신 위치를 자동으로 찾고 DM-VTON 기본 합성을 준비하고 있습니다.'
            if progress is not None:
                progress(.05, desc='옷 영역 추출 · DM-VTON 기본 합성')
            cache = make_base(person, garment)
            cache['source_key'] = key
        else:
            yield cache, [], '기본 합성을 재사용해 입력한 사이즈별 길이와 폭을 적용하고 있습니다.'
        if progress is not None:
            progress(.8, desc='사이즈별 길이 · 몸통 폭 비교')
        gallery, reports, errors = render_sizes(cache, reference, chart)
        cache['size_reports'] = reports
        metadata = cache['metadata']
        status = (f'완료 · 이번 처리 {time.perf_counter()-started:.2f}초 · '
                  + ('기본 합성 재사용' if cached else f'DM-VTON 추론 {metadata["inference_s"]:.3f}초')
                  + '\n\n자동으로 추출한 옷 영역에 총장·가슴단면 비율을 적용한 비교용 결과입니다. '
                    '어깨 치수는 보관하며 어깨 변형은 적용하지 않습니다.')
        if errors:
            status += '\n\n변형 보류: ' + ' / '.join(errors)
        if progress is not None:
            progress(1, desc='비교 완료')
        yield cache, gallery, status
    except Exception as exc:
        if not isinstance(exc, ValueError):
            traceback.print_exc()
        yield None, [], f'합성을 진행하지 못했습니다: {exc}'


def build_ui():
    import gradio as gr

    with gr.Blocks(title='FitCheck 빠른 사이즈 비교 · DM-VTON') as demo:
        gr.Markdown('# 빠른 사이즈 합성 비교')
        cache = gr.State(None)
        with gr.Row():
            with gr.Column():
                gr.Markdown('### 1. 기존 전신 사진')
                person = gr.Image(label='아래 치수를 입력할 기존 옷을 입은 사진',
                                  type='pil', image_mode='RGB', format='png', height=360,
                                  sources=['upload'])
            with gr.Column():
                gr.Markdown('### 2. 기존 옷 사이즈 (cm)')
                length = gr.Number(label='총장', value=None)
                shoulder = gr.Number(label='어깨너비', value=None)
                chest = gr.Number(label='가슴단면', value=None)
                gr.Markdown('### 3. 새 옷 사이즈별 입력 (cm)')
                columns = ({'column_count': 4, 'row_count': 3}
                           if 'column_count' in inspect.signature(gr.Dataframe).parameters
                           else {'col_count': (4, 'fixed'), 'row_count': (3, 'dynamic')})
                table = gr.Dataframe(headers=['사이즈', '총장', '어깨너비', '가슴단면'],
                                     datatype=['str', 'number', 'number', 'number'],
                                     value=[['S', None, None, None], ['M', None, None, None],
                                            ['L', None, None, None]], type='array',
                                     label='상품 치수표', **columns)
        gr.Markdown('### 4. DM-VTON 합성')
        with gr.Row():
            garment = gr.Image(label='새 옷 상품 사진 · 옷 한 벌만 있는 사진 또는 투명 PNG',
                               type='pil', image_mode='RGBA', format='png', height=320,
                               sources=['upload'])
            gallery = gr.Gallery(label='기본 합성과 사이즈별 비교', columns=3, height=420,
                                 format='png', type='pil', interactive=False)
        generate = gr.Button('사이즈별 합성 비교', variant='primary')
        status = gr.Markdown(PROMPT)
        controls = [person, length, shoulder, chest, table, garment, generate]

        def run(photo, length_value, shoulder_value, chest_value, rows, clothes, previous,
                progress=gr.Progress(track_tqdm=True)):
            latest = previous, [], PROMPT
            for state, images, text in run_comparison(photo, length_value, shoulder_value,
                                                     chest_value, rows, clothes, previous, progress):
                latest = state, images, text
                yield state, images, text, *[gr.update(interactive=False) for _ in controls]
            yield (*latest, *[gr.update(interactive=True) for _ in controls])

        generate.click(run, inputs=[person, length, shoulder, chest, table, garment, cache],
                       outputs=[cache, gallery, status, *controls], api_name='compare_sizes',
                       concurrency_limit=1, concurrency_id='dmvton', trigger_mode='once')
        for control in (person, garment):
            control.input(lambda: (None, [], PROMPT), inputs=[], outputs=[cache, gallery, status], queue=False)
        for control in (length, shoulder, chest, table):
            control.input(lambda: ([], PROMPT), inputs=[], outputs=[gallery, status], queue=False)
    return demo.queue(max_size=2, default_concurrency_limit=1)


def main():
    parser = argparse.ArgumentParser(description='DM-VTON 빠른 상의 사이즈 합성 비교')
    parser.add_argument('--ui', action='store_true')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int, default=None)
    args = parser.parse_args()
    build_ui().launch(server_name='0.0.0.0' if args.share else '127.0.0.1',
                      server_port=args.port, share=args.share)


if __name__ == '__main__':
    main()
