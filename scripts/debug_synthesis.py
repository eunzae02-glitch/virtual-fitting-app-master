"""사진 두 장으로 FASHN 상의 기본 합성만 확인하는 웹 화면.

로컬: python -X utf8 scripts/debug_synthesis.py --ui
Colab: python -u scripts/debug_synthesis.py --share
"""
import argparse
import importlib.util
import math
from pathlib import Path
import sys
import time
import traceback

from PIL import ImageOps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'app'))

from fashn_core import (DEFAULT_GUIDANCE, DEFAULT_SEED, DEFAULT_STEPS,
                        load_models, try_on)  # noqa: E402
from paths import FASHN_REPO, FASHN_WEIGHTS  # noqa: E402


PROMPT = '인물 사진과 상의 상품 사진을 올린 뒤 **합성하기**를 눌러주세요.'


def check_environment():
    """버튼을 누른 뒤에만 GPU와 설치를 확인한다. 모델은 여기서 로드하지 않는다."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU가 필요합니다. 코랩의 런타임 유형을 GPU로 설정해주세요.')
    if not (importlib.util.find_spec('fashn_vton') or
            (Path(FASHN_REPO) / 'src' / 'fashn_vton').is_dir()):
        raise RuntimeError('FASHN 설치 셀을 먼저 실행해주세요.')
    required = ('model.safetensors', 'dwpose/yolox_l.onnx', 'dwpose/dw-ll_ucoco_384.onnx')
    missing = [name for name in required if not (Path(FASHN_WEIGHTS) / name).is_file()]
    if missing:
        raise RuntimeError('가중치 다운로드 셀을 먼저 실행해주세요. 누락: ' + ', '.join(missing))
    return torch.cuda.get_device_name(0)


def run_synthesis(person, garment, seed=DEFAULT_SEED, steps=DEFAULT_STEPS,
                  guidance=DEFAULT_GUIDANCE, progress=None):
    """진행 상황을 전달하며, 같은 서버의 FASHN 모델 캐시를 재사용한다."""
    started = time.perf_counter()
    phase = '입력 확인'

    def update(value, description):
        print(f'[합성 확인] {description}', flush=True)
        if progress is not None:
            progress(value, desc=description)

    try:
        if person is None or garment is None:
            raise ValueError('인물 사진과 상의 상품 사진을 모두 올려주세요.')
        if (any(isinstance(value, bool) for value in (seed, steps, guidance)) or
                not float(seed).is_integer() or not 0 <= float(seed) < 2**32 or
                not float(steps).is_integer() or not 1 <= float(steps) <= 200 or
                not math.isfinite(float(guidance)) or float(guidance) <= 0):
            raise ValueError('시드·스텝·guidance 값을 확인해주세요.')
        person = ImageOps.exif_transpose(person).convert('RGB')
        garment = ImageOps.exif_transpose(garment).convert('RGB')
        yield None, 'GPU와 설치 상태를 확인하고 있습니다.'
        update(0, 'GPU와 설치 확인')
        gpu = check_environment()

        phase = '모델 준비'
        yield None, '모델을 준비하고 있습니다. 첫 실행은 가중치를 GPU에 올리는 시간이 추가됩니다.'
        update(.05, '모델 준비 중 (첫 실행 로딩 / 이후 캐시 재사용)')
        load_started = time.perf_counter()
        load_models()
        load_seconds = time.perf_counter() - load_started

        phase = '상의 합성'
        yield None, '상의 합성 중입니다. 진행 상황은 코랩 실행 셀에서도 확인할 수 있습니다.'
        update(.15, '상의 합성 중')
        result, timing = try_on(person, garment, cloth_type='upper', seed=int(seed),
                               steps=int(steps), guidance_scale=float(guidance),
                               garment_photo_type='flat-lay', return_timing=True)
        elapsed = time.perf_counter() - started
        update(1, '합성 완료')
        yield result, (f'합성 완료 · 전체 {elapsed:.1f}초 '
                       f'(모델 준비 {load_seconds:.1f}초 / 합성 {timing["total_s"]:.1f}초)\n\n'
                       f'GPU: {gpu} · 시드: {int(seed)} · 스텝: {int(steps)} · '
                       f'guidance: {float(guidance):g}')
    except Exception as exc:
        if phase != '입력 확인':
            traceback.print_exc()
        yield None, f'{phase} 실패: {exc}'


def build_ui():
    import gradio as gr

    with gr.Blocks(title='FitCheck 합성 확인') as demo:
        gr.Markdown('# 상의 합성 확인\n\n' + PROMPT)
        with gr.Row():
            person = gr.Image(label='인물 사진', type='pil', format='png', height=380,
                              sources=['upload'])
            garment = gr.Image(label='상의 상품 사진', type='pil', format='png', height=380,
                               sources=['upload'])
            result = gr.Image(label='합성 결과', type='pil', format='png', height=380,
                              interactive=False)
        with gr.Accordion('고급 설정', open=False):
            with gr.Row():
                seed = gr.Number(label='시드', value=DEFAULT_SEED, precision=0,
                                 minimum=0, maximum=2**32-1)
                steps = gr.Slider(label='합성 스텝', value=DEFAULT_STEPS,
                                  minimum=10, maximum=100, step=1)
                guidance = gr.Slider(label='Guidance', value=DEFAULT_GUIDANCE,
                                     minimum=.5, maximum=5, step=.1)
        generate = gr.Button('합성하기', variant='primary')
        status = gr.Markdown(PROMPT)

        def run(photo, clothes, seed_value, step_value, guidance_value,
                progress=gr.Progress(track_tqdm=True)):
            yield from run_synthesis(photo, clothes, seed_value, step_value,
                                     guidance_value, progress)

        generate.click(run, inputs=[person, garment, seed, steps, guidance],
                       outputs=[result, status], api_name='synthesize',
                       concurrency_limit=1, concurrency_id='fashn_synthesis', trigger_mode='once')
        for control in (person, garment, seed, steps, guidance):
            control.input(lambda: (None, PROMPT), inputs=[], outputs=[result, status], queue=False)
    return demo.queue(max_size=2, default_concurrency_limit=1)


def main():
    parser = argparse.ArgumentParser(description='사진 두 장으로 FASHN 상의 합성 확인')
    parser.add_argument('--ui', action='store_true', help='웹 화면 실행 (생략해도 실행)')
    parser.add_argument('--share', action='store_true', help='코랩용 공개 Gradio 링크 생성')
    parser.add_argument('--port', type=int, default=None, help='생략하면 비어 있는 포트 자동 선택')
    args = parser.parse_args()
    build_ui().launch(server_name='0.0.0.0' if args.share else '127.0.0.1',
                      server_port=args.port, share=args.share)


if __name__ == '__main__':
    main()
