"""목표 윤곽 디버거의 기본 합성·마스크 확인·사이즈 조정 화면."""
import hashlib
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from garment_outline import photo_info, prepare_photo, render_outline
from size_synthesis import (adjust_sizes, bind_base, checked_preview, draft_mask,
                            editor_mask, editor_value, engine_settings, generate_base,
                            synthesis_report)


ROOT = Path(__file__).resolve().parent.parent
PROMPT = 'FASHN 기본 합성을 준비하고 옷 영역을 확인하면 **판매 사이즈별 치수가 자동 적용**됩니다.'


def mask_editor_payload(editor, payload):
    """확인에 쓰는 원본/레이어만 읽고 불필요한 합성 PNG의 비동기 업로드를 기다리지 않는다."""
    if payload is None:
        return None
    updates = {'composite': None}
    if payload.id is not None:
        cached = editor.blob_storage.pop(payload.id, None)
        if cached is None:
            return None
        updates.update(id=None, background=cached.background, layers=cached.layers)
    return payload.model_copy(update=updates)


def preprocess_mask_editor(editor, payload, preprocess):
    payload = mask_editor_payload(editor, payload)
    for attempt in range(6):
        try:
            return preprocess(payload)
        except PermissionError as exc:
            # Windows에서 레이어 자동 업로드가 캐시 파일을 쓰는 동안 읽기 요청이 겹칠 수 있다.
            if attempt == 5:
                raise ValueError('옷 영역 업로드가 끝나지 않았습니다. 잠시 후 다시 확인해주세요.') from exc
            time.sleep(.03)


def save_synthesis(result, person, base, garment=None, output_directory=None):
    data = result['report']
    if (photo_info(person)['sha256'] != data['base']['person_sha256']
            or photo_info(base)['sha256'] != data['base']['base_sha256']):
        raise ValueError('현재 사진과 사이즈 조정 결과가 다릅니다. 다시 적용해주세요.')
    if data['base']['garment_sha256'] is not None and (
            garment is None or photo_info(garment)['sha256'] != data['base']['garment_sha256']):
        raise ValueError('기본 합성에 사용한 새 옷 사진이 달라졌습니다.')
    directory = Path(output_directory) if output_directory else ROOT / 'outputs' / 'size_synthesis'
    directory = directory / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    directory.mkdir(parents=True, exist_ok=False)
    images = {'person.png': prepare_photo(person), 'base.png': prepare_photo(base),
              'adjusted.png': result['image'], 'source_mask.png': result['mask'],
              'target_mask.png': result['target_mask'], 'repair_mask.png': result['repair_mask'],
              'adjusted_overlay.png': render_outline(result['image'], preview=data['preview'])}
    if garment is not None:
        images['garment.png'] = prepare_photo(garment)
    for name, image in images.items():
        image.save(directory / name)
    payload = dict(data, artifacts={name: name for name in images})
    (directory / 'synthesis.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                                       allow_nan=False), encoding='utf-8')
    (directory / 'report.md').write_text('# 사이즈별 합성 후처리 실험\n\n' + synthesis_report(result), encoding='utf-8')
    return [str(directory / name) for name in (*images, 'synthesis.json', 'report.md')], f'저장 완료: `{directory}`'


def add_synthesis_ui(gr, preview, person, reset_controls=(), reset_buttons=(), point_editor=None,
                     sample=None, preview_events=(), busy_controls=()):
    """사이즈를 바꿀 때 기본 합성을 재생성하지 않고 같은 사진/시드를 재사용한다."""
    gr.Markdown('## 4. 새 옷 기본 합성과 사이즈 조정\n\n'
                'FASHN 기본 합성을 한 번 만든 뒤 옷 영역을 확인하면 판매 사이즈별 길이·몸통 폭을 적용합니다. '
                '3단계의 사이즈를 바꾸면 기본 합성을 재사용해 결과를 자동 갱신합니다. '
                'GPU가 없는 PC에서는 기존 합성 결과를 불러오거나 기준 사진으로 변형만 시험할 수 있습니다.')
    metadata = gr.State(None)
    mask_signature = gr.State(None)
    with gr.Row():
        garment = gr.Image(type='pil', format='png', sources=['upload'], label='새 옷 상품 사진', height=300)
        with gr.Accordion('치수 적용 전 원본 · 기존 합성 PNG 업로드', open=False):
            base = gr.Image(type='pil', format='png', sources=['upload'],
                            label='FASHN 기본 합성 · 사이즈 미적용', height=400)
    with gr.Accordion('기본 합성 설정', open=False):
        gr.Markdown('현재 엔진: FASHN. 새 옷 상품 사진을 사용하며, 같은 시드를 유지해 사이즈 변화만 비교합니다.')
        with gr.Row():
            seed = gr.Number(value=42, precision=0, label='시드')
            steps = gr.Number(value=50, precision=0, label='생성 스텝')
            guidance = gr.Number(value=2.5, label='Guidance')
    with gr.Row():
        generate = gr.Button('FASHN 기본 합성 생성', variant='primary')
        dry_run = gr.Button('GPU 없이 기준 사진으로 변형 시험')
    base_status = gr.Markdown()
    gr.Markdown('### 옷 영역 확인\n\n'
                '초록색은 자동 추출한 옷 영역 초안입니다. 빠진 옷은 브러시로 칠하고, '
                '팔·손·얼굴·바지·배경은 지우개로 지워주세요. 사진을 자르거나 확대 변형하지 않습니다.')
    mask_editor = gr.ImageEditor(type='pil', format='png', sources=[], transforms=[], layers=False,
                                brush=gr.Brush(colors=['#20c8aa'], color_mode='fixed'),
                                eraser=gr.Eraser(), label='옷 영역 수정', height=560)
    original_preprocess = mask_editor.preprocess
    mask_editor.preprocess = lambda payload: preprocess_mask_editor(mask_editor, payload, original_preprocess)
    approved = gr.Checkbox(value=False, label='초록색 영역이 옷에만 해당하는 것을 확인했습니다')
    apply = gr.Button('선택 사이즈 다시 적용', variant='secondary')
    with gr.Row():
        adjusted = gr.Image(type='pil', format='png', interactive=False,
                            label='현재 선택 사이즈 · 치수 적용 결과', height=520)
        overlay = gr.Image(type='pil', format='png', interactive=False, label='조정 결과 + 목표선', height=520)
    gallery = gr.Gallery(label='판매 사이즈별 치수 적용 결과 · 동일한 FASHN 기본 합성 사용',
                         columns=3, height=440, format='png')
    report = gr.Markdown(PROMPT)
    with gr.Accordion('편집·복원 영역 확인', open=False):
        with gr.Row():
            target_mask = gr.Image(type='pil', format='png', interactive=False, label='변형 후 옷 영역', height=270)
            repair_mask = gr.Image(type='pil', format='png', interactive=False, label='줄어들어 복원한 영역', height=270)
    save = gr.Button('현재 합성·사이즈 조정 실험 저장')
    downloads = gr.File(file_count='multiple', label='원본 · 기본 합성 · 조정 결과 · 마스크 · 설정 · 보고서')
    saved = gr.Markdown()
    result_outputs = [adjusted, overlay, gallery, report, target_mask, repair_mask, downloads, saved]

    def clear_result():
        return None, None, [], PROMPT, None, None, [], ''

    def clear_base():
        return None, None, None, None, False, '', *clear_result()

    def prepare_editor(image, current):
        try:
            return editor_value(image, draft_mask(image, current)), '자동 추출 초안입니다. 옷 영역을 확인·수정해주세요.'
        except ValueError as exc:
            return editor_value(image, Image.new('L', image.size)), str(exc)

    def signature(edited, image):
        return hashlib.sha256(editor_mask(edited, image).tobytes()).hexdigest()

    def mask_changed(edited, previous, image):
        try:
            current = signature(edited, image)
        except (ValueError, AttributeError, TypeError):
            return None, False, *clear_result()
        if current == previous:
            # 편집기의 자동 전송/재직렬화는 실제 수정으로 취급하지 않는다.
            return tuple(gr.skip() for _ in range(2+len(result_outputs)))
        return current, False, *clear_result()

    def make_base(current, photo, clothes, seed_value, step_value, guidance_value, test=False):
        try:
            current = checked_preview(current, photo)
            if test:
                image, data = bind_base(photo, photo, 'reference_photo_dry_run')
            else:
                image, data = generate_base(photo, clothes, engine_settings(seed_value, step_value, guidance_value))
            editor, text = prepare_editor(image, current)
            prefix = ('기준 사진 변형 테스트입니다. 새 옷을 생성한 결과가 아닙니다. ' if test
                      else 'FASHN 기본 합성을 생성했습니다. ')
            return data, image, editor, signature(editor, image), False, prefix+text, *clear_result()
        except (ValueError, RuntimeError, ImportError, OSError) as exc:
            return None, None, None, None, False, f'기본 합성을 준비하지 못했습니다. {exc}', *clear_result()

    def import_base(current, photo, clothes, image):
        try:
            current = checked_preview(current, photo)
            image, data = bind_base(photo, image, 'external_unverified', clothes)
            editor, text = prepare_editor(image, current)
            return data, image, editor, signature(editor, image), False, '외부 결과를 불러왔습니다. 원본과 같은 인물·구도인지 확인해주세요. '+text, *clear_result()
        except (ValueError, TypeError, AttributeError) as exc:
            return None, image, None, None, False, f'불러오기를 보류했습니다. {exc}', *clear_result()

    def calculate(current, photo, image, data, edited, confirmed, clothes):
        if image is None or not data:
            raise ValueError('기본 합성을 생성하거나 결과 사진을 불러와주세요.')
        if data['garment_sha256'] is not None and (
                clothes is None or photo_info(clothes)['sha256'] != data['garment_sha256']):
            raise ValueError('기본 합성에 사용한 새 옷 사진이 바뀌었습니다. 다시 생성해주세요.')
        return adjust_sizes(photo, image, data, current, editor_mask(edited, image), confirmed)

    def apply_now(*values):
        try:
            result, comparisons = calculate(*values)
            return (result['image'], render_outline(result['image'], preview=result['report']['preview']),
                    comparisons, synthesis_report(result), result['target_mask'], result['repair_mask'], [], '')
        except (ValueError, TypeError, KeyError, ImportError) as exc:
            return None, None, [], f'사이즈 적용을 보류했습니다. {exc}', None, None, [], ''

    def refresh(*values):
        current, photo, image, data, edited, confirmed, clothes = values
        if current is None or image is None or not data:
            return clear_result()
        if not confirmed:
            return None, None, [], '옷 영역을 확인하면 선택 사이즈와 전체 사이즈 비교가 자동으로 표시됩니다.', None, None, [], ''
        return apply_now(*values)

    def export(*values):
        try:
            result, _ = calculate(*values)
            return save_synthesis(result, values[1], values[2], values[6])
        except (ValueError, TypeError, KeyError, ImportError, OSError) as exc:
            return [], f'저장을 보류했습니다. {exc}'

    base_outputs = [metadata, base, mask_editor, mask_signature, approved, base_status, *result_outputs]
    generation_inputs = [preview, person, garment, seed, steps, guidance]
    # ImageEditor는 interactive 갱신 시 내부 canvas를 재생성하므로 스트리밍 제어 대상에서 제외한다.
    controls = list(dict.fromkeys([*busy_controls, person, garment, base, approved,
                                   apply, save, seed, steps, guidance, generate, dry_run]))
    # None은 최초 렌더의 자동 판정값이다. update(None)으로는 잠근 입력이 다시 열리지 않는다.
    interactive = [control.interactive is not False for control in controls]

    def generate_with_status(current, photo, clothes, seed_value, step_value, guidance_value,
                             progress=gr.Progress(track_tqdm=True)):
        yield (None, None, None, None, False,
               'FASHN 기본 합성 중입니다. 완료 후 옷 영역을 확인해주세요. 사이즈 변경은 재합성하지 않습니다.',
               *clear_result(), *[gr.update(interactive=False) for _ in controls])
        progress(.05, desc='FASHN 모델 준비 · 기본 합성')
        try:
            result = make_base(current, photo, clothes, seed_value, step_value, guidance_value)
        except Exception as exc:
            traceback.print_exc()
            result = (None, None, None, None, False, f'기본 합성 오류: {exc}', *clear_result())
        progress(1, desc='기본 합성 준비 완료 · 옷 영역 확인')
        yield (*result, *[gr.update(interactive=value) for value in interactive])

    generate.click(generate_with_status, inputs=generation_inputs, outputs=[*base_outputs, *controls],
                   api_name='generate_base', concurrency_limit=1, concurrency_id='size_synthesis',
                   trigger_mode='once')

    def test_base(*values):
        return make_base(*values, test=True)

    dry_run.click(test_base, inputs=generation_inputs, outputs=base_outputs,
                  api_name='test_reference_warp', concurrency_limit=1, concurrency_id='size_synthesis')
    base.input(import_base, inputs=[preview, person, garment, base], outputs=base_outputs,
               api_name='import_base', concurrency_limit=1, concurrency_id='size_synthesis')
    mask_editor.input(mask_changed, inputs=[mask_editor, mask_signature, base],
                      outputs=[mask_signature, approved, *result_outputs], queue=False)
    apply_inputs = [preview, person, base, metadata, mask_editor, approved, garment]
    approved.input(refresh, inputs=apply_inputs, outputs=result_outputs,
                   concurrency_limit=1, concurrency_id='size_synthesis', trigger_mode='always_last')
    # 목표선 계산이 끝난 뒤 같은 이벤트 체인에서 적용한다. 별도 clear 응답과 경쟁하지 않는다.
    for event in preview_events:
        event.then(refresh, inputs=apply_inputs, outputs=result_outputs,
                   concurrency_limit=1, concurrency_id='size_synthesis', trigger_mode='always_last')
    if not preview_events:
        preview.change(refresh, inputs=apply_inputs, outputs=result_outputs,
                       concurrency_limit=1, concurrency_id='size_synthesis', trigger_mode='always_last')
    for control in reset_controls:
        control.input(clear_result, inputs=[], outputs=result_outputs, queue=False)
    for button in reset_buttons:
        button.click(clear_result, inputs=[], outputs=result_outputs, queue=False)
    if point_editor is not None:
        point_editor.select(clear_result, inputs=[], outputs=result_outputs, queue=False)
    if sample is not None:
        sample.click(clear_base, inputs=[], outputs=base_outputs, queue=False)
    person.change(clear_base, inputs=[], outputs=base_outputs, queue=False)
    garment.input(clear_base, inputs=[], outputs=base_outputs, queue=False)
    for control in (seed, steps, guidance):
        control.input(clear_base, inputs=[], outputs=base_outputs, queue=False)
    apply.click(apply_now, inputs=apply_inputs, outputs=result_outputs, api_name='apply_size',
                concurrency_limit=1, concurrency_id='size_synthesis')
    save.click(export, inputs=apply_inputs, outputs=[downloads, saved], api_name='save_synthesis',
               concurrency_limit=1, concurrency_id='size_synthesis')
