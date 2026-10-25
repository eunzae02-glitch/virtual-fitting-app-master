"""가상 피팅 웹 데모. 합성 엔진은 FASHN VTON v1.5 (app/fashn_core.py).

기본 파이썬(3.10 이상)에서 실행 — CatVTON용 3.9 venv가 아니다:
    python app/gradio_app.py
FASHN 저장소·가중치 위치는 app/paths.py (FASHN_REPO, FASHN_WEIGHTS 환경변수로 바꿀 수 있음).
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import gradio_client.utils as _gcu

# --- gradio_client 버그 우회 ---------------------------------------------
# pydantic이 만든 JSON 스키마에는 `additionalProperties: true` 처럼 값이 dict가 아니라
# bool인 경우가 있는데, gradio_client가 이를 dict로 가정하고 `"const" in schema` 를 해서
#   TypeError: argument of type 'bool' is not iterable
# 로 터진다. gradio의 메인 라우트가 show_api 설정과 무관하게 api_info()를 호출하므로
# 페이지 자체가 안 열린다. bool 스키마를 Any로 처리하도록 감싼다.
_orig_json_schema_to_python_type = _gcu._json_schema_to_python_type
_orig_get_type = _gcu.get_type


def _safe_json_schema_to_python_type(schema, defs=None):
    if isinstance(schema, bool):
        return 'Any'
    return _orig_json_schema_to_python_type(schema, defs)


def _safe_get_type(schema):
    if not isinstance(schema, dict):
        return 'Any'
    return _orig_get_type(schema)


_gcu._json_schema_to_python_type = _safe_json_schema_to_python_type
_gcu.get_type = _safe_get_type
# -------------------------------------------------------------------------

from fashn_core import (try_on, load_models, FASHN_REPO,
                        DEFAULT_STEPS, DEFAULT_GUIDANCE, DEFAULT_SEED)
from body_measure import measure
from body_profile import BodyProfile
from photo_check import check_photo
from size_fit import SizeChart, recommend

# 새 상의 추천은 합성과 독립적으로 실행한다. 디버거와 계산 경로를 공유한다.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
from debug_upper_size import analyze as analyze_upper_size
from chart_ocr import image_to_draft

CLOTH_LABELS = {
    '상의': 'upper',
    '하의': 'lower',
    '원피스 (상하 일체)': 'overall',
    '이너': 'inner',
    '아우터': 'outer',
}


CHART_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'data', 'size_charts')


def load_chart(filename):
    with open(os.path.join(CHART_DIR, filename), encoding='utf-8') as f:
        return SizeChart.from_dict(json.load(f))


def list_charts(category=None):
    """치수표 목록. category를 주면 그 부위의 치수표만 돌려준다.

    상의를 입히면서 하의 치수표를 고르면 공통 항목이 없어 추천이 안 나온다.
    선택 자체를 막는 게 낫다.
    """
    if not os.path.isdir(CHART_DIR):
        return []
    files = sorted(f for f in os.listdir(CHART_DIR) if f.endswith('.json'))
    if category is None:
        return files
    out = []
    for f in files:
        try:
            if load_chart(f).category == category:
                out.append(f)
        except Exception:
            continue
    return out


def charts_for_cloth(cloth_label):
    """옷 종류 라디오 선택에 맞는 치수표로 드롭다운을 갱신한다."""
    cloth_type = CLOTH_LABELS.get(cloth_label, 'upper')
    category = 'lower' if cloth_type == 'lower' else 'upper'
    files = list_charts(category)
    return gr.update(choices=files, value=files[0] if files else None)


def size_advice(chart_file, height, shoulder, chest, waist, hip) -> str:
    """입력한 신체 치수와 옷 치수표를 비교해 사이즈 추천 문구를 만든다."""
    if not chart_file:
        return '치수표가 없어 사이즈 추천을 건너뜁니다.'
    try:
        profile = BodyProfile.from_inputs(
            height_cm=height, shoulder_cm=shoulder, chest_circumference_cm=chest,
            waist_circumference_cm=waist, hip_circumference_cm=hip,
        )
    except ValueError as e:
        return f'**사이즈 추천 불가** — {e}'

    problems = profile.validate()
    if problems:
        return '**입력값을 확인해주세요**\n\n' + '\n'.join(f'- {p}' for p in problems)

    body = profile.to_chart_dimensions()
    if not body:
        return '신체 치수를 하나 이상 입력하면 사이즈를 추천합니다.'

    chart = load_chart(chart_file)

    rec = recommend(chart, body)
    if rec.best is None and not set(chart.dimensions()) & set(body):
        return ('**사이즈 추천 불가** — '
                f'`{chart.name}`({chart.category})와 입력한 치수가 맞지 않습니다. '
                '옷 종류에 맞는 치수표를 선택했는지 확인해주세요.')
    if rec.best is None:
        return '**사이즈 추천 불가** — ' + ' '.join(rec.notes)

    lines = [f'### {rec.message}', '', f'**{chart.name}** 기준', '',
             '| 사이즈 | 항목별 여유분 | 입을 수 있음 |', '|---|---|---|']
    for fit in rec.ranked:
        detail = ' / '.join(f'{d.dimension} {d.ease_cm:+.1f}cm ({d.label})'
                            for d in fit.dimensions)
        lines.append(f"| {fit.size} | {detail} | {'O' if fit.wearable else 'X'} |")
    for note in rec.notes:
        lines.append(f'\n> {note}')
    return '\n'.join(lines)


def auto_fill_measurements(person, height, shoulder, chest, waist, hip):
    """사진 추정값을 편집 가능한 둘레 입력칸에 채운다. 측정 불가 항목은 유지한다."""
    if person is None:
        raise gr.Error('먼저 정면 전신 사진을 올려주세요.')
    try:
        profile = BodyProfile.from_inputs(height_cm=height)
    except ValueError as e:
        raise gr.Error(str(e)) from e
    problems = profile.validate()
    if problems:
        raise gr.Error(' '.join(problems))

    try:
        estimate = measure(person, profile.height_cm)
    except (ImportError, OSError, RuntimeError) as e:
        raise gr.Error(f'사진 치수 추정을 실행하지 못했습니다: {e}') from e

    current = (shoulder, chest, waist, hip)
    if not estimate.measurements_cm:
        return shoulder, chest, waist, hip, '사진에서 치수를 추정하지 못했습니다. ' + ' '.join(estimate.warnings)
    if estimate.unsafe:
        return shoulder, chest, waist, hip, '사진 상태를 확인해주세요. 입력값은 유지했습니다. ' + ' '.join(estimate.warnings)

    values = estimate.measurements_cm
    filled = (
        values.get('shoulder', shoulder),
        round(values['chest'] * 2, 1) if estimate.has_segmentation and 'chest' in values else chest,
        round(values['waist'] * 2, 1) if 'waist' in values else waist,
        round(values['hip'] * 2, 1) if 'hip' in values else hip,
    )
    labels = ('어깨너비', '가슴둘레', '허리둘레', '엉덩이둘레')
    updated = [label for old, new, label in zip(current, filled, labels) if new != old]
    lines = ['사진 추정값을 입력칸에 채웠습니다: ' + (', '.join(updated) or '변경된 항목 없음') + '. '
             '실측치가 있다면 직접 고쳐주세요.']
    lines.extend(estimate.warnings)
    lines.append('두꺼운 옷·박시한 옷은 몸 실루엣을 가려 둘레가 크게 추정될 수 있습니다.')
    return filled[0], filled[1], filled[2], filled[3], '\n\n'.join(lines)


def auto_fill_needs_refresh(_value):
    return '사진이나 키가 바뀌었다면 이전 사진 추정값은 맞지 않습니다. 다시 자동 채우기를 누르거나 직접 수정해주세요.'


def upper_size_ocr(image):
    return image_to_draft(image)


def upper_size_reset(_image):
    return '', '', '', False


def upper_size_unconfirm(_text):
    return False


def upper_size_run(person, height, basis, calibration, manual_chest,
                   reference_json, reference_confirmed, worn_size, worn_fit,
                   target_json, target_confirmed, target_shoulder_standard):
    return analyze_upper_size(person, height, basis, calibration, manual_chest,
                              reference_json, reference_confirmed, worn_size, worn_fit,
                              target_json, target_confirmed, target_shoulder_standard,
                              None, None)


GARMENT_PHOTO_LABELS = {
    '상품 사진 (옷만 펴 놓고 찍음)': 'flat-lay',
    '착용 사진 (사람이 입고 찍음)': 'model',
}


def photo_status(person):
    """인물 사진을 올린 순간 자세·거리·역광을 검사해 안내한다 (app/photo_check.py).

    합성 한 장에 2분이 걸리므로, 규칙에 안 맞는 사진은 올리는 순간 알려주는 편이 낫다.
    포즈 인식(mediapipe)이 없거나 실패하면 photo_check 가 경고만 남기고 넘어간다.
    """
    if person is None:
        return ''
    check = check_photo(person)
    if check.ok and not check.warnings:
        return '✅ ' + check.message()
    mark = '❌' if check.errors else '⚠️'
    lines = [f'{mark} ' + line for line in check.message().splitlines()]
    if check.errors:
        lines.append('')
        lines.append('다시 찍기 어렵다면 아래 **사진 검사 무시하고 합성**을 켜고 진행할 수 있어요 (결과가 나빠질 수 있어요).')
    return '\n\n'.join(lines)


def run(person, garment, cloth_label, garment_photo_label, ignore_photo_check,
        steps, guidance_scale, seed, chart_file, height, shoulder, chest, waist, hip):
    if person is None or garment is None:
        raise gr.Error('인물 사진과 옷 사진을 모두 올려주세요.')

    if not ignore_photo_check:
        check = check_photo(person)
        if not check.ok:
            raise gr.Error(check.message() + '\n(그대로 해보려면 "사진 검사 무시하고 합성"을 켜세요)')

    advice = size_advice(chart_file, height, shoulder, chest, waist, hip)

    result, timing = try_on(
        person=person,
        garment=garment,
        cloth_type=CLOTH_LABELS[cloth_label],
        garment_photo_type=GARMENT_PHOTO_LABELS[garment_photo_label],
        steps=int(steps),
        guidance_scale=float(guidance_scale),
        seed=int(seed),
        return_timing=True,
    )
    info = f"합성 {timing['total_s']:.0f}초 · 시드 {timing['seed']} · {timing['dtype']}"
    return result, advice, info


def _examples():
    """FASHN 저장소 예시 이미지로 예시 조합을 만든다. 없으면 빈 리스트."""
    person = glob.glob(os.path.join(FASHN_REPO, 'examples/data/model.*'))
    garment = glob.glob(os.path.join(FASHN_REPO, 'examples/data/garment.*'))
    if not person or not garment:
        return []
    return [[person[0], garment[0], '상의', '착용 사진 (사람이 입고 찍음)']]


with gr.Blocks(title='사이즈 반영 가상 피팅') as demo:
    gr.Markdown(
        '# 가상 피팅 데모\n'
        '인물 사진과 옷 사진을 올리고 옷 종류를 고르면 합성 결과가 나옵니다 (FASHN VTON v1.5).\n\n'
        '- 인물은 **정면, 머리부터 발끝까지 화면을 꽉 채운** 사진일수록 결과가 좋습니다\n'
        '- T4 GPU 기준 한 장에 약 2분 (50스텝, 품질 우선). 빠르게 보려면 고급 설정에서 스텝을 30으로'
    )

    with gr.Row():
        with gr.Column():
            person_in = gr.Image(label='인물 사진', type='pil', height=400)
            photo_status_out = gr.Markdown(
                '인물 사진을 올리면 자세·거리를 먼저 검사해요 '
                '(정면 · 팔은 몸에서 주먹 하나 · 머리부터 발끝까지 화면 가득).')
            garment_in = gr.Image(label='옷 사진', type='pil', height=400)
            cloth_in = gr.Radio(
                choices=list(CLOTH_LABELS.keys()),
                value='상의',
                label='옷 종류',
            )
            garment_photo_in = gr.Radio(
                choices=list(GARMENT_PHOTO_LABELS.keys()),
                value='상품 사진 (옷만 펴 놓고 찍음)',
                label='옷 사진 종류',
            )
            with gr.Accordion('내 신체 치수 (사이즈 추천용)', open=True):
                gr.Markdown(
                    '아는 항목만 넣어도 됩니다. **둘레**로 입력하면 치수표의 단면과 '
                    '자동으로 맞춰 계산합니다 (가슴둘레 96 → 가슴단면 48).'
                )
                height_in = gr.Number(value=None, label='키 (cm, 직접 입력 필수)')
                shoulder_in = gr.Number(value=None, label='어깨너비 (cm, 폭)')
                chest_in = gr.Number(value=None, label='가슴둘레 (cm)')
                waist_in = gr.Number(value=None, label='허리둘레 (cm)')
                hip_in = gr.Number(value=None, label='엉덩이둘레 (cm)')
                auto_fill_btn = gr.Button('사진으로 치수 자동 채우기')
                auto_fill_status = gr.Markdown(
                    '사진 추정은 참고값입니다. 몸에 붙는 옷을 입고 팔을 몸에서 조금 떼어 정면 전신 사진을 찍어주세요.')
                _initial = list_charts('upper')
                chart_in = gr.Dropdown(
                    choices=_initial, value=(_initial or [None])[0],
                    label='옷 치수표 (선택한 옷 종류에 맞춰 자동 변경)',
                )

            with gr.Accordion('고급 설정', open=False):
                ignore_check_in = gr.Checkbox(
                    value=False, label='사진 검사 무시하고 합성',
                    info='가이드에 안 맞는 사진도 그대로 넣어본다. 결과가 나빠질 수 있다.',
                )
                steps_in = gr.Slider(
                    20, 50, value=DEFAULT_STEPS, step=1,
                    label='추론 스텝 (50 기본, 30이면 약 1.7배 빠름)',
                )
                guidance_in = gr.Slider(
                    1.0, 4.0, value=DEFAULT_GUIDANCE, step=0.1,
                    label='guidance scale (2.5 기본. 낮추면 글자 프린트가 휜다)',
                )
                seed_in = gr.Number(value=DEFAULT_SEED, precision=0,
                                    label='시드 (-1이면 매번 랜덤. 결과가 어색하면 바꿔서 다시)')
            run_btn = gr.Button('피팅 해보기', variant='primary')

        with gr.Column():
            result_out = gr.Image(label='합성 결과', height=520)
            info_out = gr.Markdown()
            size_out = gr.Markdown(label='사이즈 추천')

    # 옷 종류를 바꾸면 그에 맞는 치수표만 남긴다
    cloth_in.change(fn=charts_for_cloth, inputs=[cloth_in], outputs=[chart_in])
    # 사진을 올리는 순간 검사한다 (합성 한 장에 2분이라 미리 걸러야 한다)
    person_in.change(fn=photo_status, inputs=[person_in], outputs=[photo_status_out])
    person_in.change(fn=auto_fill_needs_refresh, inputs=[person_in], outputs=[auto_fill_status])
    height_in.change(fn=auto_fill_needs_refresh, inputs=[height_in], outputs=[auto_fill_status])
    auto_fill_btn.click(
        fn=auto_fill_measurements,
        inputs=[person_in, height_in, shoulder_in, chest_in, waist_in, hip_in],
        outputs=[shoulder_in, chest_in, waist_in, hip_in, auto_fill_status],
    )

    run_btn.click(
        fn=run,
        inputs=[person_in, garment_in, cloth_in, garment_photo_in, ignore_check_in,
                steps_in, guidance_in, seed_in,
                chart_in, height_in, shoulder_in, chest_in, waist_in, hip_in],
        outputs=[result_out, size_out, info_out],
    )

    examples = _examples()
    if examples:
        gr.Examples(examples=examples, inputs=[person_in, garment_in, cloth_in, garment_photo_in], label='예시')

    with gr.Accordion('새 상의 사이즈 추천 (합성 없이 실행)', open=False):
        gr.Markdown('전신 사진+키가 필수입니다. 평소 반팔 치수표는 선택, 새 옷 치수표는 상품별 추천에 필수입니다. '
                    'OCR 초안은 반드시 원본과 대조해주세요. 이 버튼은 GPU 합성을 실행하지 않습니다. '
                    '수동 입력은 `{"unit":"cm","sizes":{"S":{"가슴단면":49},"M":{"가슴단면":52}}}` '
                    '형식입니다. 옷 완성 둘레는 `옷 가슴둘레`로, 권장 신체둘레는 입력하지 마세요.')
        with gr.Row():
            with gr.Column():
                upper_person = gr.Image(label='정면 전신 사진', type='pil')
                upper_height = gr.Number(label='키 (cm)', value=None)
                upper_basis = gr.Radio(['선택 안 함', '남', '녀'], value='선택 안 함',
                                       label='가슴 경계 불명확 시 평균 기준')
                upper_calibration = gr.File(label='실측 보정 CSV (선택)', type='filepath')
                upper_manual_chest = gr.Number(label='가슴둘레 직접 입력 (cm, 선택)', value=None)
            with gr.Column():
                upper_ref_image = gr.Image(label='평소 반팔 치수표 (선택)', type='pil')
                upper_ref_ocr = gr.Button('평소 반팔 치수표 OCR')
                upper_ref_text = gr.Textbox(label='평소 반팔 OCR 원문', lines=2, interactive=False)
                upper_ref_json = gr.Textbox(label='평소 반팔 치수표 JSON 확인/수정', lines=5)
                upper_ref_status = gr.Markdown()
                upper_ref_confirm = gr.Checkbox(label='평소 반팔 숫자·라벨 확인', value=False)
                upper_worn_size = gr.Textbox(label='실제로 입는 반팔 사이즈 (예: M)')
                upper_worn_fit = gr.Radio(['작음', '적당함', '큼'], value='적당함',
                                          label='평소 반팔의 착용감')
            with gr.Column():
                upper_target_image = gr.Image(label='새 옷 치수표', type='pil')
                upper_target_ocr = gr.Button('새 옷 치수표 OCR')
                upper_target_text = gr.Textbox(label='새 옷 OCR 원문', lines=2, interactive=False)
                upper_target_json = gr.Textbox(label='새 옷 치수표 JSON 확인/수정', lines=7)
                upper_target_status = gr.Markdown()
                upper_target_confirm = gr.Checkbox(label='새 옷 숫자·라벨 확인', value=False)
                upper_standard_shoulder = gr.Checkbox(label='새 옷의 어깨선은 일반적인 봉제 어깨선',
                                                      value=False)
        upper_run_btn = gr.Button('상체 추정 + 사이즈 비교', variant='primary')
        upper_report = gr.Markdown()
        upper_table = gr.Markdown()
        upper_ref_ocr.click(fn=upper_size_ocr, inputs=[upper_ref_image],
                            outputs=[upper_ref_json, upper_ref_text, upper_ref_status])
        upper_target_ocr.click(fn=upper_size_ocr, inputs=[upper_target_image],
                               outputs=[upper_target_json, upper_target_text, upper_target_status])
        upper_ref_image.change(fn=upper_size_reset, inputs=[upper_ref_image],
                               outputs=[upper_ref_json, upper_ref_text, upper_ref_status,
                                        upper_ref_confirm])
        upper_target_image.change(fn=upper_size_reset, inputs=[upper_target_image],
                                  outputs=[upper_target_json, upper_target_text,
                                           upper_target_status, upper_target_confirm])
        upper_ref_json.change(fn=upper_size_unconfirm, inputs=[upper_ref_json],
                              outputs=[upper_ref_confirm])
        upper_target_json.change(fn=upper_size_unconfirm, inputs=[upper_target_json],
                                 outputs=[upper_target_confirm])
        upper_run_btn.click(
            fn=upper_size_run,
            inputs=[upper_person, upper_height, upper_basis, upper_calibration,
                    upper_manual_chest, upper_ref_json, upper_ref_confirm,
                    upper_worn_size, upper_worn_fit, upper_target_json,
                    upper_target_confirm, upper_standard_shoulder],
            outputs=[upper_report, upper_table],
        )


if __name__ == '__main__':
    print('모델 로딩 중... (최초 1회)', flush=True)
    load_models()
    print('로딩 완료. Gradio 실행합니다.', flush=True)
    # show_api=False: API 스키마 생성 경로(gradio_client)가 pydantic 버전에 따라 터지는 걸 회피.
    #   단 gradio 6에서 이 인자가 없어져 TypeError가 난다(Colab 2026-09-16) → 빼고 다시 띄운다.
    # prevent_thread_lock=True로 먼저 URL을 받아 직접 flush 출력한다.
    # (gradio 자체 print는 flush를 안 해서 파이프로 실행하면 링크가 안 보인다)
    options = dict(share=True, show_error=True, prevent_thread_lock=True)
    try:
        _, local_url, share_url = demo.queue().launch(show_api=False, **options)
    except TypeError:
        _, local_url, share_url = demo.queue().launch(**options)
    print('=' * 60, flush=True)
    print('로컬 주소 :', local_url, flush=True)
    print('공개 링크 :', share_url or '(생성 실패 — 아래 안내 참고)', flush=True)
    print('=' * 60, flush=True)
    if not share_url:
        print(
            'share 링크를 못 만들었습니다. Colab이라면 아래 셀로 접속하세요:\n'
            '    from google.colab.output import eval_js\n'
            '    print(eval_js("google.colab.kernel.proxyPort(7860)"))',
            flush=True,
        )
    demo.block_thread()
