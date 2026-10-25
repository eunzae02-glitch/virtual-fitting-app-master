"""가상 피팅 웹 데모 — IDM-VTON 실험 버전 (app/gradio_app_idm.py).

기존 app/gradio_app.py의 신체 치수 추천, 사진 자세 검사(MediaPipe) 기능은 그대로 유지하면서,
합성 엔진만 IDM-VTON(app/idm_core.py)으로 교체한 버전입니다.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import gradio_client.utils as _gcu

# --- gradio_client 버그 우회 ---------------------------------------------
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

# 합성 엔진: IDM-VTON 연결
from idm_core import (try_on, load_models,
                      DEFAULT_STEPS, DEFAULT_GUIDANCE, DEFAULT_SEED)
from body_profile import BodyProfile
from photo_check import check_photo
from size_fit import SizeChart, recommend

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
    cloth_type = CLOTH_LABELS.get(cloth_label, 'upper')
    category = 'lower' if cloth_type == 'lower' else 'upper'
    files = list_charts(category)
    return gr.update(choices=files, value=files[0] if files else None)


def size_advice(chart_file, height, shoulder, chest, waist, hip) -> str:
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


GARMENT_PHOTO_LABELS = {
    '상품 사진 (옷만 펴 놓고 찍음)': 'flat-lay',
    '착용 사진 (사람이 입고 찍음)': 'model',
}


def photo_status(person):
    if person is None:
        return ''
    check = check_photo(person)
    if check.ok and not check.warnings:
        return '✅ ' + check.message()
    mark = '❌' if check.errors else '⚠️'
    lines = [f'{mark} ' + line for line in check.message().splitlines()]
    if check.errors:
        lines.append('')
        lines.append('다시 찍기 어렵다면 아래 **사진 검사 무시하고 합성**을 켜고 진행할 수 있어요.')
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
    info = f"합성 {timing['total_s']:.0f}초 · 시드 {timing['seed']} · {timing['dtype']} (엔진: IDM-VTON)"
    return result, advice, info


with gr.Blocks(title='사이즈 반영 가상 피팅 (IDM-VTON)') as demo:
    gr.Markdown(
        '# 가상 피팅 데모 (IDM-VTON 실험 버전)\n'
        '인물 사진과 옷 사진을 올리고 옷 종류를 고르면 합성 결과가 나옵니다 (**IDM-VTON SDXL**).\n\n'
        '- 인물은 **정면, 머리부터 발끝까지 화면을 꽉 채운** 사진일수록 결과가 좋습니다\n'
        '- T4 GPU 기준 약 1~2분 소요됩니다'
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
                height_in = gr.Number(value=175, label='키 (cm)')
                shoulder_in = gr.Number(value=None, label='어깨너비 (cm, 폭)')
                chest_in = gr.Number(value=None, label='가슴둘레 (cm)')
                waist_in = gr.Number(value=None, label='허리둘레 (cm)')
                hip_in = gr.Number(value=None, label='엉덩이둘레 (cm)')
                _initial = list_charts('upper')
                chart_in = gr.Dropdown(
                    choices=_initial, value=(_initial or [None])[0],
                    label='옷 치수표 (선택한 옷 종류에 맞춰 자동 변경)',
                )

            with gr.Accordion('고급 설정', open=False):
                ignore_check_in = gr.Checkbox(
                    value=False, label='사진 검사 무시하고 합성',
                    info='가이드에 안 맞는 사진도 그대로 넣어본다.',
                )
                steps_in = gr.Slider(
                    20, 50, value=DEFAULT_STEPS, step=1,
                    label='추론 스텝 (30 기본)',
                )
                guidance_in = gr.Slider(
                    1.0, 4.0, value=DEFAULT_GUIDANCE, step=0.1,
                    label='guidance scale (2.0 기본)',
                )
                seed_in = gr.Number(value=DEFAULT_SEED, precision=0,
                                    label='시드 (-1이면 매번 랜덤)')
            run_btn = gr.Button('피팅 해보기', variant='primary')

        with gr.Column():
            result_out = gr.Image(label='합성 결과', height=520)
            info_out = gr.Markdown()
            size_out = gr.Markdown(label='사이즈 추천')

    cloth_in.change(fn=charts_for_cloth, inputs=[cloth_in], outputs=[chart_in])
    person_in.change(fn=photo_status, inputs=[person_in], outputs=[photo_status_out])

    run_btn.click(
        fn=run,
        inputs=[person_in, garment_in, cloth_in, garment_photo_in, ignore_check_in,
                steps_in, guidance_scale, seed_in,
                chart_in, height_in, shoulder_in, chest_in, waist_in, hip_in],
        outputs=[result_out, size_out, info_out],
    )


if __name__ == '__main__':
    print('IDM-VTON 모델 로딩 중... (최초 1회)', flush=True)
    load_models()
    print('로딩 완료. Gradio 실행합니다.', flush=True)
    options = dict(share=True, show_error=True, prevent_thread_lock=True)
    try:
        _, local_url, share_url = demo.queue().launch(show_api=False, **options)
    except TypeError:
        _, local_url, share_url = demo.queue().launch(**options)
    print('=' * 60, flush=True)
    print('로컬 주소 :', local_url, flush=True)
    print('공개 링크 :', share_url or '(생성 실패)', flush=True)
    print('=' * 60, flush=True)
    demo.block_thread()
