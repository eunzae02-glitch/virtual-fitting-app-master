"""원본 사진을 다시 렌더링/업로드하지 않는 단일 옷 마스크 편집기."""
import base64
from io import BytesIO
import json

import gradio as gr
from PIL import Image


TEMPLATE = """
<div class="mask-tools" role="toolbar" aria-label="옷 영역 편집 도구">
  <button type="button" data-tool="brush" aria-pressed="true">브러시</button>
  <button type="button" data-tool="erase" aria-pressed="false">지우개</button>
  <label>크기 <input type="range" min="2" max="80" value="20" aria-label="브러시 크기"></label>
  <button type="button" data-reset>초안으로 되돌리기</button>
</div>
<p class="mask-message">기본 합성 사진을 먼저 준비해주세요.</p>
<div class="mask-view" hidden>
  <img alt="편집하지 않는 기본 합성 원본" draggable="false">
  <canvas aria-label="옷 영역 마스크"></canvas>
</div>
"""

STYLE = """
.mask-tools { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin-bottom: 12px; }
button { padding: 7px 12px; border: 1px solid #888; border-radius: 6px; cursor: pointer; }
button[aria-pressed="true"] { background: #16745f; color: white; }
label { display: flex; gap: 8px; align-items: center; }
.mask-view { position: relative; width: fit-content; max-width: 100%; margin: auto; }
.mask-view[hidden] { display: none; }
.mask-view img { display: block; max-height: 560px; max-width: 100%; width: auto; user-select: none; }
canvas { position: absolute; inset: 0; width: 100%; height: 100%; touch-action: none; cursor: crosshair; }
"""

JAVASCRIPT = """
let data = null, ctx = null, tool = 'brush', size = 20, last = null, loading = 0;
const image = (src) => new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
});
const tools = () => {
    element.querySelectorAll('[data-tool]').forEach(button => {
        button.setAttribute('aria-pressed', String(button.dataset.tool === tool));
    });
    element.querySelector('input').value = size;
};
async function load() {
    const version = ++loading;
    data = null;
    ctx = null;
    last = null;
    tools();
    const view = element.querySelector('.mask-view');
    const message = element.querySelector('.mask-message');
    view.hidden = true;
    message.hidden = false;
    message.textContent = '기본 합성 사진을 먼저 준비해주세요.';
    if (!props.value) return;
    try {
        const value = JSON.parse(props.value);
        const [background, mask] = await Promise.all([image(value.background), image(value.layer)]);
        if (version !== loading) return;
        const canvas = element.querySelector('canvas');
        canvas.width = background.naturalWidth;
        canvas.height = background.naturalHeight;
        ctx = canvas.getContext('2d');
        ctx.drawImage(mask, 0, 0);
        element.querySelector('img').src = value.background;
        data = value;
        view.hidden = false;
        message.hidden = true;
    } catch (error) {
        message.textContent = '옷 영역을 불러오지 못했습니다. 기본 합성을 다시 불러와주세요.';
    }
}
function position(event) {
    const canvas = element.querySelector('canvas');
    const rect = canvas.getBoundingClientRect();
    return [(event.clientX - rect.left) * canvas.width / rect.width,
            (event.clientY - rect.top) * canvas.height / rect.height];
}
function stroke(point) {
    ctx.globalCompositeOperation = tool === 'erase' ? 'destination-out' : 'source-over';
    ctx.strokeStyle = tool === 'erase' ? 'rgba(0, 0, 0, 1)' : 'rgba(32, 200, 170, 0.51)';
    ctx.fillStyle = ctx.strokeStyle;
    ctx.lineWidth = size;
    ctx.lineCap = 'round';
    ctx.beginPath();
    if (last) {
        ctx.moveTo(...last);
        ctx.lineTo(...point);
        ctx.stroke();
    } else {
        ctx.arc(...point, size / 2, 0, Math.PI * 2);
        ctx.fill();
    }
    last = point;
}
element.addEventListener('click', event => {
    const button = event.target.closest('button');
    if (button?.dataset.tool) {
        tool = button.dataset.tool;
        tools();
    } else if (button?.hasAttribute('data-reset') && data) {
        props.value = JSON.stringify({...data, layer: data.draft});
        trigger('input');
    }
});
element.addEventListener('input', event => {
    if (event.target.matches('input[type="range"]')) size = Number(event.target.value);
});
element.addEventListener('pointerdown', event => {
    if (!data || !ctx || event.target.tagName !== 'CANVAS' || event.button !== 0) return;
    event.preventDefault();
    event.target.setPointerCapture(event.pointerId);
    last = null;
    stroke(position(event));
});
element.addEventListener('pointermove', event => {
    if (last && ctx) stroke(position(event));
});
function finish(event) {
    if (!last || !data || !ctx) return;
    const canvas = element.querySelector('canvas');
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    last = null;
    // 배경 데이터는 처음 받은 그대로 유지하고 투명 마스크만 갱신한다.
    props.value = JSON.stringify({...data, layer: canvas.toDataURL('image/png')});
    trigger('input');
}
element.addEventListener('pointerup', finish);
element.addEventListener('pointercancel', finish);
watch('value', load);
load();
"""


def encode_png(image):
    stream = BytesIO()
    image.save(stream, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode('ascii')


def decode_png(value):
    if not isinstance(value, str) or not value.startswith('data:image/png;base64,') or len(value) > 16_000_000:
        raise ValueError('옷 영역의 PNG 데이터를 확인해주세요.')
    try:
        raw = base64.b64decode(value.split(',', 1)[1], validate=True)
        with Image.open(BytesIO(raw)) as image:
            if image.format != 'PNG' or max(image.size) > 1200:
                raise ValueError('옷 영역의 크기와 PNG 형식을 확인해주세요.')
            return image.convert('RGBA')
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError('옷 영역의 PNG 데이터를 확인해주세요.') from exc


class GarmentMaskEditor(gr.HTML):
    """백엔드에는 기존 ImageEditor와 같은 background/layers 자료형을 제공한다."""

    def __init__(self):
        super().__init__(value='', label='옷 영역 수정', show_label=True, container=True,
                         padding=True, html_template=TEMPLATE, css_template=STYLE,
                         js_on_load=JAVASCRIPT)

    def postprocess(self, value):
        if not value:
            return ''
        background = encode_png(value['background'].convert('RGBA'))
        layer = encode_png(value['layers'][0])
        return json.dumps({'background': background, 'layer': layer, 'draft': layer})

    def preprocess(self, payload):
        if not payload:
            return None
        try:
            value = json.loads(payload)
            background, layer = decode_png(value['background']), decode_png(value['layer'])
        except (TypeError, KeyError, ValueError) as exc:
            raise ValueError('기본 합성 사진과 옷 영역을 다시 불러와주세요.') from exc
        if background.size != layer.size:
            raise ValueError('옷 영역과 기본 합성 사진의 크기가 다릅니다.')
        return {'background': background, 'layers': [layer], 'composite': None}
