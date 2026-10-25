"""공식 DM-VTON 가중치와 PyTorch correlation 연산을 사용하는 추론 어댑터.

원본: https://github.com/KiseKloset/DM-VTON (CC BY-NC-SA 4.0)
correlation의 7x7 오프셋 순서·채널 평균·zero padding은 원본과 같다.
CuPy CUDA 커널 대신 PyTorch 연산을 사용하므로 논문의 속도 수치와 별도로 측정한다.
"""
import importlib
from pathlib import Path
import sys
import time
import types

import numpy as np
from PIL import Image

from paths import DMVTON_REPO, DMVTON_WEIGHTS


UPSTREAM_COMMIT = '5dcd04ee3f5b1fb392df1184c887c1a7e910c72e'
MODEL_SIZE = (192, 256)  # PIL: width, height
BACKEND = 'dmvton_pytorch_correlation_v1'
_pipeline = None
_pipeline_key = None
_parser = None


def correlation(tenFirst, tenSecond, intStride=1):
    """원본 CUDA forward의 반경 3 cost volume. 추론에서 사용하는 stride=1만 지원한다."""
    import torch
    import torch.nn.functional as functional

    if intStride != 1 or tenFirst.ndim != 4 or tenFirst.shape != tenSecond.shape:
        raise ValueError('DM-VTON correlation은 같은 NCHW 크기와 stride=1이 필요합니다.')
    height, width = tenFirst.shape[-2:]
    padded = functional.pad(tenSecond, (3, 3, 3, 3))
    # unfold 전체를 만들지 않아 최고 해상도 피라미드에서도 임시 메모리가 작다.
    return torch.stack([(tenFirst * padded[:, :, dy:dy+height, dx:dx+width]).mean(1)
                        for dy in range(7) for dx in range(7)], dim=1)


def pipeline_class(repo):
    """별도 실행 서버 안에서 공식 모델을 불러온다. 원본 파일은 수정하지 않는다."""
    repo = Path(repo).resolve()
    if not (repo / 'pipelines' / 'dmvton_pipeline.py').is_file():
        raise RuntimeError('DM-VTON 소스가 없습니다. 코랩의 준비 셀을 먼저 실행해주세요.')
    for name in ('models', 'pipelines', 'utils'):
        loaded = sys.modules.get(name)
        if loaded is not None:
            locations = list(getattr(loaded, '__path__', []))
            if not locations or not all(Path(p).resolve().is_relative_to(repo) for p in locations):
                raise RuntimeError('다른 모델 모듈이 로드돼 있습니다. DM-VTON 실행 셀을 새로 시작해주세요.')
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    importlib.import_module('models.common')
    module = types.ModuleType('models.common.correlation')
    module.FunctionCorrelation = correlation
    sys.modules['models.common.correlation'] = module
    return importlib.import_module('pipelines.dmvton_pipeline').DMVTONPipeline


def require_gpu():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('DM-VTON 합성은 CUDA GPU가 필요합니다. 코랩에서 GPU 런타임을 선택해주세요.')
    return torch


def load_models():
    global _pipeline, _pipeline_key
    torch = require_gpu()
    weights = [Path(DMVTON_WEIGHTS) / name for name in ('dmvton_pf_warp.pt', 'dmvton_pf_gen.pt')]
    if any(not path.is_file() for path in weights):
        raise RuntimeError('DM-VTON 가중치 두 개가 필요합니다. 코랩의 모델 준비 셀을 실행해주세요.')
    key = (str(Path(DMVTON_REPO).resolve()),
           tuple((str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in weights))
    if _pipeline is None or _pipeline_key != key:
        model = pipeline_class(DMVTON_REPO)(align_corners=True)
        for target, path in zip((model.warp_model, model.gen_model), weights):
            state = torch.load(path, map_location='cpu', weights_only=True)
            if isinstance(state, dict) and 'model' in state:
                state = state['model']
            target.load_state_dict(state, strict=True)
        _pipeline = model.to('cuda').eval()
        _pipeline_key = key
    return _pipeline


def get_parser():
    global _parser
    require_gpu()
    if _parser is None:
        from fashn_human_parser import FashnHumanParser

        _parser = FashnHumanParser(device='cuda')
    return _parser


def infer(person, garment, garment_mask):
    """준비된 192x256 이미지에서 기본 합성과 실제 warped garment mask를 반환한다."""
    import torch.nn.functional as functional

    torch = require_gpu()
    if any(image.size != MODEL_SIZE for image in (person, garment, garment_mask)):
        raise ValueError('DM-VTON 입력 이미지와 마스크는 192x256이어야 합니다.')
    load_start = time.perf_counter()
    model = load_models()
    load_seconds = time.perf_counter() - load_start

    def tensor(image):
        array = np.asarray(image.convert('RGB')).copy().transpose(2, 0, 1)
        return torch.from_numpy(array).float().div_(127.5).sub_(1).unsqueeze(0).to('cuda')

    photo, clothes = tensor(person), tensor(garment)
    edge = torch.from_numpy((np.asarray(garment_mask.convert('L')) > 127).astype(np.float32))
    edge = edge[None, None].to('cuda')
    flows = []
    # 원본 forward의 흐름을 관측해서 추가 warping 추론 없이 옷 마스크를 얻는다.
    hook = model.warp_model.register_forward_hook(lambda _model, _inputs, output: flows.append(output[1]))
    try:
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            result, _warped = model(photo, clothes, edge, phase='test')
            mask = functional.grid_sample(edge, flows[-1].permute(0, 2, 3, 1),
                                          mode='bilinear', padding_mode='zeros', align_corners=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    finally:
        hook.remove()
    pixels = result[0].clamp(-1, 1).add(1).mul(127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    mask = (mask[0, 0].cpu().numpy() > .5).astype(np.uint8) * 255
    return Image.fromarray(pixels), Image.fromarray(mask), {
        'engine': 'DM-VTON', 'backend': BACKEND, 'upstream_commit': UPSTREAM_COMMIT,
        'input_size': list(MODEL_SIZE), 'model_load_s': load_seconds, 'inference_s': elapsed,
        'gpu': torch.cuda.get_device_name(0), 'dtype': 'float32',
    }
