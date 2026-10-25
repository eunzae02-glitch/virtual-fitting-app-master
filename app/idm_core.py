"""IDM-VTON 추론 엔진 — 가상 피팅 모델 교체 실험용 (app/idm_core.py).

FASHN(app/fashn_core.py)과 동일한 try_on() 인터페이스를 제공하여
gradio_app_idm.py에서 바로 교체하여 사용할 수 있도록 구성했습니다.

SDXL 기반 고화질 모델이므로 T4(16GB) 환경에서는:
  - fp16 적용
  - pipe.enable_model_cpu_offload() 및 vae slicing/tiling 활성화
가 필수입니다.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import SCRATCH, REPO_ROOT

IDM_REPO = os.environ.get('IDM_REPO', os.path.join(SCRATCH, 'IDM-VTON'))

DEFAULT_STEPS = 30
DEFAULT_GUIDANCE = 2.0
DEFAULT_SEED = 42

CATEGORY_BY_CLOTH_TYPE = {
    'upper': 'upper_body',
    'inner': 'upper_body',
    'outer': 'upper_body',
    'lower': 'lower_body',
    'overall': 'dresses',
}
CLOTH_TYPES = list(CATEGORY_BY_CLOTH_TYPE)
GARMENT_PHOTO_TYPES = ['flat-lay', 'model']

_pipeline = None
_preprocessors = None


def _to_rgb(x):
    """경로나 PIL 이미지를 EXIF 회전을 적용한 RGB로 변환."""
    from PIL import Image, ImageOps
    image = Image.open(x) if isinstance(x, str) else x
    return ImageOps.exif_transpose(image).convert('RGB')


def load_models():
    """IDM-VTON 파이프라인 및 전처리기 로드 (1회만 로드 후 캐싱)."""
    global _pipeline, _preprocessors
    if _pipeline is not None:
        return _pipeline, _preprocessors

    import torch
    from diffusers import AutoencoderKL

    # IDM-VTON 저장소 경로 등록
    if os.path.isdir(IDM_REPO) and IDM_REPO not in sys.path:
        sys.path.insert(0, IDM_REPO)

    try:
        from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
        from src.unet_hacked_tryon import UNet2DConditionModel
        from src.unet_hacked_garmnet import UNet2DConditionModel as GarmUNet
    except ImportError:
        # IDM-VTON repo 클론이 안 되어 있거나 src 구조가 다를 때 친절한 안내
        raise ImportError(
            f"IDM-VTON 저장소를 찾을 수 없습니다. (확인 경로: {IDM_REPO})\n"
            "Colab 또는 터미널에서 아래 명령을 실행해주세요:\n"
            "  git clone https://github.com/yisol/IDM-VTON.git " + IDM_REPO
        )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "yisol/IDM-VTON"

    print(f"[IDM-VTON] 가중치 로딩 중 ({model_id})...", flush=True)
    unet = UNet2DConditionModel.from_pretrained(
        model_id,
        subfolder="unet",
        torch_dtype=dtype,
    )
    garm_net = GarmUNet.from_pretrained(
        model_id,
        subfolder="garm_net",
        torch_dtype=dtype,
    )
    vae = AutoencoderKL.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=dtype,
    )

    pipe = TryonPipeline.from_pretrained(
        model_id,
        unet=unet,
        garm_net=garm_net,
        vae=vae,
        torch_dtype=dtype,
    )

    # T4 GPU(16GB) 메모리 절약 설정
    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
    else:
        pipe.to(device)

    # 전처리기 로드 (DensePose / HumanParsing / OpenPose)
    preprocessors = {}
    try:
        from preprocess.humanparsing.run_parsing import Parsing
        from preprocess.openpose.run_openpose import OpenPose
        preprocessors['parsing'] = Parsing(0) if torch.cuda.is_available() else Parsing(-1)
        preprocessors['openpose'] = OpenPose(0) if torch.cuda.is_available() else OpenPose(-1)
    except Exception as e:
        print(f"[IDM-VTON] 전처리기 로드 경고 (기본 마스크 대체 가능): {e}", flush=True)

    _pipeline = pipe
    _preprocessors = preprocessors
    print("[IDM-VTON] 로딩 완료!", flush=True)
    return _pipeline, _preprocessors


def _preprocess_inputs(person_img, cloth_category, preprocessors):
    """IDM-VTON에 필요한 마스크와 포즈 이미지를 생성."""
    from PIL import Image
    import numpy as np

    width, height = 768, 1024
    person_img = person_img.resize((width, height), Image.Resampling.LANCZOS)

    # 파싱 및 오픈포즈 전처리기가 준비되어 있는 경우
    if preprocessors and 'parsing' in preprocessors and 'openpose' in preprocessors:
        try:
            parsing_model = preprocessors['parsing']
            openpose_model = preprocessors['openpose']

            parsed_image, mask = parsing_model(person_img)
            pose_image = openpose_model(person_img)
            return person_img, mask, pose_image
        except Exception as e:
            print(f"[IDM-VTON] 전처리 자동화 실패, 기본 영역 마스크 생성: {e}")

    # Fallback: 전처리기가 없을 경우 기본 인페인팅용 단순 마스크/더미 포즈
    mask = Image.new("L", (width, height), 0)
    # 상의 영역 대략적 마스크
    mask_arr = np.array(mask)
    if cloth_category == 'lower_body':
        mask_arr[int(height * 0.45):int(height * 0.9), int(width * 0.15):int(width * 0.85)] = 255
    else:
        mask_arr[int(height * 0.15):int(height * 0.6), int(width * 0.15):int(width * 0.85)] = 255
    mask = Image.fromarray(mask_arr)
    pose_image = person_img.copy()

    return person_img, mask, pose_image


def try_on(person, garment, cloth_type='upper', steps=DEFAULT_STEPS, guidance_scale=DEFAULT_GUIDANCE,
           seed=DEFAULT_SEED, garment_photo_type='flat-lay', garment_des="a clothing item", return_timing=False):
    """인물 사진에 옷을 입힌다 (IDM-VTON).

    person/garment: 파일 경로 또는 PIL.Image
    cloth_type: CLOTH_TYPES 중 하나 ('upper', 'lower', 'overall', 등)
    steps: 추론 스텝 (기본 30)
    guidance_scale: guidance scale (기본 2.0)
    seed: -1이면 랜덤 시드
    return_timing: 실행 소요 시간 딕셔너리 반환 여부
    """
    if cloth_type not in CATEGORY_BY_CLOTH_TYPE:
        raise ValueError(f'cloth_type must be one of {CLOTH_TYPES}, got {cloth_type!r}')

    import torch
    from PIL import Image

    if seed == -1:
        import random
        seed = random.randint(0, 2 ** 31 - 1)

    pipe, preprocessors = load_models()
    start = time.perf_counter()

    person_img = _to_rgb(person)
    garment_img = _to_rgb(garment).resize((768, 1024), Image.Resampling.LANCZOS)

    category = CATEGORY_BY_CLOTH_TYPE[cloth_type]
    person_img, mask, pose_img = _preprocess_inputs(person_img, category, preprocessors)

    generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(int(seed))

    # IDM-VTON 파이프라인 호출
    prompt = f"model is wearing {garment_des}"
    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

    with torch.no_grad():
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=person_img,
            mask_image=mask,
            pose_img=pose_img,
            garm_img=garment_img,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
            height=1024,
            width=768,
        )

    result = output.images[0]

    timing = {
        'total_s': round(time.perf_counter() - start, 2),
        'seed': int(seed),
        'dtype': 'fp16' if torch.cuda.is_available() else 'fp32',
    }

    if not return_timing:
        return result
    return result, timing
