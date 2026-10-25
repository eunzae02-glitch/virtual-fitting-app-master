"""가중치가 준비된 GPU 환경에서 기본 합성을 생성한다. 네트워크 다운로드는 하지 않는다.

python -X utf8 scripts/run_size_synthesis.py --person person.png --garment garment.png --output outputs/base_run
"""
import argparse
import json
import os
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/'app'))


def main():
    parser = argparse.ArgumentParser(description='설치된 FASHN으로 새 옷 기본 합성 생성 (GPU·오프라인)')
    parser.add_argument('--person', required=True)
    parser.add_argument('--garment', required=True)
    parser.add_argument('--output', required=True, help='새 출력 폴더. 기존 폴더는 덮어쓰지 않음')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--guidance', type=float, default=2.5)
    args = parser.parse_args()
    # huggingface_hub가 임포트되기 전에 설정해 파서의 누락된 가중치 다운로드도 막는다.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from size_synthesis import bind_base, engine_settings, fashn_environment
    from garment_outline import prepare_photo

    settings = engine_settings(args.seed, args.steps, args.guidance)
    ready, message = fashn_environment()
    if not ready:
        parser.exit(1, message+'\n')
    destination = Path(args.output).resolve()
    if destination.exists():
        parser.exit(1, f'기존 폴더를 덮어쓰지 않습니다: {destination}\n')
    with Image.open(args.person) as image:
        person = prepare_photo(image)
    with Image.open(args.garment) as image:
        garment = prepare_photo(image)
    from fashn_core import try_on

    result, timing = try_on(person, garment, **settings, return_timing=True)
    result, metadata = bind_base(person, result, 'fashn', garment, settings, timing)
    destination.mkdir(parents=True, exist_ok=False)
    result.save(destination/'base.png')
    (destination/'base.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2,
                                                 allow_nan=False), encoding='utf-8')
    print(destination/'base.png')


if __name__ == '__main__':
    main()
