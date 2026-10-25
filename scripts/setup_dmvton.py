"""DM-VTON 공식 소스와 추론용 가중치 두 개만 준비한다. GPU 모델은 로드하지 않는다."""
import argparse
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'app'))

from dmvton_core import UPSTREAM_COMMIT  # noqa: E402
from paths import DMVTON_REPO, DMVTON_WEIGHTS  # noqa: E402


# 공식 README의 체크포인트 폴더에서 확인한 parser-free student 추론 가중치.
# https://drive.google.com/drive/folders/1wfWGsR0vWC5LrA26xhj92ec_GoCKV80A
CHECKPOINTS = {
    'dmvton_pf_warp.pt': '16H16AhGJrqndDrAaHWZvTFzGSsaDecpS',
    'dmvton_pf_gen.pt': '1i-Ple5L9__LV_Fbx1cZyxiOC-CRFxi0C',
}


def prepare(repo, weights, download=True):
    repo, weights = Path(repo).resolve(), Path(weights).resolve()
    fresh = not repo.exists()
    if fresh:
        subprocess.run(['git', 'clone', '--depth', '1',
                        'https://github.com/KiseKloset/DM-VTON.git', str(repo)], check=True)
    if not (repo / '.git').is_dir():
        raise RuntimeError(f'DM-VTON 경로가 Git 저장소가 아닙니다: {repo}')
    head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if fresh and head != UPSTREAM_COMMIT:
        subprocess.run(['git', '-C', str(repo), 'fetch', '--depth', '1', 'origin', UPSTREAM_COMMIT], check=True)
        subprocess.run(['git', '-C', str(repo), 'checkout', '--detach', UPSTREAM_COMMIT], check=True)
        head = UPSTREAM_COMMIT
    if head != UPSTREAM_COMMIT:
        raise RuntimeError(f'DM-VTON 소스 버전이 다릅니다. 별도 경로에 {UPSTREAM_COMMIT} 버전을 준비해주세요.')
    if download:
        import gdown

        weights.mkdir(parents=True, exist_ok=True)
        for name, identifier in CHECKPOINTS.items():
            target = weights / name
            if target.is_file() and zipfile.is_zipfile(target):
                print(f'기존 가중치 사용: {name}', flush=True)
                continue
            temporary = target.with_suffix('.pt.part')
            result = gdown.download(id=identifier, output=str(temporary), quiet=False)
            if result is None or not zipfile.is_zipfile(temporary):
                raise RuntimeError(f'{name} 다운로드를 확인해주세요. Google Drive 제한일 수 있습니다.')
            temporary.replace(target)
        from fashn_human_parser import FashnHumanParser

        print('자동 옷 영역 분할 모델 캐시 준비 (첫 실행 약 244MB)', flush=True)
        FashnHumanParser(device='cpu')
    print(f'DMVTON_REPO={repo}\nDMVTON_WEIGHTS={weights}', flush=True)


def main():
    parser = argparse.ArgumentParser(description='DM-VTON 소스·가중치 준비 (약 113MB + 분할 모델 캐시)')
    parser.add_argument('--repo', default=DMVTON_REPO)
    parser.add_argument('--weights', default=DMVTON_WEIGHTS)
    parser.add_argument('--skip-weights', action='store_true')
    args = parser.parse_args()
    prepare(args.repo, args.weights, not args.skip_weights)


if __name__ == '__main__':
    main()
