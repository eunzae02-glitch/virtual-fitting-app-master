# FitCheck — 사이즈 정보를 반영한 가상 피팅 앱

종합설계프로젝트(캡스톤). 사용자 사진과 옷 사진을 합성해 입은 모습을 보여주고,
옷 치수표와 입력한 신체 치수를 비교해 S/M/L을 추천한다.

모델은 **직접 구동(self-hosted)** 한다. 외부 합성 API는 쓰지 않는다.

> - 전체 진행 현황·실험 결과·판단 근거: **[docs/PLAN.md](docs/PLAN.md)**
> - 환경 함정 모음: **[docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)**

## 현재 확인 경로 — FASHN 사이즈별 합성

[FASHN 사이즈 합성 노트북을 Colab에서 열기](https://colab.research.google.com/github/eunzae02-glitch/virtual-fitting-app-master/blob/main/notebooks/debug_size_fashn.ipynb)

PC 변경사항을 Commit·Push한 뒤 위 노트북에서 GPU(T4)를 선택하고 셀을 순서대로 실행한다.
기존 `/content/fashn-weights`를 재사용한다. 현재 DM-VTON 실험은 보류하고 이 FASHN 경로를 확인한다.

1. 기준 옷과 새 옷 S/M/L의 총장·어깨너비·가슴단면을 입력하고 **치수 연결**을 누른다.
2. 기준 옷 착용 사진을 올려 5점, 같은 옷 여부, 총장·가슴단면 측정 기준을 확인한다.
3. **목표 윤곽 보기** 후 새 옷 상품 사진으로 **FASHN 기본 합성 생성**을 한 번 실행한다.
4. 초록색 옷 영역을 확인·수정하고 확인 체크를 켜면 **S/M/L 갤러리가 자동 생성**된다.
   판매 사이즈를 바꾸면 현재 선택 결과도 자동 갱신되며 FASHN을 다시 실행하지 않는다.

비교 대상은 **현재 선택 사이즈 · 치수 적용 결과**와 판매 사이즈별 갤러리다.
`치수 적용 전 원본`은 접힌 영역에 보관한다. `00_ME` 파일 없이 직접 올린 사진으로 진행할 수 있다.
보고서에는 선택 사이즈, 목표 밑단·몸통 폭(px), 실제 변경 화소가 표시된다. 저장 버튼은 선택
사이즈의 조정 PNG·마스크와 전체 사이즈의 좌표·해시 요약을 보존한다.

로컬 CPU 변형 확인: `python -X utf8 scripts/debug_size_preview.py --ui`.
기존 포트를 쓰려면 `--port 7867`, Colab 링크는 `--share`를 추가한다.
이미 만든 기본 합성 PNG를 불러오면 재합성 없이 비교할 수 있다.

FASHN 기본 합성 후 총장·몸통 폭을 2D 변형하는 실험이다. 어깨는 변형하지 않으며,
축소 복원·무늬·굴곡·실제 착용 핏의 품질은 실제 사진으로 검증해야 한다.
원인 조사와 수정 검증은 [SIZE_SYNTHESIS_DEBUG.md](docs/SIZE_SYNTHESIS_DEBUG.md)를 참고한다.

## 실험 기록 — DM-VTON (보류)

[DM-VTON 사이즈 비교를 Colab에서 열기](https://colab.research.google.com/github/eunzae02-glitch/virtual-fitting-app-master/blob/main/notebooks/debug_dmvton.ipynb)

PC 변경사항을 GitHub에 Commit·Push한 뒤 위 노트북을 연다. GPU(T4) 런타임에서 셀을 순서대로
실행하고 마지막 `gradio.live` 링크를 연다. 화면은 다음 네 영역으로 구성된다.

1. 치수를 알고 있는 기존 옷을 입은 전신 사진
2. 기존 옷의 총장·어깨너비·가슴단면(cm)
3. 새 옷의 판매 사이즈별 총장·어깨너비·가슴단면(cm)
4. 새 옷 상품 사진 업로드와 DM-VTON 기본 합성·사이즈별 비교 결과

`00_ME`, 수동 기준점, 마스크 편집, 어깨 구조 선택은 이 화면에 없다. 새 옷 사진은 옷 한 벌만
있는 상품 사진이나 투명 PNG를 사용한다. 자동 옷 영역과 상반신 크롭으로 192×256 입력을 만들고,
같은 사진에서는 기본 합성을 재사용해 총장·몸통 폭 비율만 다시 적용한다.
얼굴·팔 등 분할된 보호 영역과 편집 영역 밖 사진은 후처리에서 유지한다.
자동 분할 오류·가려진 신체 복원·옷 굴곡·소매 구조·실제 착용 핏은 검증되지 않았으며,
어깨 수치는 보관하되 어깨 변형은 적용하지 않는다.

설치: `python -m pip install -r requirements_dmvton.txt`.
준비: `python scripts/setup_dmvton.py`.
실행: `python -X utf8 scripts/debug_dmvton.py --ui` (코랩에서는 `--share` 추가).
포트는 자동 선택한다. `DMVTON_REPO`·`DMVTON_WEIGHTS` 환경변수로 경로를 지정할 수 있다.

공식 [DM-VTON](https://github.com/KiseKloset/DM-VTON)의 고정 소스 버전과 추론 가중치 두 개(약 113MB)를
사용한다. 자동 영역 추출에는 FASHN Human Parser 캐시(첫 준비 약 244MB)를 사용하며 FASHN 합성 모델은
로드하지 않는다. 구형 CuPy 설치 대신 원본 correlation의 7×7 오프셋·채널 평균·zero padding을
PyTorch로 계산한다. **공식 CUDA 벤치마크와 다른 경로이므로 23.3ms를 보장하지 않으며**, 화면에서
모델 추론 시간과 전체 처리 시간을 따로 확인한다. 로컬 테스트는 연산 정의·좌표·보호 영역·캐시와
가중치 없는 모델 구조 구동을 확인했다. 공식 가중치의 실제 T4 합성과 품질 평가는 코랩에서 진행한다.
기존 FASHN·윤곽·마스크 디버거도 유지한다.

## 사진 두 장으로 상의 합성만 확인

[합성 확인 노트북을 Colab에서 열기](https://colab.research.google.com/github/eunzae02-glitch/virtual-fitting-app-master/blob/main/notebooks/debug_synthesis.ipynb)

GPU 런타임을 선택하고 셀을 순서대로 실행한 뒤 `gradio.live` 링크를 연다.
인물 사진과 상의 상품 사진을 올려 **합성하기**를 누른다. `00_ME` 사진이나 기준점 입력은 필요 없다.
고급 설정에서 시드·스텝·guidance를 바꿀 수 있고, 화면에 모델 준비·합성 단계와 소요 시간을 표시한다.
같은 서버에서 반복 합성하면 로드한 FASHN 모델을 재사용한다. 결과 사진에서 PNG를 내려받을 수 있다.

기존 코랩의 설치·가중치 셀을 이미 실행했다면 마지막 실행 셀만 아래로 바꿔도 된다.
PC 변경사항을 GitHub에 Push한 뒤 코랩의 코드 받기 셀을 다시 실행해야 새 파일을 받는다.

```python
os.environ['FASHN_REPO'] = FASHN
os.environ['FASHN_WEIGHTS'] = WEIGHTS
!python -u {APP}/scripts/debug_synthesis.py --share
```

로컬 화면 실행: `python -X utf8 scripts/debug_synthesis.py --ui`.
화면은 CPU에서도 열 수 있지만 실제 합성에는 FASHN 설치·가중치·CUDA GPU가 필요하다.
포트를 생략하면 비어 있는 포트를 자동 선택한다. 이 화면은 **상의 기본 합성 확인용**이며,
치수·목표선·마스크를 이용한 사이즈 조정은 기존 `scripts/debug_size_preview.py --ui`에서 확인한다.

## 팀원용 빠른 시작 — 데모 보기

[![Colab에서 열기](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhjung03eee/virtual-fitting-app/blob/master/notebooks/run_app.ipynb)

1. 위 배지를 눌러 Colab에서 `notebooks/run_app.ipynb` 를 연다
2. **런타임 → 런타임 유형 변경 → GPU (T4)**
3. 셀을 위에서부터 실행 → 마지막 셀 출력의 **공개 링크(`*.gradio.live`)** 를 연다

최초 1회 설치·가중치 내려받기에 약 5분. 합성은 T4 기준 한 장 약 2분(품질 우선 설정).
사진은 **정면 · 팔은 몸에서 주먹 하나 간격 · 머리부터 발끝까지 화면 가득** 이어야 한다
(앱이 올리는 순간 검사해서 안내한다).

## 진행 상황

- [x] 합성 모델 구동·비교: CatVTON → HR-VITON → 하이브리드 → **FASHN VTON v1.5 채택**, FitDiT 비교 후 보류
- [x] 환경 구성 문제 전부 규명·문서화 → `docs/ENVIRONMENT.md`
- [x] 속도/품질 실험: 정밀도·스텝·guidance 실측, Kaggle T4 2장 병렬 실험 커널
- [x] 옷 치수표 대비 여유분 계산 → S/M/L 추천 (사용자 입력 방식, 단위 테스트)
- [x] 사진 자세·거리 검사(`app/photo_check.py`)를 앱에 연결
- [x] **앱에 FASHN 연결 + 실제 GPU에서 합성·사이즈 추천·서버 기동 검증** (docs/PLAN.md F-17)
- [ ] 촬영 가이드대로 찍은 새 전신 사진으로 **하의(바지)** 확인 (F-12)
- [ ] 고성능 GPU(A100 등)에서 속도 측정, 시연 환경 확정
- [ ] 모바일 앱 UI + 추론 서버 분리

## 모델 구성

| 역할 | 사용 |
|---|---|
| 가상 피팅 합성 (**현재**) | [FASHN VTON v1.5](https://github.com/fashn-AI/fashn-vton-1.5) (Apache-2.0, 마스크 없이 픽셀 공간 생성) |
| 기본 설정 | fp16 · 50스텝 · guidance 2.5 (`app/fashn_core.py`, 근거는 docs/PLAN.md F-11·F-13·F-14) |
| 가상 피팅 합성 (이전) | [CatVTON](https://github.com/Zheng-Chong/CatVTON) — `app/tryon_core.py` 에 남아 있음(벤치마크·과거 실험 재현용) |
| 사진 자세·거리 검사 | MediaPipe Pose (`app/photo_check.py`) |
| 사이즈 추천 | 치수표 대비 여유분 계산 (`app/size_fit.py`, `app/body_profile.py`) |

## 구성

```
app/fashn_core.py          합성 엔진 (FASHN). 앱·실험이 공유하는 기본값
app/gradio_app.py          웹 데모 화면 (합성 + 사진 검사 + 사이즈 추천)
app/photo_check.py         인물 사진 자세·거리·역광 검사 (GPU 불필요)
app/size_fit.py            치수표 대비 여유분 계산 → S/M/L 추천
app/tryon_core.py          CatVTON 추론 (과거 실험 재현용)
notebooks/run_app.ipynb    Colab에서 데모 띄우기
kaggle_*/                  Kaggle 커널로 돌린 실험 기록 (모델 비교·속도·품질)
docs/PLAN.md               진행 현황과 모든 실험 결과·판단 근거
data/samples/results/      실험 결과 이미지 (커밋 제외)
```

## CatVTON(이전 엔진)을 돌려야 할 때

과거 실험을 재현할 때만 필요하다. CatVTON 저장소에 **cp39 전용으로 컴파일된 detectron2 바이너리**가
들어 있어 **Python 3.9 venv** 에서만 돌아간다(`scripts/setup_env.py` 가 만든다). 자세한 이유는
`docs/ENVIRONMENT.md`. 현재 앱(FASHN)은 반대로 Python 3.10 이상이 필요하므로 두 엔진은 같이 못 쓴다.

## 주의

- GPU는 T4(16GB) 권장. FASHN 가중치는 약 2GB(최초 1회 자동 다운로드), 추론에 fp16 기준 3.2GB 사용.
- 사람 얼굴 사진(`data/person/`)과 쇼핑몰 상품 사진(`data/samples/`)은 커밋하지 않는다(.gitignore).
