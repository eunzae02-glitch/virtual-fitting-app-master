"""전처리 엑셀에서 성별 평균·Ridge·Gradient Boosting을 비교하고 결과를 저장한다.

    python -X utf8 scripts/train_body_model.py

각 실행은 outputs/body_model/<실행시각>/에 별도로 남긴다. 원본 엑셀은 수정하지 않는다.
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

APP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app')
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from body_model import (DEFAULT_DATA, DEFAULT_OUTPUT, MODEL_NAMES, SEX_NAMES, TARGETS,
                        load_dataset, report_markdown, train_models)  # noqa: E402


def save_run(directory, dataset, bundle, splits, predictions):
    import joblib

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    report = bundle['report']
    joblib.dump(bundle, directory / 'model.joblib', compress=3)
    (directory / 'report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    (directory / 'report.md').write_text(report_markdown(report), encoding='utf-8')
    with (directory / 'metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report['metrics'][0]))
        writer.writeheader()
        for metric in report['metrics']:
            writer.writerow({**metric, 'parameters': json.dumps(metric['parameters'])})
    with (directory / 'split_membership.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['human_id', 'sex', 'split'])
        for name, indices in splits.items():
            writer.writerows((dataset.ids[index], dataset.sexes[index], name) for index in indices)
    with (directory / 'test_predictions.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)


def main():
    parser = argparse.ArgumentParser(description='신체 치수 모델 학습·성능 비교 (CPU)')
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--sheet', default='전처리')
    parser.add_argument('--header-row', type=int, default=2, help='엑셀의 실제 헤더 행 번호')
    parser.add_argument('--output', type=Path, default=None, help='새 실행 폴더. 기존 폴더를 덮어쓰지 않음')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.1, help='예측 구간 목표 미포함률 (기본 0.1)')
    args = parser.parse_args()
    if args.header_row < 1 or args.folds < 2 or not 0 < args.alpha < 1:
        parser.error('헤더 행은 1 이상, folds는 2 이상, alpha는 0~1 사이여야 합니다.')
    started = datetime.now(timezone.utc)
    directory = args.output or DEFAULT_OUTPUT / started.strftime('%Y%m%dT%H%M%S_%fZ')
    if directory.exists():
        parser.error(f'이미 있는 실행 폴더입니다: {directory}')
    try:
        dataset = load_dataset(args.data, args.sheet, args.header_row)
        print(f"유효 {len(dataset.ids)}명 / 제외 {dataset.audit['dropped_rows']}명 / "
              f"성별 {dataset.audit['sex_counts']}", flush=True)
        elapsed = time.perf_counter()
        bundle, splits, predictions = train_models(dataset, args.seed, args.folds, args.alpha,
                                                   progress=lambda message: print(message, flush=True))
        bundle['report']['started_at_utc'] = started.isoformat()
        bundle['report']['training_seconds'] = round(time.perf_counter() - elapsed, 2)
        save_run(directory, dataset, bundle, splits, predictions)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f'학습 중단: {exc}\n')
    print('\nCV 기준 선택 모델과 최종 평가 결과:', flush=True)
    for row in bundle['report']['metrics']:
        if row['selected']:
            print(f"{SEX_NAMES[row['sex']]} / {TARGETS[row['target']]} / {MODEL_NAMES[row['model']]}: "
                  f"MAE {row['test_mae_cm']:.2f}cm, 구간 ±{row['interval_radius_cm']:.2f}cm, "
                  f"포함률 {row['test_coverage']:.1%}", flush=True)
    print(f'결과 폴더: {directory.resolve()}', flush=True)


if __name__ == '__main__':
    main()
