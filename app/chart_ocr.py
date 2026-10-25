"""선택적 로컬 Tesseract OCR → 수정 가능한 상의 치수표 초안.

OCR 결과는 자동 확정하지 않는다. 인식 실패/엔진 미설치 때도 수동 JSON 입력 가능.
"""
import json
import os
import re
import shutil
import subprocess
import tempfile

from chart_extract import normalize_size_name


_SIZE = re.compile(r'^(?:[2-4]?X{0,3}[SML]|S|M|L|FREE|\d{2,3})$', re.I)
_NUMBER = re.compile(r'^\d+(?:\.\d+)?$')


def _tokens(line):
    line = line.replace('가슴 단면', '가슴단면').replace('가슴 둘레', '가슴둘레')
    line = line.replace('어깨 너비', '어깨너비').replace('옷 가슴둘레', '옷가슴둘레')
    return [part for part in re.split(r'[\s|,]+', line.strip()) if part]


def draft_from_text(text):
    """단순 행/열 치수표를 읽는다. 복잡한 표는 원문을 보여주고 수동 수정한다."""
    lines = [_tokens(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    for header_index, header in enumerate(lines):
        labels = [part.lower() for part in header]
        chest_positions = [i for i, key in enumerate(labels)
                           if '가슴' in key or key in ('흉위', 'chest')]
        if not chest_positions:
            continue
        # 행형: 사이즈 | 가슴단면 | 어깨너비
        if any(key in ('사이즈', 'size', '구분') for key in labels):
            chest_index = chest_positions[0]
            shoulder_index = next((i for i, key in enumerate(labels) if '어깨' in key), None)
            sizes = {}
            for row in lines[header_index + 1:]:
                if not row or not _SIZE.fullmatch(row[0]):
                    continue
                if len(row) <= chest_index or not _NUMBER.fullmatch(row[chest_index]):
                    continue
                cells = {header[chest_index]: float(row[chest_index])}
                if (shoulder_index is not None and len(row) > shoulder_index
                        and _NUMBER.fullmatch(row[shoulder_index])):
                    cells[header[shoulder_index]] = float(row[shoulder_index])
                sizes[normalize_size_name(row[0])] = cells
            if sizes:
                return {'name': 'OCR 상의 치수표', 'category': 'upper', 'unit': 'cm', 'sizes': sizes}
        # 열형: S M L / 가슴단면 49 52 55 (첫 행에는 가슴 라벨이 없으므로 아래 분기)
    for index, header in enumerate(lines):
        labels = [normalize_size_name(token) for token in header]
        size_columns = [(i, label) for i, label in enumerate(labels)
                        if _SIZE.fullmatch(header[i])]
        if len(size_columns) < 2:
            continue
        sizes = {size: {} for _, size in size_columns}
        for row in lines[index + 1:]:
            key = row[0].lower()
            if not ('가슴' in key or '어깨' in key or key in ('chest', 'shoulder')):
                continue
            for column, size in size_columns:
                if column < len(row) and _NUMBER.fullmatch(row[column]):
                    sizes[size][row[0]] = float(row[column])
        if sizes and all(any('가슴' in key or key == 'chest' for key in cells)
                         for cells in sizes.values()):
            return {'name': 'OCR 상의 치수표', 'category': 'upper', 'unit': 'cm', 'sizes': sizes}
    return None


def image_to_draft(image):
    """(편집용 JSON 문자열, OCR 원문, 상태)를 반환한다."""
    if image is None:
        return '', '', '치수표 이미지를 올려주세요.'
    executable = os.environ.get('TESSERACT_CMD') or shutil.which('tesseract')
    if not executable:
        return '', '', ('로컬 Tesseract OCR이 없습니다. Tesseract와 한국어 언어 데이터를 설치하거나 '
                        '아래 JSON 칸에 치수표를 직접 입력해주세요.')
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
            temp_path = temp.name
        image.save(temp_path)
        result = subprocess.run([executable, temp_path, 'stdout', '-l', 'kor+eng', '--psm', '6'],
                                capture_output=True, text=True, encoding='utf-8',
                                errors='replace', timeout=30, check=False)
        if result.returncode:
            return '', '', ('OCR 실패: ' + (result.stderr.strip() or '언어 데이터/실행 파일을 확인해주세요.'))
        raw_text = result.stdout.strip()
        draft = draft_from_text(raw_text)
        if draft is None:
            return '', raw_text, ('문자는 읽었지만 표 구조를 확정할 수 없습니다. 원본과 OCR 원문을 대조해 '
                                   '아래 JSON을 직접 입력해주세요.')
        return json.dumps(draft, ensure_ascii=False, indent=2), raw_text, (
            'OCR 초안입니다. 원본과 숫자·단면/둘레 라벨을 대조해 수정한 뒤 확인 체크를 해주세요.')
    except (OSError, subprocess.TimeoutExpired) as exc:
        return '', '', f'OCR 실행 실패: {exc}'
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
