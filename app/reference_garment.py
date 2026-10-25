"""잘 맞는 기존 상의의 옷 실측값. 몸 치수로 환산하지 않는다."""
import math
from dataclasses import dataclass
from typing import Dict


FIELDS = (
    ('length_cm', '총장'),
    ('shoulder_cm', '어깨너비'),
    ('chest_flat_cm', '가슴단면'),
)


@dataclass(frozen=True)
class ReferenceGarment:
    length_cm: float
    shoulder_cm: float
    chest_flat_cm: float

    def __post_init__(self):
        for field, label in FIELDS:
            raw = getattr(self, field)
            if raw is None or (isinstance(raw, str) and not raw.strip()):
                raise ValueError(f'{label}(cm)을 입력해주세요.')
            try:
                value = float(raw)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f'{label}는 cm 단위 숫자로 입력해주세요.') from exc
            if isinstance(raw, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{label}는 0보다 큰 유한한 숫자로 입력해주세요.')
            object.__setattr__(self, field, value)

    def to_chart_dimensions(self) -> Dict[str, float]:
        """새 옷과 비교할 항목명을 사용하되, 입력한 옷 치수를 그대로 보존한다."""
        return {'length': self.length_cm, 'shoulder': self.shoulder_cm,
                'chest': self.chest_flat_cm}

    def to_dict(self) -> dict:
        return {'source': 'manual_reference_garment', 'unit': 'cm',
                'measurement_type': 'garment', 'chest_measurement': 'flat_width',
                'dimensions': self.to_chart_dimensions()}
