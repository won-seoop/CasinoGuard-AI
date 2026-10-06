"""Attribute Metadata를 실시간 파이프라인에 연결하는 호출 지점 (EXP-029, 지침 19).

EXP-028 Next Action: c_wb(Segmentation Mask + White Balance, classify_person_attributes_with_mask_and_wb)를
Track당 1회(BestShot 확정 시점)만 호출하는 방식으로 run_full_pipeline.py에 연결하고 실제 FPS/P95
Latency 영향을 재측정한다. Segmentation 모델(yolo11n-seg.pt) 호출 자체는 이 모듈에서 하지 않고
(모델 의존성을 분리해 Unit Test에서는 가짜 mask_provider로 대체 가능하게 함) 호출부가 주입한
`mask_provider`를 통해서만 접근한다.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from attributes.color import classify_person_attributes_with_mask_and_wb

MaskProvider = Callable[[np.ndarray], np.ndarray]


def classify_track_attributes_once(crop_bgr: np.ndarray, mask_provider: MaskProvider) -> dict[str, str]:
    """BestShot이 확정된 crop 1장에 대해서만 Segmentation Mask + White Balance 기반 색상
    분류(c_wb)를 수행한다. mask_provider가 crop_bgr -> bool mask(동일 HxW)를 반환한다.

    crop_bgr이 비어 있으면(0x0 등) Segmentation 모델을 호출할 필요 없이 바로 fallback
    색상("gray")을 반환한다 - classify_person_attributes_with_mask_and_wb 내부에서도
    처리되지만, 빈 crop으로 mask_provider(실제 모델 추론)를 호출하는 낭비를 피한다.
    """
    if crop_bgr.size == 0:
        return {"upper": "gray", "lower": "gray"}
    mask = mask_provider(crop_bgr)
    return classify_person_attributes_with_mask_and_wb(crop_bgr, mask)
