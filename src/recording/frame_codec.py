"""Circular Buffer에 저장할 프레임을 압축하는 Codec (FC-009 해결).

EXP-020/FC-009: `FrameCircularBuffer`가 디코드된 원본 BGR ndarray를 그대로 보관하면
Pre-Roll 10초만으로 채널당 628~753MB가 필요해(810x1080 기준) 다채널 카지노 CCTV에는
비현실적이다. `FrameCircularBuffer`는 이미 Generic(`FrameT`)이므로 자료구조 자체를
바꿀 필요는 없다 — 대신 버퍼에 넣기 "직전"에 프레임을 JPEG로 인코딩하고, 버퍼에서
꺼낸 "직후"에 디코드하면 된다. 이 모듈은 그 인코딩/디코딩 경계만 담당한다.
"""

from __future__ import annotations

import cv2
import numpy as np


def encode_frame(frame: np.ndarray, quality: int = 85) -> bytes:
    """BGR ndarray를 JPEG 바이트로 인코딩한다.

    quality(0~100)가 낮을수록 버퍼 메모리 사용량은 줄지만 디코드 후 화질 손실이 커진다 —
    이 Trade-off는 자료구조가 아니라 호출자가 quality 값으로 직접 선택한다.
    """
    if not (0 <= quality <= 100):
        raise ValueError("quality must be in [0, 100]")
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("cv2.imencode failed to encode frame as JPEG")
    return encoded.tobytes()


def decode_frame(data: bytes) -> np.ndarray:
    """JPEG 바이트를 BGR ndarray로 디코드한다. 손상된 데이터는 조용히 삼키지 않고 예외를 던진다."""
    buf = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("cv2.imdecode failed to decode JPEG bytes (corrupt or empty data)")
    return frame
