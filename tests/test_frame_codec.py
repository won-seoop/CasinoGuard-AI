import numpy as np
import pytest

from recording.adaptive import RecordingTier, TierQualityConfig
from recording.circular_buffer import FrameCircularBuffer
from recording.frame_codec import decode_frame, encode_frame


def _gradient_frame(h: int = 64, w: int = 96) -> np.ndarray:
    """decode_frame으로 shape/의미 있는 픽셀 변화를 검증할 수 있는 합성 프레임."""
    x = np.linspace(0, 255, w, dtype=np.uint8)
    y = np.linspace(0, 255, h, dtype=np.uint8)
    xv, yv = np.meshgrid(x, y)
    return np.stack([xv, yv, np.full_like(xv, 128)], axis=-1).astype(np.uint8)


def test_encode_decode_roundtrip_preserves_shape_and_dtype():
    frame = _gradient_frame()
    encoded = encode_frame(frame, quality=90)
    decoded = decode_frame(encoded)
    assert decoded.shape == frame.shape
    assert decoded.dtype == frame.dtype


def test_encode_decode_roundtrip_is_visually_close_at_high_quality():
    frame = _gradient_frame()
    decoded = decode_frame(encode_frame(frame, quality=95))
    # JPEG는 손실 압축이므로 완전히 동일하지는 않지만, 높은 quality에서는 평균 오차가 작아야 한다.
    mean_abs_error = np.abs(frame.astype(np.int16) - decoded.astype(np.int16)).mean()
    assert mean_abs_error < 5.0


def test_higher_quality_never_produces_dramatically_larger_bytes_than_raw():
    frame = _gradient_frame()
    encoded = encode_frame(frame, quality=95)
    assert len(encoded) < frame.nbytes


def test_lower_quality_reduces_or_keeps_encoded_size():
    frame = _gradient_frame()
    size_high = len(encode_frame(frame, quality=95))
    size_low = len(encode_frame(frame, quality=30))
    assert size_low <= size_high


def test_encode_rejects_out_of_range_quality():
    frame = _gradient_frame()
    with pytest.raises(ValueError):
        encode_frame(frame, quality=101)
    with pytest.raises(ValueError):
        encode_frame(frame, quality=-1)


def test_decode_rejects_corrupt_bytes():
    with pytest.raises(ValueError):
        decode_frame(b"not a real jpeg")


def test_circular_buffer_of_encoded_bytes_roundtrips_through_get_range():
    """FrameCircularBuffer[bytes]로 써도 기존 자료구조를 전혀 바꾸지 않고 그대로 재사용 가능함을 확인."""
    buf: FrameCircularBuffer[bytes] = FrameCircularBuffer(capacity_frames=5)
    base = _gradient_frame(h=32, w=32)
    frames = [np.roll(base, shift=i, axis=1) for i in range(5)]
    for i, f in enumerate(frames):
        buf.push(i, encode_frame(f, quality=90))
    recovered = [decode_frame(b) for b in buf.get_range(0, 4)]
    for original, back in zip(frames, recovered):
        assert back.shape == original.shape
        assert np.abs(original.astype(np.int16) - back.astype(np.int16)).mean() < 5.0


def test_circular_buffer_with_tier_differentiated_quality_keeps_event_frame_fidelity():
    """EXP-022: IDLE Tier는 낮은 quality로, EVENT Tier는 높은 quality로 같은 버퍼에 섞어
    넣어도(quality_for_tier), 완전성(coverage)과는 무관하게 EVENT 프레임의 화질이 IDLE
    프레임보다 항상 원본에 더 가깝게 복원되어야 한다 — Tier 차등의 핵심 목적이 실제로
    지켜지는지 회귀로 고정한다."""
    tq = TierQualityConfig(idle_quality=20, normal_quality=70, event_quality=95)
    tiers = [RecordingTier.IDLE, RecordingTier.IDLE, RecordingTier.NORMAL, RecordingTier.EVENT]
    base = _gradient_frame(h=48, w=48)
    frames = [np.roll(base, shift=i * 3, axis=1) for i in range(len(tiers))]

    buf: FrameCircularBuffer[bytes] = FrameCircularBuffer(capacity_frames=4)
    for i, (frame, tier) in enumerate(zip(frames, tiers)):
        buf.push(i, encode_frame(frame, quality=tq.quality_for_tier(tier)))

    cov = buf.coverage(0, 3)
    assert cov.is_complete
    decoded = [decode_frame(b) for b in buf.get_range(0, 3)]

    def mae(a, b):
        return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())

    idle_mae = mae(frames[0], decoded[0])
    event_mae = mae(frames[3], decoded[3])
    assert event_mae < idle_mae
