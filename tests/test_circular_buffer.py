import pytest

from recording.circular_buffer import FrameCircularBuffer, required_capacity_for_pre_roll


def test_push_and_get_range_within_capacity():
    buf = FrameCircularBuffer(capacity_frames=5)
    for i in range(5):
        buf.push(i, f"frame{i}")
    assert len(buf) == 5
    assert buf.get_range(0, 4) == ["frame0", "frame1", "frame2", "frame3", "frame4"]


def test_push_beyond_capacity_evicts_oldest():
    buf = FrameCircularBuffer(capacity_frames=3)
    for i in range(5):
        buf.push(i, f"frame{i}")
    assert len(buf) == 3
    cov = buf.coverage(0, 4)
    assert cov.missing_frames == (0, 1)
    assert not cov.is_complete


def test_coverage_true_when_all_frames_present():
    buf = FrameCircularBuffer(capacity_frames=10)
    for i in range(10):
        buf.push(i, i)
    cov = buf.coverage(2, 7)
    assert cov.is_complete
    assert cov.missing_frames == ()


def test_coverage_false_when_frame_not_yet_pushed():
    buf = FrameCircularBuffer(capacity_frames=10)
    buf.push(0, "a")
    buf.push(1, "b")
    cov = buf.coverage(0, 5)
    assert not cov.is_complete
    assert cov.missing_frames == (2, 3, 4, 5)


def test_get_range_raises_on_missing_frames_never_returns_partial():
    buf = FrameCircularBuffer(capacity_frames=3)
    for i in range(5):
        buf.push(i, i)
    with pytest.raises(KeyError):
        buf.get_range(0, 4)


def test_push_duplicate_frame_idx_raises():
    buf = FrameCircularBuffer(capacity_frames=3)
    buf.push(0, "a")
    with pytest.raises(ValueError):
        buf.push(0, "b")


def test_invalid_capacity_raises():
    with pytest.raises(ValueError):
        FrameCircularBuffer(capacity_frames=0)
    with pytest.raises(ValueError):
        FrameCircularBuffer(capacity_frames=-1)


def test_memory_bytes_scales_with_frames_actually_held():
    buf = FrameCircularBuffer(capacity_frames=4)
    for i in range(2):
        buf.push(i, i)
    assert buf.memory_bytes(frame_nbytes=1000) == 2000
    for i in range(2, 6):
        buf.push(i, i)
    # capacity=4이므로 최대 4프레임 분량만 계산되어야 한다
    assert buf.memory_bytes(frame_nbytes=1000) == 4000


def test_total_bytes_sums_variable_size_frames_individually():
    buf = FrameCircularBuffer(capacity_frames=4)
    buf.push(0, b"a")
    buf.push(1, b"bbb")
    buf.push(2, b"cc")
    assert buf.total_bytes(len) == 1 + 3 + 2


def test_total_bytes_reflects_eviction_beyond_capacity():
    buf = FrameCircularBuffer(capacity_frames=2)
    buf.push(0, b"aaaaa")  # evicted once frame 2 is pushed
    buf.push(1, b"bb")
    buf.push(2, b"c")
    assert buf.total_bytes(len) == 2 + 1


def test_required_capacity_for_pre_roll_is_off_by_one_aware():
    # 트리거 프레임 자신 포함 pre_roll_frames+1 프레임이 필요하다
    assert required_capacity_for_pre_roll(pre_roll_frames=250) == 251
    assert required_capacity_for_pre_roll(pre_roll_frames=250, safety_margin_frames=50) == 301


def test_required_capacity_for_pre_roll_rejects_negative_inputs():
    with pytest.raises(ValueError):
        required_capacity_for_pre_roll(pre_roll_frames=-1)
    with pytest.raises(ValueError):
        required_capacity_for_pre_roll(pre_roll_frames=10, safety_margin_frames=-1)


def test_capacity_exactly_pre_roll_frames_misses_trigger_frame_itself():
    """capacity를 pre_roll_frames와 '같게'(흔한 실수) 설정하면, 트리거 프레임 시점에
    가장 오래된 프레임 1개가 이미 evict되어 정확히 1프레임이 모자란다."""
    pre_roll_frames = 10
    trigger_frame = 100
    buf = FrameCircularBuffer(capacity_frames=pre_roll_frames)  # off-by-one 실수 재현
    for i in range(trigger_frame + 1):
        buf.push(i, i)
    cov = buf.coverage(trigger_frame - pre_roll_frames, trigger_frame)
    assert not cov.is_complete
    assert cov.missing_frames == (trigger_frame - pre_roll_frames,)


def test_capacity_with_required_helper_covers_trigger_frame_exactly():
    pre_roll_frames = 10
    trigger_frame = 100
    capacity = required_capacity_for_pre_roll(pre_roll_frames)
    buf = FrameCircularBuffer(capacity_frames=capacity)
    for i in range(trigger_frame + 1):
        buf.push(i, i)
    cov = buf.coverage(trigger_frame - pre_roll_frames, trigger_frame)
    assert cov.is_complete
