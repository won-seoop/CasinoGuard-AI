import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from recording.adaptive import (  # noqa: E402
    EventInterval,
    RecordingTier,
    active_stream_positions,
    clip_window_all_active,
    compute_tier_sequence,
    event_intervals_from_active_flags,
    plan_event_clips,
    should_keep_idle_frame,
    total_planned_frames,
)


def test_idle_when_no_person_and_no_event():
    tiers = compute_tier_sequence([False, False], [False, False], post_roll_frames=5)
    assert tiers == [RecordingTier.IDLE, RecordingTier.IDLE]


def test_normal_when_person_present_and_no_event():
    tiers = compute_tier_sequence([True, True], [False, False], post_roll_frames=5)
    assert tiers == [RecordingTier.NORMAL, RecordingTier.NORMAL]


def test_event_tier_while_active():
    tiers = compute_tier_sequence([True, True], [False, True], post_roll_frames=5)
    assert tiers == [RecordingTier.NORMAL, RecordingTier.EVENT]


def test_event_tier_holds_through_post_roll_then_drops():
    # frame 0: event active. post_roll_frames=2 -> frame 1,2도 EVENT 유지, frame 3부터 해제
    person_present = [True, True, True, True, False]
    event_active = [True, False, False, False, False]
    tiers = compute_tier_sequence(person_present, event_active, post_roll_frames=2)
    assert tiers == [
        RecordingTier.EVENT,
        RecordingTier.EVENT,
        RecordingTier.EVENT,
        RecordingTier.NORMAL,
        RecordingTier.IDLE,
    ]


def test_second_event_within_post_roll_extends_event_tier_without_gap():
    # frame 0 event, post_roll=3 -> frames 1~3 EVENT. frame 3에 새 event 발생 -> post_roll이 다시 3~6으로 연장
    person_present = [True] * 8
    event_active = [True, False, False, True, False, False, False, False]
    tiers = compute_tier_sequence(person_present, event_active, post_roll_frames=3)
    # frame index:            0      1      2      3      4      5      6      7
    expected = [
        RecordingTier.EVENT,  # 0: active
        RecordingTier.EVENT,  # 1: post-roll of event0
        RecordingTier.EVENT,  # 2: post-roll of event0
        RecordingTier.EVENT,  # 3: active (event1)
        RecordingTier.EVENT,  # 4: post-roll of event1
        RecordingTier.EVENT,  # 5: post-roll of event1
        RecordingTier.EVENT,  # 6: post-roll of event1
        RecordingTier.NORMAL,  # 7: post-roll expired
    ]
    assert tiers == expected


def test_compute_tier_sequence_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        compute_tier_sequence([True], [True, False], post_roll_frames=1)


@pytest.mark.parametrize(
    "frame_idx,stride,expected",
    [
        (0, 5, True),
        (1, 5, False),
        (4, 5, False),
        (5, 5, True),
        (7, 1, True),  # stride<=1 -> 항상 유지 (Frame Skip 없음)
        (7, 0, True),
    ],
)
def test_should_keep_idle_frame(frame_idx, stride, expected):
    assert should_keep_idle_frame(frame_idx, stride) is expected


def test_event_intervals_from_active_flags_extracts_contiguous_runs():
    flags = [False, True, True, False, False, True, False]
    intervals = event_intervals_from_active_flags(flags)
    assert intervals == [
        EventInterval(event_id=0, start_frame=1, end_frame=2),
        EventInterval(event_id=1, start_frame=5, end_frame=5),
    ]


def test_event_intervals_open_at_video_end_is_closed_at_last_frame():
    flags = [False, False, True, True]
    intervals = event_intervals_from_active_flags(flags)
    assert intervals == [EventInterval(event_id=0, start_frame=2, end_frame=3)]


def test_event_intervals_no_events_returns_empty():
    assert event_intervals_from_active_flags([False, False, False]) == []


def test_plan_event_clips_expands_pre_and_post_roll_and_clamps_to_bounds():
    events = [EventInterval(event_id=0, start_frame=10, end_frame=12)]
    clips = plan_event_clips(events, pre_roll_frames=5, post_roll_frames=5, total_frames=100)
    assert len(clips) == 1
    assert clips[0].start_frame == 5
    assert clips[0].end_frame == 17

    # 영상 시작/끝 근처 Event는 Pre/Post Roll이 잘려야 한다 (버그가 아니라 알려진 한계)
    edge_events = [EventInterval(event_id=0, start_frame=1, end_frame=1)]
    clips = plan_event_clips(edge_events, pre_roll_frames=5, post_roll_frames=5, total_frames=10)
    assert clips[0].start_frame == 0
    assert clips[0].end_frame == 6


def test_plan_event_clips_merges_close_events_alternative_b():
    # event0: [0,0] -> window [0,10] (post_roll=10). event1 시작(8)이 window 안에 들어옴 -> 병합
    events = [
        EventInterval(event_id=0, start_frame=0, end_frame=0),
        EventInterval(event_id=1, start_frame=8, end_frame=8),
    ]
    clips = plan_event_clips(events, pre_roll_frames=0, post_roll_frames=10, total_frames=100, merge_gap_frames=0)
    assert len(clips) == 1
    assert clips[0].event_ids == (0, 1)
    assert clips[0].start_frame == 0
    assert clips[0].end_frame == 18


def test_plan_event_clips_alternative_a_no_merge_keeps_overlap_duplicated():
    # 같은 두 Event를 병합하지 않으면(대안 A) Clip이 2개로 쪼개지고 총 프레임 수에
    # 겹치는 구간이 중복으로 카운트되어 실제 저장 공간 낭비를 만든다.
    events = [
        EventInterval(event_id=0, start_frame=0, end_frame=0),
        EventInterval(event_id=1, start_frame=8, end_frame=8),
    ]
    naive_clips = plan_event_clips(
        events, pre_roll_frames=0, post_roll_frames=10, total_frames=100, merge_gap_frames=-1_000_000
    )
    merged_clips = plan_event_clips(
        events, pre_roll_frames=0, post_roll_frames=10, total_frames=100, merge_gap_frames=0
    )
    assert len(naive_clips) == 2
    assert len(merged_clips) == 1
    # 대안 A는 겹치는 구간([10,15]구간이 두 Clip에 모두 포함)을 중복 저장하므로
    # 병합했을 때보다 총 저장 프레임 수가 더 많다.
    assert total_planned_frames(naive_clips) > total_planned_frames(merged_clips)


def test_plan_event_clips_no_events_returns_empty():
    assert plan_event_clips([], pre_roll_frames=5, post_roll_frames=5, total_frames=100) == []


def test_active_stream_positions_skips_idle_frames():
    tiers = [
        RecordingTier.IDLE,
        RecordingTier.IDLE,
        RecordingTier.NORMAL,
        RecordingTier.EVENT,
        RecordingTier.IDLE,
        RecordingTier.NORMAL,
    ]
    assert active_stream_positions(tiers) == [None, None, 0, 1, None, 2]


def test_active_stream_positions_all_active_is_identity():
    tiers = [RecordingTier.NORMAL, RecordingTier.EVENT, RecordingTier.NORMAL]
    assert active_stream_positions(tiers) == [0, 1, 2]


def test_active_stream_positions_all_idle_is_all_none():
    tiers = [RecordingTier.IDLE, RecordingTier.IDLE]
    assert active_stream_positions(tiers) == [None, None]


def test_clip_window_all_active_true_when_no_idle_inside():
    tiers = [RecordingTier.NORMAL, RecordingTier.EVENT, RecordingTier.EVENT, RecordingTier.NORMAL]
    assert clip_window_all_active(tiers, 0, 3) is True
    assert clip_window_all_active(tiers, 1, 2) is True


def test_clip_window_all_active_false_when_preroll_dips_into_idle():
    # Pre-Roll이 IDLE 구간(index 0,1)까지 파고든 경우 -> Stream Copy만으로는 불완전
    tiers = [RecordingTier.IDLE, RecordingTier.IDLE, RecordingTier.NORMAL, RecordingTier.EVENT]
    assert clip_window_all_active(tiers, 0, 3) is False
    assert clip_window_all_active(tiers, 2, 3) is True
