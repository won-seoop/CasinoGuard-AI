"""EXP-020: Adaptive Recording Circular Buffer 실제 구현 및 검증 (지침 23).

Roadmap(01. Architecture & Roadmap)과 EXP-019/PAR-010 Next Action:
"Hybrid Event Clip 생성 방식이 IDLE 겹침 구간의 원본 프레임을 결정론적 함수로
'재생성'했는데, 실제 카메라는 지나간 원본 프레임을 재생성할 수 없다 — 지침 23의
Circular Buffer(최근 N초 고화질 원본 프레임을 Tier와 무관하게 항상 보관)를 실제로
구현해 '재생성'을 '버퍼에서 꺼내기'로 바꿔야 한다."

이번 실험은 그 Circular Buffer(`src/recording/circular_buffer.py`)를 실제로 만들고,
EXP-019와 동일한 시나리오(1,600프레임, IDLE 75%, Pre-Roll이 IDLE Tier까지 파고드는
경계 사례)에 적용해서:

1. Circular Buffer로 실제로 만든 Hybrid Clip이 EXP-019의 "결정론적 재생성" Hybrid와
   완전성/CPU 비용 면에서 동등한지 확인한다 (버퍼 자체가 정확히 동작하는지 검증).
2. 버퍼 용량(capacity)을 얼마로 잡아야 하는지는 자명하지 않다 — capacity를
   "pre_roll_frames와 똑같이"(흔히 할 법한 실수) 잡으면 정확히 1프레임이 모자라는
   Off-by-One이 실제로 발생하는지 실측한다.
3. Edge Camera처럼 메모리 예산이 빠듯해 Pre-Roll 전체를 담을 용량조차 확보하지 못하면
   어떤 일이 벌어지는지(그리고 그것을 조용히 삼키지 않고 명시적으로 탐지하는지) 확인한다.
4. 각 capacity가 실제로 차지하는 메모리(바이트)를 측정해 Pre-Roll 길이와 메모리 비용의
   Trade-off를 기록한다.

Detection(YOLO11n+ByteTrack+ROI State Machine) 결과는 EXP-019가 이미 실제로 추론해
캐시해 둔 `results/EXP-019/detection_cache.json`을 그대로 재사용한다 — 이 실험은
Detection/Tracking 로직이 아니라 Recording/Circular Buffer 계층만 다루므로, 같은
입력을 다시 추론하는 것은 낭비다(EXP-019 스크립트 자체도 같은 캐시 재사용 패턴을 쓴다).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import run_exp019_event_clip_copy as exp019  # noqa: E402
from recording.adaptive import (  # noqa: E402
    RecordingTier,
    active_stream_positions,
    clip_window_all_active,
    compute_tier_sequence,
    event_intervals_from_active_flags,
    plan_event_clips,
)
from recording.circular_buffer import FrameCircularBuffer, required_capacity_for_pre_roll  # noqa: E402

FFMPEG = exp019.FFMPEG
CONFIG = exp019.CONFIG
N_FRAMES = exp019.N_FRAMES
ASSUMED_FPS = exp019.ASSUMED_FPS

OUT_DIR = ROOT / "results" / "EXP-020"
VIDEO_DIR = OUT_DIR / "videos"
GOP = 25  # EXP-019 Decision: 연속 녹화 저장량과 절단 오차의 절충점으로 채택된 기본값

# 실험할 버퍼 용량 시나리오. pre_roll_frames=250 (지침 23, 10초 @ 25fps).
CAPACITY_SCENARIOS = {
    "undersized_edge_budget": 64,  # Edge Camera 메모리 예산이 빠듯한 극단적 과소 설정
    "off_by_one_naive": None,  # placeholder, run()에서 pre_roll_frames로 채움
    "exact_minimum": None,  # placeholder, required_capacity_for_pre_roll(pre_roll_frames)
    "production_with_margin": None,  # placeholder, exact_minimum + 2초(fps*2) 안전마진
}


def cpu_time_sec() -> float:
    return exp019.cpu_time_sec()


def run() -> dict:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    base_img = cv2.imread(str(exp019.find_asset()))

    cache_path = ROOT / "results" / "EXP-019" / "detection_cache.json"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"EXP-019 detection cache가 없음: {cache_path}. "
            "먼저 scripts/run_exp019_event_clip_copy.py를 한 번 실행해 캐시를 만들어야 한다."
        )
    detection = json.loads(cache_path.read_text())
    frame_w, frame_h = detection["frame_w"], detection["frame_h"]
    person_present = detection["person_present"]
    event_active = detection["event_active"]
    frame_nbytes = frame_w * frame_h * 3  # BGR uint8

    tiers = compute_tier_sequence(person_present, event_active, post_roll_frames=CONFIG.post_roll_frames)
    positions = active_stream_positions(tiers)
    event_intervals = event_intervals_from_active_flags(event_active)
    clips = plan_event_clips(
        event_intervals,
        pre_roll_frames=CONFIG.pre_roll_frames,
        post_roll_frames=CONFIG.post_roll_frames,
        total_frames=N_FRAMES,
        merge_gap_frames=CONFIG.merge_gap_frames,
    )
    assert len(clips) == 1
    clip = clips[0]
    assert not clip_window_all_active(tiers, clip.start_frame, clip.end_frame), (
        "이 실험은 EXP-019와 동일하게 Pre-Roll이 IDLE Tier까지 파고드는 경계 사례를 전제로 한다"
    )

    idle_end_in_window = clip.start_frame - 1
    for i in range(clip.start_frame, clip.end_frame + 1):
        if tiers[i] == RecordingTier.IDLE:
            idle_end_in_window = i
        else:
            break
    idle_prefix_len = idle_end_in_window - clip.start_frame + 1
    trigger_frame = event_intervals[0].start_frame  # 첫 Event가 트리거된, 버퍼 추출이 일어나는 순간

    pre_roll_frames = CONFIG.pre_roll_frames
    exact_minimum = required_capacity_for_pre_roll(pre_roll_frames)
    production_capacity = required_capacity_for_pre_roll(pre_roll_frames, safety_margin_frames=round(ASSUMED_FPS * 2))
    scenarios = {
        "undersized_edge_budget": 64,
        "off_by_one_naive": pre_roll_frames,
        "exact_minimum": exact_minimum,
        "production_with_margin": production_capacity,
    }

    size = (frame_w, frame_h)
    results: dict = {
        "n_frames": N_FRAMES,
        "assumed_fps": ASSUMED_FPS,
        "frame_w": frame_w,
        "frame_h": frame_h,
        "frame_nbytes": frame_nbytes,
        "pre_roll_frames": pre_roll_frames,
        "clip_plan": {"start_frame": clip.start_frame, "end_frame": clip.end_frame, "n_frames": clip.end_frame - clip.start_frame + 1},
        "idle_prefix_frames_needed": idle_prefix_len,
        "trigger_frame": trigger_frame,
        "required_capacity_exact_minimum": exact_minimum,
        "required_capacity_production_with_margin": production_capacity,
        "scenarios": {},
    }

    # Active Stream + Naive Stream Copy 파일은 EXP-019와 동일한 로직으로 한 번만 만든다
    # (Circular Buffer는 IDLE 겹침 구간 조달 방식만 바꾸는 것이지, Active Stream/Stream
    # Copy 절반은 EXP-019에서 이미 검증된 그대로 재사용한다).
    active_frames = [exp019.make_frame(base_img, idx) for idx in range(N_FRAMES) if tiers[idx] != RecordingTier.IDLE]
    active_path = VIDEO_DIR / "active_stream.mp4"
    exp019.ffmpeg_encode_frames(active_frames, ASSUMED_FPS, size, GOP, active_path)
    del active_frames

    active_start_pos = positions[idle_end_in_window + 1]
    active_end_pos = positions[clip.end_frame]
    copy_start_sec = active_start_pos / ASSUMED_FPS
    copy_duration_sec = (active_end_pos - active_start_pos + 1) / ASSUMED_FPS
    naive_copy_path = VIDEO_DIR / "clip_active_portion_copy.mp4"
    exp019.ffmpeg_stream_copy(active_path, copy_start_sec, copy_duration_sec, naive_copy_path)
    active_portion_frames = active_end_pos - active_start_pos + 1

    expected_total_frames = idle_prefix_len + active_portion_frames

    for name, capacity in scenarios.items():
        buf: FrameCircularBuffer = FrameCircularBuffer(capacity_frames=capacity)
        cpu0, wall0 = cpu_time_sec(), time.perf_counter()
        for idx in range(trigger_frame + 1):
            frame = exp019.make_frame(base_img, idx)
            buf.push(idx, frame)
        push_cpu = cpu_time_sec() - cpu0
        push_wall = time.perf_counter() - wall0

        cov = buf.coverage(clip.start_frame, idle_end_in_window)
        scenario_result: dict = {
            "capacity_frames": capacity,
            "memory_bytes": buf.memory_bytes(frame_nbytes),
            "memory_mb": round(buf.memory_bytes(frame_nbytes) / 1024 / 1024, 2),
            "push_cpu_time_sec": push_cpu,
            "push_wall_time_sec": push_wall,
            "coverage_complete": cov.is_complete,
            "missing_frame_count": len(cov.missing_frames),
            "missing_frames_sample": list(cov.missing_frames[:5]),
        }

        if cov.is_complete:
            idle_prefix_frames = buf.get_range(clip.start_frame, idle_end_in_window)
            scenario_dir = VIDEO_DIR / name
            scenario_dir.mkdir(parents=True, exist_ok=True)
            idle_part_path = scenario_dir / "idle_part_from_buffer.mp4"
            idle_part_stats = exp019.ffmpeg_encode_frames(idle_prefix_frames, ASSUMED_FPS, size, GOP, idle_part_path)
            hybrid_path = scenario_dir / "clip_hybrid_from_buffer.mp4"
            concat_stats = exp019.ffmpeg_concat([idle_part_path, naive_copy_path], hybrid_path)
            actual_frames = exp019.ffprobe_frame_count(hybrid_path, size)
            scenario_result["clip_build"] = {
                "idle_part_encode_cpu_sec": idle_part_stats["cpu_time_sec"],
                "concat_cpu_sec": concat_stats["cpu_time_sec"],
                "total_clip_cpu_sec": idle_part_stats["cpu_time_sec"] + concat_stats["cpu_time_sec"],
                "file_size_bytes": os.path.getsize(hybrid_path),
                "expected_frames": expected_total_frames,
                "actual_frames": actual_frames,
                "frame_diff": actual_frames - expected_total_frames,
                "is_complete_clip": actual_frames == expected_total_frames,
            }
        else:
            scenario_result["clip_build"] = None
            scenario_result["outcome"] = (
                f"버퍼가 {len(cov.missing_frames)}개 프레임을 이미 evict함 — "
                "get_range()가 KeyError를 던져 Hybrid 조립을 명시적으로 중단시킨다 "
                "(EXP-019 Naive Stream Copy처럼 조용히 불완전한 파일을 만들지 않음)."
            )
            try:
                buf.get_range(clip.start_frame, idle_end_in_window)
                raise AssertionError("get_range()가 예외를 던지지 않음 — 예상과 다름")
            except KeyError as e:
                scenario_result["get_range_error"] = str(e)

        results["scenarios"][name] = scenario_result

    return results


if __name__ == "__main__":
    summary = run()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
