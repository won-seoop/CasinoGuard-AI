"""EXP-017: Adaptive Recording (지침 23번).

Goal:
Core MVP 이후 남은 Stretch Goal 중 "Adaptive Recording"의 Baseline과 개선안을 실제로
구현하고, 저장 용량/CPU 사용량/Event Clip 완전성을 실측 비교한다.

- Baseline: 항상 30(가정 25) FPS High Quality로 모든 프레임을 녹화.
- Adaptive: No Person -> Low FPS(Frame Skip + 저해상도) / Person Detected -> Normal FPS /
  Security Event(Intrusion ROI ENTER~EXIT) -> High FPS·High Quality + Circular Buffer
  기반 Event Clip(-10s~+10s).

Blocker / Decision (EXP-010/014/015/016과 동일):
이 원격 세션은 egress 정책상 Wikimedia 등 외부 일반 도메인이 차단되어 있어(재확인 완료),
기존 crowded_intersection_1080p.webm을 다시 받을 수 없다. 대신 ultralytics 패키지 내장
실제 사진 bus.jpg(사람 4명 이상, 1080x810)에 Pan/Jitter를 적용해 아래 3-Phase 시나리오를
합성한다. Detection은 매 프레임 실제 YOLO11n 추론 결과를 사용한다.

3-Phase 합성 시나리오 (총 900프레임 @ 가정 25fps = 36초):
  Phase 1 [0,300)   IDLE   : 사람이 없는 배경(하늘/지붕 영역)만 촬영 (카메라가 다른 곳을 봄)
  Phase 2 [300,600) NORMAL : bus.jpg 전체 + Pan/Jitter (사람 있음, ROI 밖에서 서성임)
  Phase 3 [600,900) EVENT  : 동일 + 추가 수평 이동으로 사람 무리가 ROI(중앙 대역 x=[430,620])를
                             두 번 들락날락 -> Intrusion ENTER/EXIT이 두 번 발생, 두 사건
                             사이 간격(~3초)이 Post-Roll(10초)보다 짧아 Event Clip이 하나로
                             병합되는지 검증한다.

Failure Case (실제 발견): ROI를 처음에는 "오른쪽 절반(x>405)"으로 설계했으나, bus.jpg에는
원래 x≈734 지점에 사람이 서 있어 Phase 2(의도: 사람 있음 + Event 없음)부터 이미 ROI 안에
있는 것으로 판정되어 NORMAL Tier가 단 한 프레임도 나오지 않는 문제를 실측으로 발견했다
(자세한 원인/대안은 experiment.md Failure Cases 참고). ROI를 아무도 서 있지 않은 중앙
대역(x=[430,620])으로 재설계하고, 그 대역으로 사람 무리를 밀어 넣는 진폭으로 다시 만든
시나리오가 지금 이 스크립트다.
"""

from __future__ import annotations

import csv
import json
import os
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from events.roi_state_machine import ROIStateMachine, bottom_center  # noqa: E402
from recording.adaptive import (  # noqa: E402
    AdaptiveRecordingConfig,
    RecordingTier,
    compute_tier_sequence,
    event_intervals_from_active_flags,
    plan_event_clips,
    should_keep_idle_frame,
    total_planned_frames,
)

PERSON_CLASS_ID = 0
ASSET_CANDIDATES = [
    Path(sys.prefix) / "lib/python3.11/dist-packages/ultralytics/assets/bus.jpg",
    Path(sys.prefix) / "lib/python3.11/site-packages/ultralytics/assets/bus.jpg",
]

N_FRAMES = 900
ASSUMED_FPS = 25.0
PHASE_IDLE_END = 300
PHASE_NORMAL_END = 600
IDLE_STRIDE = 5  # Low FPS: 5프레임 중 1개만 저장 (5fps 상당)
IDLE_SCALE = 0.5  # 저해상도
ROI_X_MIN, ROI_X_MAX = 430, 620  # bus.jpg 실측: 아무도 서 있지 않은 중앙 대역 (아래 event_phase_extra_dx 주석 참고)

CONFIG = AdaptiveRecordingConfig(
    fps=ASSUMED_FPS, pre_roll_sec=10.0, post_roll_sec=10.0, idle_frame_stride=IDLE_STRIDE, merge_gap_sec=0.0
)

OUT_DIR = ROOT / "results" / "EXP-017"
VIDEO_DIR = OUT_DIR / "videos"


def find_asset() -> Path:
    for p in ASSET_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(f"ultralytics 샘플 이미지를 찾을 수 없음: {ASSET_CANDIDATES}")


def make_no_person_background(base_img: np.ndarray, frame_idx: int) -> np.ndarray:
    """Phase 1(IDLE): 사람이 없는 하늘/지붕 영역만 타일링한 배경 (카메라가 다른 곳을 보는 상황)."""
    h, w = base_img.shape[:2]
    strip_h = max(1, int(h * 0.15))
    strip = base_img[0:strip_h, :]
    reps = int(np.ceil(h / strip_h))
    tiled = np.tile(strip, (reps, 1, 1))[:h]
    dx = int(6 * np.sin(frame_idx * 0.05))
    frame = np.roll(tiled, shift=dx, axis=1)
    return frame


def make_person_frame(base_img: np.ndarray, frame_idx: int, extra_dx: int) -> np.ndarray:
    """Phase 2/3: EXP-010/014/016과 동일한 Pan(진폭 12px)/밝기 Jitter + 시나리오용 추가 수평 이동."""
    max_shift = 12
    dx = int(max_shift * np.sin(frame_idx * 0.037)) + extra_dx
    dy = int(max_shift * np.cos(frame_idx * 0.053))
    shifted = np.roll(base_img, shift=(dy, dx), axis=(0, 1))
    brightness = 1.0 + 0.03 * np.sin(frame_idx * 0.011)
    frame = np.clip(shifted.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    return frame


EXTRA_DX_AMPLITUDE = 300
EXTRA_DX_PERIOD = 150  # 프레임, 약 6초 @ 25fps


def event_phase_extra_dx(local_idx: int) -> int:
    """Phase 3 전용: 사람 무리를 두 번 ROI(중앙 대역)로 밀어넣었다 빼는 수평 이동량.
    period=150 프레임짜리 사인파 -> 300프레임 구간에 약 2주기 -> ENTER/EXIT 2쌍 유도.

    최초 시도(ROI=오른쪽 절반, x>405)는 실패했다: bus.jpg에는 원래 x=734 부근에 사람이
    서 있어 Phase 2(사람 있음, Event 없음으로 의도)부터 이미 ROI 안에 있었다(FC 후보,
    experiment.md Failure Cases 참고). 대신 아무도 서 있지 않은 x=[430,620] 중앙 대역을
    ROI로 잡고, x~280 부근 사람 무리만 그 대역으로 밀어 넣도록 진폭을 재설계했다.
    """
    wave = 0.5 + 0.5 * np.sin(2 * np.pi * local_idx / EXTRA_DX_PERIOD - np.pi / 2)  # 0~1, local_idx=0에서 0
    return int(wave * EXTRA_DX_AMPLITUDE)


def make_frame(base_img: np.ndarray, frame_idx: int) -> np.ndarray:
    if frame_idx < PHASE_IDLE_END:
        return make_no_person_background(base_img, frame_idx)
    if frame_idx < PHASE_NORMAL_END:
        return make_person_frame(base_img, frame_idx, extra_dx=0)
    local_idx = frame_idx - PHASE_NORMAL_END
    extra_dx = event_phase_extra_dx(local_idx)
    return make_person_frame(base_img, frame_idx, extra_dx=extra_dx)


def run_detection_and_events(base_img: np.ndarray) -> dict:
    """Pass 1: 실제 YOLO11n+ByteTrack 추론 + Intrusion ROI 판정. 프레임은 저장하지 않는다
    (Detection 결과만 있으면 되므로 메모리를 아낀다 — Pass 2에서 make_frame()으로 재생성)."""
    model = YOLO("yolo11n.pt")
    h, w = base_img.shape[:2]
    roi_polygon = [(ROI_X_MIN, 0), (ROI_X_MAX, 0), (ROI_X_MAX, h), (ROI_X_MIN, h)]
    roi = ROIStateMachine(polygon=roi_polygon)

    person_present: list[bool] = []
    event_active: list[bool] = []
    roi_events: list[dict] = []
    last_seen: dict[int, int] = {}

    for idx in range(N_FRAMES):
        frame = make_frame(base_img, idx)
        result = model.track(
            frame, persist=True, tracker="bytetrack.yaml", classes=[PERSON_CLASS_ID], conf=0.4, imgsz=640, verbose=False
        )[0]
        active: list[tuple[int, tuple[float, float, float, float]]] = []
        if result.boxes is not None and result.boxes.id is not None:
            ids = [int(i) for i in result.boxes.id.tolist()]
            boxes = result.boxes.xyxy.tolist()
            active = [(tid, tuple(b)) for tid, b in zip(ids, boxes)]

        for tid, box in active:
            ev = roi.update(tid, box, frame_idx=idx)
            last_seen[tid] = idx
            if ev is not None:
                roi_events.append({"track_id": ev.track_id, "event_type": ev.event_type, "frame_idx": ev.frame_idx})

        stale_ids = [tid for tid, last in last_seen.items() if idx - last > 10]
        for tid in stale_ids:
            ev = roi.forget_track(tid, frame_idx=idx)
            del last_seen[tid]
            if ev is not None:
                roi_events.append({"track_id": ev.track_id, "event_type": ev.event_type, "frame_idx": ev.frame_idx})

        person_present.append(len(active) > 0)
        event_active.append(any(state == "INSIDE" for state in roi.states.values()))

    for tid in list(last_seen.keys()):
        ev = roi.forget_track(tid, frame_idx=N_FRAMES - 1)
        if ev is not None:
            roi_events.append({"track_id": ev.track_id, "event_type": ev.event_type, "frame_idx": ev.frame_idx})

    return {
        "frame_w": w,
        "frame_h": h,
        "person_present": person_present,
        "event_active": event_active,
        "roi_events": roi_events,
    }


def cpu_time_sec() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


def write_baseline(base_img: np.ndarray, frame_w: int, frame_h: int) -> dict:
    path = VIDEO_DIR / "baseline_always_high_quality.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), ASSUMED_FPS, (frame_w, frame_h))
    cpu_start, wall_start = cpu_time_sec(), time.perf_counter()
    frames_written = 0
    for idx in range(N_FRAMES):
        frame = make_frame(base_img, idx)
        writer.write(frame)
        frames_written += 1
    writer.release()
    return {
        "path": str(path.relative_to(ROOT)),
        "frames_written": frames_written,
        "file_size_bytes": os.path.getsize(path),
        "cpu_time_sec": cpu_time_sec() - cpu_start,
        "wall_time_sec": time.perf_counter() - wall_start,
    }


def write_adaptive(
    base_img: np.ndarray,
    frame_w: int,
    frame_h: int,
    tiers: list[RecordingTier],
    clip_plans: list,
) -> dict:
    idle_size = (max(1, int(frame_w * IDLE_SCALE)), max(1, int(frame_h * IDLE_SCALE)))
    idle_path = VIDEO_DIR / "adaptive_idle_low_fps.mp4"
    active_path = VIDEO_DIR / "adaptive_active_stream.mp4"
    idle_writer = cv2.VideoWriter(str(idle_path), cv2.VideoWriter_fourcc(*"mp4v"), ASSUMED_FPS / IDLE_STRIDE, idle_size)
    active_writer = cv2.VideoWriter(str(active_path), cv2.VideoWriter_fourcc(*"mp4v"), ASSUMED_FPS, (frame_w, frame_h))

    clip_writers: dict[int, cv2.VideoWriter] = {}
    clip_frame_counts: dict[int, int] = {}
    for plan in clip_plans:
        clip_path = VIDEO_DIR / f"event_clip_{plan.clip_id}.mp4"
        clip_writers[plan.clip_id] = cv2.VideoWriter(
            str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), ASSUMED_FPS, (frame_w, frame_h)
        )
        clip_frame_counts[plan.clip_id] = 0

    idle_frames_written = 0
    active_frames_written = 0

    cpu_start, wall_start = cpu_time_sec(), time.perf_counter()
    for idx in range(N_FRAMES):
        frame = make_frame(base_img, idx)
        tier = tiers[idx]
        if tier == RecordingTier.IDLE:
            if should_keep_idle_frame(idx, IDLE_STRIDE):
                small = cv2.resize(frame, idle_size, interpolation=cv2.INTER_AREA)
                idle_writer.write(small)
                idle_frames_written += 1
        else:
            active_writer.write(frame)
            active_frames_written += 1

        for plan in clip_plans:
            if plan.start_frame <= idx <= plan.end_frame:
                clip_writers[plan.clip_id].write(frame)
                clip_frame_counts[plan.clip_id] += 1
    adaptive_cpu_time = cpu_time_sec() - cpu_start
    adaptive_wall_time = time.perf_counter() - wall_start

    idle_writer.release()
    active_writer.release()
    for w in clip_writers.values():
        w.release()

    clip_results = []
    for plan in clip_plans:
        p = VIDEO_DIR / f"event_clip_{plan.clip_id}.mp4"
        expected_frames = plan.end_frame - plan.start_frame + 1
        actual_frames = clip_frame_counts[plan.clip_id]
        clip_results.append(
            {
                "clip_id": plan.clip_id,
                "event_ids": list(plan.event_ids),
                "start_frame": plan.start_frame,
                "end_frame": plan.end_frame,
                "expected_frames": expected_frames,
                "actual_frames_written": actual_frames,
                "frames_missing": expected_frames - actual_frames,
                "file_size_bytes": os.path.getsize(p),
            }
        )

    return {
        "idle_stream": {
            "path": str(idle_path.relative_to(ROOT)),
            "frames_written": idle_frames_written,
            "file_size_bytes": os.path.getsize(idle_path),
        },
        "active_stream": {
            "path": str(active_path.relative_to(ROOT)),
            "frames_written": active_frames_written,
            "file_size_bytes": os.path.getsize(active_path),
        },
        "event_clips": clip_results,
        "cpu_time_sec": adaptive_cpu_time,
        "wall_time_sec": adaptive_wall_time,
    }


def run() -> dict:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    base_img = cv2.imread(str(find_asset()))

    detection = run_detection_and_events(base_img)
    frame_w, frame_h = detection["frame_w"], detection["frame_h"]
    person_present = detection["person_present"]
    event_active = detection["event_active"]

    tiers = compute_tier_sequence(person_present, event_active, post_roll_frames=CONFIG.post_roll_frames)
    tier_counts = {t.value: tiers.count(t) for t in RecordingTier}

    event_intervals = event_intervals_from_active_flags(event_active)
    clips_merged = plan_event_clips(
        event_intervals,
        pre_roll_frames=CONFIG.pre_roll_frames,
        post_roll_frames=CONFIG.post_roll_frames,
        total_frames=N_FRAMES,
        merge_gap_frames=CONFIG.merge_gap_frames,
    )
    clips_naive_no_merge = plan_event_clips(
        event_intervals,
        pre_roll_frames=CONFIG.pre_roll_frames,
        post_roll_frames=CONFIG.post_roll_frames,
        total_frames=N_FRAMES,
        merge_gap_frames=-1_000_000,
    )

    baseline_result = write_baseline(base_img, frame_w, frame_h)
    adaptive_result = write_adaptive(base_img, frame_w, frame_h, tiers, clips_merged)

    adaptive_continuous_bytes = adaptive_result["idle_stream"]["file_size_bytes"] + adaptive_result["active_stream"][
        "file_size_bytes"
    ]
    adaptive_with_clips_bytes = adaptive_continuous_bytes + sum(c["file_size_bytes"] for c in adaptive_result["event_clips"])
    baseline_bytes = baseline_result["file_size_bytes"]

    summary = {
        "n_frames": N_FRAMES,
        "assumed_fps": ASSUMED_FPS,
        "frame_w": frame_w,
        "frame_h": frame_h,
        "config": {
            "pre_roll_frames": CONFIG.pre_roll_frames,
            "post_roll_frames": CONFIG.post_roll_frames,
            "idle_frame_stride": IDLE_STRIDE,
            "idle_scale": IDLE_SCALE,
        },
        "phase_layout": {"idle": [0, PHASE_IDLE_END], "normal": [PHASE_IDLE_END, PHASE_NORMAL_END], "event": [PHASE_NORMAL_END, N_FRAMES]},
        "person_present_frame_count": sum(person_present),
        "event_active_frame_count": sum(event_active),
        "tier_counts": tier_counts,
        "roi_events_raw": detection["roi_events"],
        "event_intervals": [
            {"event_id": e.event_id, "start_frame": e.start_frame, "end_frame": e.end_frame} for e in event_intervals
        ],
        "baseline": baseline_result,
        "adaptive": adaptive_result,
        "storage": {
            "baseline_bytes": baseline_bytes,
            "adaptive_continuous_bytes": adaptive_continuous_bytes,
            "adaptive_with_event_clips_bytes": adaptive_with_clips_bytes,
            "storage_saving_pct_continuous_only": (
                100.0 * (1 - adaptive_continuous_bytes / baseline_bytes) if baseline_bytes else 0.0
            ),
            "storage_saving_pct_with_clips": (
                100.0 * (1 - adaptive_with_clips_bytes / baseline_bytes) if baseline_bytes else 0.0
            ),
        },
        "file_write_count": {
            "baseline": baseline_result["frames_written"],
            "adaptive_idle": adaptive_result["idle_stream"]["frames_written"],
            "adaptive_active": adaptive_result["active_stream"]["frames_written"],
            "adaptive_total": adaptive_result["idle_stream"]["frames_written"] + adaptive_result["active_stream"]["frames_written"],
        },
        "cpu_usage_sec": {
            "baseline": baseline_result["cpu_time_sec"],
            "adaptive": adaptive_result["cpu_time_sec"],
        },
        "event_clip_completeness": {
            "clips": adaptive_result["event_clips"],
            "any_frames_missing": any(c["frames_missing"] > 0 for c in adaptive_result["event_clips"]),
        },
        "event_clip_merge_comparison": {
            "alternative_a_no_merge": {
                "clip_count": len(clips_naive_no_merge),
                "total_planned_frames": total_planned_frames(clips_naive_no_merge),
            },
            "alternative_b_merge_adjacent": {
                "clip_count": len(clips_merged),
                "total_planned_frames": total_planned_frames(clips_merged),
            },
        },
    }
    return summary


if __name__ == "__main__":
    result = run()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(result, f, indent=2)

    with open(OUT_DIR / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "baseline", "adaptive"])
        writer.writerow(["file_size_bytes", result["storage"]["baseline_bytes"], result["storage"]["adaptive_continuous_bytes"]])
        writer.writerow(["frames_written", result["file_write_count"]["baseline"], result["file_write_count"]["adaptive_total"]])
        writer.writerow(["cpu_time_sec", result["cpu_usage_sec"]["baseline"], result["cpu_usage_sec"]["adaptive"]])
        writer.writerow(["storage_saving_pct_continuous_only", "-", result["storage"]["storage_saving_pct_continuous_only"]])
        writer.writerow(["storage_saving_pct_with_clips", "-", result["storage"]["storage_saving_pct_with_clips"]])

    print(json.dumps({k: v for k, v in result.items() if k not in ("roi_events_raw",)}, indent=2))
    print("saved to", OUT_DIR)
