"""EXP-019: Adaptive Recording Event Clip 저장 방식 재검증 (지침 23, 27).

Roadmap(01. Architecture & Roadmap) 다음 Action:
"Adaptive Recording의 Next Action(더 긴 영상으로 재검증, Event Clip 재인코딩 대신 복사
방식 검토)을 이어서 진행".

EXP-017에서 발견한 문제: Event Clip을 Active Stream과 별개로 원본 프레임부터 다시
cv2.VideoWriter로 재인코딩하면, 이미 Active Stream에 기록된 프레임을 두 번 인코딩하는
셈이라 "Event Clip 포함" 총 저장량이 Baseline보다 오히려 47.7% 늘어났다(반직관적 결과,
정직하게 기록됨). Roadmap Next Action에서 제안한 대로, 재인코딩 대신 FFmpeg Stream
Copy(-c copy, 이미 인코딩된 바이트를 그대로 오려내는 방식)를 검토한다.

이번 실험에서 실제로 구현/검증한 것:
1. 더 긴 영상(1,600프레임=64초, EXP-017의 900프레임=36초보다 78% 김)으로 재검증.
   IDLE 구간을 훨씬 길게(1,200프레임, 전체의 75%) 만들어 "대부분 비어 있다가 갑자기
   사람이 들어와 곧바로 ROI를 침범하는" 더 현실적인 카지노/통로 카메라 시나리오를 재현.
2. 이 시나리오에서 Event Clip의 Pre-Roll(10초=250프레임)이 IDLE Tier 구간까지 파고드는
   새로운 경계 사례(EXP-017에는 없었던 케이스)를 실측으로 발견 — Active Stream은 IDLE
   프레임을 아예 저장하지 않으므로, Stream Copy만으로는 Pre-Roll 일부가 원본 화질로
   복구되지 않는다(Naive Copy는 결과물이 불완전함).
3. 세 가지 Event Clip 생성 방식을 실제로 구현하고 파일 크기/CPU 시간/프레임 정확도를 비교:
   (A) 전체 재인코딩 (EXP-017과 동일한 방식, libx264로 통일해 공정 비교)
   (B) Naive Stream Copy (Active Stream 전체를 그대로 오려냄, IDLE 겹침 무시 -> 불완전)
   (C) Hybrid: IDLE에 걸친 부분만 원본 프레임에서 재인코딩 + 나머지는 Active Stream에서
       Stream Copy, 두 조각을 FFmpeg concat demuxer로 합침 (완전 + 저비용)
4. Stream Copy는 컨테이너의 Keyframe(GOP) 경계에서만 정확히 잘리므로, Active Stream을
   인코딩할 때 GOP 크기(-g)를 다르게 하면 잘림 정확도와 파일 크기가 어떻게 trade-off
   되는지 GOP=1(전부 Keyframe)과 GOP=25(1초당 1개)로 비교 측정한다.

Blocker(EXP-010/014~018과 동일): 이 원격 세션은 egress 정책상 Wikimedia 등 외부 일반
도메인이 차단되어 있어(재확인 완료), ultralytics 내장 bus.jpg에 Pan/Jitter를 적용한
합성 시나리오를 계속 사용한다. Detection은 매 프레임 실제 YOLO11n 추론 결과다.
이번 실험은 imageio-ffmpeg(PyPI)로 받은 정적 FFmpeg 7.0.2 바이너리(libx264 포함)를
사용한다 — 지침 27(FFmpeg 사용 경험)을 실제로 충족한다.
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import imageio_ffmpeg  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from events.roi_state_machine import ROIStateMachine  # noqa: E402
from recording.adaptive import (  # noqa: E402
    AdaptiveRecordingConfig,
    RecordingTier,
    active_stream_positions,
    clip_window_all_active,
    compute_tier_sequence,
    event_intervals_from_active_flags,
    plan_event_clips,
)

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
PERSON_CLASS_ID = 0

N_FRAMES = 1600
ASSUMED_FPS = 25.0
PHASE_IDLE_END = 1200      # [0,1200) IDLE: 75% - 대부분 비어 있는 통로 카메라
PHASE_NORMAL_END = 1300    # [1200,1300) NORMAL: 사람이 잠깐 등장 (100프레임 = Pre-Roll 250보다 짧음)
# [1300,1600) EVENT-triggering: EXP-017 Phase 3과 동일한 진폭/주기로 ROI(x=[430,620])를 침범
ROI_X_MIN, ROI_X_MAX = 430, 620
EXTRA_DX_AMPLITUDE = 300
EXTRA_DX_PERIOD = 150

CONFIG = AdaptiveRecordingConfig(fps=ASSUMED_FPS, pre_roll_sec=10.0, post_roll_sec=10.0, idle_frame_stride=5, merge_gap_sec=0.0)

OUT_DIR = ROOT / "results" / "EXP-019"
VIDEO_DIR = OUT_DIR / "videos"

ASSET_CANDIDATES = [
    Path(sys.prefix) / "lib/python3.11/site-packages/ultralytics/assets/bus.jpg",
    Path(sys.prefix) / "lib/python3.11/dist-packages/ultralytics/assets/bus.jpg",
]


def find_asset() -> Path:
    for p in ASSET_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(f"ultralytics 샘플 이미지를 찾을 수 없음: {ASSET_CANDIDATES}")


def make_no_person_background(base_img: np.ndarray, frame_idx: int) -> np.ndarray:
    h, w = base_img.shape[:2]
    strip_h = max(1, int(h * 0.15))
    strip = base_img[0:strip_h, :]
    reps = int(np.ceil(h / strip_h))
    tiled = np.tile(strip, (reps, 1, 1))[:h]
    dx = int(6 * np.sin(frame_idx * 0.05))
    return np.roll(tiled, shift=dx, axis=1)


def make_person_frame(base_img: np.ndarray, frame_idx: int, extra_dx: int) -> np.ndarray:
    max_shift = 12
    dx = int(max_shift * np.sin(frame_idx * 0.037)) + extra_dx
    dy = int(max_shift * np.cos(frame_idx * 0.053))
    shifted = np.roll(base_img, shift=(dy, dx), axis=(0, 1))
    brightness = 1.0 + 0.03 * np.sin(frame_idx * 0.011)
    return np.clip(shifted.astype(np.float32) * brightness, 0, 255).astype(np.uint8)


def event_phase_extra_dx(local_idx: int) -> int:
    wave = 0.5 + 0.5 * np.sin(2 * np.pi * local_idx / EXTRA_DX_PERIOD - np.pi / 2)
    return int(wave * EXTRA_DX_AMPLITUDE)


def make_frame(base_img: np.ndarray, frame_idx: int) -> np.ndarray:
    if frame_idx < PHASE_IDLE_END:
        return make_no_person_background(base_img, frame_idx)
    if frame_idx < PHASE_NORMAL_END:
        return make_person_frame(base_img, frame_idx, extra_dx=0)
    local_idx = frame_idx - PHASE_NORMAL_END
    return make_person_frame(base_img, frame_idx, extra_dx=event_phase_extra_dx(local_idx))


def cpu_time_sec() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


def run_detection(base_img: np.ndarray) -> dict:
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

    return {"frame_w": w, "frame_h": h, "person_present": person_present, "event_active": event_active, "roi_events": roi_events}


# ---------------------------------------------------------------------------
# FFmpeg helpers (raw frame pipe encode / stream copy / concat / ffprobe)
# ---------------------------------------------------------------------------


def ffmpeg_encode_frames(frames: list[np.ndarray], fps: float, size: tuple[int, int], gop: int, out_path: Path) -> dict:
    """프레임 리스트를 libx264로 재인코딩해서 저장한다 (재인코딩 방식, GOP=-g)."""
    w, h = size
    cmd = [
        FFMPEG, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-g", str(gop), "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    cpu0, wall0 = cpu_time_sec(), time.perf_counter()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg encode failed (code {ret}): {out_path}")
    return {
        "cpu_time_sec": cpu_time_sec() - cpu0,
        "wall_time_sec": time.perf_counter() - wall0,
        "file_size_bytes": os.path.getsize(out_path),
    }


def ffprobe_frame_count(path: Path, size: tuple[int, int]) -> int:
    """imageio_ffmpeg에는 ffprobe 바이너리가 없어(ffmpeg만 제공), rawvideo로 디코드한 뒤
    출력 바이트 수를 프레임 1장 크기(w*h*3)로 나눠 정확한 프레임 수를 센다.
    (ffmpeg의 `-f null`/`-stats` 텍스트 로그 파싱은 concat 등에서 progress 라인이
    입력별로 리셋/중복 출력되어 부정확함을 실측으로 확인해 이 방식으로 교체했다.)
    """
    w, h = size
    frame_bytes = w * h * 3
    cmd = [FFMPEG, "-loglevel", "error", "-i", str(path), "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return len(proc.stdout) // frame_bytes


def ffmpeg_stream_copy(src: Path, start_sec: float, duration_sec: float, out_path: Path) -> dict:
    cmd = [
        FFMPEG, "-y", "-loglevel", "error",
        "-ss", f"{start_sec:.6f}", "-i", str(src), "-t", f"{duration_sec:.6f}",
        "-c", "copy", str(out_path),
    ]
    cpu0, wall0 = cpu_time_sec(), time.perf_counter()
    ret = subprocess.run(cmd).returncode
    if ret != 0:
        raise RuntimeError(f"ffmpeg stream copy failed (code {ret}): {out_path}")
    return {
        "cpu_time_sec": cpu_time_sec() - cpu0,
        "wall_time_sec": time.perf_counter() - wall0,
        "file_size_bytes": os.path.getsize(out_path),
    }


def ffmpeg_concat(parts: list[Path], out_path: Path) -> dict:
    list_path = out_path.with_suffix(".concat.txt")
    list_path.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy", str(out_path)]
    cpu0, wall0 = cpu_time_sec(), time.perf_counter()
    ret = subprocess.run(cmd).returncode
    if ret != 0:
        raise RuntimeError(f"ffmpeg concat failed (code {ret}): {out_path}")
    return {
        "cpu_time_sec": cpu_time_sec() - cpu0,
        "wall_time_sec": time.perf_counter() - wall0,
        "file_size_bytes": os.path.getsize(out_path),
    }


def run() -> dict:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    base_img = cv2.imread(str(find_asset()))

    cache_path = OUT_DIR / "detection_cache.json"
    if cache_path.exists():
        detection = json.loads(cache_path.read_text())
    else:
        detection = run_detection(base_img)
        cache_path.write_text(json.dumps(detection))

    frame_w, frame_h = detection["frame_w"], detection["frame_h"]
    person_present = detection["person_present"]
    event_active = detection["event_active"]

    tiers = compute_tier_sequence(person_present, event_active, post_roll_frames=CONFIG.post_roll_frames)
    tier_counts = {t.value: tiers.count(t) for t in RecordingTier}
    positions = active_stream_positions(tiers)

    event_intervals = event_intervals_from_active_flags(event_active)
    clips = plan_event_clips(
        event_intervals,
        pre_roll_frames=CONFIG.pre_roll_frames,
        post_roll_frames=CONFIG.post_roll_frames,
        total_frames=N_FRAMES,
        merge_gap_frames=CONFIG.merge_gap_frames,
    )
    assert len(clips) == 1, f"이 시나리오는 Event Clip 1개를 기대했으나 {len(clips)}개 생성됨: {clips}"
    clip = clips[0]
    fully_active = clip_window_all_active(tiers, clip.start_frame, clip.end_frame)

    # IDLE과 걸치는 지점 찾기 (Hybrid 방식에 필요)
    idle_end_in_window = clip.start_frame - 1
    for i in range(clip.start_frame, clip.end_frame + 1):
        if tiers[i] == RecordingTier.IDLE:
            idle_end_in_window = i
        else:
            break
    idle_prefix_len = idle_end_in_window - clip.start_frame + 1 if idle_end_in_window >= clip.start_frame else 0

    size = (frame_w, frame_h)
    results: dict = {
        "n_frames": N_FRAMES,
        "assumed_fps": ASSUMED_FPS,
        "frame_w": frame_w,
        "frame_h": frame_h,
        "phase_layout": {"idle": [0, PHASE_IDLE_END], "normal": [PHASE_IDLE_END, PHASE_NORMAL_END], "event_trigger": [PHASE_NORMAL_END, N_FRAMES]},
        "tier_counts": tier_counts,
        "event_intervals": [{"event_id": e.event_id, "start_frame": e.start_frame, "end_frame": e.end_frame} for e in event_intervals],
        "clip_plan": {"start_frame": clip.start_frame, "end_frame": clip.end_frame, "n_frames": clip.end_frame - clip.start_frame + 1},
        "clip_fully_inside_active_stream": fully_active,
        "clip_idle_prefix_frames": idle_prefix_len,
        "gop_results": {},
    }

    for gop in (1, 25):
        gop_dir = VIDEO_DIR / f"gop_{gop}"
        gop_dir.mkdir(parents=True, exist_ok=True)

        # 1) Active Stream 생성 (IDLE이 아닌 Tier의 프레임만, 원본 해상도, GOP 설정 적용)
        active_frames = [make_frame(base_img, idx) for idx in range(N_FRAMES) if tiers[idx] != RecordingTier.IDLE]
        active_path = gop_dir / "active_stream.mp4"
        active_stats = ffmpeg_encode_frames(active_frames, ASSUMED_FPS, size, gop, active_path)

        # 2) 방식 A: 전체 재인코딩 (원본 프레임에서 Clip 구간 전체를 다시 인코딩)
        clip_raw_frames = [make_frame(base_img, idx) for idx in range(clip.start_frame, clip.end_frame + 1)]
        reencode_path = gop_dir / "clip_full_reencode.mp4"
        reencode_stats = ffmpeg_encode_frames(clip_raw_frames, ASSUMED_FPS, size, gop, reencode_path)
        reencode_actual_frames = ffprobe_frame_count(reencode_path, size)

        # 3) 방식 B: Naive Stream Copy (Active Stream 전체 구간을 그대로 오려냄, IDLE 겹침 무시)
        active_start_pos = positions[clip.start_frame] if positions[clip.start_frame] is not None else positions[idle_end_in_window + 1]
        active_end_pos = positions[clip.end_frame]
        copy_start_sec = active_start_pos / ASSUMED_FPS
        copy_duration_sec = (active_end_pos - active_start_pos + 1) / ASSUMED_FPS
        naive_copy_path = gop_dir / "clip_naive_copy.mp4"
        naive_copy_stats = ffmpeg_stream_copy(active_path, copy_start_sec, copy_duration_sec, naive_copy_path)
        naive_copy_actual_frames = ffprobe_frame_count(naive_copy_path, size)
        naive_copy_expected_frames = active_end_pos - active_start_pos + 1  # IDLE 접두 구간은 아예 빠짐

        # 4) 방식 C: Hybrid (IDLE 접두 구간만 재인코딩 + 나머지는 Stream Copy, concat으로 병합)
        if idle_prefix_len > 0:
            idle_prefix_frames = [make_frame(base_img, idx) for idx in range(clip.start_frame, idle_end_in_window + 1)]
            idle_part_path = gop_dir / "clip_hybrid_idle_part.mp4"
            idle_part_stats = ffmpeg_encode_frames(idle_prefix_frames, ASSUMED_FPS, size, gop, idle_part_path)
            hybrid_path = gop_dir / "clip_hybrid.mp4"
            concat_stats = ffmpeg_concat([idle_part_path, naive_copy_path], hybrid_path)
            hybrid_stats = {
                "cpu_time_sec": idle_part_stats["cpu_time_sec"] + naive_copy_stats["cpu_time_sec"] + concat_stats["cpu_time_sec"],
                "wall_time_sec": idle_part_stats["wall_time_sec"] + naive_copy_stats["wall_time_sec"] + concat_stats["wall_time_sec"],
                "file_size_bytes": os.path.getsize(hybrid_path),
                "idle_part_reencoded_frames": len(idle_prefix_frames),
            }
        else:
            hybrid_stats = None

        hybrid_actual_frames = ffprobe_frame_count(gop_dir / "clip_hybrid.mp4", size) if idle_prefix_len > 0 else None

        results["gop_results"][str(gop)] = {
            "active_stream": {**active_stats, "frames_written": len(active_frames)},
            "clip_full_reencode": {
                **reencode_stats,
                "expected_frames": len(clip_raw_frames),
                "actual_frames": reencode_actual_frames,
                "frame_diff": reencode_actual_frames - len(clip_raw_frames),
            },
            "clip_naive_copy": {
                **naive_copy_stats,
                "expected_frames_from_active_portion_only": naive_copy_expected_frames,
                "actual_frames": naive_copy_actual_frames,
                "frame_diff_vs_active_portion": naive_copy_actual_frames - naive_copy_expected_frames,
                "missing_idle_prefix_frames": idle_prefix_len,
                "is_complete_clip": idle_prefix_len == 0,
            },
            "clip_hybrid": (
                {
                    **hybrid_stats,
                    "expected_frames": len(clip_raw_frames),
                    "actual_frames": hybrid_actual_frames,
                    "frame_diff": hybrid_actual_frames - len(clip_raw_frames),
                    "is_complete_clip": True,
                }
                if hybrid_stats is not None
                else None
            ),
        }

    return results


if __name__ == "__main__":
    summary = run()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
