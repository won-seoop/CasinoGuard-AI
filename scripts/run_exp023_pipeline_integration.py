"""EXP-023: run_full_pipeline 통합 — Adaptive Recording(EXP-017) + Circular Buffer(EXP-020/021/022)를
실제 Detection+Tracking+ROI+BestShot+Metadata 단일 패스 파이프라인(run_full_pipeline.py 계열)에
처음으로 결합한다.

Blocker / Decision (EXP-010/014~022와 동일):
이 원격 세션은 egress 정책상 Wikimedia 등 외부 일반 도메인이 차단되어 있어(재확인 완료),
run_full_pipeline.py가 쓰는 crowded_intersection_1080p.webm을 이 세션에서 받을 수 없다.
대신 EXP-017과 동일한 ultralytics 내장 bus.jpg + Pan/Jitter 3-Phase 시나리오(IDLE/NORMAL/EVENT,
ROI 중앙 대역 x=[430,620])를 재사용한다 — 이번 실험의 목적은 새 데이터셋이 아니라
"이미 각자 따로 검증된 두 파이프라인(Metadata Pipeline vs Adaptive Recording)을 하나의
실시간 루프로 합쳤을 때도 여전히 정상 동작하는가"이므로, Detection 결과가 이전 실험들과
동일하게 재현되는 이 시나리오가 오히려 회귀 비교에 유리하다.

이 실험 전까지 Adaptive Recording/Circular Buffer(EXP-017~022)는 모두 다음 두 가지 방식으로만
검증됐다:
  1) 전체 person_present/event_active 배열을 미리 다 모은 뒤 compute_tier_sequence()로
     한번에 Tier를 계산하는 Two-Pass 방식 (실제 카메라는 미래 프레임을 미리 알 수 없다)
  2) Pre-Roll Window를 트리거 프레임(예: EXP-022의 1339)처럼 스크립트에 미리 박아넣은 값으로
     검증 (실제로 언제 Event가 발생할지는 파이프라인이 그 순간에 처음 알게 된다)
  3) Metadata Pipeline(run_full_pipeline.py)과 완전히 분리된 별도 스크립트로만 존재 —
     BestShot/Track/Event가 SQLite에 쌓이는 동안 Circular Buffer가 같이 동작한 적이 없다.

이 실험은 위 세 가지 한계를 모두 없앤 실제 Live 단일 패스 루프를 구현한다:
`recording.adaptive.OnlineTierClassifier`로 매 프레임 Tier를 실시간으로만 계산하고,
ROI Intrusion ENTER가 실제로 발생하는 그 프레임에서 Circular Buffer의 Pre-Roll을 즉시 꺼내며,
Tier가 EVENT를 벗어나는 순간(연속된 Intrusion이 병합된 뒤)에 Event Clip 파일을 완성해
MetadataStore의 이벤트 row에 사후적으로 경로를 붙인다.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from bestshot.scorer import bestshot_score  # noqa: E402
from bestshot.tracker import BestShotTracker  # noqa: E402
from events.roi_state_machine import ROIStateMachine, bottom_center  # noqa: E402
from metadata.store import MetadataStore  # noqa: E402
from recording.adaptive import (  # noqa: E402
    AdaptiveRecordingConfig,
    OnlineTierClassifier,
    RecordingTier,
    TierQualityConfig,
)
from recording.circular_buffer import FrameCircularBuffer, required_capacity_for_pre_roll  # noqa: E402
from recording.frame_codec import decode_frame, encode_frame  # noqa: E402

PERSON_CLASS_ID = 0
ASSET_PATH = ROOT / "data" / "raw" / "bus.jpg"

# --- EXP-017과 완전히 동일한 3-Phase 합성 시나리오 (Detection 재현성 확보) -----------------
N_FRAMES = 900
ASSUMED_FPS = 25.0
PHASE_IDLE_END = 300
PHASE_NORMAL_END = 600
ROI_X_MIN, ROI_X_MAX = 430, 620
EXTRA_DX_AMPLITUDE = 300
EXTRA_DX_PERIOD = 150

# EXP-022에서 채택한 Tier별 quality (idle=50, normal=75, event=95) 를 이번 세션부터
# 기본값으로 승격한다 — 지금까지는 실험 스크립트 안에서만 비교됐고 실제 파이프라인에는
# 반영된 적이 없었다.
PRODUCTION_TIER_QUALITY = TierQualityConfig(idle_quality=50, normal_quality=75, event_quality=95)

CONFIG = AdaptiveRecordingConfig(fps=ASSUMED_FPS, pre_roll_sec=10.0, post_roll_sec=10.0, idle_frame_stride=5, merge_gap_sec=0.0)
SAFETY_MARGIN_FRAMES = 20
BUFFER_CAPACITY = required_capacity_for_pre_roll(CONFIG.pre_roll_frames, SAFETY_MARGIN_FRAMES)

OUT_DIR = ROOT / "results" / "EXP-023"
CLIP_DIR = OUT_DIR / "event_clips"


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


class ActiveClip:
    """EVENT Tier가 연속으로 유지되는 동안 프레임을 모으는 진행 중인 Event Clip."""

    def __init__(self, clip_id: int, start_frame: int, preroll_frames: list[np.ndarray]):
        self.clip_id = clip_id
        self.start_frame = start_frame
        self.frames: list[np.ndarray] = list(preroll_frames)
        self.event_ids: list[int] = []

    def append_live_frame(self, frame: np.ndarray) -> None:
        self.frames.append(frame)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CLIP_DIR.mkdir(parents=True, exist_ok=True)

    if not ASSET_PATH.exists():
        raise FileNotFoundError(f"synthetic scene base image missing: {ASSET_PATH}")
    base_img = cv2.imread(str(ASSET_PATH))
    frame_h, frame_w = base_img.shape[:2]

    roi_polygon = [(ROI_X_MIN, 0), (ROI_X_MAX, 0), (ROI_X_MAX, frame_h), (ROI_X_MIN, frame_h)]

    db_path = OUT_DIR / "metadata.db"
    if db_path.exists():
        db_path.unlink()
    store = MetadataStore(db_path)

    model = YOLO("yolo11n.pt")
    roi_sm = ROIStateMachine(polygon=roi_polygon)
    bestshot_tracker = BestShotTracker(frame_w, frame_h)
    tier_clf = OnlineTierClassifier(post_roll_frames=CONFIG.post_roll_frames)
    frame_buffer: FrameCircularBuffer[bytes] = FrameCircularBuffer(capacity_frames=BUFFER_CAPACITY)

    last_seen: dict[int, int] = {}
    tier_counts = {t: 0 for t in RecordingTier}
    active_clip: ActiveClip | None = None
    finished_clips: list[dict] = []
    clip_id_seq = 0

    lat_detect_ms: list[float] = []
    lat_events_ms: list[float] = []
    lat_buffer_ms: list[float] = []
    lat_total_ms: list[float] = []
    buffer_bytes_snapshots: dict[str, int] = {}

    t_run_start = time.perf_counter()
    for idx in range(N_FRAMES):
        frame = make_frame(base_img, idx)
        t0 = time.perf_counter()

        result = model.track(
            frame, persist=True, tracker="bytetrack.yaml", classes=[PERSON_CLASS_ID], conf=0.4, imgsz=640, verbose=False
        )[0]
        t1 = time.perf_counter()

        active_boxes = []
        if result.boxes is not None and result.boxes.id is not None:
            ids = [int(i) for i in result.boxes.id.tolist()]
            boxes = result.boxes.xyxy.tolist()
            confs = result.boxes.conf.tolist()
            active_boxes = list(zip(ids, boxes, confs))
        all_boxes_this_frame = [b for _, b, _ in active_boxes]

        event_active_this_frame = False
        entered_track_ids_this_frame: list[int] = []
        for tid, box, conf in active_boxes:
            box_t = tuple(box)
            point = bottom_center(box_t)
            last_seen[tid] = idx

            roi_ev = roi_sm.update(tid, box_t, idx)
            zone = "restricted_zone" if roi_sm.states.get(tid) == "INSIDE" else None
            store.upsert_track(tid, "person", idx, conf, box_t, point, zone)

            if roi_ev is not None and roi_ev.event_type == "ENTER":
                event_id = store.add_event(tid, "INTRUSION_ENTER", idx, {"zone": "restricted_zone"})
                entered_track_ids_this_frame.append(event_id)
            elif roi_ev is not None and roi_ev.event_type == "EXIT":
                store.add_event(tid, "INTRUSION_EXIT", idx, {"zone": "restricted_zone"})

            x1, y1, x2, y2 = [max(0, int(v)) for v in box_t]
            crop = frame[y1:y2, x1:x2].copy()
            others = [b for b in all_boxes_this_frame if b != box]
            score = bestshot_score(conf, crop, box_t, others, frame_w, frame_h)
            bestshot_tracker.observe(tid, idx, crop, score["total"])

        if roi_sm.states and any(state == "INSIDE" for state in roi_sm.states.values()):
            event_active_this_frame = True

        for tid in list(last_seen.keys()):
            if idx - last_seen[tid] > 10:
                exit_ev = roi_sm.forget_track(tid, frame_idx=idx)
                if exit_ev is not None:
                    store.add_event(tid, "INTRUSION_EXIT", idx, {"zone": "restricted_zone", "reason": "track_lost"})
                final = bestshot_tracker.forget_track(tid)
                if final is not None and final.observation_count >= 5:
                    path = OUT_DIR / f"bestshot_track_{tid}.jpg"
                    cv2.imwrite(str(path), final.crop)
                    store.update_bestshot(tid, str(path.relative_to(ROOT)), round(final.score, 3))
                del last_seen[tid]

        person_present_this_frame = len(active_boxes) > 0
        t2 = time.perf_counter()

        # --- Adaptive Recording + Circular Buffer (EXP-017/020/021/022 통합 지점) -----------
        tier = tier_clf.update(person_present_this_frame, event_active_this_frame)
        tier_counts[tier] += 1
        quality = PRODUCTION_TIER_QUALITY.quality_for_tier(tier)
        encoded = encode_frame(frame, quality=quality)
        frame_buffer.push(idx, encoded)

        if tier == RecordingTier.EVENT and active_clip is None:
            # Tier가 방금 EVENT로 전환된 순간(=실시간으로 처음 알게 된 트리거) — Pre-Roll을
            # Circular Buffer에서 즉시 꺼낸다. 이 순간을 놓치면(다음 프레임까지 기다리면)
            # 버퍼가 계속 밀려나 Pre-Roll이 유실될 수 있다.
            window_start = max(0, idx - CONFIG.pre_roll_frames)
            cov = frame_buffer.coverage(window_start, idx)
            if not cov.is_complete:
                raise RuntimeError(f"pre-roll window [{window_start},{idx}] incomplete: missing {cov.missing_frames}")
            preroll_encoded = frame_buffer.get_range(window_start, idx)
            preroll_decoded = [decode_frame(b) for b in preroll_encoded]
            clip_id_seq += 1
            active_clip = ActiveClip(clip_id=clip_id_seq, start_frame=window_start, preroll_frames=preroll_decoded)
        elif tier == RecordingTier.EVENT and active_clip is not None:
            active_clip.append_live_frame(frame)
        elif tier != RecordingTier.EVENT and active_clip is not None:
            # EVENT Tier 구간이 끝났다 (Post-Roll 만료) -> Clip 파일로 확정한다.
            finished_clips.append(_finalize_clip(active_clip, idx - 1, store, buffer_at_end=frame_buffer))
            active_clip = None

        if active_clip is not None:
            active_clip.event_ids.extend(entered_track_ids_this_frame)

        t3 = time.perf_counter()

        lat_detect_ms.append((t1 - t0) * 1000)
        lat_events_ms.append((t2 - t1) * 1000)
        lat_buffer_ms.append((t3 - t2) * 1000)
        lat_total_ms.append((t3 - t0) * 1000)

        if idx == PHASE_IDLE_END - 1:
            buffer_bytes_snapshots["idle_steady_state"] = frame_buffer.total_bytes(len)
        if idx == PHASE_NORMAL_END - 1:
            buffer_bytes_snapshots["normal_steady_state"] = frame_buffer.total_bytes(len)

    # 영상이 끝난 시점에도 EVENT Tier가 열려 있으면(Post-Roll이 아직 안 끝남) 잘린 채로 확정한다.
    if active_clip is not None:
        finished_clips.append(_finalize_clip(active_clip, N_FRAMES - 1, store, buffer_at_end=frame_buffer, truncated=True))

    for tid in list(last_seen.keys()):
        final = bestshot_tracker.forget_track(tid)
        if final is not None and final.observation_count >= 5:
            path = OUT_DIR / f"bestshot_track_{tid}.jpg"
            cv2.imwrite(str(path), final.crop)
            store.update_bestshot(tid, str(path.relative_to(ROOT)), round(final.score, 3))

    total_wall_sec = time.perf_counter() - t_run_start

    lat = {
        "detect_tracking": lat_detect_ms,
        "events_bestshot_metadata": lat_events_ms,
        "adaptive_recording_buffer": lat_buffer_ms,
        "total": lat_total_ms,
    }
    latency_summary = {
        name: {
            "p50_ms": round(float(np.percentile(vals, 50)), 3),
            "p95_ms": round(float(np.percentile(vals, 95)), 3),
            "p99_ms": round(float(np.percentile(vals, 99)), 3),
            "mean_ms": round(float(np.mean(vals)), 3),
        }
        for name, vals in lat.items()
    }

    n_tracks = len(store.query_tracks())
    n_events = len(store.query_events())
    n_intrusion_enter = len(store.query_events(event_type="INTRUSION_ENTER"))

    summary = {
        "n_frames": N_FRAMES,
        "assumed_fps": ASSUMED_FPS,
        "frame_w": frame_w,
        "frame_h": frame_h,
        "config": {
            "pre_roll_frames": CONFIG.pre_roll_frames,
            "post_roll_frames": CONFIG.post_roll_frames,
            "buffer_capacity": BUFFER_CAPACITY,
            "tier_quality": {
                "idle": PRODUCTION_TIER_QUALITY.idle_quality,
                "normal": PRODUCTION_TIER_QUALITY.normal_quality,
                "event": PRODUCTION_TIER_QUALITY.event_quality,
            },
        },
        "tier_distribution": {t.value: tier_counts[t] for t in RecordingTier},
        "buffer_memory_mb": {k: round(v / 1e6, 3) for k, v in buffer_bytes_snapshots.items()},
        "wall_clock_sec": round(total_wall_sec, 2),
        "effective_fps_incl_all_stages": round(N_FRAMES / total_wall_sec, 2),
        "latency_ms": latency_summary,
        "metadata": {"tracks": n_tracks, "events": n_events, "intrusion_enter_events": n_intrusion_enter},
        "event_clips": finished_clips,
    }

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    with (OUT_DIR / "latency_breakdown.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "p50_ms", "p95_ms", "p99_ms", "mean_ms"])
        for name, s in latency_summary.items():
            writer.writerow([name, s["p50_ms"], s["p95_ms"], s["p99_ms"], s["mean_ms"]])

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[EXP-023] wrote {OUT_DIR / 'summary.json'}, latency_breakdown.csv, {len(finished_clips)} event clip(s) in {CLIP_DIR}")

    store.close()


def _finalize_clip(
    clip: ActiveClip,
    end_frame: int,
    store: MetadataStore,
    buffer_at_end: FrameCircularBuffer,
    truncated: bool = False,
) -> dict:
    path = CLIP_DIR / f"event_clip_{clip.clip_id}.mp4"
    h, w = clip.frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), ASSUMED_FPS, (w, h))
    for f in clip.frames:
        writer.write(f)
    writer.release()

    expected_frames = end_frame - clip.start_frame + 1
    for event_id in clip.event_ids:
        store.update_event_detail(event_id, {"event_clip_path": str(path.relative_to(ROOT))})

    # 대안 A(트리거 시점이 아니라 EVENT Tier 구간이 "끝난 뒤" Pre-Roll을 꺼내는 방식)를
    # 실제로 채택했다면 지금 이 시점의 Circular Buffer 상태로 같은 Window를 복원할 수
    # 있었을지 사후 검증한다 (실측 — 가정이 아니라 실제 buffer 상태로 확인).
    alt_a_cov = buffer_at_end.coverage(clip.start_frame, end_frame)

    return {
        "clip_id": clip.clip_id,
        "start_frame": clip.start_frame,
        "end_frame": end_frame,
        "expected_frames": expected_frames,
        "actual_frames_written": len(clip.frames),
        "frames_missing": expected_frames - len(clip.frames),
        "merged_event_count": len(clip.event_ids),
        "event_ids": clip.event_ids,
        "file_size_bytes": path.stat().st_size,
        "path": str(path.relative_to(ROOT)),
        "truncated_at_video_end": truncated,
        "alternative_a_extract_at_run_end": {
            "would_be_complete": alt_a_cov.is_complete,
            "missing_frame_count": len(alt_a_cov.missing_frames),
            "missing_frames_sample": list(alt_a_cov.missing_frames[:5]),
        },
    }


if __name__ == "__main__":
    main()
