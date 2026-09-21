"""Metadata Pipeline 통합 실행 — Detection+Tracking+BestShot+Event를 하나로 묶어
SQLite에 저장한다. Phase 1~7에서 따로 만든 모듈들을 재사용/통합한다.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from bestshot.scorer import bestshot_score  # noqa: E402
from bestshot.tracker import BestShotTracker  # noqa: E402
from events.line_crossing import LineCrossingDetector  # noqa: E402
from events.loitering import LoiteringDetector  # noqa: E402
from events.roi_state_machine import ROIStateMachine, bottom_center  # noqa: E402
from metadata.store import MetadataStore  # noqa: E402

PERSON_CLASS_ID = 0
ROI_POLYGON = [(950, 650), (1750, 650), (1850, 1080), (700, 1080)]
LINE_A, LINE_B = (700, 850), (1900, 780)
LOITER_THRESHOLD_SEC = 8  # EXP-007 분석(횡단보도는 20~30초는 과함)을 반영해 데모용으로 8초 사용
LINE_CROSSING_BAND_PX = 5  # FC-007/EXP-014/PAR-005: 선 근처 흔들림으로 인한 왕복 중복 이벤트 방지


def main():
    out_dir = ROOT / "results" / "metadata_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    bestshot_dir = out_dir / "bestshot"
    bestshot_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "metadata.db"
    if db_path.exists():
        db_path.unlink()  # 매 실행마다 새로 만든다 (데모/재현성 목적)

    store = MetadataStore(db_path)
    model = YOLO("yolo11n.pt")
    video_path = ROOT / "data/raw/crowded_intersection_1080p.webm"
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    roi_sm = ROIStateMachine(polygon=ROI_POLYGON)
    line_det = LineCrossingDetector(LINE_A, LINE_B, band_px=LINE_CROSSING_BAND_PX)
    loiter_det = LoiteringDetector(polygon=ROI_POLYGON, threshold_frames=int(LOITER_THRESHOLD_SEC * fps))

    # Track별 BestShot 후보는 "지금까지 최고 점수 1개"만 O(1) 메모리로 유지한다.
    # (EXP-010/PAR-004: 관측치를 전부 리스트에 누적하던 이전 방식은 Track이 오래
    #  머물수록 Memory가 선형으로 증가하는 Leak이 Long Running Test에서 확인됨)
    bestshot_tracker = BestShotTracker(frame_w, frame_h)
    last_seen: dict[int, int] = {}
    saved = 0

    max_frames = 800
    idx = 0
    frame_latencies_sec: list[float] = []
    t_start = time.perf_counter()
    while idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        t_frame0 = time.perf_counter()
        result = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[PERSON_CLASS_ID], conf=0.4, imgsz=640, verbose=False)[0]

        active_boxes = []
        if result.boxes is not None and result.boxes.id is not None:
            ids = [int(i) for i in result.boxes.id.tolist()]
            boxes = result.boxes.xyxy.tolist()
            confs = result.boxes.conf.tolist()
            active_boxes = list(zip(ids, boxes, confs))

        all_boxes_this_frame = [b for _, b, _ in active_boxes]

        for tid, box, conf in active_boxes:
            box_t = tuple(box)
            point = bottom_center(box_t)
            last_seen[tid] = idx

            roi_ev = roi_sm.update(tid, box_t, idx)
            zone = "restricted_zone" if roi_sm.states.get(tid) == "INSIDE" else None
            line_ev = line_det.update(tid, point, idx)
            loiter_ev = loiter_det.update(tid, point, idx)

            # Track row를 먼저 upsert해서 FK 제약을 만족시킨 다음 이벤트를 기록한다.
            store.upsert_track(tid, "person", idx, conf, box_t, point, zone)

            if roi_ev is not None:
                store.add_event(tid, f"INTRUSION_{roi_ev.event_type}", idx, {"zone": "restricted_zone"})
            if line_ev is not None:
                store.add_event(tid, "LINE_CROSSING", idx, {"direction": line_ev.direction})
            if loiter_ev is not None:
                store.add_event(tid, "LOITERING", idx, {"dwell_sec": round(loiter_ev.dwell_frames / fps, 2)})

            x1, y1, x2, y2 = [max(0, int(v)) for v in box_t]
            crop = frame[y1:y2, x1:x2].copy()
            others = [b for b in all_boxes_this_frame if b != box]
            score = bestshot_score(conf, crop, box_t, others, frame_w, frame_h)
            bestshot_tracker.observe(tid, idx, crop, score["total"])

        # Track 소실 처리 (10프레임 이상 미관측)
        for tid in list(last_seen.keys()):
            if idx - last_seen[tid] > 10:
                exit_ev = roi_sm.forget_track(tid, frame_idx=idx)
                if exit_ev is not None:
                    store.add_event(tid, "INTRUSION_EXIT", idx, {"zone": "restricted_zone", "reason": "track_lost"})
                line_det.forget_track(tid)
                loiter_det.forget_track(tid)
                dwell = idx - loiter_det.enter_frame.get(tid, idx)
                store.set_dwell(tid, dwell)
                final = bestshot_tracker.forget_track(tid)
                if final is not None and final.observation_count >= 5:
                    path = bestshot_dir / f"track_{tid}.jpg"
                    cv2.imwrite(str(path), final.crop)
                    store.update_bestshot(tid, str(path.relative_to(ROOT)), round(final.score, 3))
                    saved += 1
                del last_seen[tid]

        frame_latencies_sec.append(time.perf_counter() - t_frame0)
        idx += 1

    cap.release()
    elapsed = time.perf_counter() - t_start

    # 영상이 끝난 시점까지 살아있던 Track들도 마지막으로 BestShot을 확정한다.
    for tid in list(last_seen.keys()):
        final = bestshot_tracker.forget_track(tid)
        if final is not None and final.observation_count >= 5:
            path = bestshot_dir / f"track_{tid}.jpg"
            cv2.imwrite(str(path), final.crop)
            store.update_bestshot(tid, str(path.relative_to(ROOT)), round(final.score, 3))
            saved += 1

    import numpy as np

    lat_ms = np.array(frame_latencies_sec) * 1000
    p50, p95, p99 = np.percentile(lat_ms, [50, 95, 99])

    n_tracks = len(store.query_tracks())
    n_events = len(store.query_events())
    print(f"Processed {idx} frames in {elapsed:.1f}s ({idx/elapsed:.1f} FPS incl. tracking+events+bestshot)")
    print(f"Per-frame(Detection+Tracking+Events) latency: P50={p50:.1f}ms P95={p95:.1f}ms P99={p99:.1f}ms")
    print(f"Tracks stored: {n_tracks}, BestShot saved: {saved}, Events stored: {n_events}")
    print(f"DB: {db_path}")

    import csv

    with open(out_dir / "pipeline_benchmark.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frames", "elapsed_sec", "fps", "p50_ms", "p95_ms", "p99_ms", "tracks", "bestshot_saved", "events"])
        writer.writerow([idx, round(elapsed, 2), round(idx / elapsed, 2), round(p50, 2), round(p95, 2), round(p99, 2), n_tracks, saved, n_events])

    store.close()


if __name__ == "__main__":
    main()
