"""Metadata Pipeline 통합 실행 — Detection+Tracking+BestShot+Event를 하나로 묶어
SQLite에 저장한다. Phase 1~7에서 따로 만든 모듈들을 재사용/통합한다.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from attributes.pipeline import classify_track_attributes_once  # noqa: E402
from bestshot.scorer import bestshot_score  # noqa: E402
from bestshot.tracker import BestShotTracker  # noqa: E402
from events.dwell import DwellCounter  # noqa: E402
from events.line_crossing import LineCrossingDetector  # noqa: E402
from events.loitering import LoiteringDetector  # noqa: E402
from events.roi_state_machine import ROIStateMachine, bottom_center  # noqa: E402
from metadata.store import MetadataStore  # noqa: E402


def segment_largest_instance_mask(segmenter: YOLO, crop_bgr) -> np.ndarray:
    """BestShot crop 자체에 Segmentation 모델을 다시 돌려(EXP-028의 "전체 프레임 Segmentation
    + bbox IoU 매칭" 대신 crop 단위로) person mask를 얻는다. BestShotTracker는 PAR-004 Memory
    Leak 수정 이후 Track마다 원본 프레임이 아니라 최종 crop 1장만 들고 있으므로, 원본 프레임에
    대한 Segmentation+매칭은 애초에 선택지가 아니다 - crop 자체를 다시 Segment하는 쪽이 더
    단순하고 메모리 사용량도 늘리지 않는다 (EXP-029 Decision 참고).

    crop에 다른 사람이 함께 잡혀 있을 수 있어 면적이 가장 큰 Instance를 선택한다. 매칭되는
    Instance가 없으면(세그멘테이션 실패) 전부 False인 mask를 반환해 classify_person_attributes_
    with_mask의 min_mask_fraction fallback이 고정 비율 사각형으로 되돌아가도록 한다.
    """
    result = segmenter.predict(crop_bgr, classes=[PERSON_CLASS_ID], conf=0.4, verbose=False)[0]
    if result.masks is None or len(result.masks.data) == 0:
        return np.zeros(crop_bgr.shape[:2], dtype=bool)
    masks = result.masks.data.cpu().numpy()
    areas = masks.reshape(len(masks), -1).sum(axis=1)
    best = masks[int(np.argmax(areas))]
    resized = cv2.resize(best, (crop_bgr.shape[1], crop_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def finalize_track_attributes(segmenter: YOLO, store: MetadataStore, track_id: int, crop) -> None:
    attrs = classify_track_attributes_once(crop, lambda c: segment_largest_instance_mask(segmenter, c))
    store.update_attributes(track_id, upper_color=attrs["upper"], lower_color=attrs["lower"])


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
    # EXP-029: Attribute Metadata(c_wb, 지침 19)를 Track당 1회 BestShot 확정 시점에만 호출한다.
    # 매 프레임 호출(대안 A)은 EXP-029에서 실측한 대로 FPS를 크게 떨어뜨려 기각됐다 - 세그멘터는
    # 여기서 한 번만 로드해 매 호출마다 모델을 다시 올리는 추가 비용을 만들지 않는다.
    segmenter = YOLO("yolo11n-seg.pt")
    video_path = ROOT / "data/raw/crowded_intersection_1080p.webm"
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    roi_sm = ROIStateMachine(polygon=ROI_POLYGON)
    line_det = LineCrossingDetector(LINE_A, LINE_B, band_px=LINE_CROSSING_BAND_PX)
    loiter_det = LoiteringDetector(polygon=ROI_POLYGON, threshold_frames=int(LOITER_THRESHOLD_SEC * fps))
    # FC-006/PAR-006: loiter_det.enter_frame은 "연속 체류 스트릭"만 표현하고 ROI를
    # 벗어나면 즉시 리셋되므로, VMS Search가 원하는 "총 누적 체류 시간"에는 쓸 수
    # 없다(재방문 시 이전 방문 기록이 사라짐). 책임을 분리한 DwellCounter로 누적한다.
    dwell_counter = DwellCounter(polygon=ROI_POLYGON)

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
            dwell_counter.update(tid, point)

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
                store.set_dwell(tid, dwell_counter.get(tid))
                final = bestshot_tracker.forget_track(tid)
                if final is not None and final.observation_count >= 5:
                    path = bestshot_dir / f"track_{tid}.jpg"
                    cv2.imwrite(str(path), final.crop)
                    store.update_bestshot(tid, str(path.relative_to(ROOT)), round(final.score, 3))
                    finalize_track_attributes(segmenter, store, tid, final.crop)
                    saved += 1
                del last_seen[tid]

        frame_latencies_sec.append(time.perf_counter() - t_frame0)
        idx += 1

    cap.release()
    elapsed = time.perf_counter() - t_start

    # 영상이 끝난 시점까지 살아있던 Track들도 마지막으로 BestShot/Dwell을 확정한다.
    for tid in list(last_seen.keys()):
        store.set_dwell(tid, dwell_counter.get(tid))
        final = bestshot_tracker.forget_track(tid)
        if final is not None and final.observation_count >= 5:
            path = bestshot_dir / f"track_{tid}.jpg"
            cv2.imwrite(str(path), final.crop)
            store.update_bestshot(tid, str(path.relative_to(ROOT)), round(final.score, 3))
            finalize_track_attributes(segmenter, store, tid, final.crop)
            saved += 1

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
