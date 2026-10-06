"""EXP-029: Attribute Metadata(c_wb) 실시간 파이프라인 통합 - 호출 빈도 A/B 비교 (지침 19).

EXP-028 Next Action 2번: c_wb(Segmentation Mask + White Balance, EXP-028 채택)를
Track당 1회(BestShot 확정 시점)만 호출하는 방식으로 run_full_pipeline.py에 연결하고, 실제
FPS/P95 Latency 영향을 재측정한다 - EXP-028은 오프라인 n=21 Batch(coco128 정지 이미지)만
측정했고 실시간 파이프라인에 통합했을 때의 Overhead는 전혀 측정하지 않았다.

Hypothesis:
1. Attribute 분류(c_wb)를 매 프레임마다 모든 활성 Track에 대해 호출하면(대안 A), Segmentation
   추론 비용(EXP-028 실측: Detector 단독 대비 1.29x, 절대 ~20ms/image 추가)이 프레임마다
   반복되어 전체 파이프라인 FPS가 크게 떨어질 것이다.
2. BestShot이 확정되는 순간(Track 소실/영상 종료)에만 Track당 1회 호출하면(대안 B, 지침 12/19
   설계 의도), Track 수는 프레임 수보다 훨씬 적으므로 전체 파이프라인 FPS에는 거의 영향이
   없을 것이다.

Blocker / Decision (EXP-010/014~023/029와 동일 패턴):
이 원격 세션은 egress 정책상 commons.wikimedia.org가 이번에도 403으로 차단되어(재확인,
curl 직접 테스트) run_full_pipeline.py가 쓰는 crowded_intersection_1080p.webm을 받을 수
없다. EXP-017/023과 동일하게 ultralytics 내장 bus.jpg + Pan 합성 시나리오로 대체하되,
이번에는 "Track 소실 후 재등장"을 의도적으로 3번 반복시켜(사람이 사라지는 구간을 100프레임
이상 넣어 ByteTrack의 track_buffer보다 길게 유지) 한 번의 연속 Track이 아니라 서로 다른
Track 3개가 실행 중간중간에 완성되도록 만든다 - 그래야 "Track당 1회" 호출이 영상 끝에서
한 번만 일어나는 것이 아니라 실행 전반에 걸쳐 분산되는, 더 현실적인 시나리오가 된다.

실행: python scripts/run_exp029_attribute_pipeline_integration.py
출력: results/EXP-029/{summary.json, latency_breakdown.csv}
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from attributes.pipeline import classify_track_attributes_once  # noqa: E402
from bestshot.scorer import bestshot_score  # noqa: E402
from bestshot.tracker import BestShotTracker  # noqa: E402
from metadata.store import MetadataStore  # noqa: E402

PERSON_CLASS_ID = 0
ASSET_PATH = ROOT / "data" / "raw" / "bus.jpg"
RESULTS_DIR = ROOT / "results" / "EXP-029"

# 3개의 분리된 "사람 등장" 구간(각 170프레임) 사이에 110프레임의 "사람 없음" 구간을 둔다.
# ByteTrack 기본 track_buffer(bytetrack.yaml, 30프레임)보다 훨�신 길어 각 등장 구간마다
# 새 Track ID가 할당되고, last_seen 기반 forget_track()도 그 사이에 반드시 발생한다.
PRESENCE_WINDOWS = [(30, 200), (310, 480), (590, 760)]
N_FRAMES = 900


def make_no_person_background(base_img: np.ndarray, frame_idx: int) -> np.ndarray:
    h, w = base_img.shape[:2]
    strip_h = max(1, int(h * 0.15))
    strip = base_img[0:strip_h, :]
    reps = int(np.ceil(h / strip_h))
    tiled = np.tile(strip, (reps, 1, 1))[:h]
    dx = int(6 * np.sin(frame_idx * 0.05))
    return np.roll(tiled, shift=dx, axis=1)


def make_person_frame(base_img: np.ndarray, frame_idx: int) -> np.ndarray:
    max_shift = 12
    dx = int(max_shift * np.sin(frame_idx * 0.037))
    dy = int(max_shift * np.cos(frame_idx * 0.053))
    shifted = np.roll(base_img, shift=(dy, dx), axis=(0, 1))
    brightness = 1.0 + 0.03 * np.sin(frame_idx * 0.011)
    return np.clip(shifted.astype(np.float32) * brightness, 0, 255).astype(np.uint8)


def person_present(frame_idx: int) -> bool:
    return any(start <= frame_idx < end for start, end in PRESENCE_WINDOWS)


def make_frame(base_img: np.ndarray, frame_idx: int) -> np.ndarray:
    if person_present(frame_idx):
        return make_person_frame(base_img, frame_idx)
    return make_no_person_background(base_img, frame_idx)


def segment_largest_instance_mask(segmenter: YOLO, crop_bgr: np.ndarray) -> np.ndarray:
    """run_full_pipeline.py와 동일한 crop 단위 Segmentation 호출 (EXP-029 본문 참고)."""
    result = segmenter.predict(crop_bgr, classes=[PERSON_CLASS_ID], conf=0.4, verbose=False)[0]
    if result.masks is None or len(result.masks.data) == 0:
        return np.zeros(crop_bgr.shape[:2], dtype=bool)
    masks = result.masks.data.cpu().numpy()
    areas = masks.reshape(len(masks), -1).sum(axis=1)
    best = masks[int(np.argmax(areas))]
    resized = cv2.resize(best, (crop_bgr.shape[1], crop_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def run_variant(variant: str, model: YOLO, segmenter: YOLO, base_img: np.ndarray) -> dict:
    """variant: "none" (Baseline, Attribute 없음) | "per_frame" (대안 A) | "per_track" (대안 B)."""
    frame_h, frame_w = base_img.shape[:2]
    db_path = RESULTS_DIR / f"metadata_{variant}.db"
    if db_path.exists():
        db_path.unlink()
    store = MetadataStore(db_path)

    bestshot_tracker = BestShotTracker(frame_w, frame_h)
    last_seen: dict[int, int] = {}
    attribute_calls = 0
    attribute_latencies_ms: list[float] = []
    frame_latencies_ms: list[float] = []

    def maybe_classify_per_frame(tid: int, crop: np.ndarray) -> None:
        nonlocal attribute_calls
        if variant != "per_frame":
            return
        t0 = time.perf_counter()
        attrs = classify_track_attributes_once(crop, lambda c: segment_largest_instance_mask(segmenter, c))
        attribute_latencies_ms.append((time.perf_counter() - t0) * 1000)
        attribute_calls += 1
        store.update_attributes(tid, upper_color=attrs["upper"], lower_color=attrs["lower"])

    def finalize(tid: int, crop: np.ndarray) -> None:
        nonlocal attribute_calls
        if variant != "per_track":
            return
        t0 = time.perf_counter()
        attrs = classify_track_attributes_once(crop, lambda c: segment_largest_instance_mask(segmenter, c))
        attribute_latencies_ms.append((time.perf_counter() - t0) * 1000)
        attribute_calls += 1
        store.update_attributes(tid, upper_color=attrs["upper"], lower_color=attrs["lower"])

    t_run_start = time.perf_counter()
    for idx in range(N_FRAMES):
        frame = make_frame(base_img, idx)
        t0 = time.perf_counter()

        result = model.track(
            frame, persist=True, tracker="bytetrack.yaml", classes=[PERSON_CLASS_ID], conf=0.4, imgsz=640, verbose=False
        )[0]

        active_boxes = []
        if result.boxes is not None and result.boxes.id is not None:
            ids = [int(i) for i in result.boxes.id.tolist()]
            boxes = result.boxes.xyxy.tolist()
            confs = result.boxes.conf.tolist()
            active_boxes = list(zip(ids, boxes, confs))
        all_boxes_this_frame = [b for _, b, _ in active_boxes]

        for tid, box, conf in active_boxes:
            box_t = tuple(box)
            last_seen[tid] = idx
            point = ((box_t[0] + box_t[2]) / 2, box_t[3])
            store.upsert_track(tid, "person", idx, conf, box_t, point, zone=None)

            x1, y1, x2, y2 = [max(0, int(v)) for v in box_t]
            crop = frame[y1:y2, x1:x2].copy()
            others = [b for b in all_boxes_this_frame if b != box]
            score = bestshot_score(conf, crop, box_t, others, frame_w, frame_h)
            bestshot_tracker.observe(tid, idx, crop, score["total"])
            maybe_classify_per_frame(tid, crop)

        for tid in list(last_seen.keys()):
            if idx - last_seen[tid] > 10:
                final = bestshot_tracker.forget_track(tid)
                if final is not None and final.observation_count >= 5:
                    store.update_bestshot(tid, f"track_{tid}.jpg", round(final.score, 3))
                    finalize(tid, final.crop)
                del last_seen[tid]

        frame_latencies_ms.append((time.perf_counter() - t0) * 1000)

    for tid in list(last_seen.keys()):
        final = bestshot_tracker.forget_track(tid)
        if final is not None and final.observation_count >= 5:
            store.update_bestshot(tid, f"track_{tid}.jpg", round(final.score, 3))
            finalize(tid, final.crop)

    total_wall_sec = time.perf_counter() - t_run_start
    store.close()

    lat = np.array(frame_latencies_ms)
    attr_lat = np.array(attribute_latencies_ms) if attribute_latencies_ms else np.array([0.0])
    return {
        "variant": variant,
        "n_frames": N_FRAMES,
        "wall_clock_sec": round(total_wall_sec, 2),
        "fps": round(N_FRAMES / total_wall_sec, 2),
        "per_frame_latency_ms": {
            "p50": round(float(np.percentile(lat, 50)), 3),
            "p95": round(float(np.percentile(lat, 95)), 3),
            "p99": round(float(np.percentile(lat, 99)), 3),
            "mean": round(float(np.mean(lat)), 3),
        },
        "attribute_calls": attribute_calls,
        "attribute_call_latency_ms": {
            "p50": round(float(np.percentile(attr_lat, 50)), 3),
            "mean": round(float(np.mean(attr_lat)), 3),
        }
        if attribute_latencies_ms
        else None,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not ASSET_PATH.exists():
        raise FileNotFoundError(f"synthetic scene base image missing: {ASSET_PATH}")
    base_img = cv2.imread(str(ASSET_PATH))

    model = YOLO("yolo11n.pt")
    segmenter = YOLO("yolo11n-seg.pt")
    # 모델 최초 호출 시 lazy init 비용(그래프 빌드 등)이 측정값에 섞이지 않도록 미리 1회 워밍업.
    model.predict(base_img, verbose=False)
    segmenter.predict(base_img, verbose=False)

    results = {}
    for variant in ("none", "per_track", "per_frame"):
        print(f"[EXP-029] running variant={variant} ...")
        results[variant] = run_variant(variant, model, segmenter, base_img)
        print(json.dumps(results[variant], indent=2, ensure_ascii=False))

    baseline_fps = results["none"]["fps"]
    summary = {
        "presence_windows": PRESENCE_WINDOWS,
        "n_frames": N_FRAMES,
        "variants": results,
        "fps_degradation_pct": {
            variant: round((1 - results[variant]["fps"] / baseline_fps) * 100, 2)
            for variant in ("per_track", "per_frame")
        },
    }

    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    import csv

    with (RESULTS_DIR / "latency_breakdown.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "fps", "p50_ms", "p95_ms", "p99_ms", "attribute_calls", "attribute_call_p50_ms"])
        for variant, r in results.items():
            attr_p50 = r["attribute_call_latency_ms"]["p50"] if r["attribute_call_latency_ms"] else ""
            writer.writerow(
                [
                    variant,
                    r["fps"],
                    r["per_frame_latency_ms"]["p50"],
                    r["per_frame_latency_ms"]["p95"],
                    r["per_frame_latency_ms"]["p99"],
                    r["attribute_calls"],
                    attr_p50,
                ]
            )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[EXP-029] wrote {RESULTS_DIR / 'summary.json'}, latency_breakdown.csv")


if __name__ == "__main__":
    main()
