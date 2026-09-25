"""EXP-018: BestShot Position Score 개선 (FC-003 수정 검증).

배경 (FC-003, EXP-004 Failure Cases에서 최초 발견):
기존 position_score()는 bbox가 프레임 경계 margin(기본 2%) "안에 들어오기만" 하면
실제로 잘렸는지와 무관하게 무조건 0.3점을 줬다. EXP-004 Track 1에서는 실제로 잘리지
않은 좋은 사진이 Position 0.3점을 받았는데 다른 서브 점수가 높아 우연히 상쇄됐을 뿐,
로직 자체는 부정확했다.

이 실험은 "경계 근접"과 "실제로 잘림(clipping)"을 구분하는 것이 왜 다른 문제인지,
그리고 개선한 로직(경계에 실제로 닿았는지, boundary_eps=2px)이 legacy 방식(margin 근접)
보다 실제로 더 정확한지를 통제된 Ground Truth로 정량 검증한다.

Ground Truth를 실제로 만드는 방법:
Wikimedia 원본 영상은 이 원격 세션에서 접근 불가(EXP-010/014~017과 동일한 network 제약,
Decision 참고)하므로, ultralytics 패키지 내장 실제 사진 bus.jpg에서 YOLO11n으로 실제
검출된 사람 1명(conf=0.878)을 정사각형 crop으로 잘라내 "알려진 크기의 사람 패치"를 만든다.
이 패치를 배경 위 다양한 x_off/y_off 위치에 합성하면, 우리는 각 프레임에서 이 사람이
"실제로 프레임 밖으로 몇 px 잘렸는지"를 배치 좌표만으로 정확히 알 수 있다(합성이므로
Ground Truth가 100% 정확함 — 육안 라벨링이 아니라 기하학적으로 계산됨).
각 합성 프레임에 실제 YOLO11n을 다시 돌려 검출 bbox를 얻고, 그 bbox를 legacy/new
position_score에 넣어 "실제로 안 잘렸는데 낮은 점수(False Positive)"와
"실제로 잘렸는데 만점(False Negative)"을 둘 다 셀 수 있다.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from bestshot.scorer import position_score, position_score_margin_legacy  # noqa: E402

PERSON_CLASS_ID = 0
ASSET_CANDIDATES = [
    Path(sys.prefix) / "lib/python3.11/dist-packages/ultralytics/assets/bus.jpg",
    Path(sys.prefix) / "lib/python3.11/site-packages/ultralytics/assets/bus.jpg",
]

CANVAS_W, CANVAS_H = 810, 1080  # bus.jpg와 동일한 해상도 -> margin_ratio(2%) 절대 px 그대로 유지
STEP_PX = 8
PAD_OUTSIDE = 40  # 완전히 화면 밖으로 나갈 때까지 여유를 두고 슬라이드
LEGACY_MARGIN_RATIO = 0.02
NEW_BOUNDARY_EPS = 2.0
NEAR_EDGE_BUCKET_PX = 20  # legacy margin(2%*810≈16.2px, 2%*1080≈21.6px)과 겹치는 구간

OUT_DIR = ROOT / "results" / "EXP-018"


def find_asset() -> Path:
    for p in ASSET_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(f"ultralytics 샘플 이미지를 찾을 수 없음: {ASSET_CANDIDATES}")


def make_background(base_img: np.ndarray, canvas_w: int, canvas_h: int) -> np.ndarray:
    """사람이 없는 하늘/지붕 영역(EXP-017과 동일한 방식)을 타일링해 배경으로 쓴다."""
    h, w = base_img.shape[:2]
    strip_h = max(1, int(h * 0.15))
    strip = base_img[0:strip_h, :]
    reps_y = int(np.ceil(canvas_h / strip_h))
    reps_x = int(np.ceil(canvas_w / w))
    tiled = np.tile(strip, (reps_y, reps_x, 1))[:canvas_h, :canvas_w]
    return tiled.copy()


def extract_person_patch(base_img: np.ndarray, model: YOLO) -> tuple[np.ndarray, float]:
    """bus.jpg에서 실제 YOLO 검출로 잘 보이는 사람 1명을 골라 crop한다."""
    result = model.predict(base_img, classes=[PERSON_CLASS_ID], conf=0.4, imgsz=640, verbose=False)[0]
    boxes = result.boxes.xyxy.tolist()
    confs = result.boxes.conf.tolist()
    h, w = base_img.shape[:2]
    # 프레임 경계에서 충분히 떨어져 있어(원본에서 안 잘림) 합성용 "완전한 사람" 패치로 쓸 수 있는
    # 후보 중, Occlusion 없이 가장 뚜렷한(면적 큰) 사람을 선택한다.
    best = None
    for b, c in zip(boxes, confs):
        x1, y1, x2, y2 = b
        if x1 < 5 or y1 < 5 or x2 > w - 5 or y2 > h - 5:
            continue
        area = (x2 - x1) * (y2 - y1)
        if best is None or area > best[0]:
            best = (area, b, c)
    if best is None:
        raise RuntimeError("합성에 쓸 만한(경계에서 떨어진) 사람 검출 결과가 없음")
    _, bbox, conf = best
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    patch = base_img[y1:y2, x1:x2].copy()
    return patch, conf


def composite(canvas: np.ndarray, patch: np.ndarray, x_off: int, y_off: int) -> np.ndarray:
    frame = canvas.copy()
    h, w = frame.shape[:2]
    ph, pw = patch.shape[:2]
    dst_x1, dst_y1 = max(0, x_off), max(0, y_off)
    dst_x2, dst_y2 = min(w, x_off + pw), min(h, y_off + ph)
    if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        return frame
    src_x1, src_y1 = dst_x1 - x_off, dst_y1 - y_off
    src_x2, src_y2 = src_x1 + (dst_x2 - dst_x1), src_y1 + (dst_y2 - dst_y1)
    frame[dst_y1:dst_y2, dst_x1:dst_x2] = patch[src_y1:src_y2, src_x1:src_x2]
    return frame


def gt_min_gap_px(x_off: int, y_off: int, pw: int, ph: int, canvas_w: int, canvas_h: int) -> float:
    """패치의 '참(true)' 사각형 기준 캔버스 경계까지 최소 거리.
    음수면 그만큼 실제로 잘렸다는 뜻(Ground Truth, 합성 좌표로 정확히 계산됨)."""
    return min(x_off, canvas_w - (x_off + pw), y_off, canvas_h - (y_off + ph))


def detect_person_bbox(model: YOLO, frame: np.ndarray) -> tuple[float, float, float, float] | None:
    result = model.predict(frame, classes=[PERSON_CLASS_ID], conf=0.25, imgsz=640, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None
    confs = result.boxes.conf.tolist()
    boxes = result.boxes.xyxy.tolist()
    best_idx = int(np.argmax(confs))
    return tuple(boxes[best_idx])


def bucket_for(gap: float) -> str:
    if gap < 0:
        return "clipped"
    if gap < NEAR_EDGE_BUCKET_PX:
        return "near_edge_not_clipped"
    return "far_from_edge"


def run() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_img = cv2.imread(str(find_asset()))
    model = YOLO("yolo11n.pt")

    patch, patch_conf = extract_person_patch(base_img, model)
    ph, pw = patch.shape[:2]
    canvas = make_background(base_img, CANVAS_W, CANVAS_H)

    rows = []
    not_detected = 0

    # 수평 슬라이드: 왼쪽 밖 -> 오른쪽 밖, 수직은 중앙 고정
    y_off_h = (CANVAS_H - ph) // 2
    x_start, x_end = -pw + PAD_OUTSIDE, CANVAS_W - PAD_OUTSIDE
    for x_off in range(x_start, x_end + 1, STEP_PX):
        frame = composite(canvas, patch, x_off, y_off_h)
        bbox = detect_person_bbox(model, frame)
        gap = gt_min_gap_px(x_off, y_off_h, pw, ph, CANVAS_W, CANVAS_H)
        if bbox is None:
            not_detected += 1
            continue
        rows.append(
            {
                "sweep": "horizontal",
                "x_off": x_off,
                "y_off": y_off_h,
                "gt_min_gap_px": gap,
                "gt_is_clipped": gap < 0,
                "bucket": bucket_for(gap),
                "det_bbox": [round(v, 1) for v in bbox],
                "legacy_score": position_score_margin_legacy(bbox, CANVAS_W, CANVAS_H, LEGACY_MARGIN_RATIO),
                "new_score": position_score(bbox, CANVAS_W, CANVAS_H, NEW_BOUNDARY_EPS),
            }
        )

    # 수직 슬라이드: 위쪽 밖 -> 아래쪽 밖, 수평은 중앙 고정
    x_off_v = (CANVAS_W - pw) // 2
    y_start, y_end = -ph + PAD_OUTSIDE, CANVAS_H - PAD_OUTSIDE
    for y_off in range(y_start, y_end + 1, STEP_PX):
        frame = composite(canvas, patch, x_off_v, y_off)
        bbox = detect_person_bbox(model, frame)
        gap = gt_min_gap_px(x_off_v, y_off, pw, ph, CANVAS_W, CANVAS_H)
        if bbox is None:
            not_detected += 1
            continue
        rows.append(
            {
                "sweep": "vertical",
                "x_off": x_off_v,
                "y_off": y_off,
                "gt_min_gap_px": gap,
                "gt_is_clipped": gap < 0,
                "bucket": bucket_for(gap),
                "det_bbox": [round(v, 1) for v in bbox],
                "legacy_score": position_score_margin_legacy(bbox, CANVAS_W, CANVAS_H, LEGACY_MARGIN_RATIO),
                "new_score": position_score(bbox, CANVAS_W, CANVAS_H, NEW_BOUNDARY_EPS),
            }
        )

    def confusion(flag_key: str) -> dict:
        tp = fp = tn = fn = 0
        for r in rows:
            gt = r["gt_is_clipped"]
            flagged = r[flag_key] < 1.0
            if gt and flagged:
                tp += 1
            elif gt and not flagged:
                fn += 1
            elif not gt and flagged:
                fp += 1
            else:
                tn += 1
        n = tp + fp + tn + fn
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        accuracy = (tp + tn) / n if n else None
        return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": n, "precision": precision, "recall": recall, "accuracy": accuracy}

    def bucket_false_positive_rate(flag_key: str, bucket: str) -> dict:
        sub = [r for r in rows if r["bucket"] == bucket]
        flagged = sum(1 for r in sub if r[flag_key] < 1.0)
        n = len(sub)
        return {"n": n, "flagged_as_clipped": flagged, "flagged_rate": flagged / n if n else None}

    summary = {
        "canvas": {"w": CANVAS_W, "h": CANVAS_H},
        "patch": {"w": pw, "h": ph, "source_conf": patch_conf},
        "config": {
            "legacy_margin_ratio": LEGACY_MARGIN_RATIO,
            "new_boundary_eps": NEW_BOUNDARY_EPS,
            "near_edge_bucket_px": NEAR_EDGE_BUCKET_PX,
            "step_px": STEP_PX,
        },
        "n_frames_total": len(rows) + not_detected,
        "n_frames_scored": len(rows),
        "n_frames_not_detected": not_detected,
        "confusion_legacy": confusion("legacy_score"),
        "confusion_new": confusion("new_score"),
        "near_edge_not_clipped_false_positive_rate": {
            "legacy": bucket_false_positive_rate("legacy_score", "near_edge_not_clipped"),
            "new": bucket_false_positive_rate("new_score", "near_edge_not_clipped"),
        },
        "clipped_recall_bucket": {
            "legacy": bucket_false_positive_rate("legacy_score", "clipped"),
            "new": bucket_false_positive_rate("new_score", "clipped"),
        },
    }
    return summary, rows


if __name__ == "__main__":
    summary, rows = run()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(OUT_DIR / "rows.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    with open(OUT_DIR / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "legacy", "new"])
        writer.writerow(["accuracy", summary["confusion_legacy"]["accuracy"], summary["confusion_new"]["accuracy"]])
        writer.writerow(["precision", summary["confusion_legacy"]["precision"], summary["confusion_new"]["precision"]])
        writer.writerow(["recall", summary["confusion_legacy"]["recall"], summary["confusion_new"]["recall"]])
        writer.writerow(
            [
                "near_edge_not_clipped_false_positive_rate",
                summary["near_edge_not_clipped_false_positive_rate"]["legacy"]["flagged_rate"],
                summary["near_edge_not_clipped_false_positive_rate"]["new"]["flagged_rate"],
            ]
        )
    print(json.dumps(summary, indent=2))
    print("saved to", OUT_DIR)
