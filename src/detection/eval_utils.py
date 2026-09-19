"""Phase 2: Person Detection 평가용 유틸리티.

pycocotools 등 외부 평가 라이브러리에 의존하지 않고,
YOLO 포맷 Ground Truth와 예측 박스를 직접 IoU 매칭해 TP/FP/FN을 센다.
(지침 31/32: 직접 작성한 로직이므로 근거를 명확히 남긴다)
"""

from __future__ import annotations

from dataclasses import dataclass


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """YOLO 정규화 좌표(중심,폭,높이) -> 픽셀 좌표(x1,y1,x2,y2)."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return x1, y1, x2, y2


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def match_image(
    gt_boxes: list[tuple[float, float, float, float]],
    pred_boxes: list[tuple[float, float, float, float]],
    pred_scores: list[float],
    iou_threshold: float = 0.5,
) -> Counts:
    """한 이미지 내에서 GT와 예측을 confidence 내림차순 Greedy Matching으로 매칭한다."""
    order = sorted(range(len(pred_boxes)), key=lambda i: -pred_scores[i])
    matched_gt: set[int] = set()
    tp = 0
    for i in order:
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            v = iou(pred_boxes[i], gt)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= iou_threshold:
            matched_gt.add(best_j)
            tp += 1
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)
    return Counts(tp=tp, fp=fp, fn=fn)
