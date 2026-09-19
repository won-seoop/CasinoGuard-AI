import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from detection.eval_utils import iou, match_image, yolo_to_xyxy  # noqa: E402


def test_iou_identical_boxes_is_one():
    box = (0, 0, 10, 10)
    assert iou(box, box) == 1.0


def test_iou_no_overlap_is_zero():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_partial_overlap():
    # 겹치는 영역 5x10=50, union = 100+100-50=150
    v = iou((0, 0, 10, 10), (5, 0, 15, 10))
    assert abs(v - 50 / 150) < 1e-6


def test_yolo_to_xyxy_conversion():
    # 중심(0.5,0.5), 폭/높이(0.5,0.5), 이미지 100x100 -> (25,25,75,75)
    x1, y1, x2, y2 = yolo_to_xyxy(0.5, 0.5, 0.5, 0.5, 100, 100)
    assert (round(x1), round(y1), round(x2), round(y2)) == (25, 25, 75, 75)


def test_match_image_perfect_match_all_tp():
    gt = [(0, 0, 10, 10), (20, 20, 30, 30)]
    pred = [(0, 0, 10, 10), (20, 20, 30, 30)]
    scores = [0.9, 0.8]
    c = match_image(gt, pred, scores, iou_threshold=0.5)
    assert (c.tp, c.fp, c.fn) == (2, 0, 0)


def test_match_image_false_positive_when_extra_pred():
    gt = [(0, 0, 10, 10)]
    pred = [(0, 0, 10, 10), (50, 50, 60, 60)]
    scores = [0.9, 0.7]
    c = match_image(gt, pred, scores, iou_threshold=0.5)
    assert (c.tp, c.fp, c.fn) == (1, 1, 0)


def test_match_image_false_negative_when_missed_gt():
    gt = [(0, 0, 10, 10), (50, 50, 60, 60)]
    pred = [(0, 0, 10, 10)]
    scores = [0.9]
    c = match_image(gt, pred, scores, iou_threshold=0.5)
    assert (c.tp, c.fp, c.fn) == (1, 0, 1)


def test_match_image_lower_confidence_pred_loses_duplicate_match():
    # 두 예측이 같은 GT를 두고 경쟁하면 confidence 높은 쪽만 TP
    gt = [(0, 0, 10, 10)]
    pred = [(0, 0, 10, 10), (1, 1, 11, 11)]
    scores = [0.5, 0.95]
    c = match_image(gt, pred, scores, iou_threshold=0.3)
    assert c.tp == 1
    assert c.fp == 1
