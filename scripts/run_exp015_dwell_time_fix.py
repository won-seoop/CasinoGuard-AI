"""EXP-015: Dwell Time(누적 체류 시간) 0-버그 재현 및 DwellCounter로 수정 (FC-006, PAR-006).

Part A. 통제된 시나리오(Ground Truth 확정) — LoiteringDetector/DwellCounter를
        run_full_pipeline.py와 동일한 방식으로 직접 구동해 세 가지 대안을 비교한다.
  1) Buggy   : 기존 run_full_pipeline.py 코드 그대로 (forget_track() 먼저 호출 후 read)
  2) Alt A   : forget_track() 호출 순서만 바꿔 read를 먼저 함 (Alternative A, 채택 안 함)
  3) Alt B   : DwellCounter로 책임 분리 (Alternative B, 채택)

Part B. 실제 Detection(YOLO11n+ByteTrack) — EXP-010과 동일하게 이번 세션은 네트워크
        정책상 기존 Wikimedia 영상을 다시 받을 수 없어 ultralytics 기본 제공 실제
        사진(bus.jpg, 사람 4명)에 Pan/Jitter를 준 대체 입력을 사용한다(Decision 참고).
        같은 300프레임 입력에 대해 Buggy 공식과 DwellCounter로 각각 dwell_frames를
        계산해 "0이 아닌 Track 비율"을 실측 비교한다.
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

from events.dwell import DwellCounter  # noqa: E402
from events.loitering import LoiteringDetector  # noqa: E402
from events.roi_state_machine import bottom_center  # noqa: E402

PERSON_CLASS_ID = 0
ASSET_IMG = ROOT / ".venv/lib/python3.11/site-packages/ultralytics/assets/bus.jpg"
OUT_DIR = ROOT / "results" / "EXP-015"


# ---------------------------------------------------------------------------
# Part A: 통제된 시나리오 (Ground Truth 알려진 상태)
# ---------------------------------------------------------------------------

POLYGON = [(0, 0), (100, 0), (100, 100), (0, 100)]
INSIDE = (50, 50)
OUTSIDE = (500, 500)


def run_scenario(events: list[tuple[str, int]]) -> dict:
    """events: [("inside", n_frames) | ("outside", n_frames) | ("gap", n_frames), ...]
    "gap"은 Track이 그 프레임 수만큼 전혀 관측되지 않음을 뜻한다(run_full_pipeline의
    10프레임 초과 미관측 -> forget_track() 호출과 동일).
    """
    loiter_buggy = LoiteringDetector(polygon=POLYGON, threshold_frames=999999)
    loiter_altA = LoiteringDetector(polygon=POLYGON, threshold_frames=999999)
    dwell_counter = DwellCounter(polygon=POLYGON)

    frame_idx = 0
    ground_truth_inside_frames = 0
    dwell_buggy_last = 0
    dwell_altA_last = 0
    gap_run = 0  # 연속 미관측 프레임 수 (10 초과 시 forget_track 발생, run_full_pipeline과 동일 임계값)
    forgotten = False  # run_full_pipeline의 `del last_seen[tid]`와 동일 — forget은 한 번만 일어남

    for kind, n in events:
        for _ in range(n):
            if kind == "gap":
                gap_run += 1
                if gap_run > 10 and not forgotten:
                    forgotten = True
                    # Buggy: forget_track() 먼저 호출한 뒤 enter_frame을 읽음 -> 항상 기본값(idx)
                    loiter_buggy.forget_track(1)
                    dwell_buggy_last = frame_idx - loiter_buggy.enter_frame.get(1, frame_idx)
                    # Alt A: enter_frame을 먼저 읽은 뒤 forget_track() 호출
                    dwell_altA_last = frame_idx - loiter_altA.enter_frame.get(1, frame_idx)
                    loiter_altA.forget_track(1)
            else:
                gap_run = 0
                forgotten = False  # 다시 관측되면(재방문) 다음 소실 때 또 forget이 발생할 수 있음
                point = INSIDE if kind == "inside" else OUTSIDE
                loiter_buggy.update(1, point, frame_idx)
                loiter_altA.update(1, point, frame_idx)
                dwell_counter.update(1, point)
                if kind == "inside":
                    ground_truth_inside_frames += 1
            frame_idx += 1

    return {
        "ground_truth_dwell_frames": ground_truth_inside_frames,
        "buggy_dwell_frames": dwell_buggy_last,
        "alt_a_reorder_dwell_frames": dwell_altA_last,
        "alt_b_dwell_counter_frames": dwell_counter.get(1),
    }


SCENARIOS = {
    "occluded_while_inside": [("inside", 40), ("gap", 15)],
    "left_roi_before_vanishing": [("inside", 30), ("outside", 20), ("gap", 15)],
    "multi_visit_reentry": [("inside", 20), ("outside", 15), ("gap", 13), ("inside", 20), ("outside", 5), ("gap", 15)],
}


def run_part_a() -> list[dict]:
    rows = []
    for name, events in SCENARIOS.items():
        result = run_scenario(events)
        result["scenario"] = name
        rows.append(result)
        print(f"[Part A] {name}: {result}")
    return rows


# ---------------------------------------------------------------------------
# Part B: 실제 YOLO11n + ByteTrack Detection (bus.jpg Pan/Jitter, EXP-010과 동일한 대체 입력)
# ---------------------------------------------------------------------------


def make_frame(base_img: np.ndarray, frame_idx: int) -> np.ndarray:
    h, w = base_img.shape[:2]
    max_shift = 12
    dx = int(max_shift * np.sin(frame_idx * 0.037))
    dy = int(max_shift * np.cos(frame_idx * 0.053))
    shifted = np.roll(base_img, shift=(dy, dx), axis=(0, 1))
    brightness = 1.0 + 0.03 * np.sin(frame_idx * 0.011)
    return np.clip(shifted.astype(np.float32) * brightness, 0, 255).astype(np.uint8)


def run_part_b(n_frames: int = 300) -> dict:
    base_img = cv2.imread(str(ASSET_IMG))
    frame_h, frame_w = base_img.shape[:2]
    model = YOLO("yolo11n.pt")

    # ROI: 프레임 왼쪽 60% 영역 (bus.jpg 속 인물들이 이 범위 안팎을 오가도록 Pan과 겹치게 설계)
    roi_polygon = [(0, 0), (int(frame_w * 0.6), 0), (int(frame_w * 0.6), frame_h), (0, frame_h)]

    loiter_buggy = LoiteringDetector(polygon=roi_polygon, threshold_frames=999999)
    dwell_counter = DwellCounter(polygon=roi_polygon)

    last_seen: dict[int, int] = {}
    buggy_dwell: dict[int, int] = {}
    fixed_dwell: dict[int, int] = {}

    def forget(tid: int, idx: int) -> None:
        loiter_buggy.forget_track(tid)  # Buggy: forget 먼저
        buggy_dwell[tid] = idx - loiter_buggy.enter_frame.get(tid, idx)  # 항상 0이 되는 원인
        fixed_dwell[tid] = dwell_counter.get(tid)

    for idx in range(n_frames):
        frame = make_frame(base_img, idx)
        result = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[PERSON_CLASS_ID], conf=0.4, imgsz=640, verbose=False)[0]

        active_ids = []
        if result.boxes is not None and result.boxes.id is not None:
            ids = [int(i) for i in result.boxes.id.tolist()]
            boxes = result.boxes.xyxy.tolist()
            active_ids = list(zip(ids, boxes))

        for tid, box in active_ids:
            point = bottom_center(tuple(box))
            last_seen[tid] = idx
            loiter_buggy.update(tid, point, idx)
            dwell_counter.update(tid, point)

        for tid in list(last_seen.keys()):
            if idx - last_seen[tid] > 10:
                forget(tid, idx)
                del last_seen[tid]

    for tid in list(last_seen.keys()):
        forget(tid, n_frames)

    all_tids = sorted(set(buggy_dwell) | set(fixed_dwell))
    n_total = len(all_tids)
    n_buggy_nonzero = sum(1 for t in all_tids if buggy_dwell.get(t, 0) > 0)
    n_fixed_nonzero = sum(1 for t in all_tids if fixed_dwell.get(t, 0) > 0)

    per_track = [{"track_id": t, "buggy_dwell_frames": buggy_dwell.get(t, 0), "fixed_dwell_frames": fixed_dwell.get(t, 0)} for t in all_tids]

    summary = {
        "frames_processed": n_frames,
        "n_tracks_total": n_total,
        "n_tracks_dwell_nonzero_buggy": n_buggy_nonzero,
        "n_tracks_dwell_nonzero_fixed": n_fixed_nonzero,
        "pct_dwell_nonzero_buggy": round(100 * n_buggy_nonzero / n_total, 1) if n_total else 0.0,
        "pct_dwell_nonzero_fixed": round(100 * n_fixed_nonzero / n_total, 1) if n_total else 0.0,
    }
    print("[Part B]", summary)
    return {"summary": summary, "per_track": per_track}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    part_a_rows = run_part_a()
    with open(OUT_DIR / "part_a_scenarios.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(part_a_rows[0].keys()))
        writer.writeheader()
        writer.writerows(part_a_rows)

    part_b = run_part_b()
    with open(OUT_DIR / "part_b_summary.json", "w") as f:
        json.dump(part_b["summary"], f, indent=2)
    with open(OUT_DIR / "part_b_per_track.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(part_b["per_track"][0].keys()))
        writer.writeheader()
        writer.writerows(part_b["per_track"])

    print(f"\nResults written to {OUT_DIR}")


if __name__ == "__main__":
    main()
