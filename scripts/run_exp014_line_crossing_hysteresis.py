"""EXP-014: Line Crossing Duplicate Event (FC-007) - band_px Hysteresis A/B.

Problem (FC-007, from 01. Architecture & Roadmap / EXP-012 Next Action):
LineCrossingDetector는 선을 기준으로 한 부호(side)가 바뀌는 "순간"마다 이벤트를 낸다
(src/events/line_crossing.py). 사람이 선 근처에서 멈추거나, Detection bbox가
프레임마다 몇 픽셀씩 흔들리면(Detector confidence/NMS jitter), bottom-center 포인트가
선을 여러 번 왔다갔다 건너는 것처럼 보여 같은 통과에 대해 A_TO_B/B_TO_A 이벤트가
반복 발생한다 (왕복 중복 이벤트). EXP-012(People Counting)는 같은 문제를 선까지의
거리 |d| >= band_px 일 때만 확정 side를 바꾸는 Hysteresis(Schmitt Trigger)로
해결했고, "같은 band를 LineCrossingDetector에도 적용할지 A/B"를 Next Action으로
남겼다 — 이 실험이 그 A/B를 수행한다.

Blocker / Decision (README/EXP-010과 동일):
이 원격 세션은 egress 정책상 Wikimedia 등 외부 일반 도메인이 차단되어 기존
crowded_intersection_1080p.webm을 다시 받을 수 없다. 대신 ultralytics 내장 실제
사진(사람이 여러 명 찍힌 실제 이미지)에 EXP-010과 동일한 방식으로 작은 Pan Jitter를
주어 "카메라가 고정된 채 사람이 화면 안에서 흔들리며 움직이는" 상황을 재현한다.
Detection 자체는 매 프레임 실제 YOLO11n 추론 결과이므로, "실제 검출 bbox의 프레임 간
흔들림이 선 통과 판정에 미치는 영향"이라는 이 실험의 목적에는 이 대체 입력으로 충분하다
(정확도 벤치마크가 목적이 아니라 이벤트 로직의 안정성 검증이 목적).

측정: band_px in {0, 3, 5, 10, 20}에서의 총 이벤트 수, "중복 쌍"(같은 Track에서
방향이 반대인 이벤트가 REVERSAL_WINDOW_SEC 이내에 연속 발생) 개수. 비교 대상으로
Alt A(시간 기반 Cooldown: 이벤트 발생 후 COOLDOWN_FRAMES 동안 같은 Track의 새 이벤트를
무시)도 같은 궤적으로 측정해 band_px Hysteresis와 정량 비교한다.
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

from events.line_crossing import CrossingEvent, LineCrossingDetector  # noqa: E402
from events.roi_state_machine import bottom_center  # noqa: E402

PERSON_CLASS_ID = 0
ASSET_CANDIDATES = [
    Path(sys.prefix) / "lib/python3.11/dist-packages/ultralytics/assets/bus.jpg",
    Path(sys.prefix) / "lib/python3.11/site-packages/ultralytics/assets/bus.jpg",
]
N_FRAMES = 900  # 약 36초(25fps 가정) - Pan 주기(EXP-010과 동일 주파수) 여러 번 포함
ASSUMED_FPS = 25.0
REVERSAL_WINDOW_SEC = 0.6  # Roadmap에 기록된 FC-007 관찰 기준(0.6초 내 방향 반전)과 동일
COOLDOWN_FRAMES = int(REVERSAL_WINDOW_SEC * ASSUMED_FPS)  # Alt A 비교용 (15 프레임)
BAND_VALUES = [0, 3, 5, 10, 20]


def find_asset() -> Path:
    for p in ASSET_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(f"ultralytics 샘플 이미지를 찾을 수 없음: {ASSET_CANDIDATES}")


def make_frame_generator(base_img: np.ndarray, frame_idx: int) -> np.ndarray:
    """EXP-010과 동일한 Pan/밝기 Jitter (재현성을 위해 같은 파라미터 사용)."""
    h, w = base_img.shape[:2]
    max_shift = 12
    dx = int(max_shift * np.sin(frame_idx * 0.037))
    dy = int(max_shift * np.cos(frame_idx * 0.053))
    shifted = np.roll(base_img, shift=(dy, dx), axis=(0, 1))
    brightness = 1.0 + 0.03 * np.sin(frame_idx * 0.011)
    frame = np.clip(shifted.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    return frame


class CooldownLineCrossing:
    """Alt A: band 대신 "이벤트 발생 후 N프레임 동안 같은 Track은 무시"하는 시간 기반 Debounce.
    핵심 로직(부호 판정)은 기존 band_px=0 LineCrossingDetector를 그대로 사용하고,
    이벤트 발생 여부만 Cooldown으로 걸러낸다 (검토했지만 채택하지 않은 대안, PAR-005 참고)."""

    def __init__(self, line_start, line_end, cooldown_frames: int):
        self.inner = LineCrossingDetector(line_start, line_end)
        self.cooldown_frames = cooldown_frames
        self.last_event_frame: dict[int, int] = {}

    def update(self, track_id: int, point, frame_idx: int) -> CrossingEvent | None:
        ev = self.inner.update(track_id, point, frame_idx)
        if ev is None:
            return None
        last = self.last_event_frame.get(track_id)
        self.last_event_frame[track_id] = frame_idx
        if last is not None and (frame_idx - last) < self.cooldown_frames:
            return None  # Cooldown 중이므로 이벤트 억제 (그러나 반대쪽으로 넘어간 상태 자체는 이미 확정됨)
        return ev


def count_reversal_pairs(events: list[CrossingEvent], fps: float, window_sec: float) -> int:
    """같은 track에서 direction이 반대인 이벤트가 window_sec 이내에 연속 발생하는 쌍의 수."""
    by_track: dict[int, list[CrossingEvent]] = {}
    for e in events:
        by_track.setdefault(e.track_id, []).append(e)
    pairs = 0
    for evs in by_track.values():
        evs.sort(key=lambda e: e.frame_idx)
        for a, b in zip(evs, evs[1:]):
            if a.direction != b.direction and (b.frame_idx - a.frame_idx) / fps <= window_sec:
                pairs += 1
    return pairs


def run() -> dict:
    base_img = cv2.imread(str(find_asset()))
    frame_h, frame_w = base_img.shape[:2]
    model = YOLO("yolo11n.pt")

    # 1차 패스: 실제 Track의 bottom-center y좌표 분포를 관찰해 흔들림 구간을 지나는 선을 고른다.
    ys_by_frame = []
    boxes_per_frame = []
    for idx in range(N_FRAMES):
        frame = make_frame_generator(base_img, idx)
        result = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[PERSON_CLASS_ID], conf=0.4, imgsz=640, verbose=False)[0]
        active = []
        if result.boxes is not None and result.boxes.id is not None:
            ids = [int(i) for i in result.boxes.id.tolist()]
            boxes = result.boxes.xyxy.tolist()
            active = list(zip(ids, boxes))
            for _, box in active:
                ys_by_frame.append(bottom_center(tuple(box))[1])
        boxes_per_frame.append(active)

    line_y = float(np.median(ys_by_frame))
    line_start, line_end = (0, line_y), (frame_w, line_y)
    print(f"virtual line y={line_y:.1f} (frame_h={frame_h}), observed bottom-center y std={np.std(ys_by_frame):.2f}")

    detectors = {f"band_{b}": LineCrossingDetector(line_start, line_end, band_px=b) for b in BAND_VALUES}
    detectors["cooldown_altA"] = CooldownLineCrossing(line_start, line_end, COOLDOWN_FRAMES)

    events_by_name: dict[str, list[CrossingEvent]] = {name: [] for name in detectors}
    last_seen: dict[int, int] = {}

    for idx, active in enumerate(boxes_per_frame):
        for tid, box in active:
            point = bottom_center(tuple(box))
            last_seen[tid] = idx
            for name, det in detectors.items():
                ev = det.update(tid, point, idx)
                if ev is not None:
                    events_by_name[name].append(ev)
        for tid in list(last_seen.keys()):
            if idx - last_seen[tid] > 10:
                for det in detectors.values():
                    inner = det.inner if isinstance(det, CooldownLineCrossing) else det
                    inner.forget_track(tid)
                del last_seen[tid]

    summary = {}
    for name, events in events_by_name.items():
        dup = count_reversal_pairs(events, ASSUMED_FPS, REVERSAL_WINDOW_SEC)
        summary[name] = {"total_events": len(events), "duplicate_reversal_pairs": dup}
        print(name, summary[name])

    return {
        "n_frames": N_FRAMES,
        "line_y": line_y,
        "cooldown_frames": COOLDOWN_FRAMES,
        "summary": summary,
    }


if __name__ == "__main__":
    result = run()
    out_dir = ROOT / "results" / "EXP-014"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "total_events", "duplicate_reversal_pairs"])
        for name, s in result["summary"].items():
            writer.writerow([name, s["total_events"], s["duplicate_reversal_pairs"]])
    print("saved to", out_dir)
