"""EXP-022: Circular Buffer에 Tier별 JPEG quality를 차등 적용 (EXP-021 Next Action #1).

EXP-021은 Circular Buffer에 넣는 모든 프레임에 quality를 고정값(85)으로 동일하게
적용했다. 하지만 실제 카지노 CCTV는 하루 대부분(이 시나리오 기준 1,600프레임 중
1,201프레임=75.1%)이 IDLE Tier이고, IDLE 프레임 대부분은 Pre-Roll Clip에 쓰이지 않고
그냥 evict된다 — Tier별로 quality를 다르게(IDLE은 낮게, EVENT는 높게) 적용하면 메모리를
추가로 절감할 수 있을 것이라는 게 EXP-021 Next Action의 가설이었다.

이 실험은 그 가설을 실제로 구현(`recording.adaptive.TierQualityConfig`)하고 검증한다.
핵심 질문은 "얼마나 절감되는가"뿐 아니라 "그 절감이 공짜인가"다 — 이 프로젝트의 실제
Pre-Roll 시나리오(trigger_frame=1339, window=[1089,1339])는 FC-008과 같은 "조용하다가
갑자기" 패턴이라 창문의 44.6%(112/251프레임)가 IDLE Tier와 겹친다. 즉 IDLE quality를
낮추면 그 손실이 실제 Event Clip에 그대로 남는다 — 이 트레이드오프를 실측한다.
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

from recording.adaptive import AdaptiveRecordingConfig, RecordingTier, TierQualityConfig, compute_tier_sequence  # noqa: E402
from recording.circular_buffer import FrameCircularBuffer  # noqa: E402
from recording.frame_codec import decode_frame, encode_frame  # noqa: E402

# --- EXP-019/020/021과 완전히 동일한 합성 시나리오 (프레임 내용 재현용) ----------------------
ASSET_PATH = ROOT / "data" / "raw" / "bus.jpg"
ASSUMED_FPS = 25.0
PHASE_IDLE_END = 1200
PHASE_NORMAL_END = 1300
EXTRA_DX_AMPLITUDE = 300
EXTRA_DX_PERIOD = 150

CONFIG = AdaptiveRecordingConfig(fps=ASSUMED_FPS, pre_roll_sec=10.0, post_roll_sec=10.0, idle_frame_stride=5, merge_gap_sec=0.0)

OUT_DIR = ROOT / "results" / "EXP-022"

# 대표 시나리오: Baseline(EXP-021 Uniform 85), Alt A(Uniform 낮은 quality 50 — EVENT까지 깎임),
# Tier-differentiated 후보들(IDLE만 낮추고 NORMAL/EVENT는 보존).
SCENARIOS: dict[str, TierQualityConfig] = {
    "baseline_uniform85": TierQualityConfig(idle_quality=85, normal_quality=85, event_quality=85),
    "alt_a_uniform50": TierQualityConfig(idle_quality=50, normal_quality=50, event_quality=50),
    "tier_diff_idle30": TierQualityConfig(idle_quality=30, normal_quality=75, event_quality=95),
    "tier_diff_idle50": TierQualityConfig(idle_quality=50, normal_quality=75, event_quality=95),
    "tier_diff_idle60": TierQualityConfig(idle_quality=60, normal_quality=75, event_quality=95),
    "tier_diff_idle70": TierQualityConfig(idle_quality=70, normal_quality=75, event_quality=95),
}


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


def mean_abs_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not ASSET_PATH.exists():
        raise FileNotFoundError(f"synthetic scene base image missing: {ASSET_PATH}")
    base_img = cv2.imread(str(ASSET_PATH))

    cache = json.loads((ROOT / "results" / "EXP-019" / "detection_cache.json").read_text())
    assert cache["frame_w"] == base_img.shape[1] and cache["frame_h"] == base_img.shape[0]
    frame_w, frame_h = cache["frame_w"], cache["frame_h"]
    frame_nbytes_raw = frame_w * frame_h * 3
    person_present, event_active = cache["person_present"], cache["event_active"]
    total_frames = len(person_present)

    tiers = compute_tier_sequence(person_present, event_active, CONFIG.post_roll_frames)
    tier_counts = {t: tiers.count(t) for t in RecordingTier}
    print(f"[EXP-022] tier distribution over {total_frames} frames: "
          f"IDLE={tier_counts[RecordingTier.IDLE]} NORMAL={tier_counts[RecordingTier.NORMAL]} EVENT={tier_counts[RecordingTier.EVENT]}")

    pre_roll_frames = CONFIG.pre_roll_frames  # 250
    trigger_frame = 1339  # EXP-020/021에서 실측 확정된 값
    window_start, window_end = trigger_frame - pre_roll_frames, trigger_frame
    assert (window_start, window_end) == (1089, 1339)
    window_tiers = tiers[window_start:window_end + 1]
    idle_idxs = [i for i, t in zip(range(window_start, window_end + 1), window_tiers) if t == RecordingTier.IDLE]
    normal_idxs = [i for i, t in zip(range(window_start, window_end + 1), window_tiers) if t == RecordingTier.NORMAL]
    event_idxs = [i for i, t in zip(range(window_start, window_end + 1), window_tiers) if t == RecordingTier.EVENT]
    print(f"[EXP-022] pre-roll window [{window_start},{window_end}] composition: "
          f"IDLE={len(idle_idxs)} NORMAL={len(normal_idxs)} EVENT={len(event_idxs)} "
          f"(IDLE fraction={len(idle_idxs) / len(window_tiers):.1%})")

    # 전체 1,600프레임을 실제로 생성한다 (bus.jpg 기반이라 가벼움) — 전체 Run 평균 메모리를
    # 실측하려면 IDLE/NORMAL/EVENT 각 구간의 실제 비율이 반영된 프레임이 전부 필요하다.
    print(f"[EXP-022] generating {total_frames} real frames...")
    real_frames: dict[int, np.ndarray] = {idx: make_frame(base_img, idx) for idx in range(total_frames)}

    capacity = 301  # EXP-021 "production_with_margin" 대표값

    scenario_rows = []
    for name, tq in SCENARIOS.items():
        # --- (A) 전체 Run 평균 메모리: 각 프레임을 자신의 Tier에 맞는 quality로 인코딩 ---
        encoded_sizes_by_tier: dict[RecordingTier, list[int]] = {t: [] for t in RecordingTier}
        buf: FrameCircularBuffer[bytes] = FrameCircularBuffer(capacity_frames=capacity)
        # 버퍼가 각 구간에서 실제로 갖는 메모리 스냅샷 (IDLE 정상상태 / NORMAL 정상상태 / trigger 시점)
        snapshot_points = {"idle_steady_state": PHASE_IDLE_END - 1, "normal_steady_state": PHASE_NORMAL_END - 1, "trigger": trigger_frame}
        snapshots: dict[str, int] = {}
        clip_encoded: list[bytes] | None = None
        for idx in range(total_frames):
            tier = tiers[idx]
            q = tq.quality_for_tier(tier)
            encoded = encode_frame(real_frames[idx], quality=q)
            encoded_sizes_by_tier[tier].append(len(encoded))
            buf.push(idx, encoded)
            for label, snap_idx in snapshot_points.items():
                if idx == snap_idx:
                    snapshots[label] = buf.total_bytes(len)
            if idx == trigger_frame:
                # Event가 이 프레임에서 트리거된 "그 순간" 버퍼에서 Pre-Roll을 꺼낸다 — 이후
                # 계속 push하면 오래된 프레임이 evict되어 늦게 꺼내면 창이 사라진다.
                cov = buf.coverage(window_start, window_end)
                assert cov.is_complete, f"scenario {name}: pre-roll window incomplete: {cov.missing_frames}"
                clip_encoded = buf.get_range(window_start, window_end)

        # 시간 비율 가중 평균 bytes/frame (실제 각 프레임을 인코딩해 얻은 값이므로 그 자체가
        # 이미 "전체 Run 평균"이다 — 별도 가중합이 필요 없다).
        all_sizes = [s for sizes in encoded_sizes_by_tier.values() for s in sizes]
        full_run_avg_bytes_per_frame = float(np.mean(all_sizes))

        # --- (B) 실제 Pre-Roll Clip: 트리거 시점에 꺼낸 window를 디코드해 원본과 비교 ---
        assert clip_encoded is not None
        clip_decoded = {idx: decode_frame(b) for idx, b in zip(range(window_start, window_end + 1), clip_encoded)}

        def segment_mae(idxs: list[int]) -> float | None:
            if not idxs:
                return None
            return float(np.mean([mean_abs_error(real_frames[i], clip_decoded[i]) for i in idxs]))

        idle_mae = segment_mae(idle_idxs)
        normal_mae = segment_mae(normal_idxs)
        event_mae = segment_mae(event_idxs)
        overall_mae = float(np.mean([mean_abs_error(real_frames[i], clip_decoded[i]) for i in range(window_start, window_end + 1)]))

        row = {
            "scenario": name,
            "idle_quality": tq.idle_quality,
            "normal_quality": tq.normal_quality,
            "event_quality": tq.event_quality,
            "full_run_avg_bytes_per_frame": full_run_avg_bytes_per_frame,
            "full_run_avg_MB_at_capacity301": full_run_avg_bytes_per_frame * capacity / 1e6,
            "buf_MB_idle_steady_state": snapshots["idle_steady_state"] / 1e6,
            "buf_MB_normal_steady_state": snapshots["normal_steady_state"] / 1e6,
            "buf_MB_at_trigger": snapshots["trigger"] / 1e6,
            "clip_idle_segment_mae": idle_mae,
            "clip_normal_segment_mae": normal_mae,
            "clip_event_segment_mae": event_mae,
            "clip_overall_mae": overall_mae,
        }
        scenario_rows.append(row)
        print(f"[EXP-022] {name}: full_run_avg={full_run_avg_bytes_per_frame:.0f}B/frame "
              f"buf@trigger={row['buf_MB_at_trigger']:.2f}MB "
              f"clip_mae(idle/normal/event/overall)="
              f"{idle_mae:.2f}/{normal_mae:.2f}/{event_mae:.2f}/{overall_mae:.2f}")

    baseline = next(r for r in scenario_rows if r["scenario"] == "baseline_uniform85")
    for row in scenario_rows:
        row["memory_saving_vs_baseline_pct"] = (
            (baseline["buf_MB_at_trigger"] - row["buf_MB_at_trigger"]) / baseline["buf_MB_at_trigger"] * 100
        )
        row["clip_overall_mae_delta_vs_baseline"] = row["clip_overall_mae"] - baseline["clip_overall_mae"]

    raw_memory_mb_at_capacity301 = capacity * frame_nbytes_raw / 1e6

    summary = {
        "trigger_frame": trigger_frame,
        "window": [window_start, window_end],
        "window_composition": {"idle": len(idle_idxs), "normal": len(normal_idxs), "event": len(event_idxs)},
        "tier_distribution_total_frames": {t.value: tier_counts[t] for t in RecordingTier},
        "capacity": capacity,
        "raw_uncompressed_memory_mb_at_capacity301": raw_memory_mb_at_capacity301,
        "scenarios": scenario_rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    with (OUT_DIR / "tier_quality_sweep.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scenario_rows[0].keys()))
        writer.writeheader()
        writer.writerows(scenario_rows)

    print(f"[EXP-022] wrote {OUT_DIR / 'summary.json'}, tier_quality_sweep.csv")


if __name__ == "__main__":
    main()
