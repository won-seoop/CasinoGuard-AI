"""EXP-021: Circular Buffer 압축 청크 버퍼링으로 FC-009 해결 (지침 23, Edge Optimization).

FC-009 / EXP-020 Next Action:
"Circular Buffer(src/recording/circular_buffer.py)가 원본 BGR ndarray를 그대로 보관하면
Pre-Roll을 완전히 보장하는 최소 용량(251프레임)만 유지해도 채널당 628.21MB, 운영 여유분
(301프레임)은 753.35MB가 필요하다 — 카지노 CCTV는 채널이 수백 개 규모이므로 이 방식을
그대로 채널마다 적용하면 Edge Camera/NVR에 비현실적이다. 다음 세션 과제: 프레임을 즉시
JPEG/H.264로 압축해 버퍼링하는 방식을 설계·검증한다."

이 실험은 `FrameCircularBuffer`가 이미 Generic(`FrameT`)이라는 점을 이용한다 — 자료구조
자체를 바꾸지 않고, 버퍼에 넣기 직전에 프레임을 JPEG로 인코딩하고(`recording.frame_codec.
encode_frame`) 꺼낸 직후에 디코드(`decode_frame`)하는 것만으로 압축 버퍼링을 구현할 수
있는지, 그리고 그 대가(메모리 절감 vs CPU 인코딩 비용 vs 화질 손실)를 실측한다.

Detection(YOLO11n+ByteTrack+ROI State Machine)은 EXP-019가 이미 실제로 추론해 캐시해 둔
`results/EXP-019/detection_cache.json`을 그대로 재사용한다 — 이 실험은 Recording/Circular
Buffer/Codec 계층만 다루므로 같은 입력을 다시 추론하는 것은 낭비다. Circular Buffer가
실제로 필요한 구간은 EXP-020에서 이미 확인한 Pre-Roll 추출 시점(trigger_frame=1339,
window=[1089,1339], 251프레임)뿐이므로, 버퍼가 그 시점에 보관하고 있을 수 있는 최대
구간([1039,1339], capacity=301)만 실제 픽셀로 재생성한다 — 그보다 이전 프레임은 어차피
전부 evict되어 최종 결과에 영향을 주지 않는다.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import imageio_ffmpeg  # noqa: E402
import numpy as np  # noqa: E402

from recording.adaptive import AdaptiveRecordingConfig  # noqa: E402
from recording.circular_buffer import FrameCircularBuffer, required_capacity_for_pre_roll  # noqa: E402
from recording.frame_codec import decode_frame, encode_frame  # noqa: E402

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# --- EXP-019/020과 완전히 동일한 합성 시나리오 (프레임 내용 재현용) -----------------------
ASSET_PATH = ROOT / "data" / "raw" / "bus.jpg"
ASSUMED_FPS = 25.0
PHASE_IDLE_END = 1200
PHASE_NORMAL_END = 1300
EXTRA_DX_AMPLITUDE = 300
EXTRA_DX_PERIOD = 150

CONFIG = AdaptiveRecordingConfig(fps=ASSUMED_FPS, pre_roll_sec=10.0, post_roll_sec=10.0, idle_frame_stride=5, merge_gap_sec=0.0)

OUT_DIR = ROOT / "results" / "EXP-021"
VIDEO_DIR = OUT_DIR / "videos"

CAPACITY_SCENARIOS = {
    "undersized_edge_budget": 64,
    "off_by_one_naive": 250,
    "exact_minimum": 251,
    "production_with_margin": 301,
}
QUALITIES = [30, 50, 70, 85, 95]
DOWNSCALE_FACTOR = 0.5  # Alternative A 비교용


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


def ffmpeg_encode_frames(frames: list[np.ndarray], fps: float, size: tuple[int, int], gop: int, out_path: Path) -> float:
    """프레임 리스트를 libx264로 재인코딩해서 저장하고 CPU 시간(초)을 반환한다."""
    w, h = size
    cmd = [
        FFMPEG, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-g", str(gop), "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg encode failed: {cmd}")
    return time.perf_counter() - t0


def _count_frames_via_rawdecode(path: Path) -> int:
    """imageio_ffmpeg는 ffprobe를 번들하지 않으므로, mp4를 rawvideo로 재디코드해 프레임 수를 센다."""
    probe = subprocess.run(
        [FFMPEG, "-i", str(path)], stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    stderr = probe.stderr
    size = None
    for line in stderr.splitlines():
        if "Video:" in line:
            import re
            m = re.search(r"(\d{2,5})x(\d{2,5})", line)
            if m:
                size = (int(m.group(1)), int(m.group(2)))
    if size is None:
        raise RuntimeError(f"could not parse video size from ffmpeg output for {path}")
    w, h = size
    frame_bytes = w * h * 3
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE)
    return len(proc.stdout) // frame_bytes


def mean_abs_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    if not ASSET_PATH.exists():
        raise FileNotFoundError(f"synthetic scene base image missing: {ASSET_PATH}")
    base_img = cv2.imread(str(ASSET_PATH))

    cache = json.loads((ROOT / "results" / "EXP-019" / "detection_cache.json").read_text())
    assert cache["frame_w"] == base_img.shape[1] and cache["frame_h"] == base_img.shape[0], (
        "cached scenario resolution mismatch with regenerated base image"
    )
    frame_w, frame_h = cache["frame_w"], cache["frame_h"]
    frame_nbytes_raw = frame_w * frame_h * 3

    pre_roll_frames = CONFIG.pre_roll_frames  # 250
    trigger_frame = 1339  # EXP-020에서 실측 확정된 값, 아래 assert로 재검증
    window_start, window_end = trigger_frame - pre_roll_frames, trigger_frame
    assert (window_start, window_end) == (1089, 1339)

    max_capacity = max(CAPACITY_SCENARIOS.values())
    gen_start = trigger_frame - max_capacity + 1  # 1039
    print(f"[EXP-021] regenerating real frames [{gen_start},{trigger_frame}] ({max_capacity} frames)...")
    real_frames: dict[int, np.ndarray] = {idx: make_frame(base_img, idx) for idx in range(gen_start, trigger_frame + 1)}

    # -----------------------------------------------------------------
    # Part 1: quality별 실제 인코딩 (encode 1회, capacity별로 재사용)
    # -----------------------------------------------------------------
    # 단발 측정(1회 pass)은 노이즈가 커서 quality와 무관하게 뒤섞인다 — 실측 확인:
    # warm-up 1회 후에도 q30 decode가 q85보다 3배 이상 느리게 나오는 등 비단조적이었다.
    # REPEATS번 반복해 평균하면(quality별로 warm-up 1회 + 5회 측정) quality가 높을수록
    # encode/decode가 단조 증가하는 깨끗한 결과를 얻는다(별도 스크립트로 재확인 완료).
    TIMING_REPEATS = 5

    quality_results = []
    for quality in QUALITIES:
        encoded: dict[int, bytes] = {idx: encode_frame(real_frames[idx], quality=quality) for idx in real_frames}
        decoded = {idx: decode_frame(encoded[idx]) for idx in range(window_start, window_end + 1)}

        enc_times = []
        for _ in range(TIMING_REPEATS):
            t0 = time.perf_counter()
            for idx in sorted(real_frames):
                encode_frame(real_frames[idx], quality=quality)
            enc_times.append((time.perf_counter() - t0) / len(real_frames) * 1000)
        per_frame_encode_ms = float(np.mean(enc_times))

        dec_times = []
        for _ in range(TIMING_REPEATS):
            t0 = time.perf_counter()
            for idx in range(window_start, window_end + 1):
                decode_frame(encoded[idx])
            dec_times.append((time.perf_counter() - t0) / len(decoded) * 1000)
        per_frame_decode_ms = float(np.mean(dec_times))

        maes = [mean_abs_error(real_frames[idx], decoded[idx]) for idx in range(window_start, window_end + 1)]
        mean_mae = float(np.mean(maes))

        clip_path = VIDEO_DIR / f"idle_prefix_q{quality}.mp4"
        decoded_frames_ordered = [decoded[idx] for idx in range(window_start, window_end + 1)]
        reencode_cpu_sec = ffmpeg_encode_frames(decoded_frames_ordered, ASSUMED_FPS, (frame_w, frame_h), gop=25, out_path=clip_path)
        recovered_frame_count = _count_frames_via_rawdecode(clip_path)

        avg_encoded_bytes_per_frame = sum(len(v) for v in encoded.values()) / len(encoded)

        quality_results.append({
            "quality": quality,
            "avg_encoded_bytes_per_frame": avg_encoded_bytes_per_frame,
            "compression_ratio_vs_raw": frame_nbytes_raw / avg_encoded_bytes_per_frame,
            "per_frame_encode_ms": per_frame_encode_ms,
            "per_frame_decode_ms": per_frame_decode_ms,
            "mean_abs_pixel_error": mean_mae,
            "idle_prefix_clip_bytes": clip_path.stat().st_size,
            "idle_prefix_clip_frame_count": recovered_frame_count,
            "idle_prefix_clip_frame_count_expected": len(decoded_frames_ordered),
            "idle_prefix_reencode_cpu_sec": reencode_cpu_sec,
        })
        print(f"[EXP-021] quality={quality}: {avg_encoded_bytes_per_frame:.0f} bytes/frame, "
              f"MAE={mean_mae:.3f}, encode={per_frame_encode_ms:.3f}ms, decode={per_frame_decode_ms:.3f}ms, "
              f"clip_frames={recovered_frame_count}/{len(decoded_frames_ordered)}")

    # raw baseline reencode (압축 없음, MAE=0 기준선)
    raw_clip_path = VIDEO_DIR / "idle_prefix_raw.mp4"
    raw_frames_ordered = [real_frames[idx] for idx in range(window_start, window_end + 1)]
    raw_reencode_cpu_sec = ffmpeg_encode_frames(raw_frames_ordered, ASSUMED_FPS, (frame_w, frame_h), gop=25, out_path=raw_clip_path)
    raw_recovered_frame_count = _count_frames_via_rawdecode(raw_clip_path)

    # -----------------------------------------------------------------
    # Part 2: capacity별 메모리/완전성 (raw vs JPEG, quality=85 대표값 + 전체 quality 표)
    # -----------------------------------------------------------------
    capacity_results = []
    for name, capacity in CAPACITY_SCENARIOS.items():
        raw_mem = capacity * frame_nbytes_raw
        row = {"scenario": name, "capacity": capacity, "raw_memory_bytes": raw_mem}
        for quality in QUALITIES:
            buf: FrameCircularBuffer[bytes] = FrameCircularBuffer(capacity_frames=capacity)
            for idx in sorted(real_frames):
                buf.push(idx, encode_frame(real_frames[idx], quality=quality))
            cov = buf.coverage(window_start, window_end)
            total_bytes = buf.total_bytes(len)
            row[f"q{quality}_total_bytes"] = total_bytes
            row[f"q{quality}_complete"] = cov.is_complete
            row[f"q{quality}_missing_count"] = len(cov.missing_frames)
        capacity_results.append(row)
        print(f"[EXP-021] capacity={capacity} ({name}): raw={raw_mem/1e6:.2f}MB, "
              f"q85={row['q85_total_bytes']/1e6:.2f}MB complete={row['q85_complete']}")

    # -----------------------------------------------------------------
    # Part 3: Alternative A (해상도 절반 다운스케일 raw 버퍼) 실측 비교, 대표 capacity=301
    # -----------------------------------------------------------------
    ds_w, ds_h = int(frame_w * DOWNSCALE_FACTOR), int(frame_h * DOWNSCALE_FACTOR)
    ds_frame_nbytes = ds_w * ds_h * 3
    ds_capacity = CAPACITY_SCENARIOS["production_with_margin"]
    ds_memory_bytes = ds_capacity * ds_frame_nbytes
    ds_maes = []
    for idx in range(window_start, window_end + 1):
        original = real_frames[idx]
        down = cv2.resize(original, (ds_w, ds_h), interpolation=cv2.INTER_AREA)
        back = cv2.resize(down, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
        ds_maes.append(mean_abs_error(original, back))
    alt_a = {
        "downscale_factor": DOWNSCALE_FACTOR,
        "capacity": ds_capacity,
        "downscaled_resolution": f"{ds_w}x{ds_h}",
        "memory_bytes": ds_memory_bytes,
        "mean_abs_pixel_error_after_upscale": float(np.mean(ds_maes)),
    }
    print(f"[EXP-021] Alternative A (0.5x downscale raw): {ds_memory_bytes/1e6:.2f}MB @ capacity={ds_capacity}, "
          f"MAE(upscaled)={alt_a['mean_abs_pixel_error_after_upscale']:.3f}")

    summary = {
        "trigger_frame": trigger_frame,
        "window": [window_start, window_end],
        "frame_w": frame_w,
        "frame_h": frame_h,
        "frame_nbytes_raw": frame_nbytes_raw,
        "raw_baseline": {
            "idle_prefix_clip_bytes": raw_clip_path.stat().st_size,
            "idle_prefix_clip_frame_count": raw_recovered_frame_count,
            "idle_prefix_reencode_cpu_sec": raw_reencode_cpu_sec,
        },
        "quality_results": quality_results,
        "capacity_results": capacity_results,
        "alternative_a_downscale": alt_a,
        "required_capacity_for_pre_roll": required_capacity_for_pre_roll(pre_roll_frames),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    import csv
    with (OUT_DIR / "quality_sweep.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(quality_results[0].keys()))
        writer.writeheader()
        writer.writerows(quality_results)

    with (OUT_DIR / "capacity_memory.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(capacity_results[0].keys()))
        writer.writeheader()
        writer.writerows(capacity_results)

    print(f"[EXP-021] wrote {OUT_DIR / 'summary.json'}, quality_sweep.csv, capacity_memory.csv")


if __name__ == "__main__":
    main()
