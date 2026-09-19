"""EXP-001: Video Input Pipeline Baseline 실행 스크립트.

지침 9번(Phase 1) / 34번(실험 파일) 형식에 맞춰
- 여러 조건의 테스트 영상을 각각 처리하고
- 원본 FPS, 처리 FPS, 전체 Frame 수, 평균 처리시간, P50/P95 Latency, Dropped Frame을 측정해
- results/EXP-001/metrics.csv 로 저장한다.
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from video_pipeline.reader import run_baseline  # noqa: E402

TEST_VIDEOS = [
    ("normal_480p_11s", ROOT / "data/raw/pedestrian_crossing_480p.ogg", None),
    ("crowded_1080p_56s", ROOT / "data/raw/crowded_intersection_1080p.webm", None),
    ("short_3s_480p", ROOT / "data/test/short_3s_480p.mp4", None),
    ("no_person_5s", ROOT / "data/test/no_person_synthetic.mp4", None),
    ("corrupted_truncated", ROOT / "data/test/corrupted_truncated.mp4", None),
    ("normal_480p_resized_640", ROOT / "data/raw/pedestrian_crossing_480p.ogg", (640, 480)),
]


def main() -> None:
    out_dir = ROOT / "results/EXP-001"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metrics.csv"

    rows = []
    for name, path, resize_to in TEST_VIDEOS:
        stats = run_baseline(path, resize_to=resize_to)
        row = {
            "case": name,
            "path": str(path),
            "opened_ok": stats.opened_ok,
            "source_fps": round(stats.source_fps, 2),
            "resolution": f"{stats.source_resolution[0]}x{stats.source_resolution[1]}",
            "frames_read": stats.frames_read,
            "frames_failed": stats.frames_failed,
            "processed_fps": round(stats.processed_fps, 2),
            "avg_read_time_ms": round(stats.avg_read_time_ms, 3),
            "p50_latency_ms": round(stats.percentile_ms(50), 3),
            "p95_latency_ms": round(stats.percentile_ms(95), 3),
            "total_time_sec": round(stats.total_time_sec, 3),
            "error": stats.error or "",
        }
        rows.append(row)
        print(
            f"[{name}] opened={stats.opened_ok} frames={stats.frames_read} "
            f"failed={stats.frames_failed} processed_fps={row['processed_fps']} "
            f"p95={row['p95_latency_ms']}ms error={stats.error}"
        )

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
