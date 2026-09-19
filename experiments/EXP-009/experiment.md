# EXP-009 Edge Optimization — PyTorch FP32 vs ONNX FP32 vs ONNX INT8(Dynamic) ✅ 실행 완료

## Goal
YOLO11n을 ONNX로 변환하고 INT8 Dynamic Quantization까지 적용해, MacBook Air M3에서 Accuracy/Latency/Model Size Trade-off를 실측한다.

## Hypothesis
ONNX Runtime으로 변환하면 PyTorch보다 추론 속도가 빨라지고, INT8 양자화를 적용하면 모델 크기가 줄고 속도가 더 빨라질 것이다 (일반적으로 알려진 통념).

## Dataset
coco128(person class, Precision/Recall/F1 측정용) + crowded_intersection_1080p.webm(FPS/Latency 측정용)

## Environment
MacBook Air M3 (CPU), PyTorch 2.14, ONNX Runtime 1.30.0 (CPUExecutionProvider), Ultralytics export

## Configuration
- PyTorch FP32: `yolo11n.pt` (원본)
- ONNX FP32: `yolo11n.onnx` (opset 18, onnxslim 적용)
- ONNX INT8: `onnxruntime.quantization.quantize_dynamic`(Weight-only Dynamic Quantization, Calibration 데이터 불필요)
- conf=0.4, imgsz=640 동일 조건

## Baseline
PyTorch FP32

## Result

| Model | Size(MB) | Precision | Recall | F1 | FPS | P50 | P95 |
|---|---|---|---|---|---|---|---|
| **PyTorch FP32** (Baseline) | 5.61 | 0.9400 | 0.5551 | 0.6980 | **25.14** | 36.8ms | 61.6ms |
| ONNX FP32 | 10.74 | 0.9342 | 0.5591 | 0.6995 | 19.20 | 48.1ms | 77.8ms |
| ONNX INT8 (Dynamic) | **3.05** | 0.9205 | 0.5472 | 0.6864 | 21.95 | 42.7ms | 61.2ms |

## Failure Cases / 예상과 다른 결과
**가설이 반증됨**: ONNX FP32가 PyTorch FP32보다 오히려 19% 느렸고(25.14→19.20 FPS), 모델 크기도 더 컸다(5.61MB→10.74MB). INT8 양자화로 크기는 크게 줄었지만(3.05MB, -46%) 속도는 여전히 PyTorch보다 느렸다(21.95 FPS).

**원인 분석**:
1. **속도**: PyTorch는 Apple Silicon에 최적화된 네이티브 ARM 커널(Accelerate/BNNS 등)을 사용하는 반면, ONNX Runtime의 `CPUExecutionProvider`는 범용 커널이라 M3에서 반드시 더 빠르지 않을 수 있다. CoreML이나 전용 실행 프로바이더를 쓰지 않는 한 "ONNX = 빠름"이 항상 성립하지 않는다는 것을 실측으로 확인했다.
2. **크기**: ONNX FP32가 PyTorch(.pt)보다 큰 것은, .pt가 압축된 state_dict인 반면 ONNX는 그래프 구조 전체(연산자 메타데이터 포함)를 직렬화하기 때문으로 추정.
3. **INT8 속도 한계**: `quantize_dynamic`은 가중치만 INT8로 양자화하고 실제 연산(특히 Convolution)은 여전히 FP32로 계산되는 경우가 많아, CNN 중심인 YOLO 구조에서는 속도 이득이 제한적이다(가중치 압축 → 모델 크기 감소는 확실히 얻었지만, 연산 가속은 제한적). 진짜 속도 이득을 보려면 Static Quantization(Calibration 필요) 또는 CoreML/전용 NPU 백엔드가 필요할 것으로 추정 — 이번 세션에서는 검증하지 못함(추가 조사 필요로 명시).
4. **정확도**: F1은 PyTorch(0.698) ≈ ONNX FP32(0.700) > INT8(0.686) — 양자화로 인한 정확도 손실이 약간 있지만 크지 않음.

## Analysis
"ONNX/INT8 변환하면 무조건 빠르고 작아진다"는 통념이 이 환경(Apple Silicon CPU, CPUExecutionProvider)에서는 성립하지 않았다. 이는 실제 Edge 배포에서 **반드시 타겟 하드웨어에서 직접 측정해야 하는 이유**를 보여주는 좋은 사례다. 모델 크기(Edge Device 저장 공간)는 INT8이 확실히 유리하지만, 속도(실시간성)는 오히려 PyTorch 원본이 가장 우수했다.

## Decision
현재 MacBook Air M3 환경에서는 **PyTorch FP32를 그대로 사용**하는 것이 가장 합리적이다(속도 1위, 정확도 최상위권, 크기도 가장 작진 않지만 준수). ONNX/INT8 전환은 "모델 크기가 중요한 실제 Edge Camera(임베디드 NPU) 배포 시점"에 재검토하고, 그때는 CoreML 변환이나 실제 NPU 전용 백엔드(한화비전 Wisenet 9의 NPU 등, 물론 비공개라 실제 적용은 불가하지만 일반적인 NPU 툴체인 관점에서)를 함께 고려해야 한다는 결론을 남긴다.

## Next Action
Long Running Test 또는 Person Re-ID Stretch로 진행. Static Quantization/CoreML 비교는 백로그로 등록(추가 조사 필요).
