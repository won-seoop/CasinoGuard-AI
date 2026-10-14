# EXP-036 Person Re-ID Baseline - Color Histogram vs YOLO Backbone 임베딩, Score Fusion 검증

관련 실험: EXP-027(PAR-018, coco128 person crop n=21/45 최초 확보), EXP-035(PAR-026,
Next Action에서 Person Re-ID를 다음 Stretch 후보로 명시)

## Goal

지침 18 Person Re-ID(Person Crop → Re-ID Network → Embedding → Cosine Similarity →
Similar Person Search)의 첫 Baseline을 실제로 구축한다. 지금까지 Core MVP와
Attribute Metadata(FC-012 전체 계열)에 집중해 Re-ID는 전혀 손대지 않은 상태였다
(Roadmap "[ ] (Stretch) Person Re-ID" 미착수).

## Hypothesis

1. ImageNet Pretrained CNN Backbone(YOLO11n-cls)의 GAP 특징이 고전적 Color
   Histogram Baseline보다 전반적으로 더 나은 Top-1/mAP를 낼 것이다(일반적으로
   학습된 Deep 특징이 손으로 설계한 특징보다 강건하다는 통념).
2. 두 임베딩의 Cosine Similarity를 Score-level로 가중 평균(50:50)하면 각자의
   약점을 상호 보완해 두 방법 중 더 나은 쪽보다도 전체 지표가 개선될 것이다.

## Problem

- Person Re-ID 전용 Pretrained 모델(OSNet 등, torchreid 생태계)의 공식 Weight는
  대부분 Google Drive에 호스팅되어 있다. 이 프로젝트의 egress 정책(pypi/GitHub
  Release Asset만 허용, EXP-027/028/035가 이미 실측으로 확인한 패턴)에서 접근
  가능한지 추측이 아니라 실제로 확인할 필요가 있었다.
- Market-1501/MSMT17 같은 정식 Multi-Camera Re-ID Dataset(지침 18 후보)도 전용
  서버/Google Drive 호스팅이라 이 세션에서 바로 받기 어렵다 - "같은 사람의 여러
  이미지"가 없는 상태에서 Re-ID 평가를 어떻게 설계할지가 선행 문제였다.

## Dataset

coco128 실제 person crop 45장(EXP-027이 이미 저장해 둔 `results/EXP-027/crops/*.jpg`,
32개 COCO 이미지에서 추출된 45명의 서로 다른 실제 인물 - 같은 이미지에 여러 명이
있어도 각 crop은 다른 bbox=다른 사람이므로 중복 신원 없음). 이 45명을 Gallery로
쓴다.

**Query는 실제 재촬영이 아니라 합성이다(한계, 명시)**: 사람마다 1장뿐이라 "같은
사람 다른 조건"을 실측할 수 없으므로, Gallery 원본에 조건별 변형을 가해 Query를
만든다.
- 조명 변화 4종(low_light/strong_light/warm_cast/cool_cast, EXP-025/027과 완전히
  동일한 합성 공식 재사용)
- Occlusion(bbox 폭 40~60%·높이 30~50% 영역을 중립 회색으로 덮음, 다른 사람/기둥에
  가려지는 상황 모사)
- Bbox Jitter(가장자리를 5~15% 비대칭으로 잘라내고 원래 크기로 Resize, Detector가
  Frame마다 bbox를 조금씩 다르게 잡는 상황 모사)
- 수평 Flip(다른 각도의 매우 거친 근사 - **실제 앞/뒤/옆모습 변화를 대체하지
  못하는 한계**를 Decision에서 다시 강조한다)

45명 x 7 variant = 315개 Query.

## Environment

coco128 crop은 재사용(재다운로드 없음). `yolo11n-cls.pt`(ImageNet Pretrained
분류 Backbone, GitHub Release Asset `v8.4.0/yolo11n-cls.pt` - 자동 해석됨, 5.5MB)를
신규 다운로드했다. ultralytics 8.4.177, torch 2.14.1+cu130(CPU로 실행), opencv
5.0.0. 이 컨테이너는 Python 3.11(`/usr/local/bin/python3`)과 3.13(`/usr/bin/pip`
설치 대상)이 분리되어 있어 `/usr/bin/python3.13`으로 torch/ultralytics/opencv를
설치·실행했다(혼선 방지를 위해 기록).

egress 실측(curl):
| URL | 결과 |
|---|---|
| `https://drive.google.com` | 403 (CONNECT tunnel failed) |
| `https://github.com/KaiyangZhou/deep-person-reid`(일반 저장소 페이지) | 403 |
| `https://github.com/ultralytics/assets/releases/download/.../yolo11n-cls.pt`(Release Asset) | 200 |
| `https://pypi.org/simple/...` | 200 |

→ "pypi + GitHub Release Asset만 허용, 그 외(Google Drive, GitHub 일반 페이지)는
차단"이라는 기존 세션들의 관찰이 Person Re-ID 영역에서도 그대로 재현됐다.

## Configuration

- Baseline: `color_histogram_embedding()` - HSV 3채널 Histogram(8,8,8 bin) 연결 +
  L2 정규화.
- 대안 A(검토, 미구현): OSNet(torchreid) - 위 egress 실측으로 Weight 접근 불가
  확인, 코드조차 작성하지 않고 기각.
- 대안 B(채택, 측정): `YoloBackboneEmbedder` - yolo11n-cls Backbone의 최종 Linear
  직전 GAP 특징(1280차원) + L2 정규화.
- Fusion: `rank_gallery_fused()` - 두 임베딩의 Cosine Similarity를 50:50 가중
  평균.
- 평가: `evaluate_identification()` - Top-1, Top-5, mAP(지침 18 Metric), 전체
  315개 Query와 variant별 45개 Query 단위로 각각 집계.

## Baseline

Color Histogram, 위 Configuration 참고.

## Result

### 1) 전체 평균: Hypothesis 1 확인(대안 B가 평균적으로 더 나음)

| Method | Top-1 | Top-5 | mAP |
|---|---|---|---|
| Baseline(Color Histogram) | 49.21% | 67.62% | 0.5809 |
| 대안 B(YOLO11n-cls Backbone) | **59.68%** | **71.75%** | **0.6614** |
| Fusion(50:50) | 52.38% | 70.16% | 0.6163 |

### 2) Variant별 분해: 평균의 착시 - 둘은 상호 보완이 아니라 상호 배타적 강점을 가진다

| Variant | Baseline Top-1 | 대안 B Top-1 | 더 나은 쪽 |
|---|---|---|---|
| flip | **100.0%** | 84.4% | Baseline |
| bbox_jitter | **95.6%** | 66.7% | Baseline |
| occlusion | **84.4%** | 64.4% | Baseline |
| low_light | **33.3%** | 2.2% | Baseline(큰 차이) |
| strong_light | 2.2% | 0.0% | 둘 다 붕괴 |
| warm_cast | 17.8% | **100.0%** | 대안 B(압도적) |
| cool_cast | 11.1% | **100.0%** | 대안 B(압도적) |

Baseline은 공간 변형(Flip/Bbox Jitter/Occlusion)에 거의 완벽하고(Histogram은
구조적으로 평행이동/대칭에 불변), 대안 B는 색 Cast(warm/cool_cast)에 완벽하다
(ImageNet Pretraining이 Color Jitter Augmentation을 광범위하게 겪어 채널 Cast에
강건한 특징을 학습한 것으로 보인다 - 반면 Histogram은 Hue bin이 채널 Cast로 직접
이동하므로 구조적으로 취약하다, FC-012의 "채널 Cast가 Hue 기반 분류를 틀어지게
한다"는 기존 발견과 같은 메커니즘). 둘 다 전역 노출(저노출/과다노출) 변화에는
약하지만, Baseline이 low_light에서는 그래도 대안 B보다 훨씬 낫다(33.3% vs 2.2%).

### 3) Fusion: Hypothesis 2 반증 - 50:50 평균은 대안 B 단독보다 오히려 나쁘다

Fusion 전체 Top-1(52.38%)은 대안 B 단독(59.68%)보다 **낮다**. Variant별로 보면
원인이 분명하다: warm_cast/cool_cast에서 대안 B는 100%를 내는데, 이 Variant에서
Baseline은 틀린 Gallery 인물에게 오히려 "자신 있게"(상대적으로 높은 Cosine
Similarity) 틀린 매칭을 준다 - 50:50 평균이 대안 B의 거의 완벽한 신호를 Baseline의
"확신 있게 틀린" 신호로 끌어내린다(warm_cast 100%→26.7%, cool_cast 100%→20.0%).
반대로 공간 변형(flip/bbox_jitter/occlusion)에서는 두 방법이 대체로 같은 방향으로
맞추고 있어 평균이 거의 둘 중 나은 쪽과 동일하거나 소폭 더 낫다(occlusion은 Fusion
88.9% > Baseline 84.4% > 대안 B 64.4% - 유일하게 둘 다를 능가한 사례).

## Failure Cases

**FC-013(신규)**: Person Re-ID 임베딩이 극단적 전역 노출 변화(low_light/strong_light)
에서 거의 무작위 수준으로 붕괴한다 - 특히 대안 B(YOLO Backbone)는 strong_light에서
Top-1 0.0%(45명 중 단 한 명도 자기 자신을 1위로 찾지 못함), low_light에서도 2.2%.
Baseline도 strong_light에서 2.2%로 동일하게 붕괴한다. 두 방법 모두 같은 약점을
공유하므로 Score Fusion으로도 구제되지 않는다(Fusion도 low_light 22.2%,
strong_light 13.3%로 여전히 Baseline 단독보다 못하거나 비슷함). FC-012(Attribute
색상 분류)가 이미 겪은 "노출 극단값에서 Hue/색 기반 신호가 무너진다"는 문제가
Re-ID 임베딩에도 형태를 바꿔 재현된 것으로 보인다 - 재현 조건/원인 가설은 Analysis
참고. 아직 수정하지 않음(이번 실험은 Baseline 구축과 측정이 목적).

## Analysis

1. **평균 지표의 착시(PAR-017/023/026과 같은 계열의 교훈)**: "대안 B가 평균적으로
   더 낫다"(Hypothesis 1)는 참이지만, 이 평균은 두 방법이 완전히 다른 조건에서
   이기고 지는 구조를 감춘다. 만약 전체 평균만 보고 대안 B를 production 후보로
   "그냥 교체"했다면, Baseline이 거의 완벽했던 Occlusion/Bbox Jitter/Flip에서
   오히려 Top-1이 20~35pp 하락하는 Regression을 알아차리지 못했을 것이다.
2. **Score Fusion이 실패하는 구체적 메커니즘**: 단순 가중 평균은 "두 신호가 서로
   다른 곳에서 약하다"는 상황에서만 상호 보완으로 작동한다. 한쪽이 특정 조건에서
   "확신 있게 틀리는"(단순히 약한 것이 아니라 오답에 높은 점수를 주는) 경우, 평균은
   다른 쪽의 강한 정답 신호를 깎아내리기만 한다 - warm_cast/cool_cast가 정확히
   이 사례였다. 이는 "Fusion이 항상 더 안전하다"는 암묵적 가정에 대한 반증 사례로
   기록할 가치가 있다.
3. **공간 불변성 대 판별력(Representational Trade-off)**: Color Histogram이
   공간 변형에 강한 것은 설계상 당연하다(위치 정보를 버리는 전역 통계이므로
   Flip/Crop에 불변) - 그러나 이 불변성이 바로 판별력의 한계와 같은 원인이다(위치
   정보를 버리면 다른 사람과 구별할 수 있는 정보도 함께 버려진다, 실제로
   Baseline은 45명 중 Top-1 49.2%로 대안 B보다 전체적으로는 약하다). 반대로 CNN
   GAP 특징은 판별력은 높지만(조명 Cast에 불변) 공간적 Context에 의존하는 ImageNet
   분류기 특성상 Crop 경계/좌우 반전에 의외로 취약하다 - "더 똑똑한 특징이 항상 더
   강건한 것은 아니다"라는 발견이다.

## Decision

1. 대안 A(OSNet)는 egress 실측(Google Drive 403)으로 접근 불가 확인 - 이 세션은
   구현을 시도하지 않는다(재발 방지: 이 experiment.md가 "왜 안 했는가"의 근거를
   보존).
2. 대안 B(YOLO11n-cls Backbone)를 Re-ID 임베딩의 **현재 최선 단일 후보**로
   기록한다 - 전체 Top-1/Top-5/mAP 모두에서 Baseline을 앞서고, 이미 파이프라인이
   쓰는 YOLO 계열 모델 하나만 더 로드하면 되므로 Edge 배포 비용이 낮다. 다만
   production(MetadataStore/VMS 연결)에는 아직 승격하지 않는다 - n=45(실제 신원
   기준)/합성 Query라는 Dataset 한계(지침 18 "Market-1501/MSMT17" 수준의 정식
   평가가 아님)와, low_light/strong_light에서 거의 무작위로 붕괴하는 FC-013이
   해결되지 않았기 때문이다.
3. 50:50 Score Fusion은 **채택하지 않는다** - 대안 B 단독보다 전체 지표가 낮다는
   것을 실측으로 확인했다(Hypothesis 2 반증). 조건에 따라 가중치를 동적으로
   바꾸는 더 정교한 Fusion(예: 두 임베딩이 서로 동의하지 않을 때 더 판별력이 높은
   쪽에 가중치를 더 주는 방식)은 다음 후보로 남기되, 이번 세션에서는 구현하지
   않는다(검증되지 않은 복잡도를 먼저 추가하지 않는다는 이 프로젝트의 원칙).

## Next Action

1. FC-013(저노출/과다노출 Re-ID 붕괴) 원인을 더 깊게 분석할 것 - FC-012의 배경
   기반 Gamma 보정(EXP-030 대안 B)이 Attribute에는 부분적으로 효과가 있었던
   것처럼, Re-ID 임베딩 추출 전에 같은 보정을 적용하면 개선되는지 검증.
2. 조건부(Confidence-aware) Fusion 설계 - 두 임베딩의 1위 후보가 일치하는지
   여부로 신뢰도를 추정해 가중치를 바꾸는 방식을 EXP-031(신뢰도 표시, 반증됨)의
   교훈을 참고해 설계하기 전에 먼저 "그 신호가 실제로 Accuracy를 예측하는가"를
   검증할 것(이 프로젝트가 반복적으로 확인한 "검증 없이 설계하지 않는다" 원칙).
3. 원본 Wide-angle CCTV 영상 접근이 가능해지면 이 합성 Query Protocol(조명/Occlusion/
   Bbox Jitter/Flip)을 실제 같은 사람의 여러 Track(동일 영상 내 재등장, People
   Counting/Loitering이 이미 쌓고 있는 Track 기록)으로 교체 검증할 것 - 진짜
   "같은 사람, 다른 Frame"이 되므로 이번 실험의 합성 한계를 벗어난다.
4. MetadataStore 연결은 FC-013 해결 또는 n 확장 이후로 보류.
