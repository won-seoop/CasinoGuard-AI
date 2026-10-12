"""coco128 n=21 Person Crop Attribute(상/하의 색상) Ground Truth - 단일 진실 공급원.

## 왜 이 파일이 필요한가 (EXP-035, PAR-026)

이 GROUND_TRUTH(coco128 person crop 21개, upper/lower 42개 라벨)는 EXP-027에서
처음 수동 라벨링됐다. 이후 EXP-028/030/031/032/033/034가 전부 각자의 스크립트
안에 이 dict를 그대로 복사-붙여넣기했다(지침 31이 권장하는 evaluation/ 폴더
분리가 되어 있지 않았다).

EXP-032가 `000000000086_0` upper(black→white)와 `000000000634_0` upper
(black→gray) 2건의 라벨 오류를 발견해 교정했을 때, 이 교정은 EXP-033/EXP-034
스크립트의 복사본에는 반영됐지만 EXP-027/028/030/031/032 자신의 원본 스크립트
복사본에는 반영되지 않았다 - 복사-붙여넣기 구조라서 "한 곳을 고치면 전부 고쳐진다"는
보장이 전혀 없었다. EXP-035는 42개 라벨 전체를 처음부터 다시 육안 재검증하는
과정에서 이 Drift를 직접 발견했고, 추가로 이전에는 아무도 검증하지 않았던
5개 라벨 오류(370_0 upper, 415_0 lower, 536_0 upper/lower, 623_0 lower)를
새로 찾아냈다.

재발 방지책: 이 Ground Truth를 쓰는 모든 실험/스크립트는 이 모듈에서
`GROUND_TRUTH`를 import해야 한다. 절대 다시 복사-붙여넣기하지 않는다.
라벨을 교정할 경우 `CORRECTIONS`에 근거와 함께 기록하고 `GROUND_TRUTH`를
갱신한다(둘 다 이 한 파일에서만 바뀐다).
"""

from __future__ import annotations

# EXP-035에서 42개 라벨 전체를 results/EXP-027/crops/*.jpg 원본 이미지로 육안
# 재검증했다. 아래 CORRECTIONS 중 EXP-032 출처 2건은 기존에 이미 발견된 오류를
# 이 단일 소스로 통합한 것이고, EXP-035 출처 5건은 이번 전수 재검증에서 새로
# 발견한 오류다.
CORRECTIONS: list[dict[str, str]] = [
    {
        "name": "000000000086_0",
        "region": "upper",
        "old": "black",
        "new": "white",
        "source": "EXP-032",
        "reason": "흑백 모터사이클 사진 속 남성은 밝은 크림색 셔츠를 입고 있다 (원 주석 '어두운 자켓'은 오류).",
    },
    {
        "name": "000000000634_0",
        "region": "upper",
        "old": "black",
        "new": "gray",
        "source": "EXP-032",
        "reason": "그레이스케일 스케이트보더 사진에서 상의는 연회색 티셔츠, 검정은 하의(반바지)뿐이다.",
    },
    {
        "name": "000000000370_0",
        "region": "upper",
        "old": "pink",
        "new": "white",
        "source": "EXP-035",
        "reason": (
            "브로콜리를 든 여아 사진에서 상의(끈 나시)는 명확히 흰색/크림색이다. "
            "분홍색은 하단에 살짝 보이는 치마(lower)뿐이며, 원 주석이 '분홍 원피스'로 "
            "상/하를 뭉뚱그려 upper까지 분홍으로 잘못 라벨링했다."
        ),
    },
    {
        "name": "000000000415_0",
        "region": "lower",
        "old": "white",
        "new": "blue",
        "source": "EXP-035",
        "reason": (
            "테니스 선수(Federer) 사진에서 반바지는 선명한 로열 블루다. 흰색으로 보이는 "
            "부분은 손목 밴드/신발/양말뿐이며 반바지 자체에는 흰색 영역이 없다."
        ),
    },
    {
        "name": "000000000536_0",
        "region": "upper",
        "old": "black",
        "new": "purple",
        "source": "EXP-035",
        "reason": (
            "여성이 입은 원피스는 자연광에서 뚜렷한 와인/자주색 톤을 띠는 짙은 버건디 색상이다. "
            "무채색(black)이 아니라 COLOR_BUCKETS에 존재하는 purple에 더 가깝다."
        ),
    },
    {
        "name": "000000000536_0",
        "region": "lower",
        "old": "black",
        "new": "purple",
        "source": "EXP-035",
        "reason": "upper와 동일한 원피스(상하 연결)이므로 동일한 이유로 purple로 교정한다.",
    },
    {
        "name": "000000000623_0",
        "region": "lower",
        "old": "pink",
        "new": "black",
        "source": "EXP-035",
        "reason": (
            "원 주석의 '핑크 바지'는 인접한 다른 인물(분홍 옷을 입은 아이)의 옷이 크롭에 "
            "겹쳐 보인 것이다. 라벨 대상 인물(회색 가디건을 입은 여성) 본인의 하의는 "
            "짙은 회색/검정 스키니 팬츠이며 핑크 영역이 아니다."
        ),
    },
]

# EXP-027 원본 라벨 + 위 CORRECTIONS 7건(기존 2건 + 신규 5건)을 모두 반영한 최종본.
GROUND_TRUTH: dict[str, dict[str, str]] = {
    "000000000086_0": {"upper": "white", "lower": "black"},  # B&W 오토바이 남성, 밝은 크림 셔츠/어두운 바지 (EXP-032 교정)
    "000000000113_0": {"upper": "white", "lower": "black"},  # 크림 스트라이프 셔츠, 네이비 바지(경계사례, 교정 안 함)
    "000000000165_0": {"upper": "gray", "lower": "gray"},  # ACU 디지털 카모(그레이-탄 계열)
    "000000000165_1": {"upper": "black", "lower": "black"},  # 검정 정장+레드 타이
    "000000000328_0": {"upper": "black", "lower": "black"},  # B&W 해군 제복
    "000000000328_1": {"upper": "black", "lower": "black"},  # B&W 해군 제복(앉은 모습)
    "000000000328_2": {"upper": "black", "lower": "black"},  # B&W 해군 제복(모자 든 소년)
    "000000000368_0": {"upper": "red", "lower": "black"},  # 축구 유니폼(적색)+검정 반바지
    "000000000370_0": {"upper": "white", "lower": "pink"},  # 흰 나시(EXP-035 교정)+분홍 치마
    "000000000389_0": {"upper": "white", "lower": "white"},  # 흰 드레스셔츠(바지 미노출, 셔츠 연장)
    "000000000395_0": {"upper": "orange", "lower": "orange"},  # 주황 재킷
    "000000000415_0": {"upper": "blue", "lower": "blue"},  # 테니스 선수, 블루 셔츠+블루 반바지(EXP-035 교정)
    "000000000431_0": {"upper": "white", "lower": "white"},  # 테니스 선수(올화이트)
    "000000000446_0": {"upper": "gray", "lower": "black"},  # 그레이 니트(나비무늬)+검정 바지
    "000000000459_0": {"upper": "black", "lower": "black"},  # 어두운 정장(거울 셀카, 실제로는 짙은 갈색이나 COLOR_BUCKETS에 brown 없어 black이 최선의 근사)
    "000000000536_0": {"upper": "purple", "lower": "purple"},  # 버건디/와인색 원피스(EXP-035 교정)
    "000000000564_0": {"upper": "white", "lower": "white"},  # 테니스 선수(올화이트)
    "000000000572_0": {"upper": "black", "lower": "black"},  # 어두운 니트+어두운 바지
    "000000000572_1": {"upper": "black", "lower": "blue"},  # 네이비 스웨터(경계사례, 교정 안 함)+청바지
    "000000000623_0": {"upper": "gray", "lower": "black"},  # 그레이 가디건+짙은 회색/검정 팬츠(EXP-035 교정, 기존 '핑크'는 인접 인물의 옷)
    "000000000634_0": {"upper": "gray", "lower": "black"},  # 그레이스케일 스케이트보더, 연회색 티셔츠(EXP-032 교정)+검정 반바지
}


def assert_consistent() -> None:
    """GROUND_TRUTH와 CORRECTIONS가 서로 어긋나지 않는지 확인한다(Regression Test용)."""
    for c in CORRECTIONS:
        actual = GROUND_TRUTH[c["name"]][c["region"]]
        if actual != c["new"]:
            raise AssertionError(
                f"{c['name']}.{c['region']}: CORRECTIONS는 '{c['new']}'를 가리키지만 "
                f"GROUND_TRUTH에는 '{actual}'로 되어 있다."
            )
