"""EXP-035(PAR-026): coco128 Attribute Color Ground Truth 단일 진실 공급원 재발 방지 Test.

evaluation/attribute_color/coco128_ground_truth.py가 생기기 전에는 같은 GROUND_TRUTH
dict가 6개 스크립트(EXP-027/028/030/031/032/033/034)에 복사-붙여넣기되어 있었고,
EXP-032가 찾은 라벨 교정 2건이 일부 복사본에만 반영되는 Drift가 실제로 발생했다.
이 Test는 (1) 단일 모듈이 내부적으로 일관적인지, (2) 앞으로 생기는 Drift를
`assert_consistent()`가 실제로 잡아내는지 검증한다.
"""

import pytest

from evaluation.attribute_color.coco128_ground_truth import CORRECTIONS, GROUND_TRUTH, assert_consistent
from src.attributes.color import COLOR_BUCKETS

N_EXPECTED_CROPS = 21


class TestGroundTruthStructure:
    def test_has_exactly_21_crops(self):
        assert len(GROUND_TRUTH) == N_EXPECTED_CROPS

    def test_every_crop_has_upper_and_lower(self):
        for name, regions in GROUND_TRUTH.items():
            assert set(regions.keys()) == {"upper", "lower"}, name

    def test_every_label_is_a_known_color_bucket(self):
        for name, regions in GROUND_TRUTH.items():
            for region, color in regions.items():
                assert color in COLOR_BUCKETS, f"{name}.{region} = {color!r} is not in COLOR_BUCKETS"

    def test_42_cells_total(self):
        n_cells = sum(len(regions) for regions in GROUND_TRUTH.values())
        assert n_cells == N_EXPECTED_CROPS * 2


class TestCorrectionsLog:
    def test_every_correction_changes_the_value(self):
        for c in CORRECTIONS:
            assert c["old"] != c["new"], f"{c['name']}.{c['region']}: old == new, not a real correction"

    def test_every_correction_targets_an_existing_cell(self):
        for c in CORRECTIONS:
            assert c["name"] in GROUND_TRUTH, c["name"]
            assert c["region"] in GROUND_TRUTH[c["name"]], (c["name"], c["region"])

    def test_has_both_exp032_and_exp035_sourced_corrections(self):
        """EXP-032가 찾은 2건(부분 감사)과 EXP-035가 찾은 5건(전수 감사)이 모두
        이 단일 모듈에 통합되어 있어야 한다 - 과거 Drift가 재발하지 않았음을 보증."""
        sources = {c["source"] for c in CORRECTIONS}
        assert "EXP-032" in sources
        assert "EXP-035" in sources
        assert sum(1 for c in CORRECTIONS if c["source"] == "EXP-032") == 2
        assert sum(1 for c in CORRECTIONS if c["source"] == "EXP-035") == 5


class TestAssertConsistent:
    def test_passes_on_the_real_module_state(self):
        assert_consistent() is None

    def test_catches_drift_between_ground_truth_and_corrections_log(self):
        """GROUND_TRUTH 값과 CORRECTIONS가 가리키는 값이 어긋나면 (예: 이 모듈을
        수정하다가 한쪽만 고치는 실수를 하면) 반드시 AssertionError로 드러나야 한다
        - 이것이 이번 PAR의 재발 방지 대책 자체를 테스트하는 Regression Test다."""
        original = GROUND_TRUTH["000000000086_0"]["upper"]
        try:
            GROUND_TRUTH["000000000086_0"]["upper"] = "black"  # CORRECTIONS는 "white"를 기대함
            with pytest.raises(AssertionError):
                assert_consistent()
        finally:
            GROUND_TRUTH["000000000086_0"]["upper"] = original
