from quishguard.fusion.scoring import fuse, ground_truth_tier, tier


def test_proposal_worked_example():
    # Appendix E of the proposal: URL 78, visual 61 -> 72.05 -> Phishing
    f = fuse(78, 61)
    assert f.score == 72.05 and f.tier == "Phishing"


def test_tier_boundaries():
    assert tier(0) == "Safe" and tier(25) == "Safe"
    assert tier(25.01) == "Suspicious" and tier(50) == "Suspicious"
    assert tier(51) == "Phishing" and tier(75) == "Phishing"
    assert tier(76) == "Critical" and tier(100) == "Critical"


def test_non_url_payload_uses_visual_only():
    f = fuse(None, 40)
    assert f.score == 40 and "visual only" in f.rule


def test_visual_only_is_capped_at_phishing():
    f = fuse(None, 99)                  # one signal only -> at most Phishing
    assert f.score == 75 and f.tier == "Phishing"


def test_floor_is_off_for_normal_looking_code():
    f = fuse(2, 30)                     # weighted = 11.8; visual < 50, so no floor
    assert f.score == 11.8 and "weighted" in f.rule and f.tier == "Safe"


def test_strong_tamper_floor_on_harmless_looking_link():
    f = fuse(5, 95)                     # weighted = 36.5, floor = 57
    assert f.score == 57 and f.tier == "Phishing"


def test_both_bad_is_critical():
    assert fuse(98, 90).tier == "Critical"


def test_clean_code_and_safe_link_is_safe():
    assert fuse(3, 4).tier == "Safe"


def test_ground_truth_rule():
    assert ground_truth_tier(True, True) == "Critical"
    assert ground_truth_tier(True, False) == "Phishing"
    assert ground_truth_tier(False, True) == "Phishing"
    assert ground_truth_tier(False, False) == "Safe"
