"""
Unit tests for src/url_features.py

Run:  pytest tests/ -v
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from url_features import extract_features, extract_features_vector, FEATURE_NAMES


class TestFeatureVectorShape:
    def test_returns_all_expected_keys(self):
        feats = extract_features("http://example.com")
        assert set(feats.keys()) == set(FEATURE_NAMES)

    def test_vector_matches_feature_names_order(self):
        feats = extract_features("http://example.com/path?x=1")
        vector = extract_features_vector("http://example.com/path?x=1")
        assert vector == [feats[name] for name in FEATURE_NAMES]

    def test_vector_length(self):
        vector = extract_features_vector("http://example.com")
        assert len(vector) == len(FEATURE_NAMES) == 31


class TestInputValidation:
    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            extract_features("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError):
            extract_features("   ")

    def test_non_string_raises(self):
        with pytest.raises(ValueError):
            extract_features(12345)

    def test_none_raises(self):
        with pytest.raises(ValueError):
            extract_features(None)

    def test_overlong_url_raises(self):
        with pytest.raises(ValueError):
            extract_features("http://example.com/" + "a" * 3000)

    def test_url_without_scheme_is_accepted(self):
        # app.py may receive "example.com/login" without a leading scheme
        feats = extract_features("example.com/login")
        assert feats["length_hostname"] == len("example.com")


class TestFeatureCorrectness:
    def test_ip_hostname_detected(self):
        feats = extract_features("http://192.168.1.5/login")
        assert feats["ip"] == 1

    def test_domain_hostname_not_flagged_as_ip(self):
        feats = extract_features("http://example.com/login")
        assert feats["ip"] == 0

    def test_dot_count(self):
        feats = extract_features("http://a.b.c.example.com")
        assert feats["nb_dots"] == 4

    def test_at_symbol_counted(self):
        feats = extract_features("http://example.com@evil.com/login")
        assert feats["nb_at"] == 1

    def test_https_token_flags_scheme_spoof_in_hostname(self):
        feats = extract_features("http://https-paypal.com/login")
        assert feats["https_token"] == 1

    def test_https_token_not_flagged_for_real_https(self):
        feats = extract_features("https://paypal.com/login")
        assert feats["https_token"] == 0

    def test_known_shortener_detected(self):
        feats = extract_features("http://bit.ly/abc123")
        assert feats["shortening_service"] == 1

    def test_suspicious_tld_detected(self):
        feats = extract_features("http://secure-login.xyz")
        assert feats["suspicious_tld"] == 1

    def test_common_tld_not_flagged_suspicious(self):
        feats = extract_features("http://example.com")
        assert feats["suspicious_tld"] == 0

    def test_phish_hint_words_counted(self):
        feats = extract_features("http://secure-account-verify-login.com")
        assert feats["phish_hints"] >= 3

    def test_ratio_digits_bounds(self):
        feats = extract_features("http://example.com/123456789")
        assert 0 <= feats["ratio_digits_url"] <= 1

    def test_length_url_matches_input(self):
        url = "http://example.com/some/path"
        feats = extract_features(url)
        assert feats["length_url"] == len(url)


class TestRelativeSignal:
    """Sanity check: known phishing-style URLs should score higher on
    heuristic features than a clean, short, reputable-looking URL. This is
    not a model-accuracy test — it just verifies the extractor's direction
    of effect matches domain intuition."""

    def test_phishing_style_url_scores_more_hints_than_clean_url(self):
        clean = extract_features("https://www.wikipedia.org/")
        suspicious = extract_features(
            "http://paypal-secure-login.xyz/verify/update/account?id=93af93")
        assert suspicious["phish_hints"] > clean["phish_hints"]
        assert suspicious["length_url"] > clean["length_url"]
        assert suspicious["suspicious_tld"] >= clean["suspicious_tld"]
