"""Parser checks. These URLs are fixtures, not training data."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features import FEATURE_NAMES, extract_features, feature_vector


class FeatureTests(unittest.TestCase):
    def test_feature_order_is_stable(self):
        first = extract_features("https://example.com")
        second = extract_features("http://example.org/a")
        self.assertEqual(list(first.keys()), FEATURE_NAMES)
        self.assertEqual(list(second.keys()), FEATURE_NAMES)
        self.assertEqual(len(FEATURE_NAMES), 12)
        self.assertEqual(len(set(FEATURE_NAMES)), 12)
        self.assertEqual(len(feature_vector("https://example.com")), 12)

    def test_https_and_http(self):
        secure = extract_features("https://example.com")
        plain = extract_features("http://example.com")
        self.assertEqual(secure["uses_https"], 1)
        self.assertEqual(plain["uses_https"], 0)
        self.assertEqual(secure["url_length"], len("https://example.com"))
        self.assertEqual(plain["url_length"], len("http://example.com"))
        self.assertEqual(secure["subdomain_count"], 0)
        self.assertEqual(secure["is_ip_address"], 0)

    def test_leading_www_is_removed(self):
        bare = extract_features("https://example.com/path")
        prefixed = extract_features("https://www.example.com/path")
        self.assertEqual(bare, prefixed)
        self.assertEqual(bare["url_length"], len("https://example.com/path"))
        self.assertEqual(bare["subdomain_count"], 0)

    def test_missing_scheme_becomes_https(self):
        features = extract_features("example.com")
        self.assertEqual(features["uses_https"], 1)
        self.assertEqual(features["url_length"], len("https://example.com"))

    def test_subdomains_query_and_keywords(self):
        url = "https://a.b.example.com/verify?id=42&x=3"
        features = extract_features(url)
        self.assertEqual(features["url_length"], len(url))
        self.assertEqual(features["dot_count"], url.count("."))
        self.assertEqual(features["subdomain_count"], 2)
        self.assertEqual(features["uses_https"], 1)
        self.assertEqual(features["suspicious_keyword_count"], 1)
        self.assertEqual(features["question_mark_count"], 1)
        self.assertEqual(features["equals_count"], 2)
        self.assertEqual(features["path_length"], len("/verify"))
        self.assertEqual(features["digit_count"], sum(1 for character in url if character.isdigit()))

    def test_ip_hostname_is_detected_from_the_host_only(self):
        ip_url = "http://192.168.0.1/login"
        features = extract_features(ip_url)
        self.assertEqual(features["is_ip_address"], 1)
        self.assertEqual(features["uses_https"], 0)
        self.assertEqual(features["subdomain_count"], 0)
        self.assertEqual(features["suspicious_keyword_count"], 1)
        self.assertEqual(features["digit_count"], 8)
        self.assertEqual(features["dot_count"], 3)
        self.assertEqual(features["path_length"], len("/login"))

        path_only = extract_features("https://example.com/192.168.0.1")
        self.assertEqual(path_only["is_ip_address"], 0)

    def test_at_symbol_uses_the_parsed_hostname(self):
        features = extract_features("https://user@example.com")
        self.assertEqual(features["at_count"], 1)
        self.assertEqual(features["is_ip_address"], 0)
        self.assertEqual(features["subdomain_count"], 0)

    def test_ipv6_hostname(self):
        features = extract_features("http://[2001:db8::1]/")
        self.assertEqual(features["is_ip_address"], 1)
        self.assertEqual(features["uses_https"], 0)
        self.assertEqual(features["subdomain_count"], 0)

    def test_rejects_bad_input(self):
        for bad in ["", "   ", "ftp://example.com", "http://", "javascript:alert(1)", "not a url"]:
            with self.assertRaises(ValueError):
                extract_features(bad)

    def test_rejects_very_long_api_input(self):
        long_url = "https://example.com/" + ("a" * 3000)
        with self.assertRaises(ValueError):
            extract_features(long_url, max_length=2048)
        # Training calls the extractor without the API length limit.
        features = extract_features(long_url)
        self.assertEqual(features["url_length"], len(long_url))

    def test_module_does_not_fetch_urls(self):
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "features.py")
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        self.assertNotIn("urlopen", source)
        self.assertNotIn("requests", source)
        self.assertNotIn("socket", source)


if __name__ == "__main__":
    unittest.main()
