"""Unit tests for json_sanitize (no GPU / cog required)."""

import json
import math
import unittest

from json_sanitize import sanitize_error_message, sanitize_for_json


class TestSanitizeForJson(unittest.TestCase):
    def test_nan_float_becomes_none(self):
        self.assertIsNone(sanitize_for_json(float("nan")))

    def test_inf_float_becomes_none(self):
        self.assertIsNone(sanitize_for_json(float("inf")))

    def test_nested_nan(self):
        out = sanitize_for_json({"scores": [1.0, float("nan"), 3.0]})
        self.assertEqual(out["scores"], [1.0, None, 3.0])

    def test_json_dumps_safe(self):
        payload = sanitize_for_json({"x": float("nan"), "y": [float("inf")]})
        json.dumps(payload)  # must not raise

    def test_primitives_unchanged(self):
        self.assertEqual(sanitize_for_json("en"), "en")
        self.assertEqual(sanitize_for_json(42), 42)
        self.assertIs(sanitize_for_json(True), True)


class TestSanitizeErrorMessage(unittest.TestCase):
    def test_truncates_long_messages(self):
        msg = "x" * 600
        out = sanitize_error_message(msg, max_len=100)
        self.assertLess(len(out), len(msg))
        self.assertIn("truncated", out)

    def test_binary_like_omitted(self):
        msg = "".join(chr(i) for i in range(32)) * 10
        self.assertIn("omitted", sanitize_error_message(msg))


if __name__ == "__main__":
    unittest.main()
