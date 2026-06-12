import unittest

from utils.json_utils import extract_json, require_json_fields


class JsonUtilsTests(unittest.TestCase):
    def test_extracts_fenced_json(self):
        parsed = extract_json('```json\n{"severity": "High"}\n```')
        self.assertEqual(parsed, {"severity": "High"})

    def test_rejects_missing_fields(self):
        with self.assertRaises(ValueError):
            require_json_fields('{"severity": "High"}', ["severity", "impact"])


if __name__ == "__main__":
    unittest.main()
