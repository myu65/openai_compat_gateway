from __future__ import annotations

import os
import unittest

from app.config import _parse_env_bool


class ParseEnvBoolTests(unittest.TestCase):
    def test_missing_value_uses_default(self) -> None:
        os.environ.pop("TEST_BOOL", None)
        self.assertFalse(_parse_env_bool("TEST_BOOL", default=False))
        self.assertTrue(_parse_env_bool("TEST_BOOL", default=True))

    def test_truthy_values(self) -> None:
        for value in ("1", "true", "TRUE", " yes ", "on"):
            with self.subTest(value=value):
                os.environ["TEST_BOOL"] = value
                self.assertTrue(_parse_env_bool("TEST_BOOL"))

    def test_falsy_values(self) -> None:
        for value in ("0", "false", "off", "no", ""):
            with self.subTest(value=value):
                os.environ["TEST_BOOL"] = value
                self.assertFalse(_parse_env_bool("TEST_BOOL"))


if __name__ == "__main__":
    unittest.main()
