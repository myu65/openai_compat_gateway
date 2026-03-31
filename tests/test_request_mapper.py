from __future__ import annotations

import unittest

from app.mappers.request_mapper import build_include_list
from app.schemas.compat import BuiltinToolsConfig


class BuildIncludeListTests(unittest.TestCase):
    def test_web_search_defaults_to_sources_only(self) -> None:
        include = build_include_list(BuiltinToolsConfig(web_search=True))
        self.assertEqual(include, ["web_search_call.action.sources"])

    def test_web_search_results_can_be_enabled(self) -> None:
        include = build_include_list(
            BuiltinToolsConfig(web_search=True),
            include_web_search_results=True,
        )
        self.assertEqual(
            include,
            ["web_search_call.action.sources", "web_search_call.results"],
        )

    def test_other_builtin_includes_remain_enabled(self) -> None:
        include = build_include_list(
            BuiltinToolsConfig(
                web_search=True,
                file_search={"vector_store_ids": ["vs_123"]},
                code_interpreter=True,
            )
        )
        self.assertEqual(
            include,
            [
                "web_search_call.action.sources",
                "file_search_call.results",
                "code_interpreter_call.outputs",
            ],
        )


if __name__ == "__main__":
    unittest.main()
