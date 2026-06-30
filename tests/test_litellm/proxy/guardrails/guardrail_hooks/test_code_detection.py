"""Tests for the Code Detection guardrail."""

import pytest

from litellm.proxy.guardrails.guardrail_hooks.code_detection import (
    CodeDetectionGuardrail,
)
from litellm.proxy.guardrails.guardrail_hooks.code_detection.code_detection import (
    _score_text,
)
from litellm.types.guardrails import GuardrailEventHooks


# ── _score_text unit tests ──────────────────────────────────────────────────


class TestScoreText:
    def test_python_import_scores_above_threshold(self):
        score, matches = _score_text("import pandas as pd\nfrom sklearn import metrics")
        assert score >= 3.0
        assert any(m["language"] == "python" for m in matches)

    def test_python_function_def_scores_above_threshold(self):
        score, _ = _score_text("def calculate(x, y):\n    return x + y")
        assert score >= 3.0

    def test_javascript_const_arrow_scores_above_threshold(self):
        score, matches = _score_text(
            "const fetchData = async () => {\n  const res = await fetch(url);\n};"
        )
        assert score >= 3.0
        assert any(m["language"] == "javascript" for m in matches)

    def test_sql_select_query_scores_above_threshold(self):
        score, matches = _score_text("SELECT id, name FROM users WHERE active = 1")
        assert score >= 3.0
        assert any(m["language"] == "sql" for m in matches)

    def test_bash_shebang_scores_above_threshold(self):
        score, matches = _score_text("#!/bin/bash\necho 'hello'\nexport PATH=$PATH:/usr/local/bin")
        assert score >= 3.0
        assert any(m["language"] == "shell" for m in matches)

    def test_c_include_scores_above_threshold(self):
        score, matches = _score_text("#include <stdio.h>\nint main() { return 0; }")
        assert score >= 3.0
        assert any(m["language"] == "c/c++" for m in matches)

    def test_java_class_scores_above_threshold(self):
        score, _ = _score_text(
            "package com.example;\nimport java.util.List;\npublic class Main { }"
        )
        assert score >= 3.0

    def test_go_func_scores_above_threshold(self):
        score, matches = _score_text(
            "package main\nimport \"fmt\"\nfunc main() {\n    fmt.Println(\"hi\")\n}"
        )
        assert score >= 3.0
        assert any(m["language"] == "go" for m in matches)

    def test_rust_fn_scores_above_threshold(self):
        score, _ = _score_text("fn main() {\n    println!(\"hello\");\n}")
        assert score >= 3.0

    def test_php_tag_scores_above_threshold(self):
        score, matches = _score_text("<?php\n$name = 'world';\necho 'Hello ' . $name;\n?>")
        assert score >= 3.0
        assert any(m["language"] == "php" for m in matches)

    def test_plain_english_scores_below_threshold(self):
        score, _ = _score_text(
            "Can you help me understand how machine learning works? "
            "I want to know more about neural networks and training data."
        )
        assert score < 3.0

    def test_simple_question_scores_zero_or_low(self):
        score, _ = _score_text("What is the capital of France?")
        assert score < 1.0

    def test_text_with_import_keyword_in_sentence_does_not_trigger(self):
        # "import" in natural language without the pattern `import <module>`
        score, _ = _score_text("We need to import some data from the old system.")
        assert score < 3.0

    def test_text_mentioning_function_in_sentence_does_not_trigger(self):
        # "function" in natural language
        score, _ = _score_text(
            "The main function of the liver is to filter blood. "
            "Let me explain how it works."
        )
        assert score < 3.0

    def test_multiline_semicolon_lines_add_to_score(self):
        code = "int x = 1;\nint y = 2;\nint z = x + y;\n"
        score, _ = _score_text(code)
        assert score >= 1.0

    def test_empty_text_scores_zero(self):
        score, matches = _score_text("")
        assert score == 0.0
        assert matches == []

    def test_matches_list_contains_detected_languages(self):
        _, matches = _score_text("import os\nfrom pathlib import Path\ndef run():\n    pass")
        languages = {m["language"] for m in matches}
        assert "python" in languages

    def test_each_match_has_required_fields(self):
        _, matches = _score_text("SELECT * FROM orders WHERE id = 1")
        assert len(matches) > 0
        for m in matches:
            assert "language" in m
            assert "pattern" in m
            assert "score" in m
            assert m["score"] > 0


# ── CodeDetectionGuardrail integration tests ────────────────────────────────


class TestCodeDetectionGuardrail:
    def _make_inputs(self, *texts: str) -> dict:
        return {"texts": list(texts)}

    def _make_request(self) -> dict:
        return {"metadata": {}, "messages": []}

    @pytest.mark.asyncio
    async def test_blocks_python_code_in_request(self):
        guardrail = CodeDetectionGuardrail(guardrail_name="test", score_threshold=3.0)
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=self._make_inputs("import pandas as pd\nfrom sklearn import metrics"),
                request_data=self._make_request(),
                input_type="request",
            )
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_blocks_sql_in_request(self):
        guardrail = CodeDetectionGuardrail(guardrail_name="test", score_threshold=3.0)
        from fastapi import HTTPException

        with pytest.raises(HTTPException):
            await guardrail.apply_guardrail(
                inputs=self._make_inputs("SELECT id, email FROM users WHERE active = 1"),
                request_data=self._make_request(),
                input_type="request",
            )

    @pytest.mark.asyncio
    async def test_blocks_bash_script_in_request(self):
        guardrail = CodeDetectionGuardrail(guardrail_name="test", score_threshold=3.0)
        from fastapi import HTTPException

        with pytest.raises(HTTPException):
            await guardrail.apply_guardrail(
                inputs=self._make_inputs("#!/bin/bash\necho 'starting'\nexport FOO=bar"),
                request_data=self._make_request(),
                input_type="request",
            )

    @pytest.mark.asyncio
    async def test_allows_plain_english(self):
        guardrail = CodeDetectionGuardrail(guardrail_name="test", score_threshold=3.0)
        result = await guardrail.apply_guardrail(
            inputs=self._make_inputs("Please summarize the quarterly sales report for Q3."),
            request_data=self._make_request(),
            input_type="request",
        )
        assert result["texts"] == ["Please summarize the quarterly sales report for Q3."]

    @pytest.mark.asyncio
    async def test_does_not_inspect_response(self):
        guardrail = CodeDetectionGuardrail(guardrail_name="test", score_threshold=3.0)
        code_response = "import os\nfrom sys import argv\ndef main():\n    pass"
        result = await guardrail.apply_guardrail(
            inputs=self._make_inputs(code_response),
            request_data=self._make_request(),
            input_type="response",
        )
        assert result["texts"] == [code_response]

    @pytest.mark.asyncio
    async def test_empty_texts_passes_through(self):
        guardrail = CodeDetectionGuardrail(guardrail_name="test", score_threshold=3.0)
        result = await guardrail.apply_guardrail(
            inputs={"texts": []},
            request_data=self._make_request(),
            input_type="request",
        )
        assert result["texts"] == []

    @pytest.mark.asyncio
    async def test_higher_threshold_allows_borderline_input(self):
        guardrail = CodeDetectionGuardrail(guardrail_name="test", score_threshold=10.0)
        # Single Python import (score 2.0) — below the raised threshold of 10.0
        result = await guardrail.apply_guardrail(
            inputs=self._make_inputs("import os"),
            request_data=self._make_request(),
            input_type="request",
        )
        assert "import os" in result["texts"][0]

    @pytest.mark.asyncio
    async def test_lower_threshold_blocks_borderline_input(self):
        guardrail = CodeDetectionGuardrail(guardrail_name="test", score_threshold=1.5)
        from fastapi import HTTPException

        with pytest.raises(HTTPException):
            await guardrail.apply_guardrail(
                inputs=self._make_inputs("import os"),
                request_data=self._make_request(),
                input_type="request",
            )

    def test_default_event_hook_is_pre_call(self):
        guardrail = CodeDetectionGuardrail(guardrail_name="test")
        assert guardrail.event_hook == GuardrailEventHooks.pre_call

    def test_config_model_returns_correct_type(self):
        from litellm.types.proxy.guardrails.guardrail_hooks.code_detection import (
            CodeDetectionGuardrailConfigModel,
        )

        assert CodeDetectionGuardrail.get_config_model() is CodeDetectionGuardrailConfigModel
