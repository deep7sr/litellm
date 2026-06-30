"""
Code Detection guardrail.

Detects code in user messages using heuristic pattern scoring across all major
programming languages, without requiring fenced code blocks. Each matched
pattern contributes a weight to a cumulative score; if the score meets or
exceeds score_threshold the request is blocked.
"""

import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

from fastapi import HTTPException

from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel
from litellm.types.proxy.guardrails.guardrail_hooks.code_detection import (
    CodeDetectionMatch,
)
from litellm.types.utils import (
    GenericGuardrailAPIInputs,
    GuardrailStatus,
    GuardrailTracingDetail,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

# (compiled_pattern, weight, language_label)
# Weight semantics: a single match contributes this much to the cumulative score.
# Default threshold is 3.0, so a single weight-3.0 match (or three weight-1.0 matches) blocks.
_PATTERNS: Tuple[Tuple[re.Pattern, float, str], ...] = tuple(
    (re.compile(pattern, re.MULTILINE | re.IGNORECASE), weight, lang)
    for pattern, weight, lang in [
        # ── Python ──────────────────────────────────────────────────────────────
        (r"^\s*import\s+[\w.,\s]+$", 2.0, "python"),
        (r"^\s*from\s+[\w.]+\s+import\s+", 2.0, "python"),
        (r"^\s*def\s+\w+\s*\(.*\)\s*:", 2.5, "python"),
        (r"^\s*class\s+\w+[\s:(]", 2.0, "python"),
        (r"^\s*@\w[\w.]*(\(.*\))?\s*$", 1.0, "python"),
        (r"^\s*if\s+__name__\s*==\s*['\"]__main__['\"]", 3.0, "python"),
        (r"^\s*raise\s+\w+\(", 1.5, "python"),
        (r"^\s*with\s+.+\s+as\s+\w+\s*:", 1.5, "python"),
        (r"^\s*yield\s+", 1.5, "python"),
        (r"^\s*lambda\s+[\w,\s]*:", 1.5, "python"),
        # ── JavaScript / TypeScript ──────────────────────────────────────────────
        (r"^\s*(const|let|var)\s+\w+\s*=\s*.+", 1.5, "javascript"),
        (r"^\s*function\s+\w+\s*\(", 2.0, "javascript"),
        (r"^\s*=>\s*\{", 1.5, "javascript"),
        (r"^\s*export\s+(default\s+)?(function|class|const|let|var)\b", 2.0, "javascript"),
        (r"^\s*import\s+.+\s+from\s+['\"]", 2.0, "javascript"),
        (r"^\s*require\s*\(\s*['\"]", 1.5, "javascript"),
        (r"^\s*console\.(log|error|warn|info)\s*\(", 1.5, "javascript"),
        (r"^\s*async\s+function\s+\w+", 2.0, "javascript"),
        (r"^\s*await\s+\w+", 1.0, "javascript"),
        (r"===|!==", 1.0, "javascript"),
        # ── Java ────────────────────────────────────────────────────────────────
        (r"^\s*(public|private|protected)\s+(static\s+)?[\w<>\[\]]+\s+\w+\s*\(", 2.0, "java"),
        (r"^\s*import\s+[\w.]+\.\w+\s*;", 2.0, "java"),
        (r"^\s*package\s+[\w.]+\s*;", 2.5, "java"),
        (r"^\s*@Override\b", 2.0, "java"),
        (r"System\.out\.print(ln)?\s*\(", 1.5, "java"),
        # ── C / C++ ─────────────────────────────────────────────────────────────
        (r"^\s*#include\s*[<\"][\w./]+[>\"]", 3.0, "c/c++"),
        (r"^\s*#define\s+\w+", 2.0, "c/c++"),
        (r"^\s*int\s+main\s*\(", 3.0, "c/c++"),
        (r"^\s*(printf|scanf|malloc|free|cout|cin)\s*[(\[]", 1.5, "c/c++"),
        (r"^\s*using\s+namespace\s+\w+\s*;", 2.5, "c/c++"),
        (r"::\w+", 1.0, "c++"),
        # ── C# ──────────────────────────────────────────────────────────────────
        (r"^\s*namespace\s+[\w.]+\s*\{?", 2.5, "csharp"),
        (r"^\s*using\s+[\w.]+\s*;", 1.5, "csharp"),
        (r"^\s*Console\.(Write|WriteLine)\s*\(", 1.5, "csharp"),
        # ── SQL ─────────────────────────────────────────────────────────────────
        (r"^\s*SELECT\s+.+\s+FROM\s+\w+", 2.5, "sql"),
        (r"^\s*INSERT\s+INTO\s+\w+", 2.5, "sql"),
        (r"^\s*UPDATE\s+\w+\s+SET\s+", 2.5, "sql"),
        (r"^\s*DELETE\s+FROM\s+\w+", 2.5, "sql"),
        (r"^\s*CREATE\s+(TABLE|DATABASE|INDEX|VIEW)\s+", 2.5, "sql"),
        (r"^\s*DROP\s+(TABLE|DATABASE|INDEX|VIEW)\s+", 2.5, "sql"),
        (r"^\s*ALTER\s+TABLE\s+", 2.5, "sql"),
        (r"\bWHERE\s+\w+\s*(=|!=|<|>|LIKE|IN)\s*", 1.0, "sql"),
        (r"\bJOIN\s+\w+\s+ON\b", 1.5, "sql"),
        # ── Shell / Bash ─────────────────────────────────────────────────────────
        (r"^\s*#!/(usr/)?bin/(env\s+)?(bash|sh|zsh|python|node)", 3.0, "shell"),
        (r"^\s*echo\s+['\"\$]", 1.0, "shell"),
        (r"^\s*export\s+\w+=", 1.5, "shell"),
        (r"\|\s*(grep|awk|sed|cut|sort|uniq|xargs|tee)\b", 1.5, "shell"),
        (r"^\s*for\s+\w+\s+in\s+\$", 1.5, "shell"),
        (r"\$\([\w\s\-\.]+\)", 1.5, "shell"),
        (r"^\s*sudo\s+\w+", 1.5, "shell"),
        (r"^\s*chmod\s+[0-7]{3,4}\s+", 2.0, "shell"),
        (r"^\s*curl\s+(-\w+\s+)*https?://", 1.5, "shell"),
        # ── Go ──────────────────────────────────────────────────────────────────
        (r"^\s*package\s+\w+\s*$", 2.0, "go"),
        (r"^\s*import\s+\(\s*$", 2.0, "go"),
        (r"^\s*func\s+\w+\s*\(", 2.5, "go"),
        (r"^\s*fmt\.(Print|Println|Sprintf)\s*\(", 1.5, "go"),
        (r":=\s*", 1.0, "go"),
        # ── Rust ────────────────────────────────────────────────────────────────
        (r"^\s*fn\s+\w+\s*\(", 2.5, "rust"),
        (r"^\s*use\s+[\w:]+\s*;", 1.5, "rust"),
        (r"^\s*let\s+mut\s+\w+", 2.0, "rust"),
        (r"^\s*impl\s+\w+", 2.0, "rust"),
        (r"println!\s*\(", 1.5, "rust"),
        # ── Ruby ────────────────────────────────────────────────────────────────
        (r"^\s*require\s+['\"][\w\/]+['\"]", 1.5, "ruby"),
        (r"^\s*def\s+\w+(\s*\(.*\))?\s*$", 2.0, "ruby"),
        (r"^\s*attr_(accessor|reader|writer)\s+:", 2.0, "ruby"),
        (r"^\s*puts\s+", 1.0, "ruby"),
        # ── PHP ─────────────────────────────────────────────────────────────────
        (r"<\?php\b", 3.0, "php"),
        (r"^\s*\$\w+\s*=\s*.+;", 1.5, "php"),
        (r"echo\s+['\"\$].+;", 1.0, "php"),
        # ── HTML / XML ───────────────────────────────────────────────────────────
        (r"<(html|head|body|div|span|script|style)\b", 1.5, "html"),
        (r"</\w+>", 1.0, "html"),
        (r"<!DOCTYPE\s+html>", 3.0, "html"),
        # ── Generic structural code signals ─────────────────────────────────────
        # Multiple consecutive lines ending in semicolons
        (r"(^.+;\s*$\n){2,}", 1.5, "generic"),
        # Multiple consecutive lines with only a closing brace
        (r"(^\s*\}\s*$\n?){2,}", 1.0, "generic"),
    ]
)


def _score_text(text: str) -> Tuple[float, List[CodeDetectionMatch]]:
    """
    Compute the cumulative code signal score for a single text string.

    Patterns are matched at most once each against the full text (MULTILINE).
    Returns (total_score, list_of_matched_patterns).
    """
    total = 0.0
    matches: List[CodeDetectionMatch] = []
    for pattern, weight, lang in _PATTERNS:
        if pattern.search(text):
            total += weight
            matches.append(
                CodeDetectionMatch(language=lang, pattern=pattern.pattern, score=weight)
            )
    return total, matches


class CodeDetectionGuardrail(CustomGuardrail):
    """
    Guardrail that blocks user messages containing code, detected via heuristic
    pattern scoring across all major programming languages. Does not require
    fenced code blocks — raw pasted code is caught too.
    """

    def __init__(
        self,
        guardrail_name: Optional[str] = None,
        score_threshold: float = 3.0,
        event_hook: Optional[Union[Literal["pre_call", "during_call"], List[str]]] = None,
        default_on: bool = False,
        **kwargs: Any,
    ) -> None:
        _event_hook: Optional[Union[GuardrailEventHooks, List[GuardrailEventHooks]]] = None
        if event_hook is not None:
            if isinstance(event_hook, list):
                _event_hook = [GuardrailEventHooks(h) if isinstance(h, str) else h for h in event_hook]
            else:
                _event_hook = GuardrailEventHooks(event_hook)
        super().__init__(
            guardrail_name=guardrail_name or "code_detection",
            supported_event_hooks=[
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.during_call,
            ],
            event_hook=_event_hook or GuardrailEventHooks.pre_call,
            default_on=default_on,
            **kwargs,
        )
        self.score_threshold = max(0.0, score_threshold)

    @staticmethod
    def get_config_model() -> Optional[type[GuardrailConfigModel]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.code_detection import (
            CodeDetectionGuardrailConfigModel,
        )

        return CodeDetectionGuardrailConfigModel

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        # Only inspect user input, never the LLM response.
        if input_type != "request":
            return inputs

        start_time = datetime.now()
        all_matches: List[CodeDetectionMatch] = []
        status: GuardrailStatus = "success"
        exception_str = ""

        try:
            texts = inputs.get("texts", [])
            if not texts:
                return inputs

            best_score = 0.0
            blocking_lang = "unknown"
            for text in texts:
                score, matches = _score_text(text)
                if score > best_score:
                    best_score = score
                    all_matches = matches
                    if matches:
                        blocking_lang = matches[0]["language"]

            if best_score >= self.score_threshold:
                status = "guardrail_intervened"
                self.raise_passthrough_exception(
                    violation_message=(
                        f"Content blocked: code detected in message (language: {blocking_lang}, "
                        f"score: {best_score:.1f}/{self.score_threshold:.1f})"
                    ),
                    request_data=request_data,
                    detection_info={
                        "score": best_score,
                        "threshold": self.score_threshold,
                        "language": blocking_lang,
                    },
                )

            return inputs
        except HTTPException:
            status = "guardrail_intervened"
            raise
        except Exception as e:
            status = "guardrail_failed_to_respond"
            exception_str = str(e)
            raise
        finally:
            guardrail_response: Union[List[dict], str] = [dict(m) for m in all_matches]
            if status != "success" and not all_matches:
                guardrail_response = exception_str
            tracing_kw: Dict[str, Any] = {
                "guardrail_id": self.guardrail_name,
                "detection_method": "heuristic_scoring",
                "match_details": guardrail_response,
            }
            if all_matches:
                tracing_kw["confidence_score"] = min(
                    1.0, best_score / self.score_threshold if self.score_threshold > 0 else 1.0  # type: ignore[name-defined]
                )
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_provider="code_detection",
                guardrail_json_response=guardrail_response,
                request_data=request_data,
                guardrail_status=status,
                start_time=start_time.timestamp(),
                end_time=datetime.now().timestamp(),
                duration=(datetime.now() - start_time).total_seconds(),
                event_type=GuardrailEventHooks.pre_call,
                tracing_detail=GuardrailTracingDetail(**tracing_kw),  # type: ignore[typeddict-item]
            )
