"""Types for the Code Detection guardrail."""

from typing import Any, List, Literal, Optional, TypedDict, cast

from pydantic import Field

from .base import GuardrailConfigModel


class CodeDetectionMatch(TypedDict, total=False):
    """A single language signal that contributed to the detection score."""

    language: str
    pattern: str
    score: float


class CodeDetectionGuardrailConfigModel(GuardrailConfigModel):
    """Configuration for the Code Detection guardrail."""

    score_threshold: float = Field(
        default=3.0,
        ge=0.0,
        description=(
            "Cumulative score across all language signals required to block the request. "
            "Each matched pattern contributes a weight. Higher = less sensitive."
        ),
        json_schema_extra=cast(
            Any,
            {
                "ui_type": "number",
                "min": 0.5,
                "max": 20.0,
                "step": 0.5,
                "default_value": 3.0,
            },
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Code Detection"
