"""Code Detection guardrail: blocks user messages that contain code."""

from typing import TYPE_CHECKING, Any, List, Literal, Optional, Union, cast

from litellm.types.guardrails import GuardrailEventHooks, SupportedGuardrailIntegrations

from .code_detection import CodeDetectionGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def _get_param(
    litellm_params: "LitellmParams",
    guardrail: "Guardrail",
    key: str,
    default: Any = None,
) -> Any:
    value = getattr(litellm_params, key, default)
    if value is not None:
        return value
    raw = guardrail.get("litellm_params")
    if isinstance(raw, dict) and key in raw:
        return raw[key]
    return default


def initialize_guardrail(
    litellm_params: "LitellmParams",
    guardrail: "Guardrail",
) -> CodeDetectionGuardrail:
    import litellm

    guardrail_name = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("Code Detection guardrail requires a guardrail_name")

    score_threshold = float(
        cast(
            Union[int, float, str],
            _get_param(litellm_params, guardrail, "score_threshold", 3.0),
        )
    )
    mode = _get_param(litellm_params, guardrail, "mode")
    event_hook = cast(
        Optional[Union[Literal["pre_call", "during_call"], List[str]]],
        mode if mode is not None else "pre_call",
    )

    instance = CodeDetectionGuardrail(
        guardrail_name=guardrail_name,
        score_threshold=score_threshold,
        event_hook=event_hook,
        default_on=bool(_get_param(litellm_params, guardrail, "default_on", False)),
    )
    litellm.logging_callback_manager.add_litellm_callback(instance)
    return instance


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.CODE_DETECTION.value: initialize_guardrail,
}

guardrail_class_registry = {
    SupportedGuardrailIntegrations.CODE_DETECTION.value: CodeDetectionGuardrail,
}

__all__ = [
    "CodeDetectionGuardrail",
    "initialize_guardrail",
]
