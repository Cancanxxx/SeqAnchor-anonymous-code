"""One fail-closed chat renderer shared by every scientific model call.

The confirmation, predictor, and live-agenda stages must not silently render the
same message list differently.  In particular, Qwen3 requires an explicit
``enable_thinking=False`` declaration.  This module owns that declaration's
validation, identity hash, and the only supported ``apply_chat_template`` call.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional

from .provenance import canonical_json_bytes, sha256_bytes, sha256_text

CHAT_RENDERER_ID = "shared-chat-renderer-v1"
RESERVED_TEMPLATE_KWARGS = {"add_generation_prompt", "date_string", "tokenize"}
VALID_MESSAGE_ROLES = {"assistant", "system", "user"}


def _validated_json_value(value: Any, *, path: str) -> Any:
    """Return a JSON-safe copy while rejecting ambiguous/non-canonical values."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain a non-finite float")
        return value
    if isinstance(value, list):
        return [
            _validated_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{path} object keys must be non-empty strings")
            result[key] = _validated_json_value(item, path=f"{path}.{key}")
        return result
    raise ValueError(f"{path} contains a value that is not canonical JSON")


def validate_chat_template_kwargs(value: Any) -> Dict[str, Any]:
    """Validate and copy the model-specific, prospectively frozen kwargs."""

    if not isinstance(value, Mapping):
        raise ValueError("chat_template_kwargs must be an object")
    overlap = RESERVED_TEMPLATE_KWARGS.intersection(value)
    if overlap:
        raise ValueError(
            "chat_template_kwargs may not override renderer-owned arguments: "
            f"{sorted(overlap)}"
        )
    validated = _validated_json_value(value, path="chat_template_kwargs")
    return dict(sorted(validated.items()))


def chat_template_kwargs_sha256(value: Any) -> str:
    """Return the canonical identity of validated template kwargs."""

    return sha256_bytes(canonical_json_bytes(validate_chat_template_kwargs(value)))


def validate_template_declaration(
    *, chat_template_kwargs: Any, declared_sha256: str
) -> Dict[str, Any]:
    """Fail closed if a registry's kwargs and digest do not identify one another."""

    if not isinstance(declared_sha256, str) or len(declared_sha256) != 64:
        raise ValueError("chat_template_kwargs_sha256 must be a SHA-256 digest")
    try:
        int(declared_sha256, 16)
    except ValueError as error:
        raise ValueError("chat_template_kwargs_sha256 must be a SHA-256 digest") from error
    validated = validate_chat_template_kwargs(chat_template_kwargs)
    observed = sha256_bytes(canonical_json_bytes(validated))
    if observed != declared_sha256:
        raise ValueError("chat_template_kwargs digest differs from its declaration")
    return validated


def validate_messages(messages: Any) -> List[Dict[str, str]]:
    """Validate the exact role/content representation used by every runner."""

    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise ValueError("messages must be a non-empty sequence")
    if not messages:
        raise ValueError("messages must be a non-empty sequence")
    validated: List[Dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
            raise ValueError(f"messages[{index}] must contain exactly role and content")
        role = message["role"]
        content = message["content"]
        if role not in VALID_MESSAGE_ROLES:
            raise ValueError(f"messages[{index}].role is unsupported")
        if not isinstance(content, str) or not content:
            raise ValueError(f"messages[{index}].content must be a non-empty string")
        validated.append({"role": role, "content": content})
    return validated


def render_chat_prompt(
    tokenizer: Any,
    messages: Any,
    *,
    fixed_date: str,
    chat_template_kwargs: Optional[Mapping[str, Any]] = None,
) -> str:
    """Render a prompt through the single scientific chat-template pathway."""

    if not isinstance(fixed_date, str) or not fixed_date:
        raise ValueError("fixed_date must be a non-empty string")
    validated_messages = validate_messages(messages)
    kwargs = validate_chat_template_kwargs(chat_template_kwargs or {})
    rendered = tokenizer.apply_chat_template(
        validated_messages,
        tokenize=False,
        add_generation_prompt=True,
        date_string=fixed_date,
        **kwargs,
    )
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("tokenizer chat template returned an empty non-string prompt")
    return rendered


def rendered_prompt_sha256(
    tokenizer: Any,
    messages: Any,
    *,
    fixed_date: str,
    chat_template_kwargs: Optional[Mapping[str, Any]] = None,
) -> str:
    """Render once and return the exact UTF-8 prompt identity."""

    return sha256_text(
        render_chat_prompt(
            tokenizer,
            messages,
            fixed_date=fixed_date,
            chat_template_kwargs=chat_template_kwargs,
        )
    )
