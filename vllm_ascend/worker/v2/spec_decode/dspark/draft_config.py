# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Checkpoint ABI helpers for the V2 DSpark speculator."""

from typing import Any

# The RedHat Window-GQA checkpoint was trained with an eight-row anchor
# block.  Kimi-K3's verification ABI exposes seven draft tokens, so the extra
# row is an implementation detail of the draft forward only.
_REDHAT_WINDOW_GQA_NATIVE_WIDTH = 8
_REDHAT_WINDOW_GQA_VERIFY_WIDTH = 7


def _draft_config_value(config: Any, name: str, default: Any = None) -> Any:
    """Read a draft setting from its top-level or nested decoder config."""
    nested = getattr(config, "transformer_layer_config", None)
    if isinstance(nested, dict):
        if name in nested:
            return nested[name]
    else:
        value = getattr(nested, name, None) if nested is not None else None
        if value is not None:
            return value
    value = getattr(config, name, None)
    return default if value is None else value


def _mapping_value(value: Any, name: str, default: Any = None) -> Any:
    """Read a field from dict-like or attribute-style HF config objects."""
    if isinstance(value, dict):
        return value.get(name, default)
    if value is None:
        return default
    return getattr(value, name, default)


def _get_native_window_gqa_width(config: Any, verify_width: int) -> int | None:
    """Return the native proposal width of the RedHat Window-GQA checkpoint."""
    block_size = _draft_config_value(config, "block_size")
    has_target_layers = bool(
        getattr(config, "aux_hidden_state_layer_ids", None)
        or getattr(config, "eagle_aux_hidden_state_layer_ids", None)
        or getattr(config, "target_layer_ids", None)
        or _mapping_value(_draft_config_value(config, "dflash_config"), "target_layer_ids")
    )
    is_qwen3_dspark = (
        _draft_config_value(config, "model_type") == "qwen3"
        or "Qwen3DSparkModel" in (getattr(config, "architectures", None) or ())
        or "DSparkDraftModel" in (getattr(config, "architectures", None) or ())
    )
    if (
        is_qwen3_dspark
        and bool(_draft_config_value(config, "use_sliding_window", False))
        and bool(_draft_config_value(config, "sample_from_anchor", False))
        and has_target_layers
        and isinstance(block_size, int)
        and block_size == _REDHAT_WINDOW_GQA_NATIVE_WIDTH
        and verify_width == _REDHAT_WINDOW_GQA_VERIFY_WIDTH
    ):
        return block_size
    return None
