# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Shared full-history and recurrent grouping for Kimi K3 and K3.1."""

from dataclasses import replace

from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import is_kimi_gqa_draft_spec
from vllm_ascend.utils import vllm_version_is

# vLLM main moved this helper to kv_cache_planning after v0.32.0.
if vllm_version_is("0.30.0") or vllm_version_is("0.32.0"):
    from vllm.v1.core.kv_cache_utils import create_kv_cache_group_specs
else:
    from vllm.v1.core.kv_cache_planning import create_kv_cache_group_specs


_KIMI_K3_TARGET_LAYER_PREFIX = "language_model.model.layers."
_KIMI_K3_DRAFT_LAYER_PREFIX = "model.layers."


def make_kimi_cache_groups(
    full_history_specs: dict[str, KVCacheSpec],
    recurrent_specs: dict[str, MambaSpec],
    *,
    recurrent_group_size: int,
    uniform_recurrent: bool,
) -> list[KVCacheGroupSpec] | None:
    """Share full-history ownership and balance recurrent groups by slot budget.

    Callers validate roles and geometry, and supply specs in layer order with
    target attention first for scheduler conversion. The recurrent slot budget
    is explicit: K3 uses target plus draft pages, whereas K3.1 overlays only
    target MLA pages, excluding both indexer and draft storage.

    K3 retains per-layer recurrent specs for its shared-tuple allocation;
    K3.1 keeps native merged Mamba specs for its paired-page allocation.
    """
    full_history = UniformTypeKVCacheSpecs.from_specs(full_history_specs)
    if full_history is None:
        return None
    groups = [KVCacheGroupSpec(list(full_history_specs), full_history)]
    names = list(recurrent_specs)
    count = cdiv(len(names), recurrent_group_size)
    recurrent_layers = [names[i::count] for i in range(count)]
    if uniform_recurrent:
        for layer_names in recurrent_layers:
            group_specs = {name: recurrent_specs[name] for name in layer_names}
            uniform = UniformTypeKVCacheSpecs.from_specs(group_specs)
            assert uniform is not None
            groups.append(KVCacheGroupSpec(layer_names, uniform))
    elif recurrent_layers:
        groups.extend(create_kv_cache_group_specs(recurrent_specs, recurrent_layers))
    return groups


def _get_kimi_k3_dspark_mixed_kv_cache_groups(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec] | None:
    """Build topology-independent Kimi K3 DSpark scheduler groups.

    Target and causal draft attention layers require the same full-sequence
    block ownership. Putting them in one UniformType group lets them share one
    scheduler block table while preserving a separate physical page per layer.
    Recurrent layers are split into the fewest balanced groups whose size does
    not exceed the attention group. This minimizes scheduler groups while
    keeping the recurrent groups balanced.

    Block and page sizes are resolved by the runtime and intentionally not
    fixed here: TP8 and TP16 produce different sizes but the same ownership
    relation. An unrecognized or incompatible signature falls back to vLLM's
    generic hybrid grouping.
    """
    target_attention_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if name.startswith(_KIMI_K3_TARGET_LAYER_PREFIX) and isinstance(spec, FullAttentionSpec)
    }
    draft_attention_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if name.startswith(_KIMI_K3_DRAFT_LAYER_PREFIX) and isinstance(spec, FullAttentionSpec)
    }
    mamba_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if name.startswith(_KIMI_K3_TARGET_LAYER_PREFIX) and isinstance(spec, MambaSpec)
    }

    matched_layer_count = len(target_attention_specs) + len(draft_attention_specs) + len(mamba_specs)
    if (
        not target_attention_specs
        or not draft_attention_specs
        or not mamba_specs
        or matched_layer_count != len(kv_cache_spec)
    ):
        return None

    attention_specs = [*target_attention_specs.values(), *draft_attention_specs.values()]
    all_specs = [*attention_specs, *mamba_specs.values()]
    # Only attention layers share a scheduler block table. Mamba keeps its
    # own groups and may use max_model_len as block_size when cache_mode=none.
    if (
        len({spec.block_size for spec in attention_specs}) != 1
        or len({spec.page_size_bytes for spec in all_specs}) != 1
    ):
        return None

    first_mamba_spec = next(iter(mamba_specs.values()))
    if any(spec != first_mamba_spec for spec in mamba_specs.values()):
        return None

    # Insert target attention first. generate_scheduler_kv_cache_config unwraps a
    # UniformType group to its first spec, and this representative is registered
    # with the FullAttentionManager needed by both target and draft attention.
    mixed_attention_specs = {**target_attention_specs, **draft_attention_specs}
    groups = make_kimi_cache_groups(
        mixed_attention_specs,
        mamba_specs,
        recurrent_group_size=len(mixed_attention_specs),
        uniform_recurrent=True,
    )
    if groups is None:
        return None

    logger.info(
        "Using Kimi K3 DSpark mixed KV grouping: %d target + %d draft attention layers, followed by Mamba groups %s",
        len(target_attention_specs),
        len(draft_attention_specs),
        [len(group.layer_names) for group in groups[1:]],
    )
    return groups


def append_kimi_gqa_draft_groups(groups, draft):
    """Keep each native GQA geometry separate from target and recurrent ownership."""
    by_spec = {}
    for name, spec in draft.items():
        if not is_kimi_gqa_draft_spec(spec):
            raise ValueError(f"Kimi GQA draft layer {name} has an unmarked cache spec")
        by_spec.setdefault(spec, []).append(name)
    groups.extend(KVCacheGroupSpec(names, spec, is_eagle_group=True) for spec, names in by_spec.items())
    return groups


def kimi_gqa_pool_bytes(groups):
    """Target groups alias native slots; draft planes occupy a private region."""
    target = [group for group in groups if not is_kimi_gqa_draft_spec(group.kv_cache_spec)]
    draft = [group for group in groups if is_kimi_gqa_draft_spec(group.kv_cache_spec)]
    target_bytes = max(
        (sum(_layer_spec(group, name).page_size_bytes for name in group.layer_names) for group in target),
        default=0,
    )
    draft_bytes = sum(len(group.layer_names) * group.kv_cache_spec.page_size_bytes for group in draft)
    return target_bytes + draft_bytes


def _layer_spec(group, name):
    spec = group.kv_cache_spec
    return spec.kv_cache_specs[name] if isinstance(spec, UniformTypeKVCacheSpecs) else spec


def kimi_gqa_max_memory(config, groups):
    blocks = 0
    for group in groups:
        specs = [_layer_spec(group, name) for name in group.layer_names]
        blocks += max(
            (cdiv(spec.max_memory_usage_bytes(config), spec.page_size_bytes) for spec in specs),
            default=0,
        )
    return blocks * kimi_gqa_pool_bytes(groups)


def kimi_gqa_cache_config(config, groups, available_memory, target_config_factory):
    """Use K3.1's native draft-plane layout without its SFA/indexer overlay.

    A private draft region is necessary: its all-K/all-V planes cannot alias
    the target's block-first pages at arbitrary manager block IDs.
    """
    target = [group for group in groups if group.layer_names and not is_kimi_gqa_draft_spec(group.kv_cache_spec)]
    draft = [group for group in groups if is_kimi_gqa_draft_spec(group.kv_cache_spec)]
    bytes_per_block = kimi_gqa_pool_bytes(groups)
    if not draft or bytes_per_block <= 0:
        raise ValueError("Kimi GQA cache layout requires materializable layers and draft groups")
    num_blocks = config.cache_config.num_gpu_blocks_override
    if num_blocks is None:
        num_blocks = available_memory // bytes_per_block
    draft_bytes = sum(len(group.layer_names) * group.kv_cache_spec.page_size_bytes for group in draft)
    target_bytes = bytes_per_block - draft_bytes
    if target_bytes:
        target_config = target_config_factory(config, target, num_blocks * target_bytes)
    else:
        # Worker projection can retain empty target groups on a draft-only rank.
        target_config = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[],
            kv_cache_groups=target,
            prefix_cache_retention_interval=config.cache_config.prefix_cache_retention_interval,
        )
    if target_config.num_blocks != num_blocks:
        raise ValueError("Kimi target and draft must agree on the number of shared pool IDs")
    size = bytes_per_block * num_blocks
    tensors = [replace(tensor, size=size) for tensor in target_config.kv_cache_tensors]
    offset = target_bytes * num_blocks
    for group in draft:
        for name in group.layer_names:
            page_bytes = group.kv_cache_spec.page_size_bytes
            layer_size = page_bytes * num_blocks
            tensors.append(
                KVCacheTensor(
                    size=size,
                    layers=[name],
                    offset=offset,
                    layer_stride=layer_size,
                    block_stride=page_bytes,
                )
            )
            offset += layer_size
    return replace(target_config, kv_cache_groups=groups, kv_cache_tensors=tensors)
