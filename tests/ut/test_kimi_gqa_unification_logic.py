# SPDX-License-Identifier: Apache-2.0
"""Run pure production helpers without importing torch/vLLM/CANN."""

import ast
from copy import copy
from dataclasses import dataclass, fields, replace
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).parents[2]
CORE = "vllm_ascend/core/kv_cache_interface.py"
PLANNER = "vllm_ascend/models/kimi_k3/cache_config.py"
V2 = "vllm_ascend/worker/v2/spec_decode/dspark/"


def extract(module, names, namespace=None, class_name=None):
    tree = ast.parse((ROOT / module).read_text(encoding="utf-8"))
    body = tree.body
    if class_name:
        body = next(node.body for node in body if isinstance(node, ast.ClassDef) and node.name == class_name)
    selected = [node for node in body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == set(names)
    for node in selected:
        node.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    code = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    namespace = {} if namespace is None else namespace
    exec(compile(code, module, "exec"), namespace)
    return namespace


@dataclass(frozen=True)
class FullSpec:
    block_size: int = 128
    dtype: str = "bf16"
    kv_quant_mode: str = "none"
    head_size: int = 64
    head_size_v: int = 64
    tokens_per_state: int = 1
    sliding_window: int = None
    page_size_padded: int = None

    @property
    def real_page_size_bytes(self):
        return self.block_size * self.head_size * 4

    @property
    def page_size_bytes(self):
        return self.page_size_padded or self.real_page_size_bytes

    def max_memory_usage_bytes(self, config):
        return 2 * self.page_size_bytes


class SlidingSpec(FullSpec):
    pass


class DraftFull(FullSpec):
    pass


class DraftSliding(SlidingSpec):
    pass


@dataclass
class Uniform:
    kv_cache_specs: dict


@dataclass
class Group:
    layer_names: list
    kv_cache_spec: object
    is_eagle_group: bool = False


@dataclass
class Tensor:
    size: int
    layers: list
    offset: int
    layer_stride: int
    block_stride: int


@dataclass
class Cache:
    num_blocks: int
    kv_cache_tensors: list
    kv_cache_groups: list
    prefix_cache_retention_interval: int = None


class Config(NS):
    def update(self, values):
        self.__dict__.update(values)


def core_helpers():
    return extract(
        CORE,
        {
            "is_kimi_k31_model_config",
            "uses_kimi_k31_gqa_draft",
            "uses_kimi_gqa_draft",
            "is_kimi_gqa_draft_spec",
            "get_kv_cache_dcp_size",
            "with_kimi_gqa_draft_dcp_config",
        },
        {
            "copy": copy,
            "UniformTypeKVCacheSpecs": Uniform,
            "AscendGQADraftFullAttentionSpec": DraftFull,
            "AscendGQADraftSlidingWindowSpec": DraftSliding,
        },
    )


def config(model="kimi_k3", v2=True):
    return Config(
        use_v2_model_runner=v2,
        model_config=NS(hf_config=NS(model_type=model)),
        speculative_config=NS(
            method="dspark", draft_model_config=NS(architectures=["Qwen3DSparkModel"], use_mla=False, hf_config=NS())
        ),
        parallel_config=NS(
            decode_context_parallel_size=8,
            tensor_parallel_size=8,
            prefill_context_parallel_size=1,
            cp_kv_cache_interleave_size=1664,
            dcp_kv_cache_interleave_size=1664,
        ),
        attention_config=NS(use_non_causal=False),
        cache_config=NS(block_size=128, num_gpu_blocks_override=None, prefix_cache_retention_interval=None),
        kv_transfer_config=None,
    )


class TestKimiGqaUnification(unittest.TestCase):
    def test_k3_extension_is_v2_only_and_excludes_mla(self):
        pred = core_helpers()["uses_kimi_gqa_draft"]
        self.assertTrue(pred(config()))
        self.assertFalse(pred(config(v2=False)))
        cfg = config()
        cfg.speculative_config.draft_model_config.use_mla = True
        self.assertFalse(pred(cfg))
        self.assertFalse(pred(config(model="qwen3")))

    def test_k31_legacy_predicate_is_not_broadened_to_k3(self):
        ns = core_helpers()
        self.assertFalse(ns["uses_kimi_k31_gqa_draft"](config()))
        for v2 in (True, False):
            self.assertTrue(ns["uses_kimi_k31_gqa_draft"](config("kimi_k31", v2)))

    def test_local_config_preserves_target_dcp_and_tp(self):
        cfg = config()
        draft = core_helpers()["with_kimi_gqa_draft_dcp_config"](cfg)
        self.assertIsNot(draft, cfg)
        self.assertEqual(cfg.parallel_config.decode_context_parallel_size, 8)
        self.assertEqual(cfg.parallel_config.cp_kv_cache_interleave_size, 1664)
        self.assertEqual(draft.parallel_config.decode_context_parallel_size, 1)
        self.assertEqual(draft.parallel_config.tensor_parallel_size, 8)
        self.assertEqual(draft.parallel_config.cp_kv_cache_interleave_size, 1)
        self.assertIsNot(draft.cache_config, cfg.cache_config)
        other = config(model="qwen3")
        self.assertIs(core_helpers()["with_kimi_gqa_draft_dcp_config"](other), other)

    def test_attn_config_is_cached_without_changing_target_causality(self):
        ns = core_helpers()
        extract(V2 + "speculator.py", {"attn_vllm_config"}, ns, "AscendDSparkSpeculator")
        owner = NS(vllm_config=config(), requires_non_causal=True)
        first = ns["attn_vllm_config"](owner)
        self.assertIs(ns["attn_vllm_config"](owner), first)
        self.assertTrue(first.attention_config.use_non_causal)
        self.assertFalse(owner.vllm_config.attention_config.use_non_causal)
        self.assertEqual(owner.vllm_config.parallel_config.decode_context_parallel_size, 8)

    def test_native_width_does_not_change_verify_three(self):
        ns = extract(
            V2 + "draft_config.py",
            {"_draft_config_value", "_mapping_value", "_get_native_window_gqa_width"},
            {"_REDHAT_WINDOW_GQA_NATIVE_WIDTH": 8, "_REDHAT_WINDOW_GQA_VERIFY_WIDTH": 7},
        )
        cfg = NS(
            architectures=["DSparkDraftModel"],
            transformer_layer_config={
                "model_type": "qwen3",
                "block_size": 8,
                "sample_from_anchor": True,
                "use_sliding_window": True,
            },
            aux_hidden_state_layer_ids=[1, 17],
        )
        native = ns["_get_native_window_gqa_width"]
        self.assertEqual(native(cfg, 7), 8)
        self.assertIsNone(native(cfg, 3))
        cfg.transformer_layer_config["sample_from_anchor"] = False
        self.assertIsNone(native(cfg, 7))

    def test_checkpoint_normalization_preserves_legacy_and_layer_numbering(self):
        ns = extract(
            "vllm_ascend/patch/platform/patch_speculative_config.py",
            {"_normalize_legacy_qwen3_dspark_config"},
            {"_orig_hf_config_override": lambda c: c, "_normalize_kimi_dflash_rope": lambda c: None},
        )
        normalize = ns["_normalize_legacy_qwen3_dspark_config"]
        legacy = Config(
            model_type="qwen3",
            architectures=["Qwen3DSparkModel"],
            dflash_config={"causal": True},
            target_layer_ids=[0, 16],
        )
        before = dict(legacy.__dict__)
        normalize(legacy)
        self.assertEqual(legacy.__dict__, before)
        redhat = Config(
            model_type="speculators",
            architectures=["DSparkDraftModel"],
            transformer_layer_config={
                "hidden_size": 4096,
                "use_sliding_window": True,
                "sliding_window": 512,
                "block_size": 8,
                "sample_from_anchor": True,
            },
            aux_hidden_state_layer_ids=[1, 17],
            sliding_window_non_causal=False,
        )
        normalize(redhat)
        normalize(redhat)
        self.assertEqual(redhat.model_type, "qwen3")
        self.assertEqual(redhat.hidden_size, 4096)
        self.assertEqual(redhat.target_layer_ids, [0, 16])
        self.assertEqual(redhat.eagle_aux_hidden_state_layer_ids, [1, 17])
        self.assertTrue(redhat.dflash_config["causal"])
        self.assertEqual(redhat.dflash_config["swa_window_size"], 519)

    def converter(self):
        ns = core_helpers()
        ns.update(
            fields=fields,
            replace=replace,
            FullAttentionSpec=FullSpec,
            SlidingWindowSpec=SlidingSpec,
            torch=NS(bfloat16="bf16"),
            KVQuantMode=NS(NONE="none"),
            supports_component_major_mla_pd=lambda c: True,
            uses_kimi_k31_bf16_cache=lambda c: False,
            logger=NS(info_once=lambda *a: None),
        )
        return extract("vllm_ascend/worker/v2/attn_utils.py", {"mark_kimi_gqa_draft_specs"}, ns)[
            "mark_kimi_gqa_draft_specs"
        ]

    def test_converter_keeps_native_swa_and_is_idempotent(self):
        convert = self.converter()
        specs = {"target": FullSpec(1664), "draft": FullSpec(sliding_window=512)}
        marked = convert(config(), specs, ["draft"])
        self.assertIsInstance(marked["draft"], DraftSliding)
        self.assertEqual(marked["draft"].block_size, 128)
        self.assertIs(marked["target"], specs["target"])
        self.assertEqual(convert(config(), marked, ["draft"]), marked)

    def test_pd_packing_uses_actual_target_block_not_global_minimum(self):
        cfg = config()
        cfg.kv_transfer_config = NS(get_from_extra_config=lambda key, default: default)
        marked = self.converter()(cfg, {"target": FullSpec(1664), "draft": SlidingSpec(sliding_window=512)}, ["draft"])
        self.assertEqual(marked["draft"].block_size, 1664)
        cfg.kv_transfer_config.get_from_extra_config = lambda key, default: 129
        with self.assertRaises(ValueError):
            self.converter()(cfg, {"draft": SlidingSpec(sliding_window=512)}, ["draft"])

    def test_converter_rejects_padded_draft_and_pcp(self):
        cfg = config()
        with self.assertRaises(ValueError):
            self.converter()(cfg, {"draft": SlidingSpec(page_size_padded=99999)}, ["draft"])
        cfg.parallel_config.prefill_context_parallel_size = 2
        with self.assertRaises(ValueError):
            self.converter()(cfg, {"draft": SlidingSpec()}, ["draft"])

    def planner(self):
        ns = core_helpers()
        ns.update(
            replace=replace,
            KVCacheTensor=Tensor,
            KVCacheConfig=Cache,
            KVCacheGroupSpec=Group,
            cdiv=lambda a, b: (a + b - 1) // b,
        )
        return extract(
            PLANNER,
            {
                "append_kimi_gqa_draft_groups",
                "kimi_gqa_pool_bytes",
                "_layer_spec",
                "kimi_gqa_max_memory",
                "kimi_gqa_cache_config",
            },
            ns,
        )

    def test_pool_replan_and_private_plane_offsets(self):
        ns = self.planner()
        target = FullSpec()
        draft = DraftSliding()
        groups = [Group(["t0", "t1"], Uniform({"t0": target, "t1": target})), Group(["kda"], target)]
        ns["append_kimi_gqa_draft_groups"](groups, {"draft": draft})
        page = target.page_size_bytes
        self.assertEqual(ns["kimi_gqa_pool_bytes"](groups), 3 * page)
        self.assertTrue(groups[-1].is_eagle_group)

        def target_factory(cfg, target_groups, memory):
            blocks = cfg.cache_config.num_gpu_blocks_override or memory // (2 * page)
            tensors = [
                Tensor(memory, ["t0", "t1"], 0, blocks * page, page),
                Tensor(memory, ["kda"], 0, blocks * page, page),
            ]
            return Cache(blocks, tensors, target_groups)

        planned = ns["kimi_gqa_cache_config"](config(), groups, 30 * page, target_factory)
        self.assertEqual(planned.num_blocks, 10)
        self.assertTrue(all(t.size == 30 * page for t in planned.kv_cache_tensors))
        self.assertEqual(planned.kv_cache_tensors[-1].offset, 20 * page)
        self.assertEqual(planned.kv_cache_tensors[-1].layer_stride, 10 * page)
        replanned = ns["kimi_gqa_cache_config"](config(), groups, 4 * ns["kimi_gqa_pool_bytes"](groups), target_factory)
        self.assertEqual(replanned.num_blocks, 4)
        self.assertEqual(replanned.kv_cache_tensors[-1].offset, 8 * page)
        self.assertEqual(ns["kimi_gqa_max_memory"](config(), groups), 18 * page)
        cfg = config()
        cfg.cache_config.num_gpu_blocks_override = 3
        self.assertEqual(ns["kimi_gqa_cache_config"](cfg, groups, 30 * page, target_factory).num_blocks, 3)

    def test_draft_only_projection_does_not_call_empty_target_factory(self):
        ns = self.planner()
        groups = [Group([], FullSpec()), Group(["draft"], DraftSliding())]

        def factory(*args):
            self.fail("Empty target projection must not be sent to target planner")

        plan = ns["kimi_gqa_cache_config"](config(), groups, 5 * DraftSliding().page_size_bytes, factory)
        self.assertEqual(plan.num_blocks, 5)
        self.assertEqual(plan.kv_cache_tensors[0].offset, 0)
        self.assertEqual(plan.kv_cache_groups, groups)

    def test_k31_keeps_paired_planner_and_k3_splits_draft_before_target_grouping(self):
        ns = self.planner()
        seen = []

        def target_groups(cfg, specs):
            seen.append(set(specs))
            return [Group(list(specs), Uniform(specs))]

        ns.update(
            paired_sfa_cache_groups=lambda cfg, specs: "baseline paired planner",
            _orig_get_kv_cache_groups=target_groups,
        )
        extract(
            "vllm_ascend/patch/platform/patch_kv_cache_utils.py",
            {"_get_glm5_next_kv_cache_groups"},
            ns,
        )
        specs = {"target": FullSpec(1664), "draft": DraftSliding()}
        k31 = ns["_get_glm5_next_kv_cache_groups"](config("kimi_k31"), specs)
        self.assertEqual(k31, "baseline paired planner")
        self.assertEqual(seen, [])
        cfg = config()
        cfg.scheduler_config = NS(disable_hybrid_kv_cache_manager=False)
        groups = ns["_get_glm5_next_kv_cache_groups"](cfg, specs)
        self.assertEqual(seen, [{"target"}])
        self.assertEqual([g.layer_names for g in groups], [["target"], ["draft"]])
        cfg.scheduler_config.disable_hybrid_kv_cache_manager = True
        with self.assertRaises(ValueError):
            ns["_get_glm5_next_kv_cache_groups"](cfg, specs)

    def test_typed_draft_dcp_ignores_target_dcp(self):
        ns = core_helpers()
        self.assertEqual(ns["get_kv_cache_dcp_size"](DraftSliding(), 8), 1)
        self.assertEqual(ns["get_kv_cache_dcp_size"](Uniform({"d": DraftSliding()}), 8), 1)
        self.assertEqual(ns["get_kv_cache_dcp_size"](FullSpec(), 8), 8)


if __name__ == "__main__":
    unittest.main()
