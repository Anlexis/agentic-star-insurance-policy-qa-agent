# INS-C2-001 — Unit Tests: manifest / runtime-config consistency
#
# Two config files, two jobs:
#   config/agent.yaml  — the flat AgentRegistry manifest (identity, entry
#                        point, trust level, compile-time requires). Read at
#                        ROOT level — no `agent:` nesting.
#   config/config.yaml — runtime parameters (max_retry, timeout_s, the
#                        retrieval tuning block, the llm block). Read by
#                        PolicyQAGraphNode._parent_config() and forwarded to
#                        the inner graph, where _extra_initial_state()
#                        republishes retrieval into inner state.
#
# These tests pin file ↔ code consistency so a config drift fails fast in CI.
# Deterministic — no LLM, no network.

import json
import pathlib

import yaml

from framework.schemas.trust_level import TrustLevel

from src.graph.graph import InsurancePolicyQAAgent, PolicyQAGraphNode
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MANIFEST = yaml.safe_load((_ROOT / "config" / "agent.yaml").read_text(encoding="utf-8"))
_RUNTIME = yaml.safe_load((_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))


class TestManifestIdentity:
    def test_cfg_01_template_id_is_ins_c2_001(self):
        assert _MANIFEST["id"] == "INS-C2-001"
        assert _MANIFEST["namespace"] == "ins"
        assert _MANIFEST["enabled"] is True

    def test_cfg_02_declared_class_is_the_graph_class(self):
        # Class-name contract: manifest class path == graph.py class == server import.
        assert _MANIFEST["class"] == "src.graph.graph.InsurancePolicyQAAgent"
        module_path, class_name = _MANIFEST["class"].rsplit(".", 1)
        assert class_name == InsurancePolicyQAAgent.__name__
        assert _MANIFEST["name"] == InsurancePolicyQAAgent().name

    def test_cfg_03_category_and_industry(self):
        assert _MANIFEST["category"] == "Cat 2"
        assert _MANIFEST["industry"] == "INS"
        assert _MANIFEST["base_type"] == "RAGAgent"

    def test_generation_mode_is_deterministic(self):
        # No LLM client is constructed anywhere in src/ — the manifest must
        # declare the deterministic generation mode.
        assert _MANIFEST["generation_mode"] == "deterministic"

    def test_requires_blocks_match_the_code(self):
        # No ctx.secrets.require() call and no client construction exist in
        # src/ — both compile-time requires stay empty.
        assert _MANIFEST["requires"]["secrets"] == []
        assert _MANIFEST["requires"]["extras"] == []


class TestManifestSecurity:
    def test_cfg_04_required_trust_level_matches_outer_gate_nodes(self):
        declared = TrustLevel(_MANIFEST["required_trust_level"])
        assert declared is TrustLevel.VERIFIED_EXTERNAL
        assert PreProcessNode.required_trust_level is declared
        assert PostProcessNode.required_trust_level is declared


class TestRuntimeConfig:
    def test_cfg_05_max_retry_within_framework_ceiling(self):
        max_retry = _RUNTIME["max_retry"]
        assert isinstance(max_retry, int)
        assert 0 <= max_retry < 10  # AgentBaseGraph MAX_RETRY_CEILING

    def test_timeout_is_a_positive_integer(self):
        assert isinstance(_RUNTIME["timeout_s"], int)
        assert _RUNTIME["timeout_s"] > 0

    def test_hitl_is_not_enabled(self):
        # This template declares no HITL — the interrupt-propagation boundary
        # test ships as a skip stub accordingly.
        assert (_RUNTIME.get("hitl") or {}).get("enabled", False) is False

    def test_cfg_06_retrieval_block_matches_node_defaults(self):
        # Node module defaults mirror the runtime config — a drift silently
        # changes tuning.
        retrieval = _RUNTIME["retrieval"]
        from src.nodes.retrieve_node import _DEFAULT_KB_PATH, _DEFAULT_TOP_K as _RETRIEVE_TOP_K
        from src.nodes.rerank_filter_node import (
            _DEFAULT_SCORE_THRESHOLD as _RERANK_SCORE_THRESHOLD,
            _DEFAULT_TOP_K as _RERANK_TOP_K,
        )

        assert retrieval["top_k"] == _RETRIEVE_TOP_K == _RERANK_TOP_K
        assert retrieval["score_threshold"] == _RERANK_SCORE_THRESHOLD
        assert retrieval["kb_path"] == _DEFAULT_KB_PATH
        assert (_ROOT / retrieval["kb_path"]).is_file()

    def test_cfg_07_parent_config_forwards_runtime_blocks(self):
        from src.graph.graph import _runtime_config

        cfg = PolicyQAGraphNode(runtime_config=_runtime_config())._parent_config()
        configurable = cfg["configurable"]
        assert configurable["retrieval"] == _RUNTIME["retrieval"]
        assert configurable["llm"] == _RUNTIME["llm"]
        assert configurable["max_retry"] == _RUNTIME["max_retry"]
        # timeout_s is mapped to the key the inner graph validates.
        assert configurable["timeout_seconds"] == _RUNTIME["timeout_s"]
        assert configurable["retrieval"], "_parent_config() must never forward an empty retrieval block"


class TestSeededKnowledgeBase:
    def test_kb_is_a_well_formed_entry_list(self):
        entries = json.loads((_ROOT / _RUNTIME["retrieval"]["kb_path"]).read_text(encoding="utf-8"))
        assert isinstance(entries, list)
        assert len(entries) >= 5, "seeded KB must carry a usable corpus"
        for entry in entries:
            assert set(entry.keys()) == {"id", "title", "policy_type", "source", "tags", "content"}
            assert entry["id"] and entry["title"] and entry["content"]

    def test_kb_ids_are_unique(self):
        entries = json.loads((_ROOT / _RUNTIME["retrieval"]["kb_path"]).read_text(encoding="utf-8"))
        ids = [e["id"] for e in entries]
        assert len(ids) == len(set(ids))
