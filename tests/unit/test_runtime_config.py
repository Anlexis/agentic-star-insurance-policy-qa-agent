# INS-C2-001 — Unit Tests: config/config.yaml is LIVE configuration
#
# The runtime parameters (max_retry, timeout_s, the retrieval tuning block)
# live in config/config.yaml. PolicyQAGraphNode._parent_config() reads that
# file and forwards it to the inner graph; a reader pointed at a retired file
# (or a key mapping drift) would silently return {} and every declared value
# would be dead configuration text. These tests PROBE the full chain with a
# temporary config file: file -> _runtime_config() -> _parent_config() ->
# DomainWorkflowGraph._extra_initial_state() -> a full outer invoke.
#
# Deterministic — no LLM, no network. framework.* / src.* imports only.

import pathlib

from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel

import src.graph.graph
from src.graph.domain_workflow_graph import DomainWorkflowGraph
from src.graph.graph import Graph, PolicyQAGraphNode

_PROBE_CONFIG = """\
max_retry: 5
timeout_s: 45

retrieval:
  top_k: 1
  score_threshold: 0.1
  kb_path: "config/kb/insurance_policy_kb.json"

llm:
  temperature: 0.0
  max_tokens: 1500
"""

_WAITING_PERIOD_QUERY = "waiting period before hospitalization coverage becomes effective"


def _write_probe_config(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "config.yaml"
    path.write_text(_PROBE_CONFIG, encoding="utf-8")
    return path


class TestRuntimeConfigChain:
    def test_parent_config_reads_the_runtime_file(self, tmp_path, monkeypatch):
        # _runtime_config() is the reader the entry point calls: it normalises the file
        # (timeout_s -> timeout_seconds) before the value is handed to Graph(config=...).
        monkeypatch.setattr(src.graph.graph, "_RUNTIME_CONFIG_PATH", _write_probe_config(tmp_path))
        probe = src.graph.graph._runtime_config()
        cfg = PolicyQAGraphNode(runtime_config=probe)._parent_config()["configurable"]
        assert cfg["max_retry"] == 5
        assert cfg["timeout_seconds"] == 45  # timeout_s mapped to the validated key
        assert cfg["retrieval"]["top_k"] == 1
        assert cfg["retrieval"]["score_threshold"] == 0.1

    def test_declared_values_seed_the_inner_state(self, tmp_path, monkeypatch):
        # _runtime_config() is the reader the entry point calls: it normalises the file
        # (timeout_s -> timeout_seconds) before the value is handed to Graph(config=...).
        monkeypatch.setattr(src.graph.graph, "_RUNTIME_CONFIG_PATH", _write_probe_config(tmp_path))
        probe = src.graph.graph._runtime_config()
        inner = DomainWorkflowGraph(config=PolicyQAGraphNode(runtime_config=probe)._parent_config())
        extra = inner._extra_initial_state()
        assert extra["retrieval_top_k"] == 1
        assert extra["retrieval_score_threshold"] == 0.1

    def test_declared_top_k_reaches_the_pipeline_end_to_end(self, tmp_path, monkeypatch):
        """The end-to-end probe: with the probe config (top_k=1) a full outer
        invoke returns exactly ONE citation for a query that matches several
        KB entries under the shipped default (top_k=4) — proving the file's
        declared value, not a fallback, drives retrieval."""
        ctx = InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL, caller_id="config-probe")
        baseline = Graph().invoke(_WAITING_PERIOD_QUERY, ctx=ctx)
        assert baseline["status"] == AgentStatus.SUCCESS.value
        assert (
            "- [2]" in baseline["output"]
        ), "probe precondition: the query must match multiple KB entries at the default top_k"

        # _runtime_config() is the reader the entry point calls: it normalises the file
        # (timeout_s -> timeout_seconds) before the value is handed to Graph(config=...).
        monkeypatch.setattr(src.graph.graph, "_RUNTIME_CONFIG_PATH", _write_probe_config(tmp_path))
        probe = src.graph.graph._runtime_config()
        probed = Graph(config=probe).invoke(_WAITING_PERIOD_QUERY, ctx=ctx)
        assert probed["status"] == AgentStatus.SUCCESS.value
        assert "- [1]" in probed["output"]
        assert "- [2]" not in probed["output"], "config/config.yaml retrieval.top_k=1 did not reach the pipeline"
