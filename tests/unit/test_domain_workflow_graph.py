# INS-C2-001 — Unit Tests: DomainWorkflowGraph (inner BaseGraph)
#
# Inner-graph composition + a full inner invoke() over the seeded KB. The
# inner graph runs the 5 domain nodes (all ANONYMOUS) — the outer trust boundary
# is the AgentBaseGraph backbone's concern and is covered in
# test_graph_composition.py / the PoB suite.
#
# NOTE: unlike some sibling templates, _extra_initial_state() here republishes
# THREE bare scalar State keys (retrieval_top_k / retrieval_score_threshold /
# retrieval_kb_path) — never a single JSON-encoded `retrieval_config` field.
#
# Mirrors docs/03_test_spec.md §3 (INT-01..INT-04).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

from langgraph.graph import END

import pytest

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_status import AgentStatus

from src.graph.domain_workflow_graph import (
    _DEFAULT_KB_PATH,
    _DEFAULT_SCORE_THRESHOLD,
    _DEFAULT_TOP_K,
    DomainWorkflowGraph,
)
from src.graph.graph import PolicyQAGraphNode
from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import State, from_json

_WAITING_PERIOD_QUERY = "waiting period before hospitalization coverage becomes effective"


class TestInnerGraphConstruction:
    def test_int_01_inherits_base_graph(self):
        assert issubclass(DomainWorkflowGraph, BaseGraph)

    def test_int_01_registers_the_five_domain_nodes(self):
        inner = DomainWorkflowGraph()
        inner.register_nodes()
        assert set(inner._nodes.keys()) == {
            "input_validate",
            "retrieve",
            "rerank_filter",
            "generate_answer",
            "output_format",
        }
        assert isinstance(inner._nodes["input_validate"], InputValidateNode)
        assert isinstance(inner._nodes["retrieve"], RetrieveNode)
        assert isinstance(inner._nodes["rerank_filter"], RerankFilterNode)
        assert isinstance(inner._nodes["generate_answer"], GenerateAnswerNode)
        assert isinstance(inner._nodes["output_format"], OutputFormatNode)

    def test_inner_graph_name_and_schema(self):
        inner = DomainWorkflowGraph()
        assert inner.name == "ins_c2_001_policy_qa_workflow"
        assert inner.state_schema is State

    def test_initialize_finalize_are_not_registered(self):
        # Outer backbone concerns must not leak into the inner topology.
        inner = DomainWorkflowGraph()
        inner.register_nodes()
        assert "initialize" not in inner._nodes
        assert "finalize" not in inner._nodes


class TestConfigForwarding:
    def test_int_02_extra_initial_state_republishes_retrieval_scalars(self):
        inner = DomainWorkflowGraph(
            config={"configurable": {"retrieval": {"top_k": 2, "score_threshold": 0.4, "kb_path": "x.json"}}}
        )
        extra = inner._extra_initial_state()
        # Bare scalars — NOT a single JSON-encoded dict field.
        assert extra == {
            "retrieval_top_k": 2,
            "retrieval_score_threshold": 0.4,
            "retrieval_kb_path": "x.json",
            "input_context": {},
        }

    def test_extra_initial_state_with_no_config_uses_module_defaults(self):
        extra = DomainWorkflowGraph()._extra_initial_state()
        assert extra == {
            "retrieval_top_k": _DEFAULT_TOP_K,
            "retrieval_score_threshold": _DEFAULT_SCORE_THRESHOLD,
            "retrieval_kb_path": _DEFAULT_KB_PATH,
            "input_context": {},
        }

    def test_extra_initial_state_seeds_the_bridged_input_context(self):
        # GraphNode.execute() does not forward input_context on
        # subgraph.invoke() (SDK 1.0.1) — the context bridge carries it.
        from src.graph.context_bridge import set_caller_input_context

        try:
            set_caller_input_context({"policy_type": "health", "top_k": 2})
            extra = DomainWorkflowGraph()._extra_initial_state()
            assert extra["input_context"] == {"policy_type": "health", "top_k": 2}
        finally:
            set_caller_input_context(None)


class TestConfigValidation:
    """_validate_config() rejects broken declarations at compile time."""

    def test_valid_config_passes(self):
        DomainWorkflowGraph(
            config={
                "configurable": {
                    "max_retry": 3,
                    "timeout_seconds": 30,
                    "retrieval": {"top_k": 4, "score_threshold": 0.25, "kb_path": "config/kb/insurance_policy_kb.json"},
                }
            }
        )._validate_config()

    def test_absent_keys_are_permitted(self):
        DomainWorkflowGraph()._validate_config()
        DomainWorkflowGraph(config={"configurable": {}})._validate_config()

    @pytest.mark.parametrize(
        "configurable",
        [
            {"max_retry": 0},
            {"max_retry": True},
            {"timeout_seconds": -1},
            {"timeout_seconds": "30"},
            {"retrieval": "not-a-mapping"},
            {"retrieval": {"top_k": 0}},
            {"retrieval": {"top_k": 21}},
            {"retrieval": {"top_k": 4.0}},
            {"retrieval": {"score_threshold": 1.5}},
            {"retrieval": {"score_threshold": True}},
            {"retrieval": {"kb_path": ""}},
            {"retrieval": {"kb_path": 42}},
        ],
        ids=[
            "retry-zero",
            "retry-bool",
            "timeout-neg",
            "timeout-str",
            "retrieval-str",
            "topk-zero",
            "topk-over",
            "topk-float",
            "threshold-over",
            "threshold-bool",
            "kbpath-empty",
            "kbpath-int",
        ],
    )
    def test_broken_declarations_are_rejected(self, configurable):
        with pytest.raises(ValueError):
            DomainWorkflowGraph(config={"configurable": configurable})._validate_config()


class TestOutputShape:
    def test_int_03_get_output_shapes_the_merge_contract(self):
        inner = DomainWorkflowGraph()
        out = inner.get_output(
            {
                "formatted_answer": "ANSWER",
                "citations": "[]",
                "status": AgentStatus.SUCCESS.value,
                "intake_notes": None,
                "trace_id": "t-1",
                "correlation_id": "c-1",
                "node_history": ["InputValidateNode"],
            }
        )
        assert out == {
            "formatted_answer": "ANSWER",
            "citations": "[]",
            "status": AgentStatus.SUCCESS.value,
            # Always carried so a reason can never be dropped at the boundary.
            "error_code": None,
            "intake_notes": None,
            "trace_id": "t-1",
            "correlation_id": "c-1",
            "node_history": ["InputValidateNode"],
        }

    def test_route_returns_end_on_error(self):
        inner = DomainWorkflowGraph()
        assert inner.route({"status": AgentStatus.ERROR.value}) == END
        assert inner.route({"status": AgentStatus.SUCCESS.value}) == "output_format"


class TestInnerEndToEnd:
    def _invoke(self, payload: str) -> dict:
        # Same construction path the outer GraphNode uses: manifest-derived
        # config via _parent_config(); domain nodes take NO ctor args.
        inner = DomainWorkflowGraph(config=PolicyQAGraphNode()._parent_config())
        return inner.invoke(payload, session_id="inner-e2e")

    def test_int_04_full_inner_run_produces_the_formatted_answer(self):
        result = self._invoke(_WAITING_PERIOD_QUERY)
        assert result["status"] == AgentStatus.SUCCESS.value
        answer = result["formatted_answer"]
        assert answer.startswith("# Insurance Policy Q&A Result")
        assert "[1]" in answer
        assert "does not amend or override your actual policy contract" in answer
        citations = from_json(result["citations"])
        assert citations and citations[0]["id"] == "kb-004"

    def test_int_04_inner_node_history_is_the_linear_topology(self):
        history = self._invoke(_WAITING_PERIOD_QUERY)["node_history"]
        assert history == [
            "InputValidateNode",
            "RetrieveNode",
            "RerankFilterNode",
            "GenerateAnswerNode",
            "OutputFormatNode",
        ]

    def test_no_coverage_query_still_terminates_success(self):
        result = self._invoke("quantum telepathy sandwich recipes")
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "does not contain sufficient coverage" in result["formatted_answer"]
