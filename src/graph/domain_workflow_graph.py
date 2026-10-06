"""AgentCore Platform v1.0"""

# INS-C2-001 - DomainWorkflowGraph (inner BaseGraph)
#
# This is the INNER graph for the Cat 2 two-layer nested architecture.
# It encapsulates the full insurance policy Q&A domain workflow:
#
#   START -> input_validate -> retrieve -> rerank_filter
#         -> generate_answer -> output_format -> END
#
# Called by PolicyQAGraphNode.get_subgraph() (graph.py).
# get_output() shapes the sub_result dict consumed by merge_output() there.
#
# Rules enforced:
#   - Inherits BaseGraph (fully custom topology - no forced backbone)
#   - Implements all 7 BaseGraph ABC methods
#   - register_nodes() does NOT call super() (abstract in BaseGraph)
#   - register_nodes() instantiates every domain node with NO ctor args
#   - Does NOT register initialize / finalize (outer backbone concerns)
#   - get_output() designed together with PolicyQAGraphNode.merge_output()
#   - No platform-internal SDK imports
#   - Not placed under src/subagents/

from typing import Any

from langgraph.graph import END, START

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus

from src.graph.context_bridge import get_caller_input_context
from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import State

# Module defaults - mirror the shipped `retrieval` block in
# config/config.yaml, used only if the runtime config is unreadable.
_DEFAULT_TOP_K = 4
_DEFAULT_SCORE_THRESHOLD = 0.25
_DEFAULT_KB_PATH = "config/kb/insurance_policy_kb.json"

# Bounds shared with the caller-side top_k guard (input_validate_node).
_TOP_K_MIN = 1
_TOP_K_MAX = 20


class DomainWorkflowGraph(BaseGraph):
    """Inner domain workflow graph for INS-C2-001.

    Inherits BaseGraph directly for a fully custom node topology.
    Called by PolicyQAGraphNode.get_subgraph() in graph.py, which passes the
    runtime config from config/config.yaml (`_parent_config()`) into the ctor.

    Pipeline (linear):
        START
          -> input_validate  (InputValidateNode)  - parse + validate the request
          -> retrieve        (RetrieveNode)       - keyword-score the seeded policy KB
          -> rerank_filter   (RerankFilterNode)   - boost / threshold / top_k cut
          -> generate_answer (GenerateAnswerNode) - grounded answer + citations
          -> output_format   (OutputFormatNode)   - final format + advisory disclaimer
          -> END

    All nodes are FunctionNode subclasses returning partial-dict state
    updates via execute(self, state) -> dict (NO config parameter).
    initialize / finalize are outer backbone concerns - not registered here.
    """

    # -- Identity --------------------------------------------------------------

    @property
    def name(self) -> str:
        """Unique identifier for this inner graph."""
        return "ins_c2_001_policy_qa_workflow"

    @property
    def state_schema(self) -> type:
        """TypedDict subclass shared across inner and outer graph."""
        return State

    # -- Config validation -----------------------------------------------------

    def _validate_config(self) -> None:
        """Validate the runtime config forwarded by the outer GraphNode.

        PolicyQAGraphNode._parent_config() forwards the config/config.yaml
        runtime parameters under config["configurable"]. No key is mandatory -
        the domain nodes all carry module-level defaults - but a value that IS
        present must be usable, so a broken declaration fails at compile time
        rather than silently mis-tuning retrieval at run time.
        """
        configurable = (self.config or {}).get("configurable") or {}
        for key in ("max_retry", "timeout_seconds"):
            if key not in configurable:
                continue
            value = configurable[key]
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                file_key = "timeout_s" if key == "timeout_seconds" else key
                raise ValueError(
                    f"DomainWorkflowGraph: config/config.yaml {file_key} " f"must be a positive integer, got {value!r}"
                )
        retrieval = configurable.get("retrieval")
        if retrieval is None:
            return
        if not isinstance(retrieval, dict):
            raise ValueError("DomainWorkflowGraph: config/config.yaml retrieval must be a mapping")
        top_k = retrieval.get("top_k")
        if top_k is not None and (
            not isinstance(top_k, int) or isinstance(top_k, bool) or not _TOP_K_MIN <= top_k <= _TOP_K_MAX
        ):
            raise ValueError(
                f"DomainWorkflowGraph: config/config.yaml retrieval.top_k must be an "
                f"integer between {_TOP_K_MIN} and {_TOP_K_MAX}, got {top_k!r}"
            )
        threshold = retrieval.get("score_threshold")
        if threshold is not None and (
            isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0.0 <= float(threshold) <= 1.0
        ):
            raise ValueError(
                f"DomainWorkflowGraph: config/config.yaml retrieval.score_threshold "
                f"must be a number between 0 and 1, got {threshold!r}"
            )
        kb_path = retrieval.get("kb_path")
        if kb_path is not None and (not isinstance(kb_path, str) or not kb_path.strip()):
            raise ValueError("DomainWorkflowGraph: config/config.yaml retrieval.kb_path must be a non-empty string")

    # -- Config + caller-context forwarding into state --------------------------

    def _extra_initial_state(self) -> dict[str, Any]:
        """Seed the inner initial state: retrieval scalars + caller input_context.

        Retrieval: PolicyQAGraphNode._parent_config() forwards the
        config/config.yaml `retrieval` block under config["configurable"]; this
        hook makes it reachable by the domain nodes at runtime as plain scalar
        state fields (retrieval_top_k / retrieval_score_threshold /
        retrieval_kb_path) - no JSON encoding needed, these are primitives
        (the JSON-string convention applies to dict/list containers only).
        RetrieveNode / RerankFilterNode read these state-seeded scalars
        directly (nodes take NO config parameter - config flows in only via
        State).

        input_context: GraphNode.execute() does not forward input_context on
        subgraph.invoke() (SDK 1.0.1); PolicyQAGraphNode.extract_input()
        stashes it via the context bridge immediately before the inner invoke,
        and this hook (called by BaseGraph.invoke while building initial
        state) reads it back. Inner domain nodes keep their plain
        state["input_context"] reads.
        """
        retrieval = (self.config or {}).get("configurable", {}).get("retrieval") or {}
        return {
            "retrieval_top_k": retrieval.get("top_k", _DEFAULT_TOP_K),
            "retrieval_score_threshold": retrieval.get("score_threshold", _DEFAULT_SCORE_THRESHOLD),
            "retrieval_kb_path": retrieval.get("kb_path", _DEFAULT_KB_PATH),
            "input_context": get_caller_input_context(),
        }

    # -- Node registration -----------------------------------------------------

    def register_nodes(self) -> None:
        """Register all 5 domain nodes.

        No super() call - BaseGraph.register_nodes() is abstract.
        Do NOT register initialize or finalize; those are outer backbone
        concerns handled by AgentBaseGraph in graph.py.

        Every node is instantiated with NO constructor arguments - FunctionNode
        subclasses take no __init__; config flows in per-call via State only
        (execute(self, state) -> dict). Every key registered here is
        referenced in add_edges().
        """
        self._nodes["input_validate"] = InputValidateNode()
        self._nodes["retrieve"] = RetrieveNode()
        self._nodes["rerank_filter"] = RerankFilterNode()
        self._nodes["generate_answer"] = GenerateAnswerNode()
        self._nodes["output_format"] = OutputFormatNode()

    # -- Edge wiring -----------------------------------------------------------

    def add_edges(self) -> None:
        """Wire the linear policy Q&A domain topology.

        Each step passes its partial-dict output into the shared State.
        For this template the topology is intentionally linear - no
        conditional branching between domain nodes. route() is implemented
        as required by the ABC but add_conditional_edges() is not used.
        """
        self._sg.add_edge(START, "input_validate")
        self._sg.add_edge("input_validate", "retrieve")
        self._sg.add_edge("retrieve", "rerank_filter")
        self._sg.add_edge("rerank_filter", "generate_answer")
        self._sg.add_edge("generate_answer", "output_format")
        self._sg.add_edge("output_format", END)

    # -- Routing ---------------------------------------------------------------

    def route(self, state: AgentState) -> str:
        """Conditional routing - required by BaseGraph ABC.

        For this linear topology add_conditional_edges() is not used, so
        this method is never called at runtime. It is implemented to
        satisfy the ABC contract. Returns END on error so an unexpected call
        does not re-enter a processing node.
        """
        if state.get("status") == AgentStatus.ERROR.value:
            return str(END)
        return "output_format"

    # -- Output shape ----------------------------------------------------------

    def get_output(self, state: AgentState) -> dict[str, Any]:
        """Shape the output dict returned to the outer graph as sub_result.

        This dict is received by PolicyQAGraphNode.merge_output() in
        graph.py as the `sub_result` argument. Both methods are designed
        together to guarantee field-name consistency:

            Inner get_output()  emits: "formatted_answer", "citations", "status", ...
            Outer merge_output() reads: sub_result.get("formatted_answer"),
                                        sub_result.get("citations"),
                                        sub_result.get("status")

        Additional fields (intake_notes, trace_id, correlation_id,
        node_history) are surfaced for observability / downstream extension.
        merge_output() currently maps formatted_answer + citations + status
        into the outer state delta; the remaining fields are available for
        future outer-merge extensions without an inner-graph change.
        """
        return {
            "formatted_answer": state.get("formatted_answer"),
            "citations": state.get("citations"),
            "status": state.get("status"),
            # Carried explicitly: the boundary only moves the keys named here,
            # so a run that completed without an answer would otherwise arrive
            # at the outer graph indistinguishable from one that answered.
            "error_code": state.get("error_code"),
            "intake_notes": state.get("intake_notes"),
            "trace_id": state.get("trace_id"),
            "correlation_id": state.get("correlation_id"),
            "node_history": state.get("node_history", []),
        }
