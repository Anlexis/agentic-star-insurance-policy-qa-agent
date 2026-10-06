"""AgentCore Platform v1.0"""

# INS-C2-001 - Outer graph (AgentBaseGraph; Cat 2 two-layer nested architecture)
#
# Insurance Policy Q&A Agent (Cat 2 RAG domain workflow).
#
# Architecture (Cat 2):
#
#   Outer backbone (fixed - identical to Cat 1, do NOT override add_edges()):
#     START -> initialize -> pre_process -> main -> {route} -> post_process -> finalize -> END
#                                             |  (RETRY, max 3)
#                                             -> pre_process
#
#   `main` slot is a GraphNode subclass (PolicyQAGraphNode) that delegates
#   the full policy Q&A domain workflow to DomainWorkflowGraph (inner
#   BaseGraph: input_validate -> retrieve -> rerank_filter ->
#   generate_answer -> output_format).
#
#   Domain complexity is fully encapsulated inside the inner graph. The outer
#   backbone is never modified.
#
# Directory layout:
#   src/graph/graph.py                 <- outer graph (this file)
#   src/graph/domain_workflow_graph.py <- inner graph (multi-step topology)
#   src/graph/context_bridge.py        <- input_context outer->inner hand-off
#
# Class-name contract:
#   graph.py class:           InsurancePolicyQAAgent (this file)
#   config/agent.yaml class:  "src.graph.graph.InsurancePolicyQAAgent"  <- must match
#   src/api/server.py import: from src.graph.graph import InsurancePolicyQAAgent
#
# Rules enforced:
#   - InsurancePolicyQAAgent inherits AgentBaseGraph (framework base class -
#     direct inheritance)
#   - super().register_nodes() called first (fills initialize + finalize)
#   - PolicyQAGraphNode assigned to self._nodes["main"]
#   - _parent_config() forwards the config/config.yaml runtime blocks
#   - merge_output() returns only changed keys
#   - add_edges() NOT overridden on the outer graph
#   - No platform-internal SDK imports

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.nodes.graph_node import GraphNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus

from src.graph.context_bridge import set_caller_input_context
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State

if TYPE_CHECKING:
    from src.graph.domain_workflow_graph import DomainWorkflowGraph

# Runtime-parameter file: src/graph/graph.py -> parents[2] is the repo root.
_RUNTIME_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def _runtime_config() -> dict[str, Any]:
    """Read the runtime parameters from config/config.yaml.

    Returns an empty dict - never raises - when the file is absent, unreadable,
    not valid YAML, or not a mapping. The inner graph validates the key
    `timeout_seconds`, so `timeout_s` is mapped on the way through. The
    `retrieval` and `llm` blocks are forwarded verbatim; the inner graph
    republishes `retrieval` into inner state (see
    DomainWorkflowGraph._extra_initial_state()), and `llm` is reserved for the
    documented LLM-synthesis upgrade seam (unused by the deterministic nodes).
    """
    try:
        import yaml

        loaded = yaml.safe_load(_RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(loaded, dict):
        return {}
    out: dict[str, Any] = {}
    if "max_retry" in loaded:
        out["max_retry"] = loaded["max_retry"]
    if "timeout_s" in loaded:
        out["timeout_seconds"] = loaded["timeout_s"]
    for block in ("retrieval", "llm"):
        if isinstance(loaded.get(block), dict):
            out[block] = loaded[block]
    return out


class PolicyQAGraphNode(GraphNode):
    """GraphNode subclass assigned to the `main` slot of the outer agent.

    Wraps DomainWorkflowGraph (inner Cat 2 BaseGraph RAG pipeline).
    Called by AgentBaseGraph backbone after pre_process and before post_process.

    Contracts:
      get_subgraph()    - instantiate DomainWorkflowGraph with the forwarded
                          runtime config (_parent_config())
      extract_input()   - pull validated_input from outer state and bridge the
                          caller's input_context to the inner graph
      merge_output()    - map sub_result fields into outer state delta (changed keys only)
      error_strategy    - "propagate": re-raise inner errors as SubgraphError (fail-fast)
    """

    def __init__(self, runtime_config: dict[str, Any] | None = None) -> None:
        """Receive the runtime config from the outer graph.

        A BaseNode has no config back-reference of its own, so the outer
        AgentBaseGraph reads `self.config` and threads it in here at
        register_nodes() time. Static construction input - not mutable state.
        """
        # A non-mapping runtime config degrades to {} instead of raising: reading and
        # parsing config/config.yaml belongs to the entry point, and this node only has
        # to survive whatever it is handed.
        self._runtime_config = dict(runtime_config) if isinstance(runtime_config, dict) else {}

    # "propagate": re-raise inner graph exceptions as SubgraphError (default - fail fast).
    # "handle": call on_subgraph_error() instead - use for graceful degradation.
    error_strategy: ClassVar[str] = "propagate"

    # False: HITL interrupts are handled inside the inner graph only.
    # True: surface inner HITL interrupt to the outer caller.
    propagate_hitl: ClassVar[bool] = False

    def _parent_config(self) -> dict[str, Any]:
        """Forward the config/config.yaml runtime parameters to the inner graph.

        Everything declared in config/config.yaml (max_retry, timeout_s, the
        retrieval tuning block, the llm block) is handed to DomainWorkflowGraph
        under config["configurable"], where its _validate_config() checks the
        values at compile time and _extra_initial_state() republishes the
        retrieval block into inner state. Without this the inner graph is
        constructed with an empty config and the declared values are dead
        configuration text.

        Degrades to {} - never raises - when the file is missing or malformed;
        the inner graph and the domain nodes then fall back to their own
        module-level defaults (which mirror the shipped config values).
        """
        config = self._runtime_config
        if not config:
            return {}
        return {"configurable": config}

    def get_subgraph(self) -> "DomainWorkflowGraph":
        """Instantiate and return the inner domain workflow graph.

        DomainWorkflowGraph is imported lazily (inside the method) to avoid
        circular-import risk at module load time and to match the nested
        Cat 2 sample pattern.

        The inner graph receives the runtime config via its BaseGraph ctor;
        its domain NODES still take no constructor arguments and read config
        per-call via execute(self, state) (FunctionNode contract - NO config
        parameter; the forwarded settings reach the nodes only through State,
        seeded by DomainWorkflowGraph._extra_initial_state()).
        """
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        return DomainWorkflowGraph(config=self._parent_config())

    def execute(self, state: AgentState) -> dict[str, Any]:
        """Skip the inner graph when the request was already found unacceptable.

        A request declined by pre_process has no validated input to work on, so
        running the inner graph would only produce a second, vaguer reason for
        the same rejection - and overwrite the specific one already settled.
        Completing here keeps the original reason intact.

        This override is deliberate: GraphNode.execute() is not final, and the
        marker is the only signal that distinguishes "nothing to do" from "not
        run yet".
        """
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}
        result: dict[str, Any] = super().execute(state)
        return result

    def extract_input(self, state: AgentState) -> str:
        """Return the string input passed into inner_graph.invoke().

        PreProcessNode validates the raw user_input and writes the result to
        validated_input. Prefer that; fall back to user_input if
        validated_input is absent (e.g. in unit tests).

        Also bridges the caller's input_context to the inner graph:
        GraphNode.execute() does not forward input_context on subgraph.invoke()
        (SDK 1.0.1), and extract_input is the last template-code hook that sees
        the outer state before the inner invoke - see
        src/graph/context_bridge.py. Structured parameters (policy_type /
        top_k) travel through that bridge; a legacy JSON envelope inside the
        input string is still parsed by the first inner node
        (InputValidateNode) for backwards compatibility.
        """
        set_caller_input_context(state.get("input_context") or {})
        return cast(str, state.get("validated_input") or state.get("user_input", ""))

    def merge_output(self, state: AgentState, sub_result: dict[str, Any]) -> dict[str, Any]:
        """Map inner graph sub_result back into the outer state delta.

        sub_result is the dict returned by DomainWorkflowGraph.get_output().
        Returns ONLY changed keys - never the full state.

        Key coupling (designed together with DomainWorkflowGraph.get_output()):
          Inner get_output() emits  -> "formatted_answer", "citations", "status", ...
          This merge_output() reads -> sub_result.get("formatted_answer"),
                                       sub_result.get("citations"),
                                       sub_result.get("status")

        policy_answer (str | None): final rendered policy Q&A answer;
          written by OutputFormatNode inside the inner graph.
        result: PostProcessNode (outer post_process slot) reads
          state.get("result") - the inner graph emits the rendered answer
          under "formatted_answer", so map it to "result" as well; otherwise
          the final output surfaced by PostProcessNode (and the output
          security gate) is always empty.
        status (str | None): terminal AgentStatus value from the inner graph run.
        """
        return {
            "policy_answer": sub_result.get("formatted_answer"),
            "result": sub_result.get("formatted_answer"),
            "citations": sub_result.get("citations"),
            "status": sub_result.get("status"),
            # Outer reason wins. A reason already settled before the inner run
            # is the real one; the inner graph only ever sees the downstream
            # consequence of it, so taking the inner value first would replace a
            # specific reason with a generic one - and a plain sub_result.get()
            # would erase the outer reason entirely whenever the inner run did
            # not set its own.
            "error_code": state.get("error_code") or sub_result.get("error_code", ""),
        }


class InsurancePolicyQAAgent(AgentBaseGraph):
    """Outer graph for INS-C2-001 (Cat 2 RAG).

    Inherits AgentBaseGraph directly (framework base class). Domain logic is
    fully encapsulated in PolicyQAGraphNode (main slot), which delegates to
    DomainWorkflowGraph (inner BaseGraph).

    Backbone (fixed - identical to Cat 1):
        START -> initialize -> pre_process -> main -> post_process -> finalize -> END

    register_nodes() is the ONLY override:
      - super().register_nodes() fills: initialize, finalize (framework defaults)
      - pre_process:  PreProcessNode (input validation gate)
      - main:         PolicyQAGraphNode (delegates to DomainWorkflowGraph)
      - post_process: PostProcessNode (output security gate)

    add_edges() is NOT overridden - backbone wiring belongs to the framework.
    """

    @property
    def name(self) -> str:
        """Agent identifier registered with AgentRegistry."""
        return "InsurancePolicyQAAgent"

    @property
    def state_schema(self) -> type:
        return State

    def register_nodes(self) -> None:
        """Fill all 5 backbone slots.

        super().register_nodes() MUST be called first - it injects the
        framework's default InitializeNode (sets schema_version, session_id,
        trust_level) and FinalizeNode (builds response_metadata, total_time_ms).
        """
        super().register_nodes()  # fills: initialize, finalize

        self._nodes["pre_process"] = PreProcessNode()
        self._nodes["main"] = PolicyQAGraphNode(runtime_config=self.config)
        self._nodes["post_process"] = PostProcessNode()

    # add_edges() is NOT overridden - backbone wiring belongs to the framework.


# Back-compat alias - config/agent.yaml declares the dotted class path, and
# src/api/server.py imports the class directly. Keep both names pointing at the agent.
Graph = InsurancePolicyQAAgent
