"""AgentCore Platform v1.0"""

# State must be a flat TypedDict - never Pydantic BaseModel.
# LangGraph checkpoints use msgpack serialization; Pydantic objects
# cause silent corruption.  Extend AgentState with agent-specific
# fields only.  Do NOT add credentials, secrets, or Pydantic models.
#
# msgpack safety: structured fields (dict / list[dict]) are stored as JSON
# STRINGS, not bare Python containers - a bare dict/list in a checkpointed
# State field is not checkpoint-safe. Producers serialize with to_json() on
# write; consumers deserialize with from_json() on read. Plain scalar config
# knobs (int / float / str) do NOT need JSON encoding - only dict/list
# containers do.
#
# INS-C2-001 - Insurance Policy Q&A Agent (Cat 2 RAG).
# Two-layer nested Cat 2 graph: outer backbone (AgentBaseGraph) + inner
# domain workflow (BaseGraph).  Fields below cover both layers.
#
# PII / confidentiality note: this template answers general policy-terms
# questions (coverage scope, exclusions, claim conditions, rider terms) over
# a seeded policy knowledge base. It does not read or persist policyholder
# account numbers, contract numbers, or other direct identifiers - only the
# normalised search query, KB passage summaries, and the final grounded
# answer are written to State.

import json
from typing import Any, Optional

from framework.schemas.agent_state import AgentState


def to_json(value: Any) -> Optional[str]:
    """Serialize a dict/list State field to a JSON string (msgpack safety).

    None passes through unchanged so an 'unset' field stays distinguishable
    from an empty container.
    """
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def from_json(value: Optional[str], default: Any = None) -> Any:
    """Deserialize a JSON-string State field back to its dict/list.

    None / empty / malformed input -> the supplied ``default`` so a missing or
    corrupt field is non-fatal for the consuming node.
    """
    if not value:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


class State(AgentState):
    """Flat TypedDict for INS-C2-001.

    All shared fields (user_input, status, session_id, node_history,
    error_log, hitl_*, etc.) are inherited from AgentState.
    Domain fields are Optional and unset at graph initialisation,
    before any node has written a value.
    """

    # ------------------------------------------------------------------
    # Outer layer - set by PreProcessNode / PolicyQAGraphNode.merge_output
    # ------------------------------------------------------------------

    # Validated query payload produced by PreProcessNode (input gate).
    validated_input: Optional[str]

    # Final policy Q&A answer, mapped from the inner graph's formatted_answer
    # output via merge_output().
    policy_answer: Optional[str]

    # ------------------------------------------------------------------
    # Inner layer - domain nodes (DomainWorkflowGraph)
    # ------------------------------------------------------------------

    # InputValidateNode outputs
    # Normalised free-text policy question (whitespace-collapsed, length-capped).
    search_query: Optional[str]

    # JSON STRING (to_json) of validated structured query params. Deserialised
    # dict shape: {"policy_type": str | None, "top_k": int | None}.
    # Consumers (RetrieveNode, RerankFilterNode) read it back via from_json().
    query_filters: Optional[str]

    # Config knobs forwarded by PolicyQAGraphNode._parent_config() -> the
    # inner DomainWorkflowGraph._extra_initial_state() hook. Plain scalars -
    # no JSON encoding needed (the JSON-string convention applies to
    # dict/list containers only). Nodes read these via
    # execute(self, state) -> dict (NO config parameter); unset (e.g. direct
    # unit-test construction) falls back to each node's own module-default
    # constant.
    retrieval_top_k: Optional[int]
    retrieval_score_threshold: Optional[float]
    retrieval_kb_path: Optional[str]

    # RetrieveNode output
    # JSON STRING (to_json) of scored KB candidates. Deserialised shape:
    # list[dict], each entry {"id": str, "title": str, "policy_type": str,
    # "source": str, "score": float, "excerpt": str}.
    # Consumers (RerankFilterNode) read it back via from_json().
    retrieved_documents: Optional[str]

    # RerankFilterNode output
    # JSON STRING (to_json) of reranked + threshold-filtered passages, capped
    # at top_k. Same entry shape as retrieved_documents.
    # Consumers (GenerateAnswerNode) read it back via from_json().
    ranked_documents: Optional[str]

    # GenerateAnswerNode outputs
    # Rule-assembled grounded answer body with numbered citation markers.
    grounded_answer: Optional[str]

    # JSON STRING (to_json) of citations. Deserialised shape: list[dict],
    # each entry {"ref": int, "id": str, "title": str, "source": str}.
    # Consumers (OutputFormatNode) read it back via from_json().
    citations: Optional[str]

    # OutputFormatNode output
    # Final formatted answer (body + sources + advisory disclaimer). Written by
    # OutputFormatNode; surfaced to the outer graph via get_output() ->
    # merge_output().
    formatted_answer: Optional[str]

    # Validation / parse notes accumulated during intake (no PII).
    # JSON STRING (to_json) of list[str].
    intake_notes: Optional[str]

    # ------------------------------------------------------------------
    # Degraded completion marker
    # ------------------------------------------------------------------

    # Set when the run completes WITHOUT producing an answer because the
    # caller's request could not be accepted as written - a rejection the
    # caller can correct and retry. The run still completes: no retrieval is
    # performed, no answer is assembled, and the domain audit event for the
    # rejection is still emitted. Carrying this as a completion marker rather
    # than a terminal error is what lets the caller see the reason and send a
    # corrected request on the same conversation.
    #
    # Content the agent refuses outright, and a breach of a contract the
    # caller cannot influence, are NOT reported here - those stay terminal so
    # they are not mistaken for something a reworded request would get past.
    #
    # Once set, every later domain node passes through without doing work, and
    # the value is carried across the inner/outer boundary by get_output() and
    # merge_output().
    error_code: Optional[str]

    # ------------------------------------------------------------------
    # Tracing / audit - framework-managed; do NOT write from node code
    # ------------------------------------------------------------------

    trace_id: Optional[str]
    correlation_id: Optional[str]
    # node_history inherited from AgentState; listed here for clarity
    # node_history: Optional[List[str]]
