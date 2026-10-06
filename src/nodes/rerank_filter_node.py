"""AgentCore Platform v1.0"""

# INS-C2-001 - RerankFilterNode
# Domain node 3: rerank the retrieval candidates and enforce the relevance
# floor. Deterministic: a small policy-type-match boost on top of the
# retrieval score, drop everything below score_threshold, cap the survivors
# at top_k.
#
# Config: reads retrieval_top_k / retrieval_score_threshold from State
# (execute(self, state) -> dict, NO config parameter). These scalars are
# seeded by DomainWorkflowGraph._extra_initial_state(); when unseeded (e.g.
# a unit test that constructs the node directly) the node falls back to its
# own module defaults below. A caller-supplied top_k override
# (query_filters, validated upstream) wins when it is stricter.
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).

from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress

from src.schemas.state import from_json, to_json

# Module defaults - mirror the `retrieval` block in config/config.yaml.
_DEFAULT_TOP_K = 4
_DEFAULT_SCORE_THRESHOLD = 0.25

# Boost applied when a candidate's policy_type matches the caller's filter.
_POLICY_TYPE_BOOST = 0.1


class RerankFilterNode(FunctionNode):
    """Rerank candidates, apply the score threshold, cap at top_k.

    Input state keys:
        retrieved_documents:        JSON list of scored candidates (from RetrieveNode)
        query_filters:               JSON dict with optional policy_type / top_k override
        retrieval_top_k:             forwarded config top_k (state-seeded scalar)
        retrieval_score_threshold:   forwarded config score_threshold (state-seeded scalar)

    Output state keys (partial dict):
        ranked_documents: JSON list of surviving passages (score desc, <= top_k)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        # The request was already found unacceptable upstream: this run
        # completes without an answer, so there is nothing for this step to
        # do. Returning the marker keeps it on the node's own result dict,
        # which is what the output gate inspects.
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}

        emit_progress("Ranking the results...")
        if state.get("status") == AgentStatus.ERROR.value:
            # A prior node failed closed - pass the error through untouched
            # (the linear topology has no conditional edges to skip on error).
            return {}

        candidates: list[dict[str, Any]] = from_json(state.get("retrieved_documents"), []) or []
        filters = from_json(state.get("query_filters"), {}) or {}

        try:
            top_k = int(state.get("retrieval_top_k", _DEFAULT_TOP_K))
        except (TypeError, ValueError):
            top_k = _DEFAULT_TOP_K
        top_k = max(1, min(20, top_k))
        # A stricter caller override (validated by InputValidateNode) wins.
        caller_top_k = filters.get("top_k")
        if isinstance(caller_top_k, int) and 1 <= caller_top_k < top_k:
            top_k = caller_top_k

        try:
            score_threshold = float(state.get("retrieval_score_threshold", _DEFAULT_SCORE_THRESHOLD))
        except (TypeError, ValueError):
            score_threshold = _DEFAULT_SCORE_THRESHOLD
        score_threshold = max(0.0, min(1.0, score_threshold))

        policy_type = filters.get("policy_type")

        reranked: list[dict[str, Any]] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            entry = dict(candidate)  # local copy - inputs stay immutable
            try:
                score = float(entry.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            if policy_type and str(entry.get("policy_type", "")).lower() == str(policy_type).lower():
                score = min(1.0, score + _POLICY_TYPE_BOOST)
            entry["score"] = round(score, 4)
            reranked.append(entry)

        # Deterministic ordering: score desc, then id asc for stable ties.
        reranked.sort(key=lambda c: (-c.get("score", 0.0), str(c.get("id", ""))))

        kept = [c for c in reranked if c.get("score", 0.0) >= score_threshold][:top_k]
        dropped = len(reranked) - len(kept)

        # Domain audit: rerank + relevance floor applied.
        emit_trace_event(
            "rerank_filter_complete",
            {
                "kept": len(kept),
                "dropped": dropped,
                "score_threshold": score_threshold,
                "top_k": top_k,
            },
            state,
        )

        return {"ranked_documents": to_json(kept)}
