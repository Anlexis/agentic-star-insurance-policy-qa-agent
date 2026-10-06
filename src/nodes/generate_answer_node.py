"""AgentCore Platform v1.0"""

# INS-C2-001 - GenerateAnswerNode
# Domain node 4: assemble the grounded answer from the ranked policy KB
# passages.
#
# This version is DETERMINISTIC (no live LLM call): the answer is
# rule-assembled from the ranked passages only - a fixed lead sentence plus
# one cited point per passage, each carrying a numbered citation marker [n].
# Nothing outside the ranked_documents input reaches the answer body - the
# caller's question text is deliberately NOT echoed into the rendered answer
# (caller free text in the output surface would be caller-controlled output
# injection), so the output is grounded by construction. The LLM synthesis
# upgrade seam is documented in docs/02_design.md ("v1 Implementation Note -
# LLM synthesis") and config/prompts/answer_synthesis_prompt.md: an upgraded
# node swaps the assembly for an LLM call over the same input and emits the
# same state contract.
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

# Answer body used when no KB passage cleared the relevance threshold.
_NO_COVERAGE_ANSWER = (
    "The policy knowledge base does not contain sufficient coverage to answer "
    "this question. Rephrase the question with more specific policy or rider "
    "terms, or escalate to your insurance agent / the insurer's customer "
    "service team for a manual review of your policy documents."
)

# Fixed lead sentence - never embeds caller-supplied text.
_LEAD_SENTENCE = "Based on the policy knowledge base, the most relevant passages for " "this question are:"

# Cited excerpt length per passage inside the answer body.
_POINT_EXCERPT_CHARS = 240


def _first_sentences(text: str, limit: int) -> str:
    """Trim an excerpt at a sentence boundary where possible, else hard-cap."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    period = cut.rfind(". ")
    if period > limit // 2:
        return cut[: period + 1]
    return cut.rstrip() + "..."


class GenerateAnswerNode(FunctionNode):
    """Rule-based grounded answer assembly with numbered citations.

    Input state keys:
        ranked_documents: JSON list of surviving passages (from RerankFilterNode)

    Output state keys (partial dict):
        grounded_answer: answer body with [n] citation markers
        citations:       JSON list [{ref, id, title, source}]
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

        emit_progress("Composing the answer...")
        if state.get("status") == AgentStatus.ERROR.value:
            # A prior node failed closed - pass the error through untouched
            # (the linear topology has no conditional edges to skip on error).
            return {}

        ranked: list[dict[str, Any]] = from_json(state.get("ranked_documents"), []) or []

        citations: list[dict[str, Any]] = []

        if not ranked:
            grounded_answer = _NO_COVERAGE_ANSWER
        else:
            lines: list[str] = [_LEAD_SENTENCE, ""]
            for ref, doc in enumerate(ranked, start=1):
                if not isinstance(doc, dict):
                    continue
                title = str(doc.get("title", "")).strip()
                excerpt = _first_sentences(str(doc.get("excerpt", "")), _POINT_EXCERPT_CHARS)
                lines.append(f"[{ref}] {title}: {excerpt}")
                citations.append(
                    {
                        "ref": ref,
                        "id": str(doc.get("id", "")),
                        "title": title,
                        "source": str(doc.get("source", "")),
                    }
                )
            grounded_answer = "\n".join(lines)

        # Domain audit: grounded answer assembled.
        emit_trace_event(
            "generate_answer_complete",
            {
                "citation_count": len(citations),
                "answer_chars": len(grounded_answer),
                "no_coverage": not ranked,
            },
            state,
        )

        return {
            "grounded_answer": grounded_answer,
            "citations": to_json(citations),
        }
