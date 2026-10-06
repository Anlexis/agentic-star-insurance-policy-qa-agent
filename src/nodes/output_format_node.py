"""AgentCore Platform v1.0"""

# INS-C2-001 - OutputFormatNode
# Domain node 5 (terminal): compose the final formatted answer - the grounded
# answer body, the Sources list, and the standing INS advisory disclaimer.
# The disclaimer is part of THIS node's domain output contract, not of the
# outer post_process slot (post_process only gates, it does not compose) -
# but the output gate independently ENFORCES the disclaimer's presence on
# whatever string reaches it, so a future rendering change cannot silently
# drop it (see src/nodes/post_process_node.py).
#
# Wired by the inner graph (DomainWorkflowGraph). get_output() of the inner
# graph surfaces formatted_answer + status to the outer merge_output().
# Returns only changed state keys (partial dict).

from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress

from src.schemas.state import from_json

# Standing INS advisory line - appended to EVERY answer this template emits.
# Reflects the Insurance Business Act Article 294 policy-explanation duty:
# this summary supports, but does not replace, that duty - the policyholder
# must still confirm binding interpretations with the insurer or a licensed
# agent. The output gate (post_process_node) imports this constant as the
# single source of truth for its disclaimer-presence check.
_ADVISORY_DISCLAIMER = (
    "This answer is generated from the seeded policy knowledge base for "
    "informational purposes only. It does not amend or override your actual "
    "policy contract and rider certificate. Confirm any binding "
    "interpretation - especially before filing a claim or cancelling "
    "coverage - with your insurer or a licensed agent."
)


class OutputFormatNode(FunctionNode):
    """Compose the final answer: body + sources + advisory disclaimer.

    Input state keys:
        grounded_answer: answer body with [n] citation markers
        citations:       JSON list [{ref, id, title, source}]

    Output state keys (partial dict):
        formatted_answer: final rendered answer string
        status:           AgentStatus.SUCCESS.value (plain string - State
                          carries the value, never the bare enum)
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

        emit_progress("Formatting the response...")
        if state.get("status") == AgentStatus.ERROR.value:
            # A prior node failed closed - pass the error through untouched
            # (the linear topology has no conditional edges to skip on error).
            return {}

        grounded_answer = state.get("grounded_answer") or ("No answer is available for this request.")
        citations: list[dict[str, Any]] = from_json(state.get("citations"), []) or []

        lines: list[str] = []
        lines.append("# Insurance Policy Q&A Result")
        lines.append("")
        lines.append(grounded_answer)
        lines.append("")
        lines.append("## Sources")
        if citations:
            for citation in citations:
                if not isinstance(citation, dict):
                    continue
                ref = citation.get("ref", "?")
                title = str(citation.get("title", "")).strip()
                source = str(citation.get("source", "")).strip()
                suffix = f" ({source})" if source else ""
                lines.append(f"- [{ref}] {title}{suffix}")
        else:
            lines.append("- none (no policy-knowledge-base passage cleared the relevance threshold)")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(f"*{_ADVISORY_DISCLAIMER}*")

        formatted_answer = "\n".join(lines)

        # Domain audit: final answer composed (disclaimer attached).
        emit_trace_event(
            "output_format_complete",
            {
                "answer_chars": len(formatted_answer),
                "citation_count": len(citations),
            },
            state,
        )

        return {
            "formatted_answer": formatted_answer,
            "status": AgentStatus.SUCCESS.value,
        }
