"""AgentCore Platform v1.0"""

# Output security gate: this node calls the MODULE-LEVEL
# _scan_disallowed_content() scan below from execute() itself. The output
# string is scanned for disallowed content (API keys, JWT tokens, Bearer
# tokens, raw credential assignments). On a violation, the surfaced output
# is replaced with a sanitised stub and ERROR status is returned. No
# _extra_security_gate_input/_extra_security_gate_output instance methods
# are defined on this node — the real SDK auto-wraps such hooks, which
# would break the e2e invoke path.
#
# On top of the credential scan, the gate independently ENFORCES the
# template's stated output contract: every non-empty answer carries the
# standing advisory disclaimer. A clean answer missing the disclaimer (e.g.
# after a future rendering change) has it appended here, with an audit
# event — the invariant holds for whatever string reaches the gate, not
# only for the renderer's happy path.
#
# Trust gate: this is the outer backbone gate node — required_trust_level is
# VERIFIED_EXTERNAL (matches config/agent.yaml required_trust_level).

import re
from typing import Any, ClassVar, List, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.services.failure_message import EMPTY_INPUT, INPUT_REJECTED, INVALID_VALUE, OUTPUT_BLOCKED, TOO_LONG

from src.nodes.output_format_node import _ADVISORY_DISCLAIMER

# Disallowed content patterns. Each tuple: (name, compiled regex).
_DISALLOWED_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    ("api_key", re.compile(r"\b(?:sk|pk|ak)-[A-Za-z0-9]{16,}", re.IGNORECASE)),
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    ("bearer_token", re.compile(r"Bearer\s+[A-Za-z0-9._~+/]{20,}", re.IGNORECASE)),
    (
        "credential_assignment",
        re.compile(
            r"\b(?:password|passwd|secret|api_key|token|access_key|private_key)\s*[:=]\s*\S{8,}",
            re.IGNORECASE,
        ),
    ),
]

_SANITISED_STUB = (
    "[OUTPUT BLOCKED by the output security gate — disallowed content "
    "detected. Review the generated answer and retry without "
    "credential-like strings.]"
)


def _scan_disallowed_content(content: str) -> Optional[str]:
    """Run the output content gate.

    Returns the name of the first matched violation, or None if clean.
    """
    for name, pattern in _DISALLOWED_PATTERNS:
        if pattern.search(content):
            return name
    return None


# Caller-facing wording for a run that completed without an answer. The marker
# is an internal reason code; this maps it to the sentence the caller sees.
# Static sentences only - no request value is ever substituted, so nothing the
# caller sent can be reflected back through this path.
_DEGRADED_MESSAGES = {
    "EMPTY_INPUT": EMPTY_INPUT,
    "QUESTION_TOO_LONG": TOO_LONG,
    "INVALID_REQUEST": INVALID_VALUE,
}


class PostProcessNode(FunctionNode):
    """Gate the final output: credential scan + advisory-disclaimer enforcement."""

    # Trust gate: explicit by design, not inherited implicitly.
    # Outer backbone gate slot — VERIFIED_EXTERNAL (config/agent.yaml
    # required_trust_level); inner Cat-2 domain nodes stay ANONYMOUS.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        emit_progress("Finalising the response...")

        # The run completed without an answer because the request could not be
        # accepted as written. Report the reason as the response: the caller
        # needs to know what to change, and an empty body would leave them with
        # nothing. Status stays SUCCESS - the run did what it could with the
        # request it was given, and the caller can correct it and send again on
        # the same conversation.
        marker = state.get("error_code")
        if marker:
            message = _DEGRADED_MESSAGES.get(marker, INPUT_REJECTED)
            emit_trace_event("post_process_degraded", {"reason": marker}, state)
            return {
                "formatted_output": message,
                "result": message,
                "status": AgentStatus.SUCCESS.value,
                "error_code": marker,
            }
        result = state.get("result", "")

        if not result or not str(result).strip():
            # No answer was generated — forward as-is (non-fatal).
            return {
                "formatted_output": result,
                "status": AgentStatus.SUCCESS.value,
            }

        text = str(result)

        violation = _scan_disallowed_content(text)
        if violation:
            emit_progress(OUTPUT_BLOCKED)
            return {
                "formatted_output": _SANITISED_STUB,
                "result": _SANITISED_STUB,
                "status": AgentStatus.ERROR.value,
                "error_log": [f"PostProcessNode: output blocked — " f"disallowed content detected ({violation})"],
            }

        # Output-contract enforcement: every non-empty answer carries the
        # advisory disclaimer. Enforced here independently of the renderer.
        out: dict[str, Any] = {}
        if _ADVISORY_DISCLAIMER not in text:
            text = f"{text.rstrip()}\n\n---\n\n*{_ADVISORY_DISCLAIMER}*"
            out["result"] = text
            emit_trace_event(
                "output_disclaimer_enforced",
                {"output_chars": len(text)},
                state,
            )

        # Domain audit: a finalized answer was emitted, clean.
        emit_trace_event(
            "post_process_complete",
            {"output_chars": len(text)},
            state,
        )

        out["formatted_output"] = text
        out["status"] = AgentStatus.SUCCESS.value
        return out
