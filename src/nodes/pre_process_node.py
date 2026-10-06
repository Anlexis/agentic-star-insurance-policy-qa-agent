"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - Return AgentStatus.<X>.value strings for status assignments
#  - Read input_context via state.get("input_context", {}) — read-only
#  - Never import from mediator/, api/, or other agents
#
# This is the outer backbone input gate node — required_trust_level is
# VERIFIED_EXTERNAL (matches config/agent.yaml required_trust_level), so an
# ANONYMOUS caller is denied by BaseNode.__call__ before execute() ever runs.

from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.services.failure_message import EMPTY_INPUT, TOO_LONG

# Hard cap on the raw request size (defence-in-depth; the inner intake node
# additionally normalises and caps the parsed query).
_MAX_INPUT_CHARS = 20_000

# Cap on the caller-supplied channel label carried into enriched_context.
_MAX_CHANNEL_CHARS = 64


class PreProcessNode(FunctionNode):
    """Validate and enrich incoming input before main processing."""

    # Trust gate: explicit by design, not inherited implicitly.
    # Outer backbone gate slot — VERIFIED_EXTERNAL (config/agent.yaml
    # required_trust_level); inner Cat-2 domain nodes stay ANONYMOUS.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        emit_progress("Checking the request...")
        user_input = state.get("user_input", "")
        input_context = state.get("input_context", {})  # read-only
        if not isinstance(input_context, dict):
            input_context = {}

        if not user_input or not user_input.strip():
            # Nothing to answer, but the caller can send a question and try
            # again - so the run completes carrying the reason rather than
            # terminating and surfacing only an exception type.
            emit_progress(EMPTY_INPUT)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": "EMPTY_INPUT",
                "error_log": ["PreProcessNode: user_input is empty or missing"],
            }

        if len(user_input) > _MAX_INPUT_CHARS:
            # Fail closed on oversized payloads — name the bound, never echo
            # the content.
            emit_progress(TOO_LONG)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": "QUESTION_TOO_LONG",
                "error_log": [f"PreProcessNode: user_input exceeds the maximum of " f"{_MAX_INPUT_CHARS} characters"],
            }

        # Domain audit: a request was accepted and validated.
        emit_trace_event(
            "pre_process_complete",
            {"input_chars": len(user_input.strip())},
            state,
        )

        return {
            "validated_input": user_input.strip(),
            "enriched_context": {
                "source": "InsurancePolicyQAAgent",
                # Caller-supplied label — capped, carried for observability
                # only (never rendered into the answer).
                "channel": str(input_context.get("channel", "unknown"))[:_MAX_CHANNEL_CHARS],
            },
            "status": AgentStatus.SUCCESS.value,
        }
