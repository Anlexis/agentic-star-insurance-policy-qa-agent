"""AgentCore Platform v1.0"""

# INS-C2-001 - InputValidateNode
# Domain node 1: parse and validate the incoming policy question and the
# caller's structured parameters.
#
# Caller data arrives on two channels, both untrusted:
#
#   input (string)          -> the question text. A legacy JSON envelope
#                              ({"question": ..., "policy_type": ..., "top_k": N})
#                              is still parsed for backwards compatibility.
#   input_context (dict)    -> structured parameters (policy_type / top_k),
#                              bridged from the outer graph (see
#                              src/graph/context_bridge.py). Takes precedence
#                              over envelope values when both are present.
#
# Every caller field is validated against explicit bounds and FAILS CLOSED:
# an invalid policy_type or top_k returns status=ERROR naming the field -
# the offending value is never echoed into error logs or notes. Absent
# fields degrade to the unfiltered baseline (no policy_type filter, the
# configured top_k).
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).

import json
import re
from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.services.failure_message import INVALID_VALUE

from src.schemas.state import to_json

# Hard cap on the normalised query length (defence-in-depth on input size).
_MAX_QUERY_CHARS = 2000

# Bounds for the caller-supplied top_k override (untrusted numeric guard).
_TOP_K_MIN = 1
_TOP_K_MAX = 20

# Caller-supplied policy_type is locked to an inert identifier: it is compared
# against KB policy_type values and must never carry free text.
_POLICY_TYPE_RE = re.compile(r"^[a-z0-9_]{1,32}$")

_WHITESPACE_RE = re.compile(r"\s+")

# Sentinel distinguishing "field absent" from "field present but None".
_ABSENT = object()


def _validate_top_k(value: Any) -> tuple[int | None, str | None]:
    """Validate the untrusted caller top_k. Returns (top_k, error).

    Fail-closed numeric guard: accepts only a real int (bools are rejected -
    they pass isinstance(int)) within [_TOP_K_MIN, _TOP_K_MAX]. Everything
    else - floats (including NaN / +-Infinity, which arrive both as strings
    and as raw JSON non-finite literals), strings, out-of-range magnitudes -
    is rejected with an error naming the FIELD, never the value.
    """
    if value is None:
        return None, None
    if isinstance(value, bool) or not isinstance(value, int):
        return None, (f"InputValidateNode: top_k must be an integer between " f"{_TOP_K_MIN} and {_TOP_K_MAX}")
    if not _TOP_K_MIN <= value <= _TOP_K_MAX:
        return None, (f"InputValidateNode: top_k must be an integer between " f"{_TOP_K_MIN} and {_TOP_K_MAX}")
    return value, None


def _validate_policy_type(value: Any) -> tuple[str | None, str | None]:
    """Validate the untrusted caller policy_type. Returns (policy_type, error).

    The value is canonicalised (stripped, lower-cased) and then locked to the
    inert identifier pattern [a-z0-9_]{1,32}. Anything else - non-strings,
    empty strings, free text - is rejected with an error naming the FIELD,
    never the value.
    """
    if value is None:
        return None, None
    if not isinstance(value, str):
        return None, "InputValidateNode: policy_type must be a short lowercase identifier"
    canonical = value.strip().lower()
    if not canonical:
        return None, None
    if not _POLICY_TYPE_RE.match(canonical):
        return None, "InputValidateNode: policy_type must be a short lowercase identifier"
    return canonical, None


class InputValidateNode(FunctionNode):
    """Parse and validate the request into a normalised query + bounded filters.

    Input state keys:
        validated_input | user_input: request payload (from PreProcessNode)
        input_context:               structured caller parameters (bridged)

    Output state keys (partial dict):
        search_query:  normalised free-text policy question
        query_filters: JSON dict {"policy_type": str|None, "top_k": int|None}
        intake_notes:  (when non-fatal anomalies were seen) JSON list[str]

    On an invalid caller field: {"status": ERROR, "error_log": [...]} - the
    pipeline fails closed rather than running with an unvalidated parameter.
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        emit_progress("Checking the request...")
        raw = state.get("validated_input") or state.get("user_input", "")
        input_context = state.get("input_context") or {}
        if not isinstance(input_context, dict):
            input_context = {}
        notes: list[str] = []

        query = ""
        raw_policy_type: Any = None
        raw_top_k: Any = None

        if isinstance(raw, str) and raw.strip():
            payload: Any = None
            text = raw.strip()
            if text.startswith("{"):
                try:
                    payload = json.loads(text)
                except (json.JSONDecodeError, ValueError):
                    notes.append(
                        "InputValidateNode: JSON-looking input did not parse - " "treated as plain text question."
                    )
            if isinstance(payload, dict):
                query = str(payload.get("question") or payload.get("query") or "")
                raw_policy_type = payload.get("policy_type")
                raw_top_k = payload.get("top_k")
            else:
                query = text
        else:
            notes.append("InputValidateNode: empty request - no question to search.")

        # Structured parameters from input_context win over envelope values.
        if input_context.get("policy_type", _ABSENT) is not _ABSENT:
            raw_policy_type = input_context.get("policy_type")
        if input_context.get("top_k", _ABSENT) is not _ABSENT:
            raw_top_k = input_context.get("top_k")

        # Fail-closed validation of every caller-controlled parameter.
        # A value outside its documented contract stops the retrieval path, and
        # the run then COMPLETES carrying the reason: the caller can correct the
        # value and send the request again on the same conversation.
        policy_type, error = _validate_policy_type(raw_policy_type)
        if error:
            emit_progress(INVALID_VALUE)
            return {"status": AgentStatus.SUCCESS.value, "error_code": "INVALID_REQUEST", "error_log": [error]}
        top_k, error = _validate_top_k(raw_top_k)
        if error:
            emit_progress(INVALID_VALUE)
            return {"status": AgentStatus.SUCCESS.value, "error_code": "INVALID_REQUEST", "error_log": [error]}

        # Normalise whitespace and cap length.
        query = _WHITESPACE_RE.sub(" ", query).strip()
        if len(query) > _MAX_QUERY_CHARS:
            query = query[:_MAX_QUERY_CHARS]
            notes.append(f"InputValidateNode: query truncated to {_MAX_QUERY_CHARS} chars.")

        filters = {"policy_type": policy_type, "top_k": top_k}

        # Domain audit: policy question parsed and validated.
        emit_trace_event(
            "input_validate_complete",
            {
                "query_chars": len(query),
                "has_policy_type_filter": policy_type is not None,
                "has_top_k_override": top_k is not None,
            },
            state,
        )

        out: dict[str, Any] = {
            "search_query": query,
            "query_filters": to_json(filters),
        }
        if notes:
            out["intake_notes"] = to_json(notes)
        return out
