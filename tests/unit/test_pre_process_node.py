# INS-C2-001 — Unit Tests: PreProcessNode (outer pre_process slot; input gate)
#
# Invocation canon: every test invokes the node via
# node(state) — BaseNode.__call__ → trust gate → PII mask → execute()
# → credential gate — never a bare node.execute(state). PreProcessNode
# requires VERIFIED_EXTERNAL, so its behavioural tests build the state at that
# level (the ANONYMOUS rejection lives in test_trust_gate.py).
#
# Masking layering note: the FRAMEWORK input gate (shared.security.pii_detector)
# masks user_input / validated_input / llm_response to [MASKED] BEFORE
# execute() runs — email, phone/SSN/CC digit groups, AND any two-or-more
# consecutive Title-Case words (the `name` heuristic — high recall, low
# precision: regulatory-sounding phrases count too, e.g. "Insurance Business
# Act" or "Cooling Off" match just like a person's name would). This node
# adds NO domain-specific identifier screen of its own (docs/02_design.md
# input-screen note) — the framework default is the only masking layer. Positive-path
# payloads are therefore all-lowercase / PII-free so the mask leaves them
# untouched; intentional-PII tests assert the raw identifier is gone and
# [MASKED] is present.
#
# Mirrors docs/03_test_spec.md §2.1 (PRE-01..PRE-08).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

from unittest.mock import MagicMock

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

import src.nodes.pre_process_node
from src.nodes.pre_process_node import PreProcessNode

# All-lowercase, PII-free policy question: no Title-Case bigram, no @, no
# digit run, so the framework mask leaves the payload byte-for-byte untouched.
_VALID_QUERY = "what is the waiting period before hospitalization coverage becomes effective"


def _make_state(user_input=_VALID_QUERY, **extra) -> dict:
    state = {
        "user_input": user_input,
        "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestPreProcessSuccess:
    def test_pre_01_valid_query_accepted(self):
        result = PreProcessNode()(_make_state())
        assert result["status"] == AgentStatus.SUCCESS.value
        # Regression guard: State carries the plain string, never the enum.
        assert isinstance(result["status"], str) and not isinstance(result["status"], AgentStatus)
        assert result["validated_input"] == _VALID_QUERY

    def test_enriched_context_carries_channel(self):
        result = PreProcessNode()(_make_state(input_context={"channel": "web"}))
        assert result["enriched_context"]["channel"] == "web"
        assert result["enriched_context"]["source"] == "InsurancePolicyQAAgent"

    def test_missing_channel_defaults_to_unknown(self):
        result = PreProcessNode()(_make_state())
        assert result["enriched_context"]["channel"] == "unknown"


class TestPreProcessRejection:
    def test_pre_02_empty_input_is_error(self):
        result = PreProcessNode()(_make_state(user_input=""))
        assert result["status"] == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")
        assert result["error_log"]
        # No validated_input is produced on the reject path.
        assert "validated_input" not in result

    def test_whitespace_only_is_error(self):
        result = PreProcessNode()(_make_state(user_input="   \n\t "))
        assert result["status"] == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")

    def test_pre_03_missing_user_input_is_error(self):
        state = _make_state()
        del state["user_input"]
        result = PreProcessNode()(state)
        assert result["status"] == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")

    def test_non_string_input_is_error(self):
        # A dict payload has no .strip() — execute() raises internally, and
        # BaseNode.__call__'s catch-all converts it to a status:error result
        # (never an unhandled exception; the gate pipeline never crashes).
        result = PreProcessNode()(_make_state(user_input={"malicious": "dict"}))
        assert result["status"] == AgentStatus.ERROR.value


class TestPreProcessIdentifierScreen:
    """PRE-04: framework default masking — this node adds no extra screen."""

    def test_email_masked_by_framework_input_gate(self):
        raw = "please escalate this to my agent at policyholder.name@example.com today"
        result = PreProcessNode()(_make_state(user_input=raw))
        vi = result["validated_input"]
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "policyholder.name@example.com" not in vi
        assert "[MASKED]" in vi

    def test_grouped_account_digits_masked(self):
        # 4-4-4 digit run matches the framework my_number_jp pattern.
        raw = "the reference number on my policy is 1234 5678 9012 please check it"
        result = PreProcessNode()(_make_state(user_input=raw))
        vi = result["validated_input"]
        assert "1234 5678 9012" not in vi
        assert "[MASKED]" in vi

    def test_regulatory_act_name_is_masked_as_a_title_case_bigram(self):
        # The framework `name` heuristic matches ANY two-or-more consecutive
        # Title-Case words — it does not know "Insurance Business Act" is a
        # statute, not a person. Domain-sounding regulatory phrasing is
        # exactly the false-positive class this template must design payloads
        # around (this IS the docs/02_design.md regulatory driver's own name).
        raw = "does the Insurance Business Act require written disclosure"
        result = PreProcessNode()(_make_state(user_input=raw))
        vi = result["validated_input"]
        assert "Insurance Business Act" not in vi
        assert "[MASKED]" in vi


class TestPreProcessBounds:
    def test_oversize_input_fails_closed(self):
        result = PreProcessNode()(_make_state(user_input="x" * 20_001))
        assert result["status"] == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")
        joined = " ".join(str(e) for e in result["error_log"])
        assert "user_input" in joined and "20000" in joined
        assert "validated_input" not in result

    def test_maximum_size_input_is_accepted(self):
        result = PreProcessNode()(_make_state(user_input="x" * 20_000))
        assert result["status"] == AgentStatus.SUCCESS.value

    def test_channel_label_is_capped(self):
        result = PreProcessNode()(_make_state(input_context={"channel": "c" * 200}))
        assert result["enriched_context"]["channel"] == "c" * 64

    def test_non_dict_input_context_is_tolerated(self):
        result = PreProcessNode()(_make_state(input_context="weird"))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["enriched_context"]["channel"] == "unknown"


class TestPreProcessAudit:
    def test_pre_08_domain_audit_payload(self, monkeypatch):
        """Audit: the accepted request emits pre_process_complete; the assertion
        targets call.args[1] — the event payload — never the whole call repr."""
        spy = MagicMock()
        monkeypatch.setattr(src.nodes.pre_process_node, "emit_trace_event", spy)
        PreProcessNode()(_make_state())
        events = [call.args[0] for call in spy.call_args_list]
        assert "pre_process_complete" in events
        payload = spy.call_args_list[events.index("pre_process_complete")].args[1]
        assert payload["input_chars"] == len(_VALID_QUERY)
