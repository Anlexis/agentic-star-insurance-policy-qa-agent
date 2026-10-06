# INS-C2-001 — Unit Tests: InputValidateNode (inner domain node 1)
#
# Invocation canon: node(state) via BaseNode.__call__ with an ANONYMOUS caller
# (inner Cat-2 domain node). This node's input field IS validated_input, which
# IS in the framework's PII-scan field set — payloads are lowercase / PII-free
# so the mask leaves them untouched (a masked JSON envelope would corrupt the
# json.loads() parse downstream).
#
# Caller-parameter contract under test (fail CLOSED):
#   top_k       — real int (bools rejected) within 1..20; everything else —
#                 floats, NaN/±Infinity in both string and raw-float form,
#                 strings, out-of-range magnitudes — returns status=ERROR
#                 naming the FIELD, never echoing the value.
#   policy_type — canonicalised (strip/lower) then locked to [a-z0-9_]{1,32};
#                 anything else returns status=ERROR naming the field.
#   Sources: the legacy JSON envelope in the input string AND the structured
#   input_context dict (bridged from the outer graph); input_context wins.
#
# Deterministic — no LLM, no network. framework.* / src.* imports only.

import json

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.input_validate_node import InputValidateNode
from src.schemas.state import from_json


def _make_state(payload, **extra) -> dict:
    state = {
        "validated_input": payload,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


_TOP_K_MESSAGE = "InputValidateNode: top_k must be an integer between 1 and 20"
_POLICY_TYPE_MESSAGE = "InputValidateNode: policy_type must be a short lowercase identifier"


def _assert_field_rejected(result, field: str, forbidden_value=None):
    """The node must fail CLOSED on the retrieval path, and the error message
    must be EXACTLY the static field-naming text — byte-equality proves the
    rejected value is never echoed into error logs. The run COMPLETES carrying
    the reason, so the caller can correct the value and send the request again
    on the same conversation."""
    assert result["status"] == AgentStatus.SUCCESS.value, result
    assert result.get("error_code") == "INVALID_REQUEST", result
    expected = _TOP_K_MESSAGE if field == "top_k" else _POLICY_TYPE_MESSAGE
    assert result["error_log"] == [expected]
    assert "search_query" not in result


class TestPlainTextParsing:
    def test_val_01_plain_text_becomes_query(self):
        result = InputValidateNode()(_make_state("hospitalization coverage scope"))
        assert result["search_query"] == "hospitalization coverage scope"
        filters = from_json(result["query_filters"])
        assert filters == {"policy_type": None, "top_k": None}

    def test_val_02_whitespace_is_collapsed(self):
        result = InputValidateNode()(_make_state("  hospitalization   coverage\n scope "))
        assert result["search_query"] == "hospitalization coverage scope"

    def test_query_filters_is_json_string(self):
        # Structured State fields travel as JSON strings, never bare dicts.
        result = InputValidateNode()(_make_state("hospitalization coverage"))
        assert isinstance(result["query_filters"], str)
        assert isinstance(from_json(result["query_filters"]), dict)


class TestJsonEnvelopeParsing:
    def test_val_03_envelope_query_policy_type_top_k(self):
        payload = json.dumps({"query": "waiting period before coverage starts", "policy_type": "health", "top_k": 2})
        result = InputValidateNode()(_make_state(payload))
        assert result["search_query"] == "waiting period before coverage starts"
        filters = from_json(result["query_filters"])
        assert filters == {"policy_type": "health", "top_k": 2}

    def test_question_alias_accepted(self):
        payload = json.dumps({"question": "what does the hospitalization rider cover?"})
        result = InputValidateNode()(_make_state(payload))
        assert result["search_query"] == "what does the hospitalization rider cover?"

    def test_policy_type_is_canonicalised(self):
        payload = json.dumps({"query": "rider terms", "policy_type": "  HEALTH "})
        result = InputValidateNode()(_make_state(payload))
        assert from_json(result["query_filters"])["policy_type"] == "health"

    def test_val_04_malformed_json_falls_back_to_plain_text(self):
        payload = "{ this is not valid json but starts like it"
        result = InputValidateNode()(_make_state(payload))
        assert result["search_query"] == payload
        notes = from_json(result.get("intake_notes"), [])
        assert any("did not parse" in n for n in notes)


class TestInputContextParameters:
    """Structured parameters arrive via input_context (bridged from the outer
    graph) and take precedence over the legacy envelope values."""

    def test_input_context_parameters_are_honoured(self):
        result = InputValidateNode()(
            _make_state(
                "waiting period before coverage starts",
                input_context={"policy_type": "health", "top_k": 2},
            )
        )
        filters = from_json(result["query_filters"])
        assert filters == {"policy_type": "health", "top_k": 2}

    def test_input_context_wins_over_the_envelope(self):
        payload = json.dumps({"query": "rider terms", "policy_type": "life", "top_k": 9})
        result = InputValidateNode()(_make_state(payload, input_context={"policy_type": "health", "top_k": 2}))
        filters = from_json(result["query_filters"])
        assert filters == {"policy_type": "health", "top_k": 2}

    def test_non_dict_input_context_is_ignored(self):
        result = InputValidateNode()(_make_state("hospitalization coverage", input_context="not-a-dict"))
        assert from_json(result["query_filters"]) == {"policy_type": None, "top_k": None}

    def test_invalid_input_context_parameter_fails_closed(self):
        result = InputValidateNode()(_make_state("hospitalization coverage", input_context={"top_k": 99}))
        _assert_field_rejected(result, "top_k")


class TestTopKFailsClosed:
    """Every caller-controlled number goes through the finite+bounded guard."""

    @pytest.mark.parametrize(
        "bad_top_k",
        [99, -5, 0, 21],
        ids=["over-max", "negative", "zero", "just-over"],
    )
    def test_out_of_range_top_k_is_rejected(self, bad_top_k):
        payload = json.dumps({"query": "hospitalization coverage", "top_k": bad_top_k})
        result = InputValidateNode()(_make_state(payload))
        _assert_field_rejected(result, "top_k")

    @pytest.mark.parametrize(
        "bad_top_k",
        [
            "many",
            "NaN",
            "Infinity",
            "-Infinity",
            float("nan"),
            float("inf"),
            float("-inf"),
            3.7,
            5.0,
            True,
            False,
            [3],
            {"n": 3},
            10**9,
        ],
        ids=[
            "str",
            "str-nan",
            "str-inf",
            "str-neginf",
            "raw-nan",
            "raw-inf",
            "raw-neginf",
            "float",
            "integral-float",
            "bool-true",
            "bool-false",
            "list",
            "dict",
            "over-magnitude",
        ],
    )
    def test_non_integer_top_k_is_rejected(self, bad_top_k):
        result = InputValidateNode()(_make_state("hospitalization coverage", input_context={"top_k": bad_top_k}))
        _assert_field_rejected(result, "top_k")

    def test_boundary_values_are_accepted(self):
        for good in (1, 20):
            result = InputValidateNode()(_make_state("hospitalization coverage", input_context={"top_k": good}))
            assert from_json(result["query_filters"])["top_k"] == good

    def test_absent_top_k_degrades_to_none(self):
        result = InputValidateNode()(_make_state("hospitalization coverage"))
        assert from_json(result["query_filters"])["top_k"] is None


class TestPolicyTypeFailsClosed:
    """Caller strings that influence retrieval are locked to inert identifiers."""

    @pytest.mark.parametrize(
        "bad_policy_type",
        ["health insurance", "health-plan", "a" * 33, "路線", "x;drop", 42, ["health"], {"t": "x"}],
        ids=["space", "hyphen", "too-long", "non-ascii", "punctuation", "int", "list", "dict"],
    )
    def test_non_identifier_policy_type_is_rejected(self, bad_policy_type):
        result = InputValidateNode()(
            _make_state("hospitalization coverage", input_context={"policy_type": bad_policy_type})
        )
        _assert_field_rejected(result, "policy_type")

    def test_rejected_value_is_never_echoed(self):
        marker = "zzmarkerzz not an identifier"
        result = InputValidateNode()(_make_state("hospitalization coverage", input_context={"policy_type": marker}))
        assert result["status"] == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")
        assert result["error_log"] == [_POLICY_TYPE_MESSAGE]
        assert "zzmarkerzz" not in " ".join(str(e) for e in result["error_log"])

    def test_blank_policy_type_degrades_to_none(self):
        result = InputValidateNode()(_make_state("hospitalization coverage", input_context={"policy_type": "   "}))
        assert from_json(result["query_filters"])["policy_type"] is None


class TestSizeAndEmptyGuards:
    def test_val_08_oversize_query_is_truncated(self):
        payload = "hospitalization " * 200  # ~3400 chars after collapse
        result = InputValidateNode()(_make_state(payload))
        assert len(result["search_query"]) == 2000
        notes = from_json(result.get("intake_notes"), [])
        assert any("truncated" in n for n in notes)

    def test_val_09_empty_request_yields_note_not_error(self):
        result = InputValidateNode()(_make_state(""))
        assert result["search_query"] == ""
        notes = from_json(result.get("intake_notes"), [])
        assert any("empty request" in n for n in notes)
