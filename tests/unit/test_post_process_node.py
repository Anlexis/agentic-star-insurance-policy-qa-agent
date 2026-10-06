# INS-C2-001 — Unit Tests: PostProcessNode (outer post_process slot; output gate)
#
# Invocation canon: node(state) via BaseNode.__call__. PostProcessNode is the
# second outer gate slot and requires VERIFIED_EXTERNAL (like PreProcessNode),
# so its behavioural tests build the state at that level; the ANONYMOUS
# rejection lives in test_trust_gate.py.
#
# Gate layering: the node's own module-level _scan_disallowed_content() scan
# runs INSIDE execute() and replaces a violating answer with the sanitised
# stub (returned dict — no exception). The framework's FunctionNode
# credential scan then sees only the clean stub. Intentional-credential
# tests assert the raw secret never survives into formatted_output OR
# result. On top of the credential scan the gate enforces the output
# contract: every non-empty answer carries the advisory disclaimer.
#
# Mirrors docs/03_test_spec.md §2.7 (POST-01..POST-06).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.output_format_node import _ADVISORY_DISCLAIMER
from src.nodes.post_process_node import PostProcessNode

_CLEAN_REPORT = (
    "# Insurance Policy Q&A Result\n\n"
    "[1] the hospitalization rider pays a daily benefit from the first day of admission.\n"
    "\n---\n\n*" + _ADVISORY_DISCLAIMER + "*\n"
)

# JWT-shaped token built at runtime so no credential-shaped literal ever sits
# in the repository (credential-scan hygiene).
_FAKE_JWT = "eyJ" + "a" * 12 + "." + "b" * 12 + "." + "c" * 12


def _make_state(result_text, **extra) -> dict:
    state = {
        "result": result_text,
        "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestPostProcessClean:
    def test_post_01_clean_output_passes_through(self):
        result = PostProcessNode()(_make_state(_CLEAN_REPORT))
        assert result["status"] == AgentStatus.SUCCESS.value
        # Regression guard: State carries the plain string, never the enum.
        assert isinstance(result["status"], str) and not isinstance(result["status"], AgentStatus)
        assert result["formatted_output"] == _CLEAN_REPORT

    def test_post_02_empty_result_is_non_fatal(self):
        result = PostProcessNode()(_make_state(""))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["formatted_output"] == ""


class TestPostProcessCredentialGate:
    def _assert_blocked(self, result, secret):
        assert result["status"] == AgentStatus.ERROR.value
        assert any("output blocked" in str(e) for e in result["error_log"])
        # The raw secret must not survive into either surfaced field.
        assert secret not in str(result.get("formatted_output", ""))
        assert secret not in str(result.get("result", ""))
        assert "[OUTPUT BLOCKED by the output security gate" in result["formatted_output"]

    def test_post_03_api_key_is_blocked(self):
        secret = "sk-ABCDEF0123456789abcdef"
        result = PostProcessNode()(_make_state(f"# Report\n\n<!-- debug api_key={secret} -->\n"))
        self._assert_blocked(result, secret)

    def test_post_04_credential_assignment_is_blocked(self):
        secret = "password=super_secret_value_123"
        result = PostProcessNode()(_make_state(f"# Report\n\ninternal note: {secret}\n"))
        self._assert_blocked(result, "super_secret_value_123")

    def test_post_05_jwt_is_blocked(self):
        result = PostProcessNode()(_make_state(f"# Report\n\nsession token {_FAKE_JWT}\n"))
        self._assert_blocked(result, _FAKE_JWT)

    def test_post_06_bearer_token_is_blocked(self):
        secret = "Bearer abcdefghijklmnopqrstuvwxyz0123456789"
        result = PostProcessNode()(_make_state(f"# Report\n\nauthorization: {secret}\n"))
        self._assert_blocked(result, secret)


class TestDisclaimerEnforcement:
    """The gate enforces the output contract independently of the renderer:
    a clean answer missing the advisory disclaimer has it appended here, so
    a future rendering change cannot silently drop it."""

    def test_clean_answer_with_disclaimer_is_untouched(self):
        result = PostProcessNode()(_make_state(_CLEAN_REPORT))
        assert result["formatted_output"] == _CLEAN_REPORT
        assert "result" not in result  # unchanged keys are not re-emitted

    def test_missing_disclaimer_is_appended(self):
        bare = "# Insurance Policy Q&A Result\n\n[1] a cited passage.\n"
        result = PostProcessNode()(_make_state(bare))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert _ADVISORY_DISCLAIMER in result["formatted_output"]
        assert _ADVISORY_DISCLAIMER in result["result"]
        assert result["formatted_output"].startswith(bare.rstrip())

    def test_empty_result_is_not_padded_with_a_disclaimer(self):
        result = PostProcessNode()(_make_state(""))
        assert result["formatted_output"] == ""

    def test_blocked_output_is_not_disclaimer_padded(self):
        secret = "sk-ABCDEF0123456789abcdef"
        result = PostProcessNode()(_make_state(f"api_key={secret}"))
        assert result["status"] == AgentStatus.ERROR.value
        assert _ADVISORY_DISCLAIMER not in result["formatted_output"]
