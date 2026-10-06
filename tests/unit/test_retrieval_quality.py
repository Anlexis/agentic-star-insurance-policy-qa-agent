# INS-C2-001 — Unit Tests: retrieval quality over the seeded KB
#
# Golden-query suite: drives the REAL inner retrieval chain
# (InputValidateNode → RetrieveNode → RerankFilterNode) via node(state) /
# __call__ (ANONYMOUS inner nodes) against config/kb/insurance_policy_kb.json
# and pins the expected top hit per domain query. The scorer is deterministic
# (keyword field-weights, stable tie-break), so exact top-1 assertions are
# safe and catch KB / scorer / threshold regressions. Every (query, expected
# top-1) pair below was verified by actually running the real node code
# against the real seeded KB — not hand-computed.
#
# Payloads are lowercase / PII-free: the entry field (validated_input) IS a
# framework PII-scan target, and a [MASKED] substitution before
# InputValidateNode.execute() parses a JSON envelope would corrupt the parse.
#
# Mirrors docs/03_test_spec.md §2.9 (QUAL-01..QUAL-07).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

import json
import pathlib

import pytest

from framework.schemas.trust_level import TrustLevel

from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import from_json

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_KB_IDS = {
    entry["id"]
    for entry in json.loads((_ROOT / "config" / "kb" / "insurance_policy_kb.json").read_text(encoding="utf-8"))
}

_DEFAULT_SCORE_THRESHOLD = 0.25  # mirrors config/agent.yaml retrieval block


def _search(payload: str) -> list[dict]:
    """Run the real inner retrieval chain and return the surviving passages."""
    state = {
        "validated_input": payload,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "quality-session",
        "execution_time": {},
    }
    state.update(InputValidateNode()(state))
    state.update(RetrieveNode()(state))
    state.update(RerankFilterNode()(state))
    return from_json(state["ranked_documents"], [])


# (query, expected top-1 KB entry id) — verified against the deterministic scorer.
_GOLDEN_QUERIES = [
    ("insurer's duty to explain policy terms before contract formation", "kb-001"),
    ("hospitalization rider coverage scope for inpatient admission", "kb-002"),
    ("waiting period before hospitalization coverage becomes effective", "kb-004"),
    ("dread disease rider critical illness lump sum payout", "kb-006"),
    ("disability income rider benefit conditions elimination period", "kb-007"),
    ("premium payment grace period lapse reinstatement", "kb-009"),
]


class TestGoldenQueries:
    @pytest.mark.parametrize(("query", "expected_id"), _GOLDEN_QUERIES)
    def test_qual_01_top_hit_per_golden_query(self, query, expected_id):
        kept = _search(query)
        assert kept, f"no passage cleared the relevance floor for: {query!r}"
        assert kept[0]["id"] == expected_id

    def test_qual_02_all_survivors_clear_the_relevance_floor(self):
        for query, _expected in _GOLDEN_QUERIES:
            for doc in _search(query):
                assert doc["score"] >= _DEFAULT_SCORE_THRESHOLD

    def test_qual_03_survivor_ids_exist_in_the_seeded_kb(self):
        for query, _expected in _GOLDEN_QUERIES:
            for doc in _search(query):
                assert doc["id"] in _KB_IDS


class TestPrecision:
    def test_qual_04_coordination_of_benefits_query_keeps_only_that_entry(self):
        # Off-topic passages score below the floor and are cut — precision, not
        # just recall.
        kept = _search("coordination of benefits with other insurance public healthcare")
        assert [d["id"] for d in kept] == ["kb-010"]

    def test_qual_05_policy_type_filter_restricts_to_health_or_general(self):
        payload = json.dumps({"query": "coverage rider benefit conditions", "policy_type": "health"})
        kept = _search(payload)
        assert kept, "health policy_type carries seeded entries"
        assert {d["policy_type"] for d in kept} <= {"health", "general"}
        assert kept[0]["id"] == "kb-002"


class TestNoCoverage:
    def test_qual_06_out_of_domain_query_yields_no_survivors(self):
        assert _search("quantum telepathy sandwich recipes") == []

    def test_qual_07_no_coverage_produces_the_escalation_answer(self):
        state = {
            "ranked_documents": "[]",
            "search_query": "quantum telepathy sandwich recipes",
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
            "node_history": [],
            "error_log": [],
            "session_id": "quality-session",
            "execution_time": {},
        }
        result = GenerateAnswerNode()(state)
        assert "does not contain sufficient coverage" in result["grounded_answer"]
        assert from_json(result["citations"]) == []
