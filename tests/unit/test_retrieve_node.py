# INS-C2-001 — Unit Tests: RetrieveNode (inner domain node 2)
#
# Invocation canon: node(state) via BaseNode.__call__ with an ANONYMOUS caller.
# NO carve-out (the `execute(state, config=…)`
# 2-arg exception): FunctionNode.execute() takes NO config parameter anywhere
# in this repo. Every knob RetrieveNode reads (retrieval_top_k,
# retrieval_kb_path) is a bare scalar seeded onto State by
# DomainWorkflowGraph._extra_initial_state() — so these tests seed the SAME
# state keys and invoke uniformly through node(state).
#
# Expected top hits below were verified against the REAL deterministic scorer
# and the real seeded config/kb/insurance_policy_kb.json (not hand-computed) —
# see docs/03_test_spec.md §2.3 for the query -> id table.
#
# Mirrors docs/03_test_spec.md §2.3 (RET-01..RET-08).
# Deterministic — keyword scoring over the seeded KB; no LLM, no network.
# framework.* / src.* imports only.

from framework.schemas.trust_level import TrustLevel

from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import from_json, to_json

_HOSPITALIZATION_QUERY = "hospitalization rider coverage scope for inpatient admission"


def _make_state(query=_HOSPITALIZATION_QUERY, **extra) -> dict:
    state = {
        "search_query": query,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestRetrieveHappyPath:
    def test_ret_01_top_hit_is_hospitalization_rider_entry(self):
        result = RetrieveNode()(_make_state())
        docs = from_json(result["retrieved_documents"])
        assert docs, "expected candidates for the hospitalization-rider query"
        assert docs[0]["id"] == "kb-002"

    def test_ret_02_scores_sorted_descending(self):
        docs = from_json(RetrieveNode()(_make_state())["retrieved_documents"])
        scores = [d["score"] for d in docs]
        assert scores == sorted(scores, reverse=True)
        assert all(s > 0.0 for s in scores)

    def test_ret_03_entry_shape_and_excerpt_cap(self):
        docs = from_json(RetrieveNode()(_make_state())["retrieved_documents"])
        for doc in docs:
            assert set(doc.keys()) == {"id", "title", "policy_type", "source", "score", "excerpt"}
            assert len(doc["excerpt"]) <= 400

    def test_retrieved_documents_is_json_string(self):
        # List-shaped State fields travel as JSON strings.
        result = RetrieveNode()(_make_state())
        assert isinstance(result["retrieved_documents"], str)


class TestRetrieveFilters:
    def test_ret_04_policy_type_filter_restricts_pool_to_health_or_general(self):
        # RetrieveNode's filter keeps an EXACT policy_type match OR "general"
        # (general entries are cross-cutting — waiting period / claims /
        # cooling-off apply regardless of policy_type). life-only entries
        # (kb-006 dread-disease, kb-007 disability-income) must never appear.
        state = _make_state(
            query="coverage rider benefit conditions",
            query_filters=to_json({"policy_type": "health", "top_k": None}),
        )
        docs = from_json(RetrieveNode()(state)["retrieved_documents"])
        assert docs, "health policy_type has seeded entries"
        policy_types = {d["policy_type"] for d in docs}
        assert policy_types <= {"health", "general"}
        assert "health" in policy_types
        ids = {d["id"] for d in docs}
        assert "kb-006" not in ids and "kb-007" not in ids  # life-only, excluded
        assert docs[0]["id"] == "kb-002"

    def test_ret_05_empty_query_yields_no_candidates(self):
        docs = from_json(RetrieveNode()(_make_state(query=""))["retrieved_documents"])
        assert docs == []


class TestRetrieveStateSeededConfig:
    """RET-06/07: retrieval_kb_path is read directly from State (no config
    parameter — SDK v1.0.0rc1 canon); an unreadable path degrades gracefully."""

    def test_ret_06_bad_kb_path_degrades_to_empty_with_note(self):
        state = _make_state(retrieval_kb_path="config/kb/does_not_exist.json")
        result = RetrieveNode()(state)
        assert from_json(result["retrieved_documents"]) == []
        notes = from_json(result.get("intake_notes"), [])
        assert any("not readable" in n for n in notes)

    def test_ret_07_unseeded_kb_path_falls_back_to_module_default(self):
        # No retrieval_kb_path in state at all — falls back to the node's own
        # _DEFAULT_KB_PATH (mirrors config/agent.yaml), so retrieval still works.
        docs = from_json(RetrieveNode()(_make_state())["retrieved_documents"])
        assert docs, "module-default kb_path must resolve the real seeded KB"


class TestRetrieveNotesAccumulation:
    def test_ret_08_notes_append_never_clobber(self):
        state = _make_state(
            retrieval_kb_path="config/kb/does_not_exist.json",
            intake_notes=to_json(["earlier note from input validation"]),
        )
        result = RetrieveNode()(state)
        notes = from_json(result["intake_notes"])
        assert notes[0] == "earlier note from input validation"
        assert len(notes) == 2
