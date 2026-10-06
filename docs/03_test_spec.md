# Test Specification — INS-C2-001

**Template ID:** INS-C2-001
**Template Name:** InsurancePolicyQAAgent
**Category:** Cat 2 (nested RAG)

This document defines the test cases for the implementation (state, nodes,
inner/outer graphs, config, server). The test code lives in `tests/unit/` +
`tests/proof_of_boundary/`; this spec is the contract those tests implement.

## 1. Scope & Invocation Conventions

- Per-node unit tests for the 5 inner domain nodes + the 2 outer gate nodes.
- Manifest (`config/agent.yaml`) and runtime-config (`config/config.yaml`)
  consistency with code, and seeded-KB integrity.
- Runtime-config liveness probe (the declared values must actually drive the
  pipeline — `tests/unit/test_runtime_config.py`).
- Retrieval quality (golden queries over `config/kb/insurance_policy_kb.json`).
- Inner-graph (`DomainWorkflowGraph`) and outer-graph
  (`InsurancePolicyQAAgent`) composition / integration.
- Proof-of-Boundary (PoB): import isolation, State msgpack safety, invoke
  order (PB-6), end-to-end `/invoke` behaviour (auth, caller-data contract,
  fail-closed validation), HITL propagation (PB-7, conditional), server boot.

**Trust-gate invocation canon.** Every per-node test invokes the node via
`node(state)` — through `BaseNode.__call__`, which runs the trust gate →
input gate → `execute()` → output gate — never a bare `node.execute(state)`.
The state builder sets `caller_trust_level` to
`TrustLevel.VERIFIED_EXTERNAL.value` for the two outer gate slots
(PreProcessNode / PostProcessNode — the manifest's declared caller level) and
`TrustLevel.ANONYMOUS.value` for the five inner domain nodes.
`FunctionNode.execute(self, state) -> dict` takes no `config` parameter
anywhere in this repo. Every retrieval knob (`retrieval_top_k`,
`retrieval_score_threshold`, `retrieval_kb_path`) is a bare scalar seeded onto
State by `DomainWorkflowGraph._extra_initial_state()`; tests exercise a knob
by seeding the same State key and invoking uniformly through `node(state)`.

**Framework masking expectations.** The framework input gate masks
`user_input`/`validated_input`/`llm_response` (e-mail, phone/SSN/CC digit
groups, and — high-recall / low-precision — ANY two-or-more consecutive
Title-Case words) to `[MASKED]` before `execute()` runs. This template's own
nodes add no domain-specific identifier screen beyond the framework default
(docs/02_design.md input-screen note). Positive-path payloads are therefore
lowercase, PII-free policy phrasing; intentional-PII tests assert the raw
identifier is gone and `[MASKED]` is present — including the false-positive
case where a regulatory phrase like "Insurance Business Act" matches the
Title-Case heuristic exactly like a person's name would.
Domain fields (`grounded_answer`, `formatted_answer`, `retrieved_documents`,
`search_query`, …) are not masking targets.

**Audit muting.** `shared.*` is never sys.modules-stubbed (the framework
imports `shared.security` at load time). The domain audit emitter is muted via
an autouse fixture patching `src.nodes.<mod>.emit_trace_event`; the audit
assertion test re-patches the same attribute with a spy and asserts on
`call.args[1]` (the event payload).

## 2. Unit Test Cases

### 2.1 PreProcessNode (outer pre_process slot; input gate) — `test_pre_process_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| PRE-01 | Valid query | lowercase policy question | `status=SUCCESS`, `validated_input` set, `enriched_context` carries channel/source |
| PRE-02 | Empty input | `""` / whitespace | `status=ERROR`, `error_log` non-empty, no `validated_input` |
| PRE-03 | Missing / non-string input | `user_input` absent; dict payload | `status=ERROR` (dict payload: `execute()` raises `AttributeError` internally; `BaseNode.__call__`'s catch-all converts it to `status=ERROR` — never an unhandled exception) |
| PRE-04 | Framework identifier screen | e-mail / 4-4-4 digit run / a Title-Case regulatory phrase ("Insurance Business Act") | raw identifier absent from `validated_input`; `[MASKED]` present in all three cases (this node adds no node-level screen of its own) |
| PRE-05 | Oversize input | > 20,000 chars | `status=ERROR` naming the bound; content never echoed; exactly 20,000 chars accepted |
| PRE-06 | Channel label bound | 200-char `input_context.channel` | capped at 64 chars; non-dict `input_context` tolerated |
| PRE-08 | Audit | valid query | `pre_process_complete` emitted; payload (`call.args[1]`) carries `input_chars` |

### 2.2 InputValidateNode (inner node 1) — `test_input_validate_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| VAL-01 | Plain text | free-text query | whole string becomes `search_query`; filters `{policy_type: None, top_k: None}` |
| VAL-02 | Whitespace | ragged spacing/newlines | collapsed to single spaces |
| VAL-03 | JSON envelope | `{"query","policy_type","top_k"}` | all three parsed; `question` alias accepted; `policy_type` canonicalised (strip/lower) |
| VAL-04 | Malformed JSON | `{`-prefixed non-JSON | treated as plain-text query + parse note |
| VAL-05 | input_context params | `{"policy_type","top_k"}` via `input_context` | honoured; input_context wins over envelope values; non-dict input_context ignored |
| VAL-06 | top_k fail-closed matrix | 0 / -5 / 21 / 99 / `"many"` / `"NaN"` / `"Infinity"` / `"-Infinity"` / `float("nan")` / `float("inf")` / 3.7 / 5.0 / bools / list / dict / 1e9 | `status=ERROR`; `error_log` is EXACTLY the static field-naming message (byte-equality proves no value echo); boundary values 1 and 20 accepted; absent → `None` |
| VAL-07 | policy_type fail-closed matrix | free text / hyphen / >32 chars / non-ASCII / punctuation / non-strings | `status=ERROR`, static field-naming message, value never echoed; blank canonicalises to `None` |
| VAL-08 | Oversize query | > 2000 chars | truncated to 2000 + note |
| VAL-09 | Empty request | `""` | `search_query=""` + "empty request" note (non-fatal) |
| — | JSON-string convention | any | `query_filters` is a JSON string, never a bare dict |

### 2.3 RetrieveNode (inner node 2) — `test_retrieve_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| RET-01 | Happy path | hospitalization-rider query | top-1 candidate is `kb-002` |
| RET-02 | Ordering | hospitalization-rider query | scores strictly sorted desc; all > 0 |
| RET-03 | Entry shape | any hit | keys `{id,title,policy_type,source,score,excerpt}`; excerpt ≤ 400 chars |
| RET-04 | policy_type filter | `query_filters.policy_type="health"` | pool restricted to `{health, general}` (general is a cross-cutting fallback — waiting period / claims / cooling-off apply regardless of policy type); `life`-only entries excluded; top-1 `kb-002` |
| RET-05 | Empty query | `""` | no candidates |
| RET-06 | State-seeded `retrieval_kb_path` override | unreadable path | `[]` + "not readable" note |
| RET-07 | Unseeded state fallback | no `retrieval_kb_path` in state | resolves the real seeded KB via the node's own module default |
| RET-08 | Notes accumulation | prior `intake_notes` | appended, never clobbered |

### 2.4 RerankFilterNode (inner node 3) — `test_rerank_filter_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| RRF-01 | Relevance floor | scores 0.9 / 0.1 | 0.1 dropped (default 0.25 floor) |
| RRF-02 | State-seeded `retrieval_score_threshold` override | `0.5` | 0.3 dropped |
| RRF-03 | State-seeded `retrieval_top_k` override | `1` | one survivor, highest score |
| RRF-04 | policy_type boost | matching policy_type | +0.1, re-ranked ahead |
| RRF-05 | Boost cap | 0.95 + boost | capped at 1.0 |
| RRF-06 | Caller top_k | stricter (1) wins; looser (10) does not widen the state-seeded baseline | enforced |
| RRF-07 | Garbage entries | non-dict / uncoercible score | skipped / coerced to 0.0 and dropped |
| RRF-08 | Tie-break | equal scores | deterministic id-ascending order |

### 2.5 GenerateAnswerNode (inner node 4) — `test_generate_answer_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| GEN-01 | Citation markers | 2 ranked passages | `[1]`/`[2]` markers with titles |
| GEN-02 | No caller echo | query containing marker text | marker ABSENT from the answer; fixed lead sentence used (caller free text in the answer surface would be output injection) |
| GEN-03 | Citations list | ranked passages | refs 1..n mirror ranked order; id/title/source carried |
| GEN-04 | Groundedness | single passage | answer body traces to ranked passages only |
| GEN-05 | No coverage | empty/missing `ranked_documents` | escalation answer; `citations=[]` |
| GEN-06 | Error pass-through | prior `status=error` | returns `{}` — the fail-closed error survives the linear topology |

### 2.6 OutputFormatNode (inner node 5, terminal) — `test_output_format_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| FMT-01 | Full compose | body + citations | header + body + `## Sources` rows + advisory disclaimer; `status=SUCCESS` |
| FMT-02 | Blank source | citation without source | no `()` suffix |
| FMT-03 | Disclaimer | every input | disclaimer rides with every answer |
| FMT-04 | No citations | empty list | explicit "- none (…)" sources line |
| FMT-05 | Missing body | no `grounded_answer` | fallback text; `status=SUCCESS` |

### 2.7 PostProcessNode (outer post_process slot; output gate) — `test_post_process_node.py`

| ID | Case | Input (`result`) | Expected |
|----|------|------------------|----------|
| POST-01 | Clean output | normal policy answer (with disclaimer) | `formatted_output=result` untouched, `status=SUCCESS` |
| POST-02 | Empty result | `""` | forwarded as-is, `status=SUCCESS` (non-fatal), no disclaimer padding |
| POST-03..06 | Credential leak | `sk-` API key / `password=` assignment / JWT (built at runtime) / Bearer token | `formatted_output` + `result` replaced with the sanitised stub, `status=ERROR`, raw secret absent from both; no disclaimer padding on the blocked stub |
| POST-07 | Disclaimer enforcement | clean answer WITHOUT the disclaimer | disclaimer appended to `formatted_output` AND `result`; `output_disclaimer_enforced` audit path |

### 2.8 Manifest / runtime config consistency — `test_config_manifest.py`

| ID | Case | Expected |
|----|------|----------|
| CFG-01 | Template id | manifest root `id` = `INS-C2-001`; `namespace` = `ins`; `enabled` true (flat manifest — no `agent:` nesting) |
| CFG-02 | Class contract | manifest `class` = `src.graph.graph.InsurancePolicyQAAgent`; class name and `name` match the code |
| CFG-03 | Classification | Cat 2 / INS / RAGAgent; `generation_mode` = `deterministic`; `requires.secrets` = `[]`, `requires.extras` = `[]` (code-derived — no client construction, no secret reads in `src/`) |
| CFG-04 | Trust level | manifest `VERIFIED_EXTERNAL` == PreProcessNode & PostProcessNode `required_trust_level` |
| CFG-05 | Runtime values | `config/config.yaml` `max_retry` int `0 ≤ v < 10` (framework ceiling); `timeout_s` positive int; hitl not enabled (PB-7 waiver contract) |
| CFG-06 | Retrieval block | `top_k`/`score_threshold`/`kb_path` mirror the node modules' OWN separate default constants (`retrieve_node._DEFAULT_TOP_K`/`_DEFAULT_KB_PATH`, `rerank_filter_node._DEFAULT_TOP_K`/`_DEFAULT_SCORE_THRESHOLD` — no combined dict) |
| CFG-07 | `_parent_config()` | forwards `config/config.yaml` retrieval + llm blocks + `max_retry` + `timeout_s`→`timeout_seconds`; never an empty retrieval block |
| — | KB integrity | JSON list ≥ 5 entries; unique ids; required keys per entry (`id,title,policy_type,source,tags,content`) |

### 2.9 Runtime-config liveness — `test_runtime_config.py`

| ID | Case | Expected |
|----|------|----------|
| RTC-01 | File read | a probe `config.yaml` (top_k=1, max_retry=5, timeout_s=45) is read by `_parent_config()`; `timeout_s` mapped to `timeout_seconds` |
| RTC-02 | Inner seed | the probe values reach `_extra_initial_state()` scalars |
| RTC-03 | End-to-end probe | with the probe config, a full outer `invoke()` returns exactly ONE citation for a query that yields several under the shipped default — the file's declared value, not a fallback, drives retrieval |

### 2.10 Retrieval quality (golden queries) — `test_retrieval_quality.py`

| ID | Case | Expected |
|----|------|----------|
| QUAL-01 | 6 golden domain queries | expected KB entry is top-1 (kb-001/002/004/006/007/009) |
| QUAL-02 | Relevance floor | every survivor ≥ 0.25 |
| QUAL-03 | Citation integrity | every survivor id exists in the seeded KB |
| QUAL-04 | Precision | coordination-of-benefits query keeps ONLY `kb-010` |
| QUAL-05 | policy_type filter | `health` filter → survivors restricted to `{health, general}`, top-1 `kb-002` |
| QUAL-06 | No coverage | out-of-domain query → zero survivors |
| QUAL-07 | Escalation answer | no-coverage → explicit escalation text, no citations |

### 2.11 Trust matrix — `test_trust_gate.py`

| ID | Case | Expected |
|----|------|----------|
| TRUST-01 | ANONYMOUS on inner node | passes (inner domain nodes declare ANONYMOUS) |
| TRUST-02 | ANONYMOUS on outer gate slots | denial dict (never raises); `trust gate denied` in `error_log`; execute-only output keys ABSENT |
| TRUST-03 | VERIFIED_EXTERNAL on outer gate slots | passes; node output produced |
| TRUST-04 | Declared matrix | outer gates VERIFIED_EXTERNAL; all five inner nodes ANONYMOUS |

### 2.12 Framework compliance — `test_framework_compliance_tc06_tc07.py`

| ID | Case | Expected |
|----|------|----------|
| TC-06 / TC-07 | Overriding the default input / output gate on a `FunctionNode` subclass | `TypeError` at class definition (gates are non-bypassable) |

## 3. Integration / Composition

### 3.1 Inner graph — `test_domain_workflow_graph.py`

| ID | Case | Expected |
|----|------|----------|
| INT-01 | Composition | inherits `BaseGraph`; registers exactly the 5 domain nodes; no initialize/finalize |
| INT-02 | Config forwarding | `_extra_initial_state()` republishes THREE bare scalar State keys (`retrieval_top_k`, `retrieval_score_threshold`, `retrieval_kb_path`) — never a single JSON-string field — plus the bridged `input_context` |
| INT-02b | Config validation | `_validate_config()` accepts the shipped shape and absent keys; rejects broken declarations (non-int/bool `max_retry`/`timeout_seconds`, out-of-range `top_k`/`score_threshold`, empty `kb_path`) with `ValueError` at compile time |
| INT-03 | Output shape | `get_output()` emits `formatted_answer`/`citations`/`status`/`intake_notes`/`trace_id`/`correlation_id`/`node_history` (the merge contract); `route()` → END on error |
| INT-04 | Inner e2e | full inner `invoke()` → SUCCESS; formatted answer + disclaimer + `kb-004` citation (waiting-period query); inner `node_history` = the 5 domain nodes in linear order |

### 3.2 Outer graph + e2e — `test_graph_composition.py`

| ID | Case | Expected |
|----|------|----------|
| INT-05 | Outer composition | inherits `AgentBaseGraph` (direct framework inheritance); `Graph` alias; `add_edges()` NOT overridden |
| INT-06 | Backbone slots | compile() fills all 5; pre/main/post are PreProcessNode / PolicyQAGraphNode / PostProcessNode |
| INT-07 | `get_subgraph()` | returns `DomainWorkflowGraph` carrying the forwarded retrieval config + `timeout_seconds` |
| INT-08 | `extract_input()` | prefers `validated_input`, falls back to `user_input` |
| INT-09 | `merge_output()` | inner `formatted_answer` → outer `policy_answer` AND `result`; `citations`/`status` mapped; changed keys only |
| INT-10 | Unreadable runtime config | `_parent_config()` degrades to `{}` (never raises); inner seeding falls back to module defaults |
| INT-11 | e2e happy path | VERIFIED_EXTERNAL invoke → SUCCESS; `output` = gated formatted answer; PostProcessNode traversed |
| INT-12 | e2e trust denial | ANONYMOUS invoke → ERROR; empty `output`; PostProcessNode NOT traversed |
| — | JSON-string helpers | `to_json`/`from_json` round-trip; None/malformed handling |

## Marketplace Entry Point — `tests/unit/test_cli_entry_point.py`

| ID | Case | Expected |
|----|------|----------|
| CLI-01 | `cli.py` imports | module loads; `run_agent_marketplace`, `load_agent_config` and `InsurancePolicyQAAgent` are present |
| CLI-02 | override seam ships empty | `extend_config == {}`; a stray value would silently outrank `config/config.yaml` on the Marketplace path only |
| CLI-03 | the runner receives what the image's CMD would send | executing `cli.py` as `__main__` with the runner replaced captures the call: the graph class, `agent_name`, `namespace`, and every value declared in `config/config.yaml`. Loading the module alone never runs that block, so a wrong class or a dropped config there would otherwise ship unnoticed |

`cli.py` is imported by no other module, so nothing else in the suite would
notice if its import path, graph class or config assembly broke; the image
would build and fail only when the Pod starts. Skipped where the platform
events package is absent.

## 4. Proof-of-Boundary

| ID | Case | Expected |
|----|------|----------|
| PB-IMPORT | `test_import_isolation.py` | no `agenticstar` / platform-internal import anywhere under `src/` |
| PB-STATE | `test_state_safety.py` | `State` has no credential-named fields and no `BaseModel` / `InvocationContext` annotations |
| PB-6 | `test_pb_invoke_order.py` | full `Graph().invoke()` with `InvocationContext(caller_trust_level=VERIFIED_EXTERNAL)` (never `for_internal()`) over the payload byte-equal to `deploy/invoke_payload.json`'s `input` → SUCCESS with outer `node_history` exactly `[InitializeNode, PreProcessNode, PolicyQAGraphNode, PostProcessNode, FinalizeNode]` |
| PB-E2E | `test_pb_invoke_e2e.py` | through the REAL ASGI `/invoke` (Bearer auth): grounded cited answer with disclaimer from a plain question; `input_context.top_k=1` caps citations (context-bridge proof); `policy_type` filter works; invalid `top_k` (incl. the non-finite matrix `"NaN"`/`"Infinity"`/`"-Infinity"`/raw NaN/Inf/float/bool/1e9) → `status=error` with no output (fail closed, never clamp); free-text `policy_type` rejected and never echoed anywhere in the response; caller question text never echoed into the answer; empty input rejected; missing/wrong Bearer → 401; oversized `input_context` → 413 |
| PB-5 | `test_state_safety.py` | **Auto-waived — checkpointing disabled** (`config/config.yaml` enables neither `memory_enabled` nor `hitl.enabled`); the conditional gate and the non-lossy traversal helper ship with the stub |
| PB-7 | `test_pb7_hitl_interrupt_propagation.py` | **Auto-waived — non-HITL** (`PolicyQAGraphNode.propagate_hitl = False`; no graph class declares `propagate_hitl=True`); skip stub retained |
| PB-BOOT | `test_server_boot.py` | `import src.api.server` does not raise; module-level agent is this template's class, compiled; fresh ctor→`compile()` fills the 5 backbone slots; `/health` reports the agent |

> **Gate checklist:** PB-IMPORT, PB-STATE, PB-6, PB-E2E and PB-BOOT are
> mandatory. PB-7 applies only to HITL-enabled templates — this template is
> non-HITL, so PB-7 is **Auto-waived — non-HITL** and its skip must not block
> the gate.

## 5. Test Execution Summary

> **Pending re-run.** The figures below predate the Marketplace entry point work.
> The entry-point test and the PB-5 / PB-6 additions were added after this run and
> have not been executed locally — the framework wheel is not installed in the
> authoring environment. **They are not yet verified anywhere**; this summary is
> updated once a pipeline run has executed them.

- Runner: real SDK (`agenticstar-agentcore==1.0.1`), `python -m pytest tests/`
- Total tests: 215
- Pass: 214 / Fail: 0 / Skip: 1 (PB-7 conditional stub — auto-waived, non-HITL)
- Determinism: no LLM, no network; retrieval + answer assembly are rule-based
