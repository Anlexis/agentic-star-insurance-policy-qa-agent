# Template Design Specification — INS-C2-001

**Template ID:** INS-C2-001
**Template Name:** InsurancePolicyQAAgent
**Category:** Cat 2 (multi-step domain workflow — RAG pattern)
**Industry:** INS

## Position in AgentCore Architecture

| Layer | Binding |
|-------|---------|
| L1 Base (framework base class) | `AgentBaseGraph` — direct framework inheritance |
| Inner graph base | `BaseGraph` (framework) — `DomainWorkflowGraph` |
| Agent class | `InsurancePolicyQAAgent` (alias `Graph`) |

- **Pattern:** Cat 2 two-layer nested architecture (outer fixed 5-node backbone +
  `GraphNode` in the `main` slot wrapping an inner `BaseGraph` domain workflow)
- **Three-Layer Separation:**
  - State: flat `TypedDict` composition (no Pydantic — msgpack incompatible);
    structured fields stored as JSON strings via `to_json()` / `from_json()`;
    plain scalar config knobs stored as bare primitives (no JSON encoding needed)
  - Node: framework inheritance via `FunctionNode` (override
    `execute(self, state) -> dict` only — no `config` parameter)
  - Graph: composition (`register_nodes()` for node substitution); outer
    `add_edges()` is NOT overridden

## Purpose

Insurance policy Q&A agent: policyholders, agents, CSRs, and claims
adjusters ask natural-language questions about policy terms — coverage
scope, exclusions, claim conditions, and rider terms — over a seeded
insurance-policy knowledge base, with an optional policy-type hint (e.g.
health / life / general). The agent retrieves the relevant policy clauses,
reranks/filters them, and generates a plain-language, cited answer with a
mandatory advisory disclaimer. Regulatory driver: Insurance Business Act
(保険業法) Article 294 requires insurers to explain policy terms clearly
before contract formation — grounded, cited RAG Q&A directly supports this
statutory duty. This version is fully deterministic (keyword retrieval +
rule-based grounded answer assembly; no live LLM call — see the
Implementation Note below).

## Architecture Overview

### Outer backbone (AgentBaseGraph)

```
START → initialize → pre_process → main → {route} → post_process → finalize → END
                                     ↓ (retry, max 3)
                                   pre_process
```

| Slot | Class | Responsibility | required_trust_level |
|------|-------|----------------|----------------------|
| initialize | InitializeNode (framework default) | session_id, trust_level, schema_version | — (framework) |
| pre_process | `PreProcessNode` | input gate — reject empty / oversized input → `validated_input`; cap the caller channel label | `TrustLevel.VERIFIED_EXTERNAL` |
| main | `PolicyQAGraphNode` (`GraphNode`) | delegates to inner `DomainWorkflowGraph`; bridges `input_context`; maps inner `formatted_answer` → outer `result` | — (GraphNode delegation) |
| post_process | `PostProcessNode` | output gate — module-level `_scan_disallowed_content()` scans `result` for credentials → ERROR + sanitised stub; enforces the advisory-disclaimer output contract | `TrustLevel.VERIFIED_EXTERNAL` |
| finalize | FinalizeNode (framework default) | response_metadata, total_time_ms | — (framework) |

### Inner graph (DomainWorkflowGraph — BaseGraph, linear)

```
START → input_validate → retrieve → rerank_filter → generate_answer → output_format → END
```

All five inner domain nodes declare `required_trust_level = TrustLevel.ANONYMOUS`
(the external trust gate lives on the outer backbone gate node; a stricter inner
level would deny a real VERIFIED_EXTERNAL invoke at runtime). The linear inner
topology has no conditional edges, so every node after `input_validate` starts
with an error pass-through guard: when a prior node failed closed
(`status=error`), the node returns `{}` untouched so the error survives to the
inner terminal state instead of being overwritten downstream.

| Node | Responsibility | required_trust_level | Input State | Output State |
|------|----------------|----------------------|-------------|--------------|
| `InputValidateNode` | Parse the question (plain text or legacy JSON envelope); merge structured params from `input_context` (which wins); normalise whitespace; cap length; validate every caller field FAIL-CLOSED (`top_k` strict int 1–20, `policy_type` inert identifier `[a-z0-9_]{1,32}`) | `TrustLevel.ANONYMOUS` | `validated_input` \| `user_input`, `input_context` | `search_query`, `query_filters`, `intake_notes` (or `status=error` + `error_log`) |
| `RetrieveNode` | Deterministic keyword retrieval over the seeded KB (`config/kb/insurance_policy_kb.json`): tokenise query, score title/tags/content overlap, apply policy_type filter | `TrustLevel.ANONYMOUS` | `search_query`, `query_filters`, `retrieval_top_k`, `retrieval_kb_path` | `retrieved_documents`, `intake_notes` |
| `RerankFilterNode` | Rerank candidates (policy_type-match boost), drop entries below `score_threshold`, cap at `top_k` (a stricter validated caller override wins) | `TrustLevel.ANONYMOUS` | `retrieved_documents`, `query_filters`, `retrieval_top_k`, `retrieval_score_threshold` | `ranked_documents` |
| `GenerateAnswerNode` | Rule-based grounded answer assembly from the ranked KB passages only, with numbered citation markers. The lead sentence is FIXED — caller text is never echoed into the answer surface (caller free text there would be caller-controlled output injection) | `TrustLevel.ANONYMOUS` | `ranked_documents` | `grounded_answer`, `citations` |
| `OutputFormatNode` | Compose the final answer: body + Sources list + the standing INS advisory disclaimer (the disclaimer is composed here AND independently enforced by post_process) | `TrustLevel.ANONYMOUS` | `grounded_answer`, `citations` | `formatted_answer`, `status` |

### Caller-data contract

Structured invocation parameters travel as `input_context` (a first-class
`BaseGraph.invoke()` parameter, exposed by `src/api/server.py` as the
`input_context` request field, size-capped at the adapter):

```json
{"input": "What is the waiting period before hospitalization coverage starts?",
 "input_context": {"policy_type": "health", "top_k": 3}}
```

- `policy_type` — restricts retrieval to one policy type; canonicalised
  (strip/lower) then locked to the inert identifier pattern `[a-z0-9_]{1,32}`.
  Anything else is rejected with `status=error` naming the field — the value
  is never echoed.
- `top_k` — per-request passage cap; must be a real integer (bools rejected)
  in 1–20. Floats — including NaN/±Infinity in both string and raw-JSON
  form — strings, and out-of-range magnitudes are rejected fail-closed: a
  non-finite number silently defaulting the tuning would be a fail-open on
  the exact parameter the caller tried to control.
- Absent parameters degrade to the baseline: unfiltered retrieval at the
  configured `top_k`.

**Completion is not the same as answering.** A run that ends with
`AgentStatus.SUCCESS` reports that the request was handled safely to a defined
end, not that an answer was produced. A value the caller can correct (an
out-of-contract parameter, an empty or over-long request) ends this way so the
caller receives the reason and can send a corrected request on the same
conversation; terminating instead would end the calling surface's turn and
surface only an exception type, leaving the reason reachable solely from the
audit trail. The reason travels as `error_code` in State, every later domain
node passes through without doing work once it is set, the structured output
fields are withheld, and `PostProcessNode` renders the reason as a static
caller-facing sentence.

Two classes keep terminating, and must not be folded into the above: content
the agent refuses outright (an instruction-override payload — re-sending a
reworded variant is not a correction), and a breach of a contract the caller
cannot influence.

A legacy JSON envelope inside `input`
(`{"question": ..., "policy_type": ..., "top_k": N}`) is still parsed for
backwards compatibility and goes through the SAME validators; `input_context`
values win when both are present.

**input_context bridge (SDK 1.0.1).** The framework `GraphNode.execute()`
invokes the inner graph as `subgraph.invoke(user_input, session_id=..., ctx=...)`
WITHOUT forwarding the outer state's `input_context`. The sanctioned subclass
hooks bridge it (`src/graph/context_bridge.py`, a per-task `ContextVar`):
`PolicyQAGraphNode.extract_input()` stashes `state["input_context"]` immediately
before the inner invoke; `DomainWorkflowGraph._extra_initial_state()` reads it
back while building the inner initial state.

### Data Flow

```
user_input (+ input_context)
  → PreProcessNode (input gate)                    → validated_input
  → PolicyQAGraphNode.extract_input                → stashes input_context; inner invoke(validated_input)
        → input_validate                          → search_query / query_filters (fail-closed)
        → retrieve                                → retrieved_documents
        → rerank_filter                           → ranked_documents
        → generate_answer                         → grounded_answer / citations
        → output_format                           → formatted_answer (+ advisory disclaimer)
     get_output() → {formatted_answer, citations, status, ...}
  → PolicyQAGraphNode.merge_output                 → result = formatted_answer, policy_answer
  → PostProcessNode (output gate)                  → formatted_output (gated)
```

### Runtime config forwarding (`_parent_config`)

Runtime parameters live in `config/config.yaml` (separate from the static
AgentRegistry manifest `config/agent.yaml`, which is read at ROOT level — no
`agent:` nesting). `PolicyQAGraphNode._parent_config()` receives
the runtime config from the outer graph (threaded in at `register_nodes()` time
from `self.config`, which the entry point loaded) and forwards everything under
`config["configurable"]`
(`timeout_s` is mapped to `timeout_seconds`, the key the inner graph
validates):

```
{"configurable": {"max_retry": 3, "timeout_seconds": 30,
                  "retrieval": {top_k, score_threshold, kb_path, hybrid_search},
                  "llm": {...}}}
```

`get_subgraph()` passes this into `DomainWorkflowGraph(config=...)`;
`_validate_config()` rejects broken declarations at compile time, and the
inner graph republishes the `retrieval` block into the inner initial state via
`_extra_initial_state()` — **as plain scalar State fields**
(`retrieval_top_k`, `retrieval_score_threshold`, `retrieval_kb_path`), so the
declared values are live at runtime. `FunctionNode.execute()` takes NO
`config` parameter — every runtime knob a node needs must arrive through
State, seeded once at graph-init time by `_extra_initial_state()`, never
passed per-call. `RetrieveNode` and `RerankFilterNode` read
`retrieval_top_k` / `retrieval_score_threshold` / `retrieval_kb_path`
directly from State, falling back to their own module-level defaults
(mirroring the shipped config values) when those keys are unseeded — e.g.
a unit test that constructs a node directly without going through
`DomainWorkflowGraph`'s init. An unreadable `config/config.yaml` degrades to
an empty forwarded config (never raises); the test suite probes end-to-end
that a declared `retrieval.top_k` actually drives retrieval
(`tests/unit/test_runtime_config.py`).

### State Definition

| Field | Type | Purpose | Layer |
|-------|------|---------|-------|
| `validated_input` | `Optional[str]` | validated question payload | outer |
| `policy_answer` | `Optional[str]` | final answer, mapped from inner `formatted_answer` | outer |
| `search_query` | `Optional[str]` | normalised policy question | inner |
| `query_filters` | `Optional[str]` (JSON) | validated structured params (`policy_type`, `top_k`) | inner |
| `retrieval_top_k` | `Optional[int]` | forwarded `retrieval.top_k` (scalar, state-seeded) | inner |
| `retrieval_score_threshold` | `Optional[float]` | forwarded `retrieval.score_threshold` (scalar, state-seeded) | inner |
| `retrieval_kb_path` | `Optional[str]` | forwarded `retrieval.kb_path` (scalar, state-seeded) | inner |
| `retrieved_documents` | `Optional[str]` (JSON) | scored KB candidates | inner |
| `ranked_documents` | `Optional[str]` (JSON) | reranked + threshold-filtered passages | inner |
| `grounded_answer` | `Optional[str]` | rule-assembled grounded answer body | inner |
| `citations` | `Optional[str]` (JSON) | `[{ref, id, title, source}]` | inner |
| `formatted_answer` | `Optional[str]` | final answer + sources + advisory disclaimer | inner |
| `intake_notes` | `Optional[str]` (JSON) | validation / parse notes (no PII) | inner |
| `trace_id` / `correlation_id` | `Optional[str]` | framework-managed tracing | both |

**State Constraints (mandatory):**
- Flat `TypedDict` only (primitives + JSON-serializable types).
- Structured fields (dict / list[dict]) stored as JSON STRINGS via `to_json()` /
  `from_json()` — used consistently by every producer AND consumer (msgpack
  checkpoint safety). Plain scalar config knobs (`retrieval_top_k` etc.) are
  bare primitives — the JSON-string convention applies to dict/list
  containers only.
- Domain fields are `Optional[...]` and unset before any node writes.
- `formatted_output` is NOT re-declared (backbone field stays framework-owned).
- No JWT, API keys, credentials in State (checkpoint DB leakage). No raw
  policyholder account / contract numbers either.
- `InvocationContext` via `config["configurable"]` only (not in State).
- No Pydantic models, dataclass, arbitrary Python objects (msgpack incompatible).

## Framework Utilization

### Shared Components Used
- [x] InvocationContext (correlation_id, session_id, permissions, credential handle)
- [ ] Input-gate extension hook `_extra_security_gate_input()` — not added; the
      default framework PII scan on `user_input` / `validated_input` is
      sufficient (this template does not collect direct policyholder
      identifiers)
- [ ] Output-gate extension hook `_extra_security_gate_output()` — not added on
      any `FunctionNode` (the framework auto-wraps such hooks); the output
      gate is a module-level `_scan_disallowed_content()` scan called directly
      from `PostProcessNode.execute()`
- [x] `emit_trace_event()` — one domain-specific audit event inside every
      node's `execute()` (see Security Gates below for the full event list)

> **Gate behaviour by node type (framework contract):**
> - `FunctionNode` subclass → the framework's `@final` input/output gates always
>   run automatically; extend via `_extra_security_gate_input()` /
>   `_extra_security_gate_output()` only
> - `GraphNode` / `RemoteAgentNode` → deliberate no-op (upstream or remote node's gate already applied)
> - Custom `BaseNode` subclass → must implement `_security_gate_input()` and
>   `_security_gate_output()` directly (`@abstractmethod` — omission raises `TypeError` at instantiation)

## Security Gates

- **Trust gate / input validation:** every node declares
  `required_trust_level` (see tables above); `PreProcessNode`
  (VERIFIED_EXTERNAL) rejects empty, non-string, or oversized (> 20,000 chars)
  `user_input` before the inner workflow runs. The standalone server elevates
  authenticated Bearer callers to VERIFIED_EXTERNAL (`INVOKE_AUTH_TOKEN`).
- **Caller-parameter validation (fail closed):** `InputValidateNode` validates
  every caller-controlled parameter against explicit bounds (see Caller-data
  contract above); rejected values are never echoed into error logs (the
  error names the field only).
- **Input screen:** the framework `FunctionNode` default PII scan masks
  `user_input` / `validated_input` / `llm_response` at every node boundary.
  This template does not collect direct policyholder identifiers (account
  / contract numbers) as part of its Q&A flow, so no domain-specific
  identifier-stripping hook is added beyond the framework default.
- **Output gate:** `PostProcessNode` calls the module-level
  `_scan_disallowed_content()` scan from `execute()` — API keys / JWT / Bearer
  tokens / credential assignments in the final `result` replace the output with
  a sanitised stub and return `AgentStatus.ERROR`. On top of the credential
  scan the gate enforces the output contract: every non-empty answer carries
  the advisory disclaimer (appended with an audit event if a future rendering
  change drops it). No `_extra_security_gate_input` /
  `_extra_security_gate_output` instance methods are defined on any node.
- **Output-surface inertness:** the rendered answer derives only from KB
  passages and fixed template strings; the caller's question text and free-form
  parameters are never rendered into the answer.
- **Audit logging:** every node's `execute()` emits exactly ONE
  domain-specific `emit_trace_event("<node>_complete", {small non-PII payload},
  state)` on its success path (plus `output_disclaimer_enforced` when the
  output gate repairs a missing disclaimer). Nodes do NOT emit `node_start` /
  `node_complete` / `node_error` — `BaseNode.__call__()` emits those. Domain
  event names (also listed in the operation guide's monitoring table):
  - `pre_process_complete`
  - `input_validate_complete`
  - `retrieve_complete`
  - `rerank_filter_complete`
  - `generate_answer_complete`
  - `output_format_complete`
  - `post_process_complete`
  - `output_disclaimer_enforced` (only when the gate had to repair)

## Advisory Disclaimer

Every answer carries the standing INS advisory line (informational only,
does not amend or override the actual policy contract and rider
certificate, confirm binding interpretation with the insurer or a licensed
agent). It is appended by `OutputFormatNode` as part of the domain output
contract, and `PostProcessNode` independently enforces its presence on
whatever string reaches the gate.

## Implementation Note — LLM synthesis

This version of the template is **deterministic end-to-end**: retrieval is
keyword scoring over the seeded KB and `GenerateAnswerNode` assembles the
grounded answer rule-based from the ranked passages (fixed lead sentence +
cited passage excerpts). There is NO live LLM call and no LLM client
dependency — the `llm` block in `config/config.yaml` is forwarded through
`_parent_config()` for forward-compatibility but is not consumed by any
node, and no `system_prompt` is read at runtime. The LLM synthesis upgrade
seam is documented in `config/prompts/answer_synthesis_prompt.md`: an
upgraded `GenerateAnswerNode` swaps the rule-based assembly for an LLM call
that synthesises over the same `ranked_documents` input and emits the same
`grounded_answer` / `citations` state contract, so no other node changes.

## Entry Points

The agent is reachable through three entry points, all of which build the graph
from the same `config/config.yaml`:

| Entry point | Construction | Notes |
|---|---|---|
| Platform registry | `Graph(config=...)` by the registry | Reads `config/config.yaml` itself |
| Standalone HTTP (`src/api/server.py`) | Loads `config/config.yaml`, passes `Graph(config=...)` | Caller-auth boundary; see Security Design |
| Marketplace (`cli.py`) | `run_agent_marketplace(...)` is handed the graph class and the resolved config | The runner constructs the graph itself, so `cli.py` resolves `config/config.yaml` with `load_agent_config()` and passes it in; `extend_config` is the seam for deployment-specific overrides |

`cli.py` sits at the repository root because the deployment image starts it as
`CMD ["python", "cli.py"]`. It adds no business logic: graph construction,
lifecycle, secret provisioning and the invocation loop belong to
`run_agent_marketplace()`.

## Caller-Facing Events

Nodes report progress and rejection reasons to the caller as non-terminal
events, so a caller watching a run sees the pipeline advance instead of a
silent wait, and learns what to change when a request is refused.

- **Progress** — each node reports its phase at the top of `execute()`.
- **Rejection reason** — a node that returns `status: error` sends the reason
  first. It has to happen there: once the run carries an error status the
  framework skips `execute()` on every later node, so no downstream node could
  send it. Wording separates what the caller can fix (missing question,
  oversized request, malformed value) from what they cannot (retrieval or
  output failures), so a caller is not invited into a pointless retry.

Both are best-effort: the emitter is resolved lazily and failures are
swallowed, because reporting must never change the outcome of a run. Messages
are static phase and reason labels — no request value, record value or
internal identifier is ever included, since these events leave the process and
are not covered by the S-3 output gate. Terminal delivery (success/failure)
belongs to the platform runner alone.

## Composition Pattern

- **Pattern**: GraphNode (subgraph)
- **Composition target**: `DomainWorkflowGraph` (inner `BaseGraph`)
- **Error propagation strategy**: propagate (inner errors re-raised as `SubgraphError`)
- Inner domain nodes run at `TrustLevel.ANONYMOUS`; outer pre/post_process run
  at `TrustLevel.VERIFIED_EXTERNAL`.

## Import Isolation Confirmation
- [x] Template does not import the platform-internal SDK (agenticstar-platform)
- [x] Import targets: framework/ and shared/ only
- [x] No retired base-agent class names (`VectorRAGAgent`, `ChatAgent`, …) in any base position

## Design Decision Record

| Decision | Option A | Option B | Chosen | Rationale |
|----------|----------|----------|--------|-----------|
| Framework base type | AgentBaseGraph | AutonomousBaseGraph | **AgentBaseGraph** | Fixed multi-step RAG workflow, no autonomous loop |
| Composition pattern | Standalone Cat 1 slots | GraphNode → inner BaseGraph | **GraphNode → inner BaseGraph** | 5-step domain workflow exceeds a single `main` node; nested keeps the outer backbone untouched |
| Config-knob flow | Per-call `config` parameter | State-seeded scalars via `_extra_initial_state()` | **State-seeded scalars** | The `execute(self, state) -> dict` node contract has no `config` parameter |
| Structured caller params | JSON envelope only | `input_context` + ContextVar bridge | **`input_context` (envelope kept for compat)** | `input_context` is the SDK's first-class channel; the bridge closes the GraphNode forwarding gap (SDK 1.0.1) |
| Caller-parameter errors | Clamp / ignore with a note | Fail closed naming the field | **Fail closed** | A silently clamped or defaulted parameter is a fail-open on the exact value the caller tried to control |
| Answer synthesis | Rule-based assembly | LLM call | **Rule-based (this version)** | Deterministic assembly is testable; the LLM swaps in at the documented seam |
| KB storage | External vector store | Seeded JSON KB | **Seeded JSON KB** | Self-contained, deterministic CI; the retrieval contract (`retrieved_documents` JSON) is store-agnostic for a later vector-store upgrade |
