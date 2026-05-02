# Ontology4Agile-backed AGES-lite Scrum Agent Architecture

## 1. Purpose

This document describes the AGES-lite Scrum architecture currently
implemented in CogniTwin. It is **not** a full ZT4SWE/AGES system. It
is a deliberately small set of additions on top of the existing
`sprint_loop` runtime that introduce ontology-grounded Scrum
governance without rewriting the orchestration layer.

Goals:

- Keep the existing `src/loop/sprint_loop.py` runtime stable and in
  charge of phase-stepping (ANALYZE → PLAN → EXECUTE → VALIDATE).
- Add an Ontology4Agile-backed Scrum-shape contract (`C3_AGILE`) and
  wire it into governance for the two agents whose output participates
  in Scrum-shape validation (`ScrumMasterAgent`, `POLLMAgent`).
  `ScrumMasterAgent` participates in sprint-level facilitation and
  compliance validation; ProductIncrement data is derived from sprint
  state, not produced by the agent itself.
- Surface agent capabilities as declarative manifests so future audit
  and UI layers can reason about what each agent does.
- Run a single sprint-level advisory compliance check at the end of
  every sprint, attach the result to `SprintResult`, and never block
  sprint completion.

Out of scope for this iteration:

- Full agent registry / dynamic orchestration.
- Mode-switching governance routers.
- A persistent message bus.
- OWL reasoning over the ontology.
- Moving `run_sprint` into `ScrumMasterAgent`.

---

## 2. Architecture Summary

```
┌─────────────────────────────────────────────────────────────┐
│  sprint_loop.run_sprint                  (orchestration)    │
│   ANALYZE → PLAN → EXECUTE → VALIDATE                       │
└──────────┬──────────────────────────────┬───────────────────┘
           │                              │
           │ DeveloperAgent gates         │ end-of-sprint advisory
           │ (unchanged)                  ▼
           │                   sprint_compliance.validate_sprint_compliance
           │                              │
           │                              ▼
           │                   build_sprint_payload (state → agile_payload)
           │                              │
           ▼                              ▼
   evaluate_all_gates            evaluate_all_gates_rich
   role=DeveloperAgent           role=ScrumMasterAgent
                                 agile_payload=<built>
                                          │
                                          ▼
                                  C3_AGILE  ──►  agile_contract (Ontology4Agile)
                                          │
                                          ▼
                                  advisory dict
                                  → SprintResult.agile_compliance
                                  → non-blocking event log
```

Layers:

- **Agent capability layer** — `src/agents/capability_manifest.py`
  defines the frozen `CapabilityManifest` dataclass and a small
  registry. Every agent class can declare what it does without coupling
  to runtime behaviour.
- **BaseAgent contract layer** — `src/agents/base_agent.py` exposes
  `capability_manifest()` so callers can introspect any agent
  uniformly.
- **Ontology4Agile contract layer** — `src/ontology/agile_contract.py`
  loads `ontologies/ontology4agile_v1_3_0.ttl` lazily and exposes
  read-only SPARQL helpers for events, roles, facilitators, sprint
  goal states, artefact relations, and DoD conditions.
- **C3_AGILE gate layer** — `src/gates/c3_agile_compliance.py` is the
  Scrum-shape gate. It validates a structured `agile_payload` against
  the ontology contract.
- **Governance policy layer** — `src/governance/policy.py` decides
  which gates run for which agent role. C3_AGILE is opt-in for two
  roles only.
- **Agile payload builder layer** —
  `src/pipeline/scrum_team/agile_payload_builder.py` is a pure-data
  adapter that converts `SprintStateStore` snapshots into the
  `agile_payload` shape that C3_AGILE consumes.
- **Advisory sprint compliance layer** —
  `src/loop/sprint_compliance.py` is the single helper that runs one
  C3_AGILE check per sprint and returns a fixed-shape advisory dict.
- **Sprint orchestration layer** — `src/loop/sprint_loop.py` is
  unchanged in flow. It calls the advisory helper exactly once,
  emits a non-blocking event, and attaches the result to
  `SprintResult`.

---

## 3. Why AGES-lite, Not Full AGES?

A full AGES-style system would require:

- A dynamic agent registry with capability discovery.
- A governance mode router that switches policies at runtime based on
  context.
- A persistent message bus with replay.
- Deeper ontology reasoning (OWL DL / SHACL).
- Orchestration owned by the agents themselves, not by a top-level
  loop.

CogniTwin currently uses a much smaller surface:

- **Manifests are declarative, not runtime-resolved.** Each agent
  returns a static `CapabilityManifest`; nothing dispatches on it yet.
- **Ontology4Agile is consulted via SPARQL lookups, not reasoning.**
  No reasoner is invoked; the gate uses local-name set membership.
- **C3_AGILE is opt-in per role.** Most roles still run the existing
  C-series gates and behave identically to before.
- **Compliance is advisory only.** The gate result is attached to the
  sprint output but never blocks completion or triggers REDO.
- **`sprint_loop` keeps the orchestration role.** Promoting it into
  `ScrumMasterAgent` is explicitly deferred.

This is intentional: it lets us add ontology grounding without
destabilising the developer pipeline that today produces real
artefacts and runs C8 acceptance-criteria validation.

---

## 4. Agent Capability Manifests

`CapabilityManifest` is a frozen dataclass:

```python
@dataclass(frozen=True)
class CapabilityManifest:
    role: str
    intents: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    gates_consumed: tuple[str, ...] = ()
    ontology_classes_referenced: tuple[str, ...] = ()
```

The module-level registry (`register_manifest`, `get_manifest`,
`list_manifests`) stores one manifest per role and rejects duplicate
registrations. `BaseAgent.capability_manifest()` returns an
`Optional[CapabilityManifest]` for any subclass.

Concrete agents that expose manifests:

- `ProductOwnerAgent` — rule-based PO; its 8 intent handlers are the
  `intents` tuple.
- `POLLMAgent` — LLM-backed PO; declares `decompose_goal`,
  `generate_stories`, `review_story`, plus the gates it now runs
  (`C4`, `C3_AGILE`).
- `ScrumMasterAgent` — rule-based SM; lists its 12 facilitation
  intents and the same gate set.
- `DeveloperAgent` — declares the developer gates
  (`C2_DEV`, `C3`, `C4`, `C5`, `C8`, `A1`).
- `ComposerOrchestrator` — declares meta-intents (`analyze`,
  `reroute`, `synthesize_state`).

Field meanings:

- `role` — canonical agent role identifier; matches `GATE_POLICY`.
- `intents` — high-level operations the agent can be asked to perform.
- `inputs` — the data shapes the agent accepts.
- `outputs` — the data shapes the agent produces.
- `gates_consumed` — which gates evaluate this agent's output.
- `ontology_classes_referenced` — Scrum / domain ontology classes
  that appear in the agent's prompt or output schema.

Why this matters:

- **Audit** — a future audit layer can validate that any output
  actually maps to an intent declared in the manifest.
- **UI** — the cockpit can render agent capabilities without
  hard-coding a list.
- **Governance** — `gates_consumed` can be cross-checked against
  `GATE_POLICY` to detect drift between declared and active gates.

> **CapabilityManifest vs GATE_POLICY.** These are intentionally
> distinct concepts and are cross-checked, not unified.
> `CapabilityManifest` is a *declarative* description of an agent's
> intents, inputs/outputs, related capabilities, and the gates it
> *claims* to consume. `GATE_POLICY` is the *active runtime* mapping
> that decides which gates actually fire for a given role. Drift
> between the two is allowed (and tested for); they are kept
> consistent by convention, not by construction.

---

## 5. Ontology4Agile Runtime Contract

`ontologies/ontology4agile_v1_3_0.ttl` is the canonical Scrum
reference. It is **read-only**: nothing in the runtime mutates it.

`src/ontology/agile_contract.py` is the runtime adapter:

- Lazy load: `load_agile_graph()` parses the TTL on first call, caches
  it, and re-uses the cached `rdflib.Graph` for every later query.
- Failure policy mirrors `src/ontology/loader.py`:
  - Default — missing TTL or `rdflib` returns `None`, logs a warning,
    and downstream helpers return empty results.
  - `COGNITWIN_ONTOLOGY_REQUIRED=1` — raises `OntologyLoadError`
    instead of degrading silently.

Public helpers:

- `valid_scrum_events()` — local names of `agile:ScrumEvent`
  subclasses (e.g. `SprintPlanning`, `DailyScrum`, `SprintReview`,
  `SprintRetrospective`).
- `valid_scrum_roles()` — local names of `agile:ScrumRole` subclasses
  (e.g. `ProductOwner`, `ScrumMaster`, `Developer`).
- `valid_facilitators(event_name)` — set of role names allowed to
  facilitate the given event (`agile:facilitatedBy` triples).
- `valid_sprint_goal_states()` — local names of
  `agile:SprintGoalState` individuals.
- `valid_artefact_relations()` — `(domain, range)` pairs for a curated
  set of object properties (`hasSprintBacklog`, `hasSprintGoal`,
  `producesIncrement`, `validatesIncrement`, `produces`).
- `dod_conditions()` — local names of
  `agile:DefinitionOfDoneCondition` individuals.

There is no OWL reasoner, no SHACL engine, no inferencing. Every
helper is a single SPARQL `SELECT` over the materialised graph.

---

## 6. C3_AGILE Gate

`src/gates/c3_agile_compliance.py` exposes one function:

```python
check_agile_compliance(payload: Optional[dict]) -> tuple[bool, str]
```

Payload sections (all optional — absent sections are not validated):

```python
{
  "event":      {"name": "SprintReview", "facilitator": "ProductOwner"},
  "sprint":     {"goal": "...", "backlog": [...], "tasks": [...]},
  "increment":  {"items": [...], "dod_acknowledged": bool,
                 "acceptance_evidence": str | list,
                 "acceptance_criteria_validated": list},
  "impediment": {"owner": "ScrumMaster", "description": "..."}
}
```

Validation rules:

- **Event** — `name` must be a known Scrum event class. If
  `facilitator` is set, it must be a role allowed by
  `agile:facilitatedBy`.
- **Sprint** — must declare a non-empty `goal` and at least one of
  `backlog` / `tasks`.
- **Increment** — must show DoD acknowledgement *or* acceptance
  evidence (DoD ack flag, DoD evidence text, or
  `acceptance_evidence` / `acceptance_criteria_validated`).
- **Impediment** — `owner` must be a known Scrum role.

Outcomes:

- Empty / `None` payload → `(True, "C3_AGILE not applicable …")`.
- Ontology unavailable and no structural failure →
  `(True, "C3_AGILE DEGRADED PASS — Ontology4Agile unavailable …")`.
- Any structural or ontology-grounded violation →
  `(False, "C3_AGILE FAIL — …")` with each failure joined by
  `"; "`.

The gate is wired into `src/gates/evaluator.py` through
`gate_c3_agile_compliance()` and reads `agile_payload` from the
keyword-only argument on `evaluate_all_gates` /
`evaluate_all_gates_rich`. A revision hint is registered in
`src/gates/gate_result.py::_REVISION_HINTS["C3_AGILE"]` so REDO and
audit downstream can present a recovery instruction.

---

## 7. Governance Policy

`src/governance/policy.py` is the single source of truth for which
gates run for which role. C3_AGILE activation is deliberately
restricted:

| Role                | Gates                               | C3_AGILE |
|---------------------|-------------------------------------|:--------:|
| StudentAgent        | C2, C3, C4, C5, C6, C7, A1          | no       |
| InstructorAgent     | C2, C3, C4, C5, C6, C7, A1          | no       |
| ResearcherAgent     | C2, C3, C4, C5, C6, C7, A1          | no       |
| DeveloperAgent      | C2_DEV, C3, C4, C5, C8, A1          | no       |
| ProductOwnerAgent   | C4                                  | no       |
| ScrumMasterAgent    | C4, **C3_AGILE**                    | yes      |
| POLLMAgent          | C4, **C3_AGILE**                    | yes      |
| `DEFAULT_GATE_POLICY` | C4, A1                            | no       |

Why this scope:

- `ScrumMasterAgent` participates in sprint-level facilitation and
  compliance validation; `POLLMAgent` emits Scrum-shaped LLM output
  (story decompositions, facilitation summaries). ProductIncrement
  data is derived from sprint state, not produced by these agents.
  Activating C3_AGILE for them grounds the validated payload in the
  ontology.
- `ProductOwnerAgent` is rule-based and does not currently emit a
  Scrum facilitation payload — `POLLMAgent` is the LLM-backed PO that
  does, so only the LLM path picks up C3_AGILE.
- `DeveloperAgent` is purposely excluded. The developer pipeline runs
  inside `sprint_loop`'s REDO loop and must not gain a new failure
  mode tied to ontology availability.
- `DEFAULT_GATE_POLICY` and the student-path roles are excluded so
  unknown roles and the academic flows stay safe.

---

## 8. Agile Payload Builder

`src/pipeline/scrum_team/agile_payload_builder.py` is a pure-data
adapter. It does no I/O, no ontology loading, and never mutates its
input.

Public API:

- `build_sprint_payload(sprint_state)` — converts a
  `SprintStateStore` snapshot into a full agile payload (`event`
  excluded; `sprint` / `increment` derived).
- `build_event_payload(event_name, facilitator, extra=None)` —
  constructs the `event` fragment.
- `build_increment_payload(increment, *, tasks=None)` — constructs
  the `increment` fragment, optionally inferring DoD evidence from
  task fields.

Mappings used by `build_sprint_payload`:

- **Sprint goal**: first non-empty of `sprint_goal`, `goal`,
  `sprint.goal`. The Turkish placeholder `"… tanımlanmamış"` is
  treated as no-goal.
- **Backlog**: first non-empty of `sprint_backlog`, `backlog`,
  `sprint.backlog`.
- **Tasks**: `tasks[]` from the snapshot.
- **Increment**: first non-empty of `product_increment`, `increment`.

DoD / acceptance evidence inference:

- A task is considered "accepted" when `accepted is True` *or*
  `po_status ∈ {"accepted", "agent_accepted", "human_accepted"}`.
- If at least one task is both accepted and `ac_validated`, the
  increment fragment gains `dod_acknowledged: True`.
- Acceptance lines are emitted per accepted-or-validated task.
- `acceptance_criteria` from validated tasks accumulate into
  `acceptance_criteria_validated`.
- Explicit `dod_evidence` / `acceptance_evidence` on the snapshot are
  preserved verbatim.

The builder output round-trips through `check_agile_compliance` so
its tests can assert end-to-end shape correctness.

---

## 9. Advisory Sprint Compliance

`src/loop/sprint_compliance.py` exposes one helper:

```python
validate_sprint_compliance(state_store) -> dict
```

Behaviour:

- Loads the current sprint snapshot via `state_store.load()`.
- Calls `build_sprint_payload(state)` to produce the agile payload.
- Calls `evaluate_all_gates_rich(...)` with
  `agent_role="ScrumMasterAgent"` and `agile_payload=payload` so
  C3_AGILE actually fires.
- Reads `report["gates"]["C3_AGILE"]` and packs it into a fixed
  advisory shape.

Critical advisory contract:

- It runs as `ScrumMasterAgent` so the gate dispatches.
- It is advisory only — the returned dict carries
  `advisory=True` and `blocking=False`.
- It does not block sprint completion.
- It does not trigger REDO or reroute.
- It does not alter the DeveloperAgent gate flow inside
  `sprint_loop` — that call is left untouched.
- Every internal failure (state load error, evaluator exception,
  non-dict state) is wrapped into a `gate_pass=True` advisory note
  so the helper can never crash a sprint.

`run_sprint` calls the helper exactly once, after sprint-summary
computation and before `_persist_and_build_result`, emits a single
non-blocking event (`_emit("sm", "gate", "C3_AGILE (advisory,
non-blocking): …")`) and forwards the dict to
`SprintResult.agile_compliance`.

### 9.1 Operational Modes

C3_AGILE / advisory sprint compliance currently has two operational
modes; a third is reserved for future work.

- **Default mode (advisory, non-blocking).** C3_AGILE runs once per
  sprint, attaches its result to `SprintResult.agile_compliance`, and
  never blocks completion. If the ontology TTL or `rdflib` is
  unavailable and no structural failure was found, the gate returns
  `DEGRADED PASS` so sprints continue to run.
- **Strict ontology mode (`COGNITWIN_ONTOLOGY_REQUIRED=1`).** A
  missing TTL or missing `rdflib` raises `OntologyLoadError` instead
  of degrading silently. The gate itself stays advisory; only the
  ontology-load contract becomes strict.
- **Blocking mode — not implemented yet.** A future flag could promote
  C3_AGILE failures to a blocking decision (REDO / abort). Today the
  advisory dict's `blocking=False` invariant is enforced by tests.

---

## 10. `pass` vs `gate_pass`

The advisory dict has two boolean flags that look similar but mean
different things:

- **`pass`** (`True` always) — "the advisory check does not block
  sprint continuation." This is the runner contract: any caller that
  treats `pass` like a normal gate dict will see `True` and let the
  sprint complete. Even when C3_AGILE itself fails, `pass` stays
  `True`.
- **`gate_pass`** (`True` / `False`) — the actual outcome of the
  C3_AGILE check. This is what UI, audit, and humans should look at
  when judging Scrum compliance.
- **`blocking`** (`False` always) — explicit non-blocking marker so
  any future router cannot accidentally promote this dict to a
  blocking decision.
- **`advisory`** (`True` always) — declares that this dict is
  observational, not authoritative.
- **`evidence`** — the human-readable reason from the gate.
- **`revision_hint`** — actionable instruction registered in
  `_REVISION_HINTS["C3_AGILE"]`.
- **`gate`** (`"C3_AGILE"`) — gate identifier.

Example — advisory FAIL with continued sprint:

```json
{
  "gate": "C3_AGILE",
  "advisory": true,
  "blocking": false,
  "pass": true,
  "gate_pass": false,
  "evidence": "C3_AGILE FAIL — sprint context missing SprintGoal",
  "revision_hint": "Align your output with Ontology4Agile: use canonical Scrum event names …"
}
```

Example — advisory PASS:

```json
{
  "gate": "C3_AGILE",
  "advisory": true,
  "blocking": false,
  "pass": true,
  "gate_pass": true,
  "evidence": "C3_AGILE PASS — Scrum-shape contract satisfied.",
  "revision_hint": "Align your output with Ontology4Agile: …"
}
```

---

## 11. Runtime Flow

```
SprintStateStore snapshot
        │
        ▼
build_sprint_payload(state)              (pure-data adapter)
        │
        ▼
validate_sprint_compliance(state_store)  (advisory helper)
        │
        ▼
evaluate_all_gates_rich(
    agent_role="ScrumMasterAgent",
    agile_payload=<built>,
    …)
        │
        ▼
C3_AGILE  ──►  agile_contract.load_agile_graph()
                              valid_scrum_events()
                              valid_facilitators(event)
                              valid_scrum_roles()
        │
        ▼
advisory dict
{ gate, advisory, blocking, pass, gate_pass, evidence, revision_hint }
        │
        ├──►  SprintResult.agile_compliance
        │
        └──►  _emit("sm", "gate", "C3_AGILE (advisory, non-blocking): …")
                              (event log only — never blocks)
```

The DeveloperAgent gate path inside `run_sprint` runs in parallel and
is unchanged: `evaluate_all_gates(agent_role="DeveloperAgent")` with
no `agile_payload`, feeding the existing REDO loop.

---

## 12. Current Invariants

The following invariants are enforced by tests and must remain true:

- `src/loop/sprint_loop.py` still owns sprint orchestration; the
  4-phase flow is not delegated to any agent.
- The DeveloperAgent gate call in `sprint_loop` is unchanged: same
  argument list, no `agile_payload` threaded through.
- C3_AGILE never runs inside the DeveloperAgent gate flow.
- `GATE_POLICY` activates `C3_AGILE` for **only** `ScrumMasterAgent`
  and `POLLMAgent`. `DeveloperAgent`, `ProductOwnerAgent`,
  student-path roles, and `DEFAULT_GATE_POLICY` are excluded.
- The advisory helper is called at most once per sprint
  (`source.count("validate_sprint_compliance(") == 1`).
- An advisory FAIL never blocks sprint completion: `pass=True`,
  `blocking=False`, `gate_pass=False`.
- `ontologies/ontology4agile_v1_3_0.ttl` is treated as read-only.
- Missing TTL / missing `rdflib` degrades to `DEGRADED PASS`; only
  `COGNITWIN_ONTOLOGY_REQUIRED=1` upgrades the miss to a hard
  failure.
- `evaluate_all_gates` and `evaluate_all_gates_rich` keep
  `agile_payload` keyword-only with a `None` default, so legacy
  callers continue to work.

---

## 13. Tests

Key test files and what they pin:

- `tests/test_capability_manifest.py` — frozen-ness, registry
  round-trip, duplicate-role rejection.
- `tests/test_base_agent_contract.py` — `BaseAgent.capability_manifest()`
  exists and returning `None` is acceptable.
- `tests/test_agent_manifests.py` — each agent's manifest declares
  intents and `gates_consumed` consistent with `GATE_POLICY`.
- `tests/test_agile_contract.py` — TTL load, helper queries return
  the expected canonical Scrum vocabulary, degraded mode behaves.
- `tests/test_c3_agile_gate.py` — gate semantics: empty payload,
  unknown event, sprint missing goal, DoD evidence, impediment owner,
  evaluator integration via `evaluate_all_gates`.
- `tests/test_c3_agile_active_policy.py` — Phase 6 invariants:
  C3_AGILE active for the two scoped roles only; baseline gate sets
  preserved; sprint_loop wiring is advisory-only and the
  DeveloperAgent gate call is untouched.
- `tests/test_agile_payload_builder.py` — purity, mappings, DoD
  inference, round-trip with C3_AGILE.
- `tests/test_sprint_compliance.py` — advisory shape contract,
  `pass=True` always, `blocking=False`, `gate_pass` reflects real
  outcome, every failure mode degrades to `gate_pass=True`.
- `tests/test_sprint_loop_compliance_wiring.py` — `SprintResult`
  exposes `agile_compliance`; the helper is called exactly once;
  emit reads `gate_pass`; `GATE_POLICY` invariant; the advisory
  contract holds end-to-end.

---

## 14. Future Work

Possible next iterations, in rough order of value:

- **EventBus-lite** — replace ad-hoc `event_callback` calls in
  `sprint_loop` with an in-process pub/sub log so per-sprint events
  are typed and replayable.
- **Stronger agent audit** — per-agent decision log keyed by
  `event_id`, emitted on phase boundaries and reconciled with
  manifest intents.
- **Optional blocking mode for C3_AGILE** — gated behind an explicit
  configuration flag; default stays advisory.
- **Gradually move orchestration toward `ScrumMasterAgent`** — keep
  `sprint_loop` as a thin shim that delegates to
  `sm_agent.run_sprint`. Only after the advisory layer has been
  stable across multiple real sprints.
- **UI display for `agile_compliance`** — surface
  `gate_pass`, `evidence`, and `revision_hint` in the cockpit so
  humans see the advisory result on the sprint review screen.
- **Retro actions feeding Product Backlog** — turn the
  `retro_actions` already collected by `SprintStateStore` into typed
  backlog entries via `ProductBacklogStore`.
- **`ProductIncrementStore`** — separate the increment artefact from
  `SprintStateStore` so it can carry its own DoD evidence, history,
  and ontology-typed metadata.
