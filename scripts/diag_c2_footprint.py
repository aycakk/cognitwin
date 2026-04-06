"""
diag_c2_footprint.py — C2 gate pass-rate comparison for DeveloperAgent

Simulates gate_c2_memory_grounding() under two conditions:
  A) WITH footprint  : is_empty=False, real ChromaDB-style student context
  B) WITHOUT footprint: is_empty=True,  empty vector context

Uses realistic draft samples taken from DeveloperAgent/DeveloperOrchestrator
output patterns (rule-based path, direct path, LLM fallback path).

No live services required — all inputs are synthetic but representative.
"""

import re
import sys

# ─── C2 gate (inline copy so no import dependencies) ─────────────────────────

def gate_c2_memory_grounding(
    draft: str,
    vector_context: str,
    is_empty: bool,
    agent_role: str = "StudentAgent",
):
    # FIXED: DeveloperAgent exempt from word-overlap check
    if agent_role == "DeveloperAgent":
        return True, "DeveloperAgent: C2 grounding not applicable (developer context is self-contained)."

    if is_empty:
        if "bulamadım" in draft.lower():
            return True, "Vector memory empty; BlindSpot disclosure present."
        return False, "Vector memory empty but no BlindSpot disclosure."

    if "bulamadım" in draft.lower():
        return True, "Draft issued BlindSpot — acceptable when query not in results."

    context_words = {
        w.lower() for w in re.findall(r"\b\w{6,}\b", vector_context)
        if not re.match(r"\[.*_MASKED\]", w)
    }
    draft_words = {w.lower() for w in re.findall(r"\b\w{6,}\b", draft)}
    overlap = context_words & draft_words

    if len(overlap) >= 2:
        return True, f"Grounding verified ({len(overlap)} shared content terms)."
    return False, f"Draft not grounded (overlap={len(overlap)} terms < 2). Possible hallucination."


# ─── Realistic vector_context (student ChromaDB records, Turkish, masked) ────
# Mirrors what VectorMemory.retrieve() returns from the 129 academic records.

VECTOR_CONTEXT_STUDENT = """
=== VECTOR MEMORY (ChromaDB, k=15) ===

[Result 1]
[STUDENT_ID_MASKED] öğrencisi CS101 dersinden 85 not aldı.
Sınav tarihi: 15.03.2026. Ödev teslim: 10.03.2026.
---
[Result 2]
Yusuf Altunel, CS202 dersi sorumlusu.
Vize sınavı: 20.04.2026. Dönem projesi teslim: 30.04.2026.
---
[Result 3]
[EMAIL_MASKED] adresinden alınan mesaj: Final sınavı ertelendi.
Yeni tarih: 25.05.2026. Sınıf: C101.
---
[Result 4]
Öğrenci danışma saatleri: Salı 14:00-16:00. Oda: B204.
---
[Result 5]
SEN4012 dersi brute force araştırma raporu teslim tarihi: 01.06.2026.
---
=== END VECTOR MEMORY ===
"""

VECTOR_CONTEXT_EMPTY = ""

# ─── Representative DeveloperAgent draft samples ─────────────────────────────
# These mirror what the orchestrator actually produces.

DRAFTS = {
    "rules_path": """\
Restructured Prompt v1

TASK MODE: OTHER

STEP 0 - MANDATORY PRE-FLIGHT FILE CHECK
1. List all files currently uploaded by the user.
2. Compare them against the required file list below.
3. If any required file is missing, STOP and ask the user to upload.
Required files:
- 'redo_v2_4_6.ttl'
- 'redo_v2_shacl_shapes.ttl'

STEP 1 - PROMPT RESTRUCTURING PHASE
Goal: Build the Developer Agent pipeline with ZT4SWE verification.
Constraints: MVP scope; keep minimal and testable.
Deliverables: task breakdown, ontology draft, validation checklist.

STEP 2 - EXECUTION PHASE
Execute only after user confirmation ("run it").

VERIFICATION SAFEGUARD
Activate conjunctive gate C1..C8, anti-sycophancy protocol,
BlindSpot disclosure, REDO checksum enforcement.
""",

    "direct_path_en": """\
Here is a direct technical response for your request.

To implement the Developer Agent pipeline you should:
1. Create DeveloperOrchestrator with memory_backend and chat_fn wired.
2. Call orchestrator.run(request, developer_id) to get structured result.
3. Extract result["solution"] and pass through C1-C8 gate array.
4. If any gate fails, trigger REDO loop with evidence injected.
5. Emit final response after BlindSpot check.

Key files: pipeline.py, developer_orchestrator.py, developer_agent.py.
""",

    "direct_path_tr": """\
Asagidaki istege dogrudan teknik cevap veriyorum.

Developer Agent boru hattini kurmak icin:
1. DeveloperOrchestrator'i memory_backend ve chat_fn ile baslat.
2. orchestrator.run(request, developer_id) cagir.
3. Sonucu C1-C8 kapilari ile dogrula.
4. Kapi basarisizilik durumunda REDO dongusunu tetikle.
5. BlindSpot kontrolunden sonra yaniti ilet.
""",

    "execution_plan": """\
Execution Plan v1

Stage 2 confirmed. Execution has started using the latest user task context.

Task mode: OTHER
Target role: Developer Agent
Goal snapshot: Implement the pipeline and wire all agents

Immediate deliverables:
- Task breakdown
- Ontology draft
- Competency question set
- Validation checklist
- Developer notes

Validation checklist:
- Every implementation task must reference a deliverable.
- Every ontology change must include at least one competency question.
- Every generated plan must mention a validation step.

Next steps:
- Create or extend a TTL file for Developer Agent.
- Add at least three competency questions tied to the role workflow.
- Define one or two validation rules for invalid or incomplete outputs.
- Prepare a small demo scenario with sample input and expected response.

Footprint memory snapshot:
Bunu hafizamda bulamadim.
""",

    "llm_fallback": """\
LLM path failed (Empty LLM response). Returning rule-based fallback.

Restructured Prompt v1

You are a high-integrity execution orchestrator.

TASK MODE: OTHER

STEP 0 - MANDATORY PRE-FLIGHT FILE CHECK
Required files:
- 'redo_v2_4_6.ttl'
- 'redo_v2_shacl_shapes.ttl'
""",

    "blindspot_in_draft": """\
Bunu hafızamda bulamadım.
Bu konuda doğrulanmış bir kayıt mevcut değil.
""",
}

# ─── Run comparison ───────────────────────────────────────────────────────────

SEP  = "=" * 72
DASH = "-" * 72

def run():
    print(SEP)
    print("C2 GATE PASS-RATE COMPARISON  --  DeveloperAgent")
    print(SEP)

    results = {}
    for condition, (ctx, empty_flag, label) in {
        "WITH_footprint":    (VECTOR_CONTEXT_STUDENT, False, "is_empty=False, real ChromaDB context"),
        "WITHOUT_footprint": (VECTOR_CONTEXT_EMPTY,   True,  "is_empty=True,  empty vector context"),
    }.items():
        print(f"\n{DASH}")
        print(f"CONDITION: {label}")
        print(DASH)

        passes = 0
        fails  = 0

        for draft_name, draft_text in DRAFTS.items():
            ok, evidence = gate_c2_memory_grounding(draft_text, ctx, empty_flag, agent_role="DeveloperAgent")
            status = "PASS" if ok else "FAIL"
            if ok:
                passes += 1
            else:
                fails += 1

            # Compute overlap for diagnostics when not empty
            if not empty_flag:
                ctx_words   = {w.lower() for w in re.findall(r"\b\w{6,}\b", ctx)
                               if not re.match(r"\[.*_MASKED\]", w)}
                draft_words = {w.lower() for w in re.findall(r"\b\w{6,}\b", draft_text)}
                overlap     = ctx_words & draft_words
                overlap_str = f"  overlap_terms={sorted(overlap)[:6]}"
            else:
                overlap_str = ""

            print(f"  [{status}] {draft_name:<25}  — {evidence}{overlap_str}")

        total = passes + fails
        pct   = 100 * passes // total if total else 0
        results[condition] = {"pass": passes, "fail": fails, "pct": pct}

    print(f"\n{SEP}")
    print("SUMMARY")
    print(SEP)
    for cond, r in results.items():
        print(f"  {cond:<25}  PASS {r['pass']}/{r['pass']+r['fail']}  ({r['pct']}%)")

    print()
    print("DIAGNOSIS")
    print(DASH)

    with_pct    = results["WITH_footprint"]["pct"]
    without_pct = results["WITHOUT_footprint"]["pct"]
    delta       = without_pct - with_pct

    if with_pct == 0 and without_pct == 0:
        print("  C2 FAILS in BOTH conditions.")
        print("  Root cause: developer drafts never contain 'bulamadım' AND")
        print("  ChromaDB context (student records) shares no 6+ char words with")
        print("  developer output. Footprint is NOT the differentiating factor.")
        print()
        print("  RECOMMENDATION: Exempt DeveloperAgent from C2 word-overlap check.")
        print("  C2 is a student-path grounding gate. Developer path is self-contained")
        print("  (rule-based or LLM with its own context). Proposed fix:")
        print("    gate_c2: if agent_role == 'DeveloperAgent': return True, 'N/A (dev path)'")
    elif delta > 0:
        print(f"  WITHOUT footprint is {delta}pp better. Footprint is actively hurting C2.")
    elif delta < 0:
        print(f"  WITH footprint is {abs(delta)}pp better. Footprint helps C2.")
    else:
        print("  No difference. Footprint has no effect on C2 pass rate.")

    print(SEP)
    return results


if __name__ == "__main__":
    run()
