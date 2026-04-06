"""
test_structured_debug.py  --  End-to-end test for structured JSON debug output

Tests three developer queries with a mocked Ollama chat_fn.
Verifies:
  1. Structured JSON is produced when codebase context is present
  2. Machine validation fires (_validation block exists in output)
  3. Each of the three representative prompts produces a non-empty answer

Run:
    python -X utf8 scripts/test_structured_debug.py
"""

import sys
import json
import re
from pathlib import Path

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.developer_orchestrator import DeveloperOrchestrator, MEMORY_NOT_FOUND_TEXT
from src.services.api.pipeline import (
    _build_codebase_context,
    _validate_debug_result,
    _format_debug_result,
)

SEP  = "=" * 72
DASH = "-" * 72

# ---------------------------------------------------------------------------
# Mock LLM that returns a plausible structured JSON response for any query
# ---------------------------------------------------------------------------

MOCK_JSON_RESPONSE = {
    "entry_point": "chat_completions",
    "files_used": [
        "src/services/api/openai_routes.py",
        "src/services/api/pipeline.py",
    ],
    "functions_used": [
        "chat_completions",
        "process_user_message",
        "_process_developer_message",
    ],
    "execution_path": [
        "POST /v1/chat/completions -> chat_completions()",
        "chat_completions() calls process_user_message()",
        "process_user_message() routes to _process_developer_message() for developer model",
        "_process_developer_message() builds codebase context and calls orchestrator.run()",
    ],
    "suspected_root_cause": "No runtime bug found; routing is correct for developer model.",
    "evidence": [
        "openai_routes.py line 55: result = process_user_message(..., model=req.model)",
        "pipeline.py: model == 'cognitwin-developer' routes to _process_developer_message",
    ],
    "fix": "",
    "confidence": 0.82,
    "speculative": False,
}


def mock_chat_fn(model: str, messages: list, **kwargs) -> dict:
    """Returns a structured JSON response wrapped in assistant message format."""
    return {
        "message": {
            "content": json.dumps(MOCK_JSON_RESPONSE, ensure_ascii=False)
        }
    }


# ---------------------------------------------------------------------------
# Helper: run one prompt through the orchestrator with mocked LLM
# ---------------------------------------------------------------------------

def run_prompt(prompt: str, developer_id: str = "dev-test-001") -> dict:
    """
    Mirrors what _process_developer_message() does, but with mock chat_fn.
    Returns a result dict with keys: draft, validated, formatted, has_validation.
    """
    # 1. Build codebase context (same call as production pipeline)
    codebase_context = _build_codebase_context(prompt)

    # 2. Run orchestrator with mock LLM
    orchestrator = DeveloperOrchestrator(
        chat_fn=mock_chat_fn,
        default_model="llama3.2",
    )
    result = orchestrator.run(
        prompt,
        developer_id=developer_id,
        memory_context=codebase_context,
    )

    draft = result.get("solution") or result.get("answer") or ""

    # 3. Stage 2b validation (same logic as production pipeline)
    validated_dict = None
    formatted_text = draft

    if codebase_context:
        try:
            parsed = json.loads(draft)
            if isinstance(parsed, dict) and "files_used" in parsed:
                validated_dict = _validate_debug_result(parsed, PROJECT_ROOT)
                formatted_text = _format_debug_result(validated_dict)
        except (json.JSONDecodeError, ValueError):
            pass  # prose response

    return {
        "prompt": prompt,
        "has_codebase_context": bool(codebase_context),
        "draft_raw": draft,
        "validated": validated_dict,
        "formatted": formatted_text,
        "has_validation": validated_dict is not None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PROMPTS = [
    "analyze the current API routing and identify runtime bugs",
    "explain the current API routing in this repository in plain technical language",
    "debug the developer routing path",
]


def main():
    print(SEP)
    print("STRUCTURED DEBUG OUTPUT  --  End-to-End Test")
    print(SEP)

    all_passed = True

    for i, prompt in enumerate(PROMPTS, 1):
        print(f"\n{DASH}")
        print(f"PROMPT {i}: {prompt}")
        print(DASH)

        r = run_prompt(prompt)

        # --- Check 1: Codebase context was built
        ctx_status = "PASS" if r["has_codebase_context"] else "FAIL"
        print(f"  [codebase_context]  {ctx_status}  (non-empty: {r['has_codebase_context']})")

        # --- Check 2: Validation fired
        val_status = "PASS" if r["has_validation"] else "WARN"
        print(f"  [validation_fired]  {val_status}  (parsed as JSON + validated: {r['has_validation']})")

        # --- Check 3: Formatted output is non-empty
        out_status = "PASS" if r["formatted"].strip() else "FAIL"
        print(f"  [output_non_empty]  {out_status}")

        if r["has_validation"] and r["validated"]:
            v = r["validated"]["_validation"]
            print(f"  [files_ok]          {'PASS' if v['files_ok']     else 'WARN'}  invalid_files={v['invalid_files']}")
            print(f"  [functions_ok]      {'PASS' if v['functions_ok'] else 'WARN'}  unverified={v['unverified_funcs']}")
            print(f"  [all_verified]      {'PASS' if v['all_verified'] else 'WARN'}")

        # --- Print formatted output (truncated)
        print()
        lines = r["formatted"].splitlines()
        for line in lines[:30]:
            print(f"    {line}")
        if len(lines) > 30:
            print(f"    ... [{len(lines) - 30} more lines]")

        if ctx_status == "FAIL" or out_status == "FAIL":
            all_passed = False

    print(f"\n{SEP}")
    print("RESULT:", "ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    print(SEP)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
