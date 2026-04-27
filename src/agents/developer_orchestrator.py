from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from src.agents.developer_agent import DeveloperAgent
from src.core.llm_config import DEFAULT_MODEL


MEMORY_NOT_FOUND_TEXT = "Bunu hafizamda bulamadim."

# ── Structured debug output schema ───────────────────────────────────────────
# Used by _generate_direct_solution() to request machine-parseable analysis.
_DEBUG_SCHEMA = """\
{
  "entry_point": "<first route/function called for this request, or empty string>",
  "files_used": ["<relative path from project root, e.g. src/services/api/pipeline.py>"],
  "functions_used": ["<exact function name as it appears in the source>"],
  "execution_path": ["<step 1: description>", "<step 2: description>"],
  "suspected_root_cause": "<concise root cause, or empty string if none found>",
  "evidence": ["<direct quote or line reference from the codebase context>"],
  "fix": "<proposed fix, or empty string>",
  "confidence": 0.0,
  "speculative": false
}"""


class DeveloperOrchestrator:
    """
    Staged orchestrator for developer-role requests (Scrum team).

    Flow (role-based — no personal footprint/profile/identity):
    1) Understand request
    2) Retrieve project context (task type, language, strategy)
    3) Retrieve ontology constraints (developer_ontology.ttl)
    4) Generate solution (via DeveloperAgent + injected team context)
    5) Validate solution
    6) Return structured response

    Context sources (role-appropriate):
      - Codebase context (injected by developer_runner from _DEV_CONTEXT_MAP)
      - Sprint context   (injected by developer_runner from SprintStateStore)
      - Ontology constraints (developer_ontology.ttl)

    NOT used (architectural invariant — Scrum team is role-based):
      - Personal footprint signals
      - Personal developer profile / DeveloperProfileStore
      - developer_id as a person identity key

    NOTE: chat_fn must be a callable that returns a plain dict
    with {"message": {"content": str}} — use _safe_chat() from pipeline.py
    as the wrapper when Ollama is the backend.
    """

    def __init__(
        self,
        *,
        generator: DeveloperAgent | None = None,
        chat_fn: Callable[..., dict[str, Any]] | None = None,
        default_model: str = DEFAULT_MODEL,
        ontology_path: str | Path | None = None,
    ) -> None:
        # Pass chat_fn into DeveloperAgent so its process() method can call the
        # LLM directly.  This makes DeveloperAgent a genuine agent rather than
        # a pure template renderer.
        self.generator = generator or DeveloperAgent(chat_fn=chat_fn)
        self.chat_fn = chat_fn
        self.default_model = default_model

        if ontology_path is None:
            project_root = Path(__file__).resolve().parents[2]
            ontology_path = project_root / "ontologies" / "developer_ontology.ttl"
        self.ontology_path = Path(ontology_path)

    def run(self, request: str, **runtime: Any) -> dict[str, Any]:
        task_understanding = self._understand_request(request=request, **runtime)

        context_runtime = dict(runtime)
        context_runtime.pop("task_understanding", None)
        context = self.retrieve_context(
            request=request,
            task_understanding=task_understanding,
            **context_runtime,
        )

        ontology_runtime = dict(runtime)
        ontology_runtime.pop("context", None)
        ontology_constraints = self.retrieve_ontology_constraints(
            request=request,
            context=context,
            **ontology_runtime,
        )

        generate_runtime = dict(runtime)
        generate_runtime.pop("context", None)
        generate_runtime.pop("ontology_constraints", None)
        solution = self.generate(
            request=request,
            context=context,
            ontology_constraints=ontology_constraints,
            **generate_runtime,
        )

        validate_runtime = dict(runtime)
        validate_runtime.pop("context", None)
        validate_runtime.pop("ontology_constraints", None)
        validation_report = self.validate(
            request=request,
            solution=solution,
            context=context,
            ontology_constraints=ontology_constraints,
            **validate_runtime,
        )

        return self.build_response(
            task_understanding=task_understanding,
            ontology_constraints=ontology_constraints,
            solution=solution,
            validation_report=validation_report,
        )

    def retrieve_context(
        self,
        request: str,
        task_understanding: str = "",
        **runtime: Any,
    ) -> dict[str, Any]:
        task_type = str(runtime.get("task_type") or self._infer_task_type(request))
        language = str(runtime.get("language") or "en").lower()
        strategy = str(runtime.get("strategy") or "auto").lower()
        if strategy == "auto":
            strategy = self._infer_strategy(request)
        is_execution_phase = bool(runtime.get("is_execution_phase", False))

        source_prompt = str(runtime.get("source_prompt") or request or "")
        uploaded_files = list(runtime.get("uploaded_files") or [])
        missing_files = list(runtime.get("missing_files") or [])
        if not missing_files:
            missing_files = self.generator.check_uploaded_files(uploaded_files, task_type=task_type)

        project_context_signals = []
        lower_request = (request or "").lower()
        if "ontology" in lower_request or ".ttl" in lower_request:
            project_context_signals.append("ontology-driven request")
        if "uploaded files" in lower_request:
            project_context_signals.append("file-gated workflow")
        if is_execution_phase:
            project_context_signals.append("execution phase confirmed")

        return {
            "task_type": task_type,
            "language": language,
            "strategy": strategy,
            "is_execution_phase": is_execution_phase,
            "source_prompt": source_prompt,
            "target_role": self._infer_target_role(source_prompt),
            "constraints": self._extract_constraints(source_prompt),
            "goal_snapshot": self._extract_goal_snapshot(source_prompt),
            "uploaded_files": uploaded_files,
            "missing_files": missing_files,
            "task_understanding": task_understanding,
            "project_context_signals": project_context_signals,
        }

    def retrieve_ontology_constraints(
        self,
        request: str = "",
        context: dict[str, Any] | None = None,
        **_: Any,
    ) -> list[str]:
        context = context or {}
        task_type = str(context.get("task_type") or "ontology")
        constraints: list[str] = [
            "Reuse existing ontology concepts before adding new ones.",
            "Keep the MVP minimal, testable, and role-aligned.",
        ]

        required_files = self.generator.required_files(task_type)
        if required_files:
            constraints.append(
                "Required files for this task: " + ", ".join(required_files)
            )

        if self.ontology_path.exists():
            try:
                text = self.ontology_path.read_text(encoding="utf-8")
                classes = self._collect_ontology_terms(text, "Class")
                props = self._collect_ontology_terms(text, "Property")
                if classes:
                    constraints.append(
                        "Available ontology classes: " + ", ".join(classes[:6])
                    )
                if props:
                    constraints.append(
                        "Available ontology properties: " + ", ".join(props[:6])
                    )
            except Exception:
                constraints.append("Developer ontology file exists but could not be parsed.")
        else:
            constraints.append("Developer ontology file not found; follow existing role library constraints.")

        return constraints

    def generate(
        self,
        request: str,
        context: dict[str, Any] | None = None,
        ontology_constraints: list[str] | None = None,
        **runtime: Any,
    ) -> str:
        del ontology_constraints
        context = context or {}

        language = str(context.get("language") or "en")
        task_type = str(context.get("task_type") or "ontology")
        strategy = str(context.get("strategy") or "auto")
        is_execution_phase = bool(context.get("is_execution_phase"))
        system_prompt_override = str(runtime.get("system_prompt_override") or "").strip()

        # memory_context here is codebase + sprint context, NOT personal footprint.
        # It is injected by developer_runner from _build_codebase_context() and
        # SprintStateStore.read_context_block().
        memory_context = str(runtime.get("memory_context") or "").strip()
        if not memory_context:
            memory_context = MEMORY_NOT_FOUND_TEXT

        if is_execution_phase:
            packet = self.generator.build_role_packet(
                target_role=str(context.get("target_role") or "Developer Agent"),
                constraints=str(context.get("constraints") or ""),
            )
            return self.generator.generate_execution_plan(
                task_type=task_type,
                role_packet=packet,
                goal_snapshot=str(context.get("goal_snapshot") or "No explicit goal provided."),
                memory_context=memory_context,
                language=language,
            )

        if strategy == "direct":
            _debug_hints = (
                "debug", "bug", "fix", "hata", "error", "exception",
                "traceback", "stack trace", "neden çalışmıyor", "neden calısmiyor",
                "why is it failing", "why does it fail",
            )
            is_debug = any(h in request.lower() for h in _debug_hints)
            return self._generate_direct_solution(
                request=request,
                language=language,
                memory_context=memory_context,
                model=str(runtime.get("model") or self.default_model),
                system_prompt_override=system_prompt_override,
                is_debug=is_debug,
            )

        # Strategy = rules path (deterministic)
        if strategy == "rules":
            title = "Yeniden Yapilandirilmis Prompt v1" if language == "tr" else "Restructured Prompt v1"
            prompt = self.generator.generate_restructured_prompt(
                request=request,
                task_type=task_type,
                language=language,
            )
            return f"{title}\n\n{prompt}"

        # Strategy = llm path (with deterministic fallback)
        if self.chat_fn is None:
            return self._build_llm_fallback(
                request=request,
                task_type=task_type,
                language=language,
                error_message="ollama is not installed",
            )

        system_prompt = system_prompt_override or self.generator.build_system_prompt_template(task_type=task_type)
        user_prompt = self.generator.build_restructured_execution_prompt(
            zero_time_prompt=request,
            task_type=task_type,
        )

        try:
            response = self.chat_fn(
                model=str(runtime.get("model") or self.default_model),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            "Return only 'Restructured Prompt v1' as executable text. "
                            "Do not skip file-check and verification constraints.\n\n"
                            + ("Reply in Turkish.\n\n" if language == "tr" else "")
                            + self._build_memory_instruction_block(
                                memory_context=memory_context,
                                language=language,
                            )
                            + "\n\n"
                            + user_prompt
                        ),
                    },
                ],
            )
            answer = str((response or {}).get("message", {}).get("content", "")).strip()
            if not answer:
                raise ValueError("Empty LLM response")
            return self._localize_common_headers(answer, language=language)
        except Exception as exc:
            return self._build_llm_fallback(
                request=request,
                task_type=task_type,
                language=language,
                error_message=str(exc),
            )

    @staticmethod
    def _extract_json_block(text: str) -> str:
        """
        Extract the first valid JSON object from an LLM response.

        llama3.2 often wraps JSON in prose or code fences. Three attempts:
          1. Direct parse — model output is already clean JSON.
          2. ```json ... ``` fence — extract inner block.
          3. First { ... } span — pull outermost braces regardless of fencing.

        Returns the raw JSON string if a valid object is found, else "".
        """
        stripped = text.strip()

        # Attempt 1: already clean JSON
        try:
            json.loads(stripped)
            return stripped
        except (json.JSONDecodeError, ValueError):
            pass

        # Attempt 2: fenced block  ```json ... ``` or ``` ... ```
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
        if m:
            candidate = m.group(1).strip()
            try:
                json.loads(candidate)
                return candidate
            except (json.JSONDecodeError, ValueError):
                pass

        # Attempt 3: outermost { ... } span
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end > start:
            candidate = stripped[start : end + 1]
            try:
                json.loads(candidate)
                return candidate
            except (json.JSONDecodeError, ValueError):
                pass

        return ""

    def _generate_direct_solution(
        self,
        *,
        request: str,
        language: str,
        memory_context: str,
        model: str,
        system_prompt_override: str = "",
        is_debug: bool = False,
    ) -> str:
        """
        Generate a direct technical answer for the given request.

        When codebase context is available (memory_context non-empty and not the
        MEMORY_NOT_FOUND sentinel), the model is asked to return a structured JSON
        object matching _DEBUG_SCHEMA so that the caller can machine-validate the
        cited files and functions.  If JSON extraction fails the raw prose is
        returned as a fallback so the function never silently drops a response.
        """
        if self.chat_fn is None:
            if language == "tr":
                return (
                    "Dogrudan teknik yanit uretilemedi: LLM backend kullanilamiyor "
                    "(ollama is not installed)."
                )
            return (
                "Direct technical response is unavailable because the LLM backend "
                "is not configured (ollama is not installed)."
            )

        # Use structured JSON output when real codebase context has been injected.
        # Fall back to prose when there is no codebase context (no code to reason about).
        has_codebase = (
            memory_context.strip()
            and memory_context.strip() != MEMORY_NOT_FOUND_TEXT
            and "CODEBASE CONTEXT" in memory_context
        )

        if has_codebase and is_debug and not system_prompt_override:
            system_prompt = (
                "You are the CogniTwin Developer Agent performing a structured code analysis.\n"
                "You will be given source code from the actual repository.\n"
                "Return ONLY a single raw JSON object — no prose, no markdown, no code fences.\n"
                "The JSON must match this exact schema:\n"
                + _DEBUG_SCHEMA
                + "\n\nRules:\n"
                "- Only include files and functions that explicitly appear in the CODEBASE CONTEXT.\n"
                "- If a file or function is not visible in the context, do not invent it.\n"
                "- Set speculative=true if you cannot find direct textual evidence.\n"
                "- confidence must be between 0.0 and 1.0.\n"
                "- evidence items must be direct quotes or identifiers from the source code shown."
            )
            user_prompt = (
                f"REQUEST: {request}\n\n"
                "CODEBASE CONTEXT (only use what is shown here):\n"
                f"{memory_context}"
            )
        elif language == "tr":
            system_prompt = (
                "Sen CogniTwin Developer Agent'sin. Kullanicinin teknik istegine dogrudan final cevap ver. "
                "Asla wrapper formatlar kullanma: 'Yeniden Yapilandirilmis Prompt v1', "
                "'Restructured Prompt v1', 'Execution Plan v1', 'TASK MODE', 'STEP 0', "
                "'ADIM 0', 'run it'."
            )
            user_prompt = (
                "Asagidaki istege dogrudan teknik cevap ver.\n\n"
                f"ISTEK:\n{request}\n\n"
                "BELLEK BAGLAMI:\n"
                f"{memory_context}"
            )
        else:
            system_prompt = (
                "You are the CogniTwin Developer Agent. Return a direct final technical response. "
                "Never output wrappers such as 'Restructured Prompt v1', "
                "'Yeniden Yapilandirilmis Prompt v1', "
                "'Execution Plan v1', 'TASK MODE', 'STEP 0', 'ADIM 0', or 'run it'."
            )
            user_prompt = (
                "Provide a direct technical response for this request.\n\n"
                f"REQUEST:\n{request}\n\n"
                "MEMORY CONTEXT:\n"
                f"{memory_context}"
            )

        if system_prompt_override:
            system_prompt = system_prompt_override

        try:
            response = self.chat_fn(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = str((response or {}).get("message", {}).get("content", "")).strip()
            if not raw:
                raise ValueError("Empty direct response")

            # When structured output was requested, try to extract the JSON block.
            # If extraction fails, fall back to raw prose — never drop the answer.
            if has_codebase and is_debug and not system_prompt_override:
                json_str = self._extract_json_block(raw)
                if json_str:
                    return json_str
                # Extraction failed: return prose but mark it as unstructured
                return raw

            return raw
        except Exception as exc:
            if language == "tr":
                return f"Dogrudan teknik yanit uretilemedi ({exc})."
            return f"Direct technical response could not be generated ({exc})."

    def validate(
        self,
        request: str,
        solution: str,
        context: dict[str, Any] | None = None,
        ontology_constraints: list[str] | None = None,
        **runtime: Any,
    ) -> dict[str, Any]:
        del request
        context = context or {}
        ontology_constraints = ontology_constraints or []

        language = str(context.get("language") or "en")
        is_execution_phase = bool(context.get("is_execution_phase"))
        strategy = str(context.get("strategy") or "auto").lower()
        missing_files = list(context.get("missing_files") or [])

        # memory_context carries the injected codebase + sprint text
        memory_context = str(runtime.get("memory_context") or "").strip()

        expected_header = ""
        if strategy != "direct":
            expected_header = "Yurutme Plani v1" if language == "tr" and is_execution_phase else (
                "Execution Plan v1" if is_execution_phase else (
                    "Yeniden Yapilandirilmis Prompt v1" if language == "tr" else "Restructured Prompt v1"
                )
            )

        # ── Content grounding checks ──
        # Measure how much of the solution is grounded in injected context.
        has_codebase_context = bool(
            memory_context
            and memory_context != MEMORY_NOT_FOUND_TEXT
            and "CODEBASE CONTEXT" in memory_context
        )
        has_sprint_context = "SPRINT CONTEXT" in memory_context if memory_context else False

        # Word overlap: count meaningful terms shared between solution and context
        # Uses regex to extract word tokens (handles punctuation correctly)
        import re as _re
        _stop = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
                 "to", "for", "of", "and", "or", "but", "not", "with", "this",
                 "that", "it", "be", "as", "by", "from", "has", "have", "had",
                 "will", "would", "could", "should", "can", "may", "must",
                 "been", "being", "about", "into", "over", "than", "then",
                 "when", "where", "which", "while", "return", "self", "none",
                 "true", "false", "import", "class", "print",
                 "ve", "bir", "bu", "da", "de", "ile", "için", "olan", "var",
                 "yok", "ise"}
        solution_words = {
            w.lower() for w in _re.findall(r"\b\w+\b", solution or "")
            if len(w) >= 4 and w.lower() not in _stop
        }
        context_words = {
            w.lower() for w in _re.findall(r"\b\w+\b", memory_context)
            if len(w) >= 4 and w.lower() not in _stop
        } if memory_context else set()
        overlap = solution_words & context_words
        grounding_overlap = len(overlap) >= 5

        # Check for file path patterns in solution vs. context
        _file_pat = _re.compile(
            r'(?:src|tests|scripts|ontologies|data|infra)/[\w/]+\.(?:py|ttl|json|yaml|yml)'
        )
        cited_in_solution = set(_file_pat.findall(solution or ""))
        cited_in_context = set(_file_pat.findall(memory_context)) if memory_context else set()
        files_grounded = cited_in_solution.issubset(cited_in_context) if cited_in_solution else True

        checks = {
            "non_empty_solution": bool(solution and solution.strip()),
            "expected_header_present": True if not expected_header else expected_header in (solution or ""),
            "file_gate_passed": len(missing_files) == 0,
            "ontology_constraints_available": len(ontology_constraints) > 0,
            "codebase_context_available": has_codebase_context,
            "sprint_context_available": has_sprint_context,
            "grounding_overlap": grounding_overlap,
            "files_grounded": files_grounded,
        }

        issues: list[str] = []
        if not checks["non_empty_solution"]:
            issues.append("Generated solution is empty.")
        if expected_header and not checks["expected_header_present"]:
            issues.append(f"Expected section header missing: {expected_header}")
        if not checks["file_gate_passed"]:
            issues.append("File gate is not satisfied (missing required files).")
        if not checks["codebase_context_available"] and not checks["sprint_context_available"]:
            issues.append("No codebase or sprint context was available — response may be ungrounded.")
        if not checks["files_grounded"]:
            invented = cited_in_solution - cited_in_context
            issues.append(f"File paths cited but not in context: {', '.join(invented)}")
        if not checks["grounding_overlap"] and checks["non_empty_solution"]:
            issues.append("Low word overlap between solution and context — weak grounding.")

        # ── Grounding-based confidence scoring ──
        # Baseline is 0.0. Confidence must be EARNED through verifiable grounding.
        score = 0.0

        # Tier 1: Context availability (max 0.35)
        if has_codebase_context:
            score += 0.20
        if has_sprint_context:
            score += 0.10
        if checks["ontology_constraints_available"]:
            score += 0.05

        # Tier 2: Content grounding (max 0.40)
        if checks["files_grounded"] and cited_in_solution:
            score += 0.20
        if grounding_overlap:
            score += 0.20

        # Tier 3: Structural quality (max 0.15)
        if checks["non_empty_solution"]:
            score += 0.05
        if checks["expected_header_present"]:
            score += 0.05
        if not issues:
            score += 0.05

        # Tier 4: Penalties
        if not has_codebase_context and not has_sprint_context and checks["non_empty_solution"]:
            # Non-empty response with no context = likely hallucination
            score -= 0.30
        if not checks["files_grounded"]:
            score -= 0.20
        solution_len = len((solution or "").split())
        if solution_len > 200 and not has_codebase_context:
            score -= 0.15

        confidence = max(0.0, min(0.99, round(score, 2)))

        return {
            "checks": checks,
            "issues": issues,
            "confidence": confidence,
        }

    def build_response(
        self,
        task_understanding: str,
        ontology_constraints: list[str],
        solution: str,
        validation_report: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "task_understanding": task_understanding,
            "ontology_constraints_used": ontology_constraints,
            "solution": solution,
            "validation_report": validation_report,
            "confidence": float(validation_report.get("confidence", 0.0)),
        }

    def _understand_request(self, request: str, **runtime: Any) -> str:
        text = (request or "").strip()
        if not text:
            return "No request provided."

        is_execution_phase = bool(runtime.get("is_execution_phase"))
        task_type = str(runtime.get("task_type") or self._infer_task_type(text)).upper()
        target_role = self._infer_target_role(text)

        if is_execution_phase:
            return (
                f"Execute stage for {target_role} in {task_type} mode using the latest approved prompt context."
            )
        return f"Prepare a role-aligned {task_type} developer response for {target_role}."

    def _collect_ontology_terms(self, text: str, term_type: str) -> list[str]:
        if term_type == "Class":
            pattern = r"([A-Za-z_][A-Za-z0-9_]*)\s+a\s+owl:Class"
        else:
            pattern = r"([A-Za-z_][A-Za-z0-9_]*)\s+a\s+owl:(?:ObjectProperty|DatatypeProperty|AnnotationProperty)"
        found = re.findall(pattern, text)
        return self._dedupe_nonempty(found, limit=10)

    def _build_llm_fallback(self, request: str, task_type: str, language: str, error_message: str) -> str:
        fallback = self.generator.generate_restructured_prompt(
            request=request,
            task_type=task_type,
            language=language,
        )
        if language == "tr":
            return (
                f"LLM yolu basarisiz oldu ({error_message}). Kural tabanli yedek yanit donuluyor.\n\n"
                "Yeniden Yapilandirilmis Prompt v1\n\n"
                f"{fallback}"
            )
        return (
            f"LLM path failed ({error_message}). Returning rule-based fallback.\n\n"
            "Restructured Prompt v1\n\n"
            f"{fallback}"
        )

    def _build_memory_instruction_block(self, memory_context: str, language: str) -> str:
        if language == "tr":
            return (
                "Proje ve sprint baglam bilgisini kullan:\n"
                f"{memory_context}\n"
                "Uretimde bu baglamla celismeyecek sekilde ilerle."
            )
        return (
            "Use this project and sprint context as evidence:\n"
            f"{memory_context}\n"
            "Do not contradict this context in your response."
        )

    def _localize_common_headers(self, answer: str, language: str) -> str:
        if language != "tr" or not answer:
            return answer
        localized = answer.replace("Restructured Prompt v1", "Yeniden Yapilandirilmis Prompt v1")
        return localized.replace("Execution Plan v1", "Yurutme Plani v1")

    def _infer_strategy(self, request: str) -> str:
        """
        Resolve 'auto' strategy to a concrete generation path based on intent.

        direct — analytic, explanatory, debug, comparison requests
                 → _generate_direct_solution() (LLM answers the question)
        rules  — explicit meta-prompt / restructure requests
                 → deterministic restructured-prompt template
        direct — default (safe: always better to answer than to produce a meta-prompt)
        """
        text = (request or "").lower()

        # Explicit meta-prompt requests only → rules path
        restructure_hints = (
            "rewrite this prompt",
            "restructure this",
            "restructure prompt",
            "turn into execution prompt",
            "turn this into a prompt",
            "execution prompt",
            "zero-time prompt",
            "yeniden yapilandir",
            "prompt yapilandir",
            "sifir zaman prompt",
        )
        if any(hint in text for hint in restructure_hints):
            return "rules"

        # Analytical / direct-answer requests → direct path
        direct_hints = (
            "analyze", "analyse", "explain", "debug", "compare",
            "describe", "what is", "what are", "what does", "what's",
            "how does", "how do", "how is", "how to",
            "why does", "why is", "why",
            "show me", "list", "summarize", "review", "check",
            "diagnose", "trace", "find", "locate", "identify",
            "difference", "vs",
            # Turkish equivalents
            "analiz", "acikla", "incele", "karsilastir",
            "nedir", "nasil", "neden", "goster", "bul", "kontrol",
        )
        if any(hint in text for hint in direct_hints):
            return "direct"

        # Default: direct — avoids the meta-prompt trap on ambiguous input
        return "direct"

    def _infer_task_type(self, user_text: str) -> str:
        text = (user_text or "").lower()
        debug_hints = (
            "debug",
            "bug",
            "fix",
            "hata",
            "pipeline.py",
            "repository",
            "stack trace",
            "traceback",
            "exception",
            "log",
        )
        if any(hint in text for hint in debug_hints):
            return "other"
        if "ontology" in text or ".ttl" in text or "rdf" in text or "owl" in text:
            return "ontology"
        return "other"

    def _infer_target_role(self, source_prompt: str) -> str:
        text = (source_prompt or "").lower()
        if "product owner" in text or re.search(r"\bpo\b", text):
            return "Product Owner Agent"
        if "scrum master" in text:
            return "Scrum Master Agent"
        if "project manager" in text or re.search(r"\bpm\b", text):
            return "Project Manager Agent"
        if "developer" in text:
            return "Developer Agent"
        return "Developer Agent"

    def _extract_constraints(self, source_prompt: str) -> str:
        match = re.search(r"constraints?\s*:\s*(.+)", source_prompt or "", flags=re.I)
        if match:
            return match.group(1).strip()
        return ""

    def _extract_goal_snapshot(self, source_prompt: str) -> str:
        text = source_prompt or ""
        goal_match = re.search(r"goal\s*:\s*(.+)", text, flags=re.I)
        if goal_match:
            return goal_match.group(1).strip()

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            return lines[0][:180]
        return "No explicit goal provided."

    def _dedupe_nonempty(self, values: list[str], limit: int = 5) -> list[str]:
        selected: list[str] = []
        seen: set[str] = set()
        for raw in values:
            cleaned = " ".join(str(raw or "").split()).strip()
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            selected.append(cleaned)
            if len(selected) >= limit:
                break
        return selected

