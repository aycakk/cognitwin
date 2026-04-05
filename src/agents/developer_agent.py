from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent
from typing import Iterable


@dataclass(frozen=True)
class RoleBlueprint:
    role_name: str
    purpose: str
    responsibilities: list[str]
    concepts: list[str]
    outputs: list[str]
    competency_questions: list[str]
    validation_rules: list[str]


class DeveloperAgent:
    """
    Team-internal builder agent for COGNITWIN.

    This agent does not answer end-user questions. It helps the team turn a
    target role into a structured delivery packet containing ontology concepts,
    competency questions, validation ideas, and a compact agent specification.
    """

    def __init__(self) -> None:
        self.agent_name = "COGNITWIN Developer Agent"
        self._role_library = self._build_role_library()
        self._required_files = {
            "ontology": [
                "core_v1_1_0.ttl",
                "meta_v6_1_0.ttl",
                "oqc_v2_1_0.ttl",
                "redo_v2_4_6.ttl",
                "redo_v2_shacl_shapes.ttl",
                "chapter 3 brute force and exhaustive search.txt",
                "sen4012_brute_force_research_report.html",
            ],
            "other": [
                "redo_v2_4_6.ttl",
                "redo_v2_shacl_shapes.ttl",
            ],
        }

    def supported_roles(self) -> list[str]:
        return sorted(self._role_library.keys())

    def build_role_packet(self, target_role: str, constraints: str = "") -> dict:
        normalized_role = self._normalize_role(target_role)
        blueprint = self._role_library.get(normalized_role, self._generic_blueprint(normalized_role))

        constraints = constraints.strip() or "MVP scope; keep the first version minimal and testable."

        return {
            "developer_agent": self.agent_name,
            "target_role": blueprint.role_name,
            "purpose": blueprint.purpose,
            "responsibilities": blueprint.responsibilities,
            "ontology_concepts": blueprint.concepts,
            "expected_outputs": blueprint.outputs,
            "competency_questions": blueprint.competency_questions,
            "validation_rules": blueprint.validation_rules,
            "constraints": constraints,
            "next_steps": self._build_next_steps(blueprint.role_name),
        }

    def render_role_packet(self, target_role: str, constraints: str = "") -> str:
        packet = self.build_role_packet(target_role, constraints)

        sections = [
            f"# {packet['target_role']} Design Packet",
            f"## Purpose\n{packet['purpose']}",
            "## Core Responsibilities\n" + self._render_bullets(packet["responsibilities"]),
            "## Ontology Concepts\n" + self._render_bullets(packet["ontology_concepts"]),
            "## Expected Outputs\n" + self._render_bullets(packet["expected_outputs"]),
            "## Competency Questions\n" + self._render_bullets(packet["competency_questions"]),
            "## Validation Rules\n" + self._render_bullets(packet["validation_rules"]),
            f"## Constraints\n- {packet['constraints']}",
            "## Next Steps\n" + self._render_bullets(packet["next_steps"]),
        ]
        return "\n\n".join(sections)

    def build_agent_spec(self, target_role: str, constraints: str = "") -> str:
        packet = self.build_role_packet(target_role, constraints)

        return dedent(
            f"""
            # Agent Specification

            ## Agent Name
            {packet['target_role']}

            ## Role in COGNITWIN
            Domain-facing role agent generated with support from {packet['developer_agent']}.

            ## Purpose
            {packet['purpose']}

            ## Inputs
            - Role-specific task or request
            - Relevant ontology context
            - Team or project constraints
            - Human feedback for iteration

            ## Outputs
            {self._render_bullets(packet['expected_outputs'])}

            ## Core Responsibilities
            {self._render_bullets(packet['responsibilities'])}

            ## Critical Rules
            - Reuse existing ontology concepts before proposing new ones.
            - Keep the MVP narrow and demonstrable.
            - Separate project facts from assumptions.
            - Produce outputs that can be checked by competency questions or validation rules.

            ## Starter Validation
            {self._render_bullets(packet['validation_rules'])}
            """
        ).strip()

    def required_files(self, task_type: str = "ontology") -> list[str]:
        normalized = self._normalize_task_type(task_type)
        return list(self._required_files[normalized])

    def check_uploaded_files(self, uploaded_files: Iterable[str], task_type: str = "ontology") -> list[str]:
        normalized = self._normalize_task_type(task_type)
        uploaded_set = {self._normalize_filename(name) for name in uploaded_files}
        required = self._required_files[normalized]
        return [name for name in required if self._normalize_filename(name) not in uploaded_set]

    def build_restructured_execution_prompt(self, zero_time_prompt: str, task_type: str = "ontology") -> str:
        normalized = self._normalize_task_type(task_type)
        required = self._required_files[normalized]
        required_text = "\n".join(f"- '{name}'" for name in required)

        return dedent(
            f"""
            You are a high-integrity execution orchestrator.

            TASK MODE: {normalized.upper()}

            STEP 0 - MANDATORY PRE-FLIGHT FILE CHECK
            1. List all files currently uploaded by the user.
            2. Compare them against the required file list below.
            3. If any required file is missing, STOP and ask the user to upload the missing files.
            4. Do not start execution until all required files are present.

            Required files:
            {required_text}

            STEP 1 - PROMPT RESTRUCTURING PHASE
            Rewrite the original zero-time prompt into a robust execution prompt using prompt engineering best practices:
            - clear goal and scope
            - explicit assumptions and constraints
            - staged workflow
            - strict deliverable format
            - quality and evidence criteria
            - fail-fast behavior on missing inputs
            Output this as "Restructured Prompt v1".

            STEP 2 - EXECUTION PHASE
            Execute only after user confirmation ("run it" / equivalent).

            VERIFICATION SAFEGUARD (MUST BE ENFORCED)
            Keep REDO in safeguard and activate all verification stages via:
            - a conjunctive gate (C1 and C2 and ... and C8)
            - an anti-sycophancy protocol with negative exemplars
            - 8 enumerated dimensions with evidence sources and pass/fail rules
            - a 4-stage verification pipeline with gates
            - REDO checksum enforcement
            - mandatory BlindSpot disclosure

            EVIDENCE POLICY
            - All verification evidence must come from real runtime execution:
              headless browser for HTML/JS, Node for server-side, Python for scripts.
            - String-matching, grep-based, and static text-presence checks are prohibited as pass/fail evidence.
            - Those checks may be used only for pre-flight sanity checks.
            - Build and run the test harness before declaring any fix complete.
            - Show all test failures to the user before attempting fixes.
            - Keep the fix -> test -> fix cycle visible.

            Original zero-time prompt:
            \"\"\"
            {zero_time_prompt.strip()}
            \"\"\"
            """
        ).strip()

    def build_system_prompt_template(self, task_type: str = "ontology") -> str:
        normalized = self._normalize_task_type(task_type)
        files = self._required_files[normalized]
        files_text = "\n".join(f"- {name}" for name in files)
        return dedent(
            f"""
            SYSTEM PROMPT TEMPLATE - COGNITWIN VERIFIED EXECUTION MODE

            Role:
            You are an execution-grade prompt orchestrator for {normalized} tasks.

            Hard rule 1:
            Before execution, check file completeness and request missing files.

            Required files for this mode:
            {files_text}

            Hard rule 2:
            Work in two stages:
            Stage A - Restructure the provided zero-time prompt with prompt-engineering best practices.
            Stage B - Execute only after explicit user confirmation.

            Hard rule 3:
            REDO safeguards are mandatory:
            C1..C8 conjunctive gate, anti-sycophancy protocol, 8 dimensions with evidence and pass/fail,
            4-stage verification pipeline, REDO checksum, BlindSpot disclosure.

            Hard rule 4:
            Verification evidence must come from runtime execution.
            Static checks are not valid pass/fail evidence.

            Hard rule 5:
            Build and run tests before claiming completion, show failures first, then run fix -> test -> fix transparently.
            """
        ).strip()

    def generate_restructured_prompt(
        self,
        request: str,
        task_type: str = "ontology",
        language: str = "en",
    ) -> str:
        normalized_language = (language or "en").strip().lower()
        if normalized_language == "tr":
            return self._build_restructured_execution_prompt_tr(
                zero_time_prompt=request,
                task_type=task_type,
            )
        return self.build_restructured_execution_prompt(
            zero_time_prompt=request,
            task_type=task_type,
        )

    def generate_execution_plan(
        self,
        *,
        task_type: str,
        role_packet: dict,
        goal_snapshot: str,
        memory_context: str,
        language: str = "en",
    ) -> str:
        normalized_language = (language or "en").strip().lower()
        if normalized_language == "tr":
            return dedent(
                f"""
                Yurutme Plani v1

                Asama 2 onaylandi. Yurutme en guncel kullanici baglami ile baslatildi.

                Gorev modu: {task_type.upper()}
                Hedef rol: {role_packet['target_role']}
                Hedef ozeti: {goal_snapshot}

                Anlik teslimatlar:
                {self._render_bullets(role_packet['expected_outputs'])}

                Dogrulama kontrol listesi:
                {self._render_bullets(role_packet['validation_rules'])}

                Siradaki adimlar:
                {self._render_bullets(role_packet['next_steps'])}

                Dijital ayak izi bellek ozeti:
                {memory_context}
                """
            ).strip()

        return dedent(
            f"""
            Execution Plan v1

            Stage 2 confirmed. Execution has started using the latest user task context.

            Task mode: {task_type.upper()}
            Target role: {role_packet['target_role']}
            Goal snapshot: {goal_snapshot}

            Immediate deliverables:
            {self._render_bullets(role_packet['expected_outputs'])}

            Validation checklist:
            {self._render_bullets(role_packet['validation_rules'])}

            Next steps:
            {self._render_bullets(role_packet['next_steps'])}

            Footprint memory snapshot:
            {memory_context}
            """
        ).strip()

    def _build_restructured_execution_prompt_tr(self, zero_time_prompt: str, task_type: str = "ontology") -> str:
        required = self.required_files(task_type)
        required_text = "\n".join(f"- '{name}'" for name in required)
        normalized = task_type.strip().upper()

        return dedent(
            f"""
            Sen yuksek dogruluklu bir yurutme orkestratorusun.

            GOREV MODU: {normalized}

            ADIM 0 - ZORUNLU ON KONTROL DOSYA KONTROLU
            1. Kullanicinin yukledigi dosyalari listele.
            2. Asagidaki zorunlu dosya listesiyle karsilastir.
            3. Eksik dosya varsa DUR ve kullanicidan yuklemesini iste.
            4. Tum dosyalar tamamlanmadan yurutmeye baslama.

            Zorunlu dosyalar:
            {required_text}

            ADIM 1 - PROMPT YENIDEN YAPILANDIRMA
            Sifir-zaman promptunu guclu bir yurutme promptuna cevir:
            - net hedef ve kapsam
            - acik varsayimlar ve kisitlar
            - asamali is akisi
            - net teslim formati
            - kalite ve kanit kriterleri
            - eksik girdide fail-fast davranisi
            Cikti basligi: "Yeniden Yapilandirilmis Prompt v1".

            ADIM 2 - YURUTME
            Sadece kullanici onayi ("run it" veya esdegeri) sonrasinda calistir.

            DOGRULAMA KORUMASI (ZORUNLU)
            REDO korumalarini uygula:
            - C1..C8 birlesik gecit
            - anti-sycophancy protokolu
            - 8 boyutlu kanit ve pass/fail kurallari
            - 4 asamali dogrulama hatti
            - REDO checksum
            - BlindSpot bildirimi

            KANIT POLITIKASI
            - Kanitlar calisan ortamdan gelmeli (browser/node/python).
            - Sadece metin arama/grep pass-fail kaniti olamaz.
            - Test hatti calismadan is tamamlandi denemez.

            Orijinal prompt:
            \"\"\"
            {zero_time_prompt.strip()}
            \"\"\"
            """
        ).strip()

    def _build_next_steps(self, role_name: str) -> list[str]:
        return [
            f"Create or extend a TTL file for {role_name}.",
            "Add at least three competency questions tied to the role workflow.",
            "Define one or two validation rules for invalid or incomplete outputs.",
            "Prepare a small demo scenario with sample input and expected response.",
        ]

    def _render_bullets(self, items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items)

    def _normalize_role(self, target_role: str) -> str:
        cleaned = target_role.strip().lower().replace("_", " ").replace("-", " ")
        aliases = {
            "pm": "project manager agent",
            "po": "product owner agent",
            "product owner": "product owner agent",
            "scrum master": "scrum master agent",
            "developer": "developer agent",
        }
        return aliases.get(cleaned, cleaned)

    def _normalize_filename(self, file_name: str) -> str:
        normalized = file_name.strip().lower().replace("\\", "/")
        return normalized.split("/")[-1]

    def _normalize_task_type(self, task_type: str) -> str:
        cleaned = task_type.strip().lower()
        if cleaned in self._required_files:
            return cleaned
        return "ontology"

    def _generic_blueprint(self, normalized_role: str) -> RoleBlueprint:
        role_name = normalized_role.title()
        return RoleBlueprint(
            role_name=role_name,
            purpose=f"Support the {role_name} workflow inside COGNITWIN with structured, ontology-aligned outputs.",
            responsibilities=[
                "Interpret role-specific requests.",
                "Produce a minimal task breakdown.",
                "Suggest ontology concepts and role outputs.",
            ],
            concepts=["Task", "Priority", "Dependency", "Constraint", "Deliverable"],
            outputs=["Task summary", "Ontology draft", "Validation note"],
            competency_questions=[
                f"What decisions must the {role_name} make?",
                f"What artifacts should the {role_name} produce?",
                f"Which dependencies block the {role_name} workflow?",
            ],
            validation_rules=[
                "Every output must reference at least one known role task.",
                "Every plan must include an explicit next step.",
            ],
        )

    def _build_role_library(self) -> dict[str, RoleBlueprint]:
        return {
            "developer agent": RoleBlueprint(
                role_name="Developer Agent",
                purpose="Turn implementation requests into technical task plans, ontology changes, validation ideas, and build-ready developer notes.",
                responsibilities=[
                    "Break a feature or role request into implementable technical tasks.",
                    "Suggest ontology classes, properties, and relations for new features.",
                    "Draft competency questions and validation rules for the implementation.",
                    "Prepare concise developer-facing notes for the next coding step.",
                ],
                concepts=[
                    "ImplementationTask",
                    "OntologyClass",
                    "OntologyProperty",
                    "CompetencyQuestion",
                    "ValidationRule",
                    "Deliverable",
                ],
                outputs=[
                    "Task breakdown",
                    "Ontology draft",
                    "Competency question set",
                    "Validation checklist",
                    "Developer notes",
                ],
                competency_questions=[
                    "Which ontology classes are required for this feature?",
                    "What should be validated before merging the change?",
                    "What artifacts must the developer produce for completion?",
                ],
                validation_rules=[
                    "Every implementation task must reference a deliverable.",
                    "Every ontology change must include at least one competency question.",
                    "Every generated plan must mention a validation step.",
                ],
            ),
            "product owner agent": RoleBlueprint(
                role_name="Product Owner Agent",
                purpose="Transform feature ideas into refined backlog items, acceptance criteria, and sprint-ready work packages.",
                responsibilities=[
                    "Refine backlog items into clear user stories.",
                    "Define acceptance criteria and scope boundaries.",
                    "Prioritize work based on value, dependency, and MVP constraints.",
                    "Flag missing requirements before sprint planning.",
                ],
                concepts=[
                    "BacklogItem",
                    "UserStory",
                    "AcceptanceCriteria",
                    "Priority",
                    "Dependency",
                    "SprintGoal",
                ],
                outputs=[
                    "Refined backlog item",
                    "Acceptance criteria set",
                    "Priority recommendation",
                    "Sprint goal suggestion",
                ],
                competency_questions=[
                    "Which backlog items are ready for sprint planning?",
                    "What acceptance criteria define completion for a story?",
                    "Which items carry unresolved dependency or requirement risks?",
                ],
                validation_rules=[
                    "Every backlog item must have acceptance criteria.",
                    "Every sprint-ready story must include a priority and owner.",
                    "Blocked items cannot be marked ready without a dependency note.",
                ],
            ),
            "scrum master agent": RoleBlueprint(
                role_name="Scrum Master Agent",
                purpose="Support sprint flow by summarizing standups, surfacing blockers, and keeping team work aligned with the sprint goal.",
                responsibilities=[
                    "Summarize daily updates in a clean, team-readable format.",
                    "Identify blockers, risks, and missing follow-ups.",
                    "Keep sprint work aligned with the sprint goal and team capacity.",
                    "Prepare review and retrospective prompts for the team.",
                ],
                concepts=[
                    "Sprint",
                    "StandupUpdate",
                    "Blocker",
                    "Risk",
                    "TeamMember",
                    "RetrospectiveNote",
                ],
                outputs=[
                    "Standup summary",
                    "Blocker list",
                    "Risk note",
                    "Review summary",
                    "Retro prompt set",
                ],
                competency_questions=[
                    "Which blockers are currently preventing sprint progress?",
                    "Which tasks are drifting away from the sprint goal?",
                    "What should be discussed in the next review or retrospective?",
                ],
                validation_rules=[
                    "Every blocker must be linked to a task or owner.",
                    "Every standup summary must distinguish progress from risk.",
                    "Sprint notes must reference the current sprint goal.",
                ],
            ),
            "project manager agent": RoleBlueprint(
                role_name="Project Manager Agent",
                purpose="Coordinate delivery by converting project needs into scoped tasks, milestone plans, and dependency-aware execution notes.",
                responsibilities=[
                    "Translate project goals into phased task plans.",
                    "Track dependencies, milestones, and execution risks.",
                    "Maintain delivery notes for the team.",
                    "Recommend the next highest-value implementation step.",
                ],
                concepts=[
                    "Milestone",
                    "Timeline",
                    "Dependency",
                    "Risk",
                    "Deliverable",
                    "Owner",
                ],
                outputs=[
                    "Milestone plan",
                    "Dependency map",
                    "Risk summary",
                    "Execution notes",
                ],
                competency_questions=[
                    "What deliverables belong to the current milestone?",
                    "Which dependencies threaten the timeline?",
                    "What is the next implementable project step?",
                ],
                validation_rules=[
                    "Every milestone must list at least one deliverable.",
                    "Every risk entry must include impact or mitigation.",
                    "Every task plan must include an owner or responsible role.",
                ],
            ),
        }
