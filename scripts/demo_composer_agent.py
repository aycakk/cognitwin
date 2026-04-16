"""Minimal demo for ComposerAgent usage."""

from __future__ import annotations

from src.agents.composer_agent import ComposerAgent


def main() -> None:
    agent_outputs = [
        {"agent": "DeveloperAgent", "draft": "API endpoint implementation completed."},
        {"agent": "ScrumMasterAgent", "draft": "T-014 tamamlandı."},
        {"agent": "ProductOwnerAgent", "draft": "T-014 tamamlanmadı; acceptance criteria eksik."},
        {"agent": "StudentAgent", "draft": " "},  # filtered as empty
    ]

    composer = ComposerAgent()
    result = composer.compose(agent_outputs)
    print(result["response_text"])


if __name__ == "__main__":
    main()

