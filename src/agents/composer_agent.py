"""agents/composer_agent.py — Compose multi-agent outputs into one response.

First working version:
  - Accepts outputs from multiple agents.
  - Filters empty / whitespace / known-useless responses.
  - Removes obvious duplicates.
  - Detects basic contradictions and reports them.
  - Returns a structured final response.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence


_USELESS_OUTPUTS = {
    "",
    "n/a",
    "none",
    "null",
    "no answer",
    "bunu hafızamda bulamadım.",
    "bunu hafizamda bulamadim.",
}

_CLAIM_PATTERNS: list[tuple[str, list[str], list[str]]] = [
    (
        "completion-status",
        ["tamamlandı", "done", "completed"],
        ["tamamlanmadı", "tamamlanmadi", "not done", "incomplete", "bitmedi"],
    ),
    (
        "existence-status",
        [" var", " mevcut", " exists"],
        [" yok", " mevcut değil", " mevcut degil", "not found", "does not exist"],
    ),
    (
        "approval-status",
        ["onaylandı", "onaylandi", "accepted", "approved"],
        ["reddedildi", "rejected", "declined"],
    ),
    (
        "result-status",
        ["başarılı", "basarili", "successful"],
        ["başarısız", "basarisiz", "failed", "error"],
    ),
]


@dataclass
class NormalizedOutput:
    agent: str
    text: str
    normalized: str
    index: int


@dataclass
class HandoffResult:
    """Result of a Composer gate validation between agent handoffs."""
    ok: bool          # True → proceed to next agent with payload
    from_agent: str
    to_agent: str
    reason: str       # human-readable explanation
    payload: str      # cleaned text to forward (empty when ok=False)


class ComposerAgent:
    """Simple, extensible output composer for multi-agent pipelines."""

    def normalize_output(self, output: Any, index: int = 0) -> NormalizedOutput | None:
        """Normalize a raw output item into a common shape."""
        if output is None:
            return None

        agent = "UnknownAgent"
        text = ""

        if isinstance(output, str):
            text = output
        elif isinstance(output, dict):
            agent = str(
                output.get("agent")
                or output.get("agent_role")
                or output.get("role")
                or output.get("name")
                or agent
            )
            text = str(
                output.get("draft")
                or output.get("output")
                or output.get("response")
                or output.get("answer")
                or output.get("content")
                or output.get("text")
                or ""
            )
        else:
            # AgentResponse-like object support without importing schemas here.
            if hasattr(output, "agent_role"):
                role_value = getattr(output.agent_role, "value", output.agent_role)
                agent = str(role_value or agent)
            if hasattr(output, "draft"):
                text = str(getattr(output, "draft") or "")
            elif hasattr(output, "content"):
                text = str(getattr(output, "content") or "")
            else:
                text = str(output)

        cleaned = self._clean_text(text)
        if not cleaned:
            return None
        if cleaned.lower() in _USELESS_OUTPUTS:
            return None

        return NormalizedOutput(
            agent=agent,
            text=cleaned,
            normalized=self._normalize_for_dedup(cleaned),
            index=index,
        )

    def detect_conflicts(self, outputs: Sequence[NormalizedOutput]) -> list[str]:
        """Detect simple, explicit contradictions between agent claims."""
        claims: dict[tuple[str, str], dict[bool, list[str]]] = {}

        for output in outputs:
            for sentence in self._split_sentences(output.text):
                sentence_norm = self._normalize_for_dedup(sentence)
                if not sentence_norm:
                    continue
                sentence_low = f" {sentence_norm} "
                for claim_type, positives, negatives in _CLAIM_PATTERNS:
                    polarity = None
                    # Check negative markers first so "not done" is not
                    # misclassified as positive via the "done" token.
                    if any(token in sentence_low for token in negatives):
                        polarity = False
                    elif any(token in sentence_low for token in positives):
                        polarity = True
                    if polarity is None:
                        continue

                    base_key = sentence_low
                    for token in positives + negatives:
                        base_key = base_key.replace(token, " <state> ")
                    base_key = self._normalize_for_dedup(base_key)
                    if not base_key:
                        continue

                    bucket = claims.setdefault((claim_type, base_key), {True: [], False: []})
                    bucket[polarity].append(output.agent)

        warnings: list[str] = []
        for (claim_type, base_key), buckets in claims.items():
            positive_agents = sorted(set(buckets[True]))
            negative_agents = sorted(set(buckets[False]))
            if positive_agents and negative_agents:
                warnings.append(
                    f"Conflict ({claim_type}): '{base_key}' "
                    f"-> positive: {', '.join(positive_agents)} | "
                    f"negative: {', '.join(negative_agents)}"
                )
        return warnings

    def merge_outputs(self, outputs: Sequence[NormalizedOutput]) -> list[NormalizedOutput]:
        """Merge outputs by dropping obvious duplicates (normalized equality)."""
        seen: set[str] = set()
        merged: list[NormalizedOutput] = []
        for output in outputs:
            if output.normalized in seen:
                continue
            seen.add(output.normalized)
            merged.append(output)
        return merged

    def format_final_response(
        self,
        *,
        summary: str,
        key_points: list[str],
        warnings: list[str],
        final_answer: str,
    ) -> str:
        """Render the required structured response format."""
        key_points_block = "\n".join(f"- {point}" for point in key_points) if key_points else "- None"
        warnings_block = "\n".join(f"- {warning}" for warning in warnings) if warnings else "- None detected."
        return (
            "Summary:\n"
            f"{summary}\n\n"
            "Key Points:\n"
            f"{key_points_block}\n\n"
            "Warnings or Conflicts:\n"
            f"{warnings_block}\n\n"
            "Final Answer:\n"
            f"{final_answer}"
        )

    def compose(self, raw_outputs: Sequence[Any]) -> dict[str, Any]:
        """Compose multiple raw outputs into a single structured response."""
        normalized: list[NormalizedOutput] = []
        for idx, item in enumerate(raw_outputs):
            parsed = self.normalize_output(item, index=idx)
            if parsed is not None:
                normalized.append(parsed)

        merged = self.merge_outputs(normalized)
        warnings = self.detect_conflicts(merged)

        if not merged:
            final_answer = "No meaningful agent output was provided."
            summary = "0 useful outputs were available; nothing could be merged."
            key_points: list[str] = []
        elif warnings:
            summary = (
                f"Collected {len(merged)} useful outputs, but detected "
                f"{len(warnings)} contradiction(s)."
            )
            key_points = [self._first_sentence(entry.text) for entry in merged]
            evidence_lines = [f"- [{entry.agent}] {entry.text}" for entry in merged]
            final_answer = (
                "Conflicting claims were detected across agent outputs. "
                "A single definitive answer cannot be produced safely.\n"
                "Evidence:\n"
                + "\n".join(evidence_lines)
            )
        else:
            lines = [f"[{entry.agent}] {entry.text}" for entry in merged]
            final_answer = "\n".join(lines)
            summary = (
                f"Merged {len(merged)} useful outputs from {len(raw_outputs)} inputs "
                f"after filtering empty/useless entries and duplicates."
            )
            key_points = [self._first_sentence(entry.text) for entry in merged]

        response_text = self.format_final_response(
            summary=summary,
            key_points=key_points,
            warnings=warnings,
            final_answer=final_answer,
        )
        return {
            "response_text": response_text,
            "useful_count": len(normalized),
            "merged_count": len(merged),
            "conflicts": warnings,
        }

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r"\s+", " ", (text or "")).strip()
        return text

    @staticmethod
    def _normalize_for_dedup(text: str) -> str:
        lowered = text.lower().strip()
        lowered = re.sub(r"[^\w\s\-]", " ", lowered, flags=re.UNICODE)
        lowered = re.sub(r"\s+", " ", lowered)
        return lowered.strip()

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        parts = re.split(r"[.!?\n]+", text)
        return [p.strip() for p in parts if p.strip()]

    def validate_handoff(self, from_agent: str, text: str, to_agent: str) -> "HandoffResult":
        """Lightweight Composer gate between agent handoffs. No LLM call.

        Checks whether the outgoing text is usable before passing it to the
        next agent.  Called explicitly in agile_workflow.py at every hop so
        that Composer is an active coordinator, not a passive end-of-chain merger.
        """
        cleaned = self._clean_text(text)
        if not cleaned or cleaned.lower() in _USELESS_OUTPUTS or len(cleaned) < 15:
            return HandoffResult(
                ok=False,
                from_agent=from_agent,
                to_agent=to_agent,
                reason=(
                    f"{from_agent} çıktısı yetersiz veya boş — "
                    f"{to_agent} orijinal istek bağlamıyla devam edecek."
                ),
                payload="",
            )
        return HandoffResult(
            ok=True,
            from_agent=from_agent,
            to_agent=to_agent,
            reason=f"{from_agent} → {to_agent} geçişi onaylandı ({len(cleaned)} karakter).",
            payload=cleaned,
        )

    def _first_sentence(self, text: str) -> str:
        sentences = self._split_sentences(text)
        return sentences[0] if sentences else text
