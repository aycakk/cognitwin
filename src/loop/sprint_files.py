"""loop/sprint_files.py — extract and persist developer-generated code files.

Parses markdown code blocks from developer LLM responses and writes them
to runtime/sprint_runs/{sprint_id}/generated_files/.

Security: all writes are validated to stay within the sprint's generated_files
directory (no path traversal).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME_BASE = _PROJECT_ROOT / "runtime" / "sprint_runs"

# Map common language tags → default filenames when no explicit name is given
_LANG_TO_FILENAME: dict[str, str] = {
    "html":       "index.html",
    "css":        "style.css",
    "javascript": "main.js",
    "js":         "main.js",
    "typescript": "main.ts",
    "ts":         "main.ts",
    "python":     "main.py",
    "py":         "main.py",
    "markdown":   "README.md",
    "md":         "README.md",
    "json":       "data.json",
    "bash":       "run.sh",
    "sh":         "run.sh",
    "sql":        "schema.sql",
}

# Regex: ```optional-filename.ext  OR  ```lang
_FENCE_RE = re.compile(
    r"```[ \t]*([^\n`]*)\n(.*?)```",
    re.DOTALL,
)


def extract_code_blocks(text: str) -> list[dict]:
    """Parse markdown fenced code blocks from developer response text.

    Returns a list of dicts: {"filename": str, "lang": str, "content": str}
    Skips empty blocks.
    """
    blocks: list[dict] = []
    seen_filenames: dict[str, int] = {}

    for m in _FENCE_RE.finditer(text):
        hint    = m.group(1).strip()   # e.g. "html", "index.html", "python"
        content = m.group(2)

        if not content.strip():
            continue

        # Determine filename and language
        if "." in hint and "/" not in hint and not hint.startswith("#"):
            # Treat as explicit filename
            filename = hint
            lang     = hint.rsplit(".", 1)[-1].lower()
        else:
            lang     = hint.lower()
            filename = _LANG_TO_FILENAME.get(lang, f"output.{lang or 'txt'}")

        # Deduplicate filenames within this response
        base = filename
        n = seen_filenames.get(base, 0)
        if n > 0:
            stem, _, ext = base.rpartition(".")
            filename = f"{stem}_{n}.{ext}" if ext else f"{base}_{n}"
        seen_filenames[base] = n + 1

        blocks.append({"filename": filename, "lang": lang, "content": content})

    return blocks


def _safe_path(base: Path, filename: str) -> Optional[Path]:
    """Resolve a filename within base, rejecting any path traversal."""
    # Strip leading slashes/dots to prevent traversal
    clean = Path(filename).name
    if not clean or clean.startswith("."):
        return None
    candidate = (base / clean).resolve()
    try:
        candidate.relative_to(base.resolve())
        return candidate
    except ValueError:
        logger.warning("sprint_files: path traversal rejected: %s", filename)
        return None


def write_developer_files(
    sprint_id: str,
    task_id: str,
    response_text: str,
) -> list[str]:
    """Extract code blocks from response_text and write to generated_files dir.

    Returns list of relative filenames written (e.g. ["index.html", "style.css"]).
    Returns empty list when no code blocks are found.
    """
    blocks = extract_code_blocks(response_text)
    if not blocks:
        logger.debug(
            "sprint_files: no code blocks found for sprint=%s task=%s",
            sprint_id, task_id,
        )
        return []

    # Validate sprint_id contains no path separators
    if "/" in sprint_id or "\\" in sprint_id or ".." in sprint_id:
        logger.error("sprint_files: invalid sprint_id rejected: %s", sprint_id)
        return []

    out_dir = _RUNTIME_BASE / sprint_id / "generated_files"
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for block in blocks:
        dest = _safe_path(out_dir, block["filename"])
        if dest is None:
            continue
        try:
            dest.write_text(block["content"], encoding="utf-8")
            written.append(block["filename"])
            logger.info(
                "sprint_files: wrote %s (sprint=%s task=%s)",
                block["filename"], sprint_id, task_id,
            )
        except OSError as exc:
            logger.error("sprint_files: write failed %s: %s", dest, exc)

    return written


def write_fallback_artefact(
    sprint_id: str,
    goal: str,
    accepted_stories: list[dict],
) -> list[str]:
    """When the developer never produced parsable code blocks, synthesise a
    minimal index.html + manifest.json from the goal and accepted stories so
    the cockpit's Live Preview / Code panes always have something concrete
    to show.

    Returns the list of filenames written (relative to generated_files/).
    """
    if "/" in sprint_id or "\\" in sprint_id or ".." in sprint_id:
        return []

    out_dir = _RUNTIME_BASE / sprint_id / "generated_files"
    out_dir.mkdir(parents=True, exist_ok=True)

    # If files already exist, do not overwrite them.
    existing = [p.name for p in out_dir.iterdir() if p.is_file()]
    if existing:
        return []

    def _esc(s: str) -> str:
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    items = "\n    ".join(
        f"<li><strong>{_esc(s.get('title', 'Story'))}</strong>"
        f"<div style='color:#94a3b8;font-size:13px;margin-top:4px'>"
        f"{_esc(s.get('description', '') or '')}</div></li>"
        for s in (accepted_stories or [])
    ) or "<li><em>No accepted stories yet.</em></li>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{_esc(goal)[:80]}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; background:#0f1117; color:#e2e8f0;
            max-width:760px; margin:40px auto; padding:0 24px; line-height:1.6; }}
    h1   {{ color:#d4ff3a; font-size:22px; margin-bottom:8px; }}
    .sub {{ color:#64748b; font-size:13px; text-transform:uppercase;
            letter-spacing:.12em; margin-bottom:24px; }}
    ul   {{ list-style:none; padding:0; }}
    li   {{ background:#1e2130; border:1px solid #2d3148; border-radius:8px;
            padding:14px 16px; margin-bottom:10px; }}
  </style>
</head>
<body>
  <h1>{_esc(goal)}</h1>
  <div class="sub">Sprint deliverable · auto-generated outline</div>
  <ul>
    {items}
  </ul>
</body>
</html>
"""
    manifest = {
        "sprint_id": sprint_id,
        "goal": goal,
        "accepted_stories": [
            {"title": s.get("title", ""), "story_id": s.get("story_id", "")}
            for s in (accepted_stories or [])
        ],
        "note": "Fallback artefact generated because developer output contained no parsable code blocks.",
    }

    written: list[str] = []
    try:
        (out_dir / "index.html").write_text(html, encoding="utf-8")
        written.append("index.html")
    except OSError as exc:
        logger.error("sprint_files: fallback index.html write failed: %s", exc)

    try:
        import json as _json  # noqa: PLC0415
        (out_dir / "manifest.json").write_text(
            _json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        written.append("manifest.json")
    except OSError as exc:
        logger.error("sprint_files: fallback manifest.json write failed: %s", exc)

    return written


def list_generated_files(sprint_id: str) -> list[dict]:
    """Return metadata + content for all files in the sprint's generated_files dir.

    Returns list of dicts: {path, name, kind, content}
    """
    if "/" in sprint_id or "\\" in sprint_id or ".." in sprint_id:
        return []

    out_dir = _RUNTIME_BASE / sprint_id / "generated_files"
    if not out_dir.exists():
        return []

    _KIND_MAP = {
        ".html": "html",
        ".htm":  "html",
        ".css":  "css",
        ".js":   "javascript",
        ".ts":   "typescript",
        ".py":   "python",
        ".md":   "markdown",
        ".json": "json",
        ".sh":   "bash",
        ".sql":  "sql",
    }

    files: list[dict] = []
    for f in sorted(out_dir.iterdir()):
        if not f.is_file():
            continue
        # Only read files directly inside generated_files (no subdirs for now)
        try:
            f.relative_to(out_dir)
        except ValueError:
            continue
        try:
            content = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            content = ""
        kind = _KIND_MAP.get(f.suffix.lower(), "other")
        files.append({
            "path":    f"generated_files/{f.name}",
            "name":    f.name,
            "kind":    kind,
            "content": content,
        })
    return files


def get_preview_content(sprint_id: str) -> Optional[str]:
    """Return the best previewable HTML content for the sprint.

    Priority:
      1. index.html
      2. any other .html file
      3. simple HTML wrapper around the primary non-HTML file
      4. None when generated_files dir is empty
    """
    if "/" in sprint_id or "\\" in sprint_id or ".." in sprint_id:
        return None

    out_dir = _RUNTIME_BASE / sprint_id / "generated_files"
    if not out_dir.exists():
        return None

    files = list_generated_files(sprint_id)
    if not files:
        return None

    # Priority 1: index.html
    for f in files:
        if f["name"].lower() == "index.html":
            return f["content"]

    # Priority 2: any .html
    for f in files:
        if f["kind"] == "html":
            return f["content"]

    # Priority 3: simple wrapper
    primary = files[0]
    file_list_html = "\n".join(
        f'<li><code>{f["name"]}</code></li>' for f in files
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <style>
    body {{ font-family: monospace; background: #0f1117; color: #e2e8f0; padding: 24px; }}
    h1 {{ color: #d4ff3a; font-size: 16px; text-transform: uppercase; letter-spacing: 0.1em; }}
    ul {{ margin: 12px 0; padding-left: 20px; color: #94a3b8; }}
    pre {{ background: #1e2130; border: 1px solid #2d3148; padding: 16px;
           border-radius: 6px; overflow: auto; font-size: 13px; line-height: 1.5; }}
  </style>
</head>
<body>
  <h1>Generated Files</h1>
  <ul>{file_list_html}</ul>
  <h1>{primary["name"]}</h1>
  <pre>{primary["content"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")}</pre>
</body>
</html>"""
