"""
main_cli.py — CogniTwin Student Agent CLI  (v2.2)

All ZT4SWE pipeline logic lives in src/services/api/pipeline.py.
This file contains only the interactive REPL and display helpers.
"""

from __future__ import annotations

import os
import sys

# Force UTF-8 stdout so box-drawing chars and Turkish text render correctly
# on Windows terminals that default to a legacy code page (e.g. cp1254).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is on sys.path for src.* imports
_src_dir      = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_src_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.services.api.pipeline import (
    CHROMA,
    VECTOR_MEM,
    VECTOR_TOP_K,
    CHROMA_PATH,
    ONTOLOGY_AGENT_ROLES,
    _get_ontology_graph,
    build_ontology_context,
    evaluate_all_gates,
    run_pipeline,
    build_blindspot_block,
)

VALID_ROLES = list(ONTOLOGY_AGENT_ROLES.keys())


# ─────────────────────────────────────────────────────────────────────────────
#  DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def print_gate_report(gate_report: dict) -> None:
    conj = "✅ PASS" if gate_report["conjunction"] else "❌ FAIL"
    print(f"\n{'─' * 64}")
    print(f"  GATE ARRAY — C1∧C2∧…∧C8 = {conj}")
    print(f"{'─' * 64}")
    for gate, info in gate_report["gates"].items():
        status = "✅ PASS" if info["pass"] else "❌ FAIL"
        ev     = info["evidence"][:70]
        print(f"  {gate}: {status}  |  {ev}")
    print(f"{'─' * 64}\n")


def print_startup_banner() -> None:
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   COGNITWIN — Student Agent CLI  (v2.2)                      ║")
    print("║   Conjunctive Gate: C1∧C2∧C3∧C4∧C5∧C6∧C7∧C8               ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    if CHROMA is not None:
        try:
            count = CHROMA.collection.count()
            print(f"  ✅ ChromaDB     : {count} records @ '{CHROMA_PATH}' (k={VECTOR_TOP_K})")
        except Exception:
            print(f"  ✅ ChromaDB     : connected @ '{CHROMA_PATH}' (k={VECTOR_TOP_K})")
    else:
        print("  ⚠  ChromaDB     : unavailable")

    g = _get_ontology_graph()
    if g is not None:
        print(f"  ✅ Ontoloji     : {len(g)} triple (cognitwin-upper + student)")
    else:
        print("  ⚠  Ontoloji     : unavailable — install rdflib and check ontologies/")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

def ask() -> None:
    print_startup_banner()
    print(f"\n  Geçerli roller : {', '.join(VALID_ROLES)}")
    print("  Komutlar       : /exit | /gates | /role <ROL> | /context\n")

    last_gate_report: dict = {}
    current_role = "StudentAgent"

    while True:
        try:
            q = input(f"[{current_role}] Soru > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nÇıkılıyor…")
            break

        if not q:
            continue

        if q.lower() in ("/exit", "exit", "quit"):
            print("Oturum sonlandırıldı.")
            break

        if q.lower() == "/gates":
            if last_gate_report:
                print_gate_report(last_gate_report)
            else:
                print("Henüz bir gate raporu yok.\n")
            continue

        if q.lower() == "/context":
            preview_q = input("  Önizleme sorgusu > ").strip()
            ctx, empty = VECTOR_MEM.retrieve(preview_q, k=3)
            print(ctx if not empty else "  (boş sonuç)\n")
            continue

        if q.lower().startswith("/role "):
            requested = q.split(" ", 1)[1].strip()
            if requested in VALID_ROLES:
                current_role = requested
                print(f"  ✅ Rol: {current_role}\n")
            else:
                print(f"  ❌ Geçersiz rol. Seçenekler: {', '.join(VALID_ROLES)}\n")
            continue

        # ── Main pipeline ─────────────────────────────────────────────────────
        response = run_pipeline(q, agent_role=current_role)

        # Gate report for /gates display (re-retrieves context for evaluation)
        vec_ctx, vec_empty = VECTOR_MEM.retrieve(q, k=VECTOR_TOP_K)
        last_gate_report   = evaluate_all_gates(
            response, vec_ctx, vec_empty, current_role, []
        )

        print(f"\nYanıt:\n{response}\n")

        conj       = last_gate_report["conjunction"]
        symbol     = "✅" if conj else "❌"
        fail_gates = [k for k, v in last_gate_report["gates"].items() if not v["pass"]]

        if conj:
            print(f"  {symbol} Tüm doğrulama kapıları geçti (C1–C8).\n")
        else:
            print(f"  {symbol} Başarısız kapılar: {', '.join(fail_gates)} — /gates ile detay.\n")


if __name__ == "__main__":
    ask()
