import os
import sys

# Proje kök dizinini Python path'ine ekler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.services.ontology_service import OntologyService
from src.services.llm_service import LLMService


class Router:
    # Intent sonucuna göre isteği uygun işleme hattına yönlendirir

    def __init__(self):
        self.ontology_service = OntologyService()
        self.llm_service = LLMService()

    def route(self, intent: str, question: str) -> dict:
        # DATA sorularında doğrudan ontoloji katmanını kullanır
        if intent == "DATA":
            print("[ROUTER] Ontology route activated. LLM skipped.")

            ontology_answer = self.ontology_service.query(question)

            return {
                "route": "ONTOLOGY",
                "llm_used": False,
                "answer": ontology_answer
            }

        # COMMENT sorularında LLM katmanına yönlendirir
        print("[ROUTER] LLM route activated.")

        return {
            "route": "LLM",
            "llm_used": True,
            "answer": None
        }


if __name__ == "__main__":
    router = Router()

    print(router.route("DATA", "Notumu getir"))
    print(router.route("DATA", "Vize ne zaman"))
    print(router.route("COMMENT", "Bu derse nasıl çalışmalıyım"))