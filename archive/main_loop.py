import os
import sys

# Proje kök dizinini Python path'ine ekler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.classifiers.intent_classifier import IntentClassifier
from src.agents.router import Router
from src.agents.context_builder import HybridContextBuilder
from src.services.llm_service import LLMService


def main_loop():
    # Ana akış bileşenlerini başlatır
    classifier = IntentClassifier()
    router = Router()
    context_builder = HybridContextBuilder()
    llm_service = LLMService()

    while True:
        # Kullanıcıdan soru alır
        question = input("User: ").strip()

        # Çıkış komutlarını kontrol eder
        if question.lower() in ["exit", "quit"]:
            print("System terminated.")
            break

        # Sorunun intent türünü belirler
        intent = classifier.classify(question)
        print(f"[INTENT] {intent}")

        # Router ile uygun yola yönlendirir
        route_result = router.route(intent, question)

        if route_result["route"] == "ONTOLOGY":
            # DATA sorularında doğrudan ontoloji cevabını döndürür
            answer = route_result["answer"]
        else:
            # COMMENT sorularında önce context üretir sonra LLM'e gönderir
            context = context_builder.build_context(question)
            answer = llm_service.ask(context)

        print(f"Agent: {answer}")


if __name__ == "__main__":
    main_loop()