import os
from owlready2 import get_ontology


class OntologyService:
    # Ontoloji katmanını yöneten servis sınıfı

    def __init__(self):
        # Ontology dosyasının tam yolunu oluşturur
        base_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../ontologies")
        )
        student_ontology_path = os.path.join(base_path, "student_ontology.ttl")

        # Ontolojiyi Turtle formatında yükler
        self.onto = get_ontology(student_ontology_path).load(format="turtle")

        # Gerekli sınıf ve property referanslarını alır
        self.grade_prop = self.onto.search_one(iri="*grade")
        self.exam_date_prop = self.onto.search_one(iri="*hasExamDate")
        self.status_prop = self.onto.search_one(iri="*status")
        self.is_masked_prop = self.onto.search_one(iri="*isMasked")

    def check_security(self, individual) -> bool:
        # Bireyin maskeli olup olmadığını kontrol eder
        if self.is_masked_prop is None:
            return False

        values = self.is_masked_prop[individual]
        if values:
            return bool(values[0])

        return False

    def get_student_grade(self) -> str:
        # Öğrenci not bilgisini ontolojiden çeker
        if self.grade_prop is None:
            return "Grade property not found"

        for individual, values in self.grade_prop.get_relations():
            if values:
                if self.check_security(individual):
                    return "[PROTECTED]"
                return str(values)

        return "Grade not found"

    def get_exam_date(self) -> str:
        # Sınav tarihini ontolojiden çeker
        if self.exam_date_prop is None:
            return "Exam date property not found"

        for individual, values in self.exam_date_prop.get_relations():
            if values:
                if self.check_security(individual):
                    return "[PROTECTED]"
                return str(values)

        return "Exam date not found"

    def get_task_status(self) -> str:
        # Görev durumunu ontolojiden çeker
        if self.status_prop is None:
            return "Status property not found"

        for individual, values in self.status_prop.get_relations():
            if values:
                if self.check_security(individual):
                    return "[PROTECTED]"
                return str(values)

        return "Task status not found"

    def debug_ontology(self):
        # Ontoloji içindeki bireyleri ve propertyleri ekrana basar
        print("\n--- ONTOLOGY DEBUG ---")
        print("Individuals:")
        for individual in self.onto.individuals():
            print(" -", individual)

        print("\nProperties:")
        print("grade_prop:", self.grade_prop)
        print("exam_date_prop:", self.exam_date_prop)
        print("status_prop:", self.status_prop)
        print("is_masked_prop:", self.is_masked_prop)

        if self.grade_prop:
            print("\nGrade relations:")
            for subject, obj in self.grade_prop.get_relations():
                print(" -", subject, "->", obj)

        if self.exam_date_prop:
            print("\nExam date relations:")
            for subject, obj in self.exam_date_prop.get_relations():
                print(" -", subject, "->", obj)

        if self.status_prop:
            print("\nStatus relations:")
            for subject, obj in self.status_prop.get_relations():
                print(" -", subject, "->", obj)

        print("--- END DEBUG ---\n")

    def query(self, question: str) -> str:
        # Soru içeriğine göre uygun ontoloji fonksiyonunu çağırır
        normalized_question = question.lower()

        if "not" in normalized_question or "grade" in normalized_question or "puan" in normalized_question:
            return self.get_student_grade()

        if (
            "ne zaman" in normalized_question
            or "tarih" in normalized_question
            or "exam" in normalized_question
            or "kaçta" in normalized_question
            or "saat" in normalized_question
        ):
            return self.get_exam_date()

        if "durum" in normalized_question or "status" in normalized_question:
            return self.get_task_status()

        return "No ontology data found for this question."


if __name__ == "__main__":
    service = OntologyService()
    service.debug_ontology()
    print("Grade:", service.get_student_grade())
    print("Exam Date:", service.get_exam_date())
    print("Task Status:", service.get_task_status())