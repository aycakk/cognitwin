import os
import sys

# Proje kök dizinini Python path'ine ekler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.services.ontology_service import OntologyService


class HybridContextBuilder:
    # Yorum gerektiren sorular için ontolojiden bağlam üretir

    def __init__(self):
        self.ontology_service = OntologyService()

    def build_context(self, question: str) -> str:
        # Ontolojiden temel bilgileri çekerek LLM için bağlam hazırlar
        exam_date = self.ontology_service.get_exam_date()
        student_grade = self.ontology_service.get_student_grade()
        task_status = self.ontology_service.get_task_status()
        course_name = self.ontology_service.get_course_name()

        context = f"""
Knowledge Context:
- Exam Date: {exam_date}
- Student Grade: {student_grade}
- Task Status: {task_status}
- Course Name: {course_name}

User Question:
{question}
"""
        return context


if __name__ == "__main__":
    builder = HybridContextBuilder()
    sample_context = builder.build_context("How should I study for the exam this week?")
    print(sample_context)