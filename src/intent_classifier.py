import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "student_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

def predict_intent(text: str) -> str:
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return pred

def get_model_paths() -> dict[str, str | bool]:
    return {
        "model_path": model_path,
        "vectorizer_path": vectorizer_path,
        "model_exists": os.path.exists(model_path),
        "vectorizer_exists": os.path.exists(vectorizer_path),
    }