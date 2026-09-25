"""Simple Flask interface for the newborn health risk demo model."""
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "newborn_model.pkl"
app = Flask(__name__)

FEATURES = [
    "Gender",
    "Gestational_Age",
    "Birth_Weight",
    "Temperature",
    "Heart_Rate",
    "Respiratory_Rate",
    "Oxygen_Saturation",
    "Apgar_Score",
    "Jaundice_Level",
    "Feeding_Frequency",
    "Urine_Count",
    "Stool_Count",
]
NUMERIC_FIELDS = FEATURES[1:]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Run 'python train_model.py' before starting Flask."
        )
    return joblib.load(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    form_data = {}

    if request.method == "POST":
        form_data = request.form.to_dict()
        try:
            values = {"Gender": form_data["Gender"]}
            values.update({field: float(form_data[field]) for field in NUMERIC_FIELDS})
            input_df = pd.DataFrame([values], columns=FEATURES)
            model = load_model()
            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]
            class_probabilities = dict(zip(model.classes_, probabilities))
            result = {
                "label": prediction,
                "probability": class_probabilities.get(prediction, 0.0) * 100,
                "healthy_probability": class_probabilities.get("Healthy", 0.0) * 100,
                "risk_probability": class_probabilities.get("At Risk", 0.0) * 100,
            }
        except (KeyError, ValueError, TypeError, FileNotFoundError) as exc:
            error = f"Please enter valid values for every field. Details: {exc}"

    return render_template("index.html", result=result, error=error, form_data=form_data)


if __name__ == "__main__":
    app.run(debug=True)
