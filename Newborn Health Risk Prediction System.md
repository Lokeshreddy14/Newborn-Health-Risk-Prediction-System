# Newborn Health Risk Prediction System

A simple end-to-end **college-level healthcare ML prototype** using Python, Pandas, Scikit-learn, Random Forest, Flask, HTML, and CSS.

> **Important disclaimer:** This project uses a clearly labeled synthetic/demo dataset generated for education. It is **not real clinical data**, is not clinically validated, and must not be used to diagnose, triage, or make treatment decisions. It is also not a cardiac-arrest prediction system or a medical decision-support system.

## Project structure

```text
newborn-health-risk/
├── data/
│   └── newborn_data.csv          # generated synthetic/demo CSV
├── models/
│   ├── newborn_model.pkl         # saved full preprocessing + model pipeline
│   └── risk_distribution.png     # basic EDA output
├── generate_dataset.py           # reproducible sample CSV generator
├── train_model.py                # cleaning, EDA, training, evaluation, saving
├── app.py                        # Flask web application
├── requirements.txt
├── templates/
│   └── index.html
├── static/
│   └── style.css
└── README.md
```

## 1. Install dependencies on Windows

Open **Command Prompt** or PowerShell:

```bat
cd path\to\newborn-health-risk
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS/Linux, the equivalent activation command is `source .venv/bin/activate`.

## 2. Generate or refresh the prototype CSV

The generator creates deterministic synthetic data and adds a few missing values to demonstrate imputation:

```bat
python generate_dataset.py
```

The CSV includes: `Gender`, `Gestational_Age`, `Birth_Weight`, `Temperature`, `Heart_Rate`, `Respiratory_Rate`, `Oxygen_Saturation`, `Apgar_Score`, `Jaundice_Level`, `Feeding_Frequency`, `Urine_Count`, `Stool_Count`, and `Risk`.

## 3. Train and evaluate the model

```bat
python train_model.py
```

The script:

1. Loads the CSV with Pandas (and generates it if absent).
2. Checks required columns, removes duplicate/invalid-label rows, and converts numeric values.
3. Reports missing values and leaves them for the pipeline's median/most-frequent imputers.
4. Prints basic EDA summaries and saves a risk-label bar chart.
5. One-hot encodes `Gender` and keeps numeric newborn measurements as numeric features.
6. Creates an 80/20 train/test split with `stratify=y`.
7. Trains a `RandomForestClassifier`.
8. Prints accuracy, precision, recall, F1-score, confusion matrix, ROC-AUC, and a classification report.
9. Prints feature importance from the fitted Random Forest.
10. Saves the entire preprocessing + model pipeline using Joblib to `models/newborn_model.pkl`.

Because the dataset is synthetic and intentionally simple, good-looking metrics do **not** indicate clinical usefulness or generalization to real newborns.

## 4. Run the Flask application

```bat
python app.py
```

Open <http://127.0.0.1:5000> in a browser. Fill in all fields and select **Predict risk**. The request path is:

```text
HTML form → Flask → saved preprocessing + Random Forest pipeline → prediction → result page
```

The result shows `Healthy` or `At Risk`, the model's predicted-class probability, and both class probabilities. These are model outputs, not medical probabilities.

## Expected output

Training prints output similar to:

```text
Dataset shape: (180, 13)
Evaluation on stratified test set:
Accuracy : 0.9xx
Precision: 0.9xx
Recall   : 0.9xx
F1-score : 0.9xx
ROC-AUC  : 0.9xx
Saved complete preprocessing + model pipeline to .../models/newborn_model.pkl
```

Exact metrics can vary if the generator settings change. The Flask page should show a prediction card after a valid submission.

## Common errors and fixes

- **`ModuleNotFoundError`**: activate `.venv` and run `pip install -r requirements.txt` again.
- **`Model file not found`**: run `python train_model.py` before `python app.py`.
- **Port 5000 already in use**: stop the other Flask process, or change the last line to `app.run(debug=True, port=5001)`.
- **`python` is not recognized on Windows**: use `py` instead, for example `py train_model.py`.
- **CSV column error**: restore the generated `data/newborn_data.csv` by running `python generate_dataset.py`.
- **Browser cannot connect**: confirm the terminal is still running Flask and visit `http://127.0.0.1:5000`.

## Educational scope

This prototype demonstrates the mechanics of a tabular classification project. It does not include clinical validation, prospective testing, calibration, fairness review, privacy controls, clinician oversight, regulatory review, or integration with medical records. Do not enter real patient-identifying information.
