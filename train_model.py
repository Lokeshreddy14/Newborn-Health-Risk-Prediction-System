"""Train and evaluate the newborn health risk prototype model.

Educational use only: the dataset and model are synthetic and not clinically validated.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from generate_dataset import build_demo_dataset


BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "newborn_data.csv"
MODEL_PATH = BASE_DIR / "models" / "newborn_model.pkl"
EDA_PATH = BASE_DIR / "models" / "risk_distribution.png"

TARGET = "Risk"
CATEGORICAL_FEATURES = ["Gender"]
NUMERIC_FEATURES = [
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
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def load_and_clean_data() -> pd.DataFrame:
    """Load CSV, standardize labels, coerce numeric fields, and fill invalid labels."""
    if not DATA_PATH.exists():
        print("Dataset not found; generating the synthetic prototype dataset...")
        df = build_demo_dataset()
        df.to_csv(DATA_PATH, index=False)
    else:
        df = pd.read_csv(DATA_PATH)

    required_columns = FEATURES + [TARGET]
    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {missing_columns}")

    df = df[required_columns].copy()
    df["Gender"] = df["Gender"].astype("string").str.strip().str.title()
    df[TARGET] = df[TARGET].astype("string").str.strip()
    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df[df[TARGET].isin(["Healthy", "At Risk"])].drop_duplicates().reset_index(drop=True)
    return df


def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
    )
    return Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", classifier)]
    )


def save_basic_eda(df: pd.DataFrame) -> None:
    """Save a small class-distribution chart as a basic EDA artifact."""
    ax = df[TARGET].value_counts().plot(kind="bar", color=["#2563eb", "#dc2626"])
    ax.set_title("Prototype dataset: risk label distribution")
    ax.set_xlabel("Risk label")
    ax.set_ylabel("Number of rows")
    ax.figure.tight_layout()
    ax.figure.savefig(EDA_PATH, dpi=140)
    plt.close(ax.figure)


def print_feature_importance(pipeline: Pipeline) -> None:
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()
    importances = pd.Series(classifier.feature_importances_, index=feature_names)
    print("\nFeature importance (highest first):")
    print(importances.sort_values(ascending=False).head(15).to_string())


def main() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = load_and_clean_data()
    print("Dataset shape:", df.shape)
    print("Missing values before pipeline imputation:\n", df.isna().sum().to_string())
    print("\nRisk distribution:\n", df[TARGET].value_counts().to_string())
    print("\nNumeric summary:\n", df[NUMERIC_FEATURES].describe().round(2).to_string())
    save_basic_eda(df)

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)
    positive_index = list(pipeline.classes_).index("At Risk")

    print("\nEvaluation on stratified test set:")
    print(f"Accuracy : {accuracy_score(y_test, predictions):.3f}")
    print(f"Precision: {precision_score(y_test, predictions, pos_label='At Risk', zero_division=0):.3f}")
    print(f"Recall   : {recall_score(y_test, predictions, pos_label='At Risk', zero_division=0):.3f}")
    print(f"F1-score : {f1_score(y_test, predictions, pos_label='At Risk', zero_division=0):.3f}")
    print(f"ROC-AUC  : {roc_auc_score((y_test == 'At Risk').astype(int), probabilities[:, positive_index]):.3f}")
    print("\nConfusion matrix (rows=true, columns=predicted; order=classes):")
    print(pd.DataFrame(confusion_matrix(y_test, predictions, labels=pipeline.classes_), index=pipeline.classes_, columns=pipeline.classes_))
    print("\nClassification report:\n", classification_report(y_test, predictions, zero_division=0))
    print_feature_importance(pipeline)

    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nSaved complete preprocessing + model pipeline to {MODEL_PATH}")
    print(f"Saved basic EDA chart to {EDA_PATH}")


if __name__ == "__main__":
    main()
