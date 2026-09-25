"""Generate a clearly labeled educational/demo newborn dataset.

This data is synthetic and is NOT real clinical data. It is only for demonstrating
an end-to-end machine-learning workflow.
"""
from pathlib import Path

import numpy as np
import pandas as pd


COLUMNS = [
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
    "Risk",
]


def build_demo_dataset(n_rows: int = 180, seed: int = 42) -> pd.DataFrame:
    """Create synthetic rows using a simple rule plus small random variation."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_rows):
        gestational_age = int(np.clip(rng.normal(38.5, 2.0), 30, 42))
        birth_weight = round(float(np.clip(rng.normal(3.1, 0.55), 1.2, 4.8)), 2)
        temperature = round(float(np.clip(rng.normal(36.8, 0.35), 35.5, 38.5)), 1)
        heart_rate = int(np.clip(rng.normal(140, 18), 90, 190))
        respiratory_rate = int(np.clip(rng.normal(44, 10), 20, 75))
        oxygen_saturation = int(np.clip(rng.normal(97, 2), 88, 100))
        apgar_score = int(np.clip(round(rng.normal(8.2, 1.2)), 3, 10))
        jaundice_level = round(float(np.clip(rng.normal(5.0, 2.2), 0.5, 14.0)), 1)
        feeding_frequency = int(np.clip(round(rng.normal(8, 2)), 3, 14))
        urine_count = int(np.clip(round(rng.normal(5, 2)), 0, 10))
        stool_count = int(np.clip(round(rng.normal(3, 1.5)), 0, 8))
        gender = rng.choice(["Female", "Male"])

        risk_score = 0
        risk_score += gestational_age < 37
        risk_score += birth_weight < 2.5
        risk_score += temperature < 36.3 or temperature > 37.5
        risk_score += heart_rate < 110 or heart_rate > 170
        risk_score += respiratory_rate < 30 or respiratory_rate > 60
        risk_score += oxygen_saturation < 95
        risk_score += apgar_score < 7
        risk_score += jaundice_level > 10
        risk_score += feeding_frequency < 6
        risk_score += urine_count < 3
        risk_score += stool_count < 1

        # A small amount of label noise keeps this a realistic classroom demo.
        risk = "At Risk" if (risk_score >= 2 or rng.random() < 0.04) else "Healthy"
        rows.append(
            [
                gender,
                gestational_age,
                birth_weight,
                temperature,
                heart_rate,
                respiratory_rate,
                oxygen_saturation,
                apgar_score,
                jaundice_level,
                feeding_frequency,
                urine_count,
                stool_count,
                risk,
            ]
        )

    return pd.DataFrame(rows, columns=COLUMNS)


def main() -> None:
    output_path = Path(__file__).parent / "data" / "newborn_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = build_demo_dataset()
    # Add a few missing values so the training script demonstrates imputation.
    df.loc[[7, 38, 91], "Temperature"] = np.nan
    df.loc[[14, 64], "Jaundice_Level"] = np.nan
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} synthetic prototype rows to {output_path}")
    print(df["Risk"].value_counts().to_string())


if __name__ == "__main__":
    main()
