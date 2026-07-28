from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


# app/predictor.py

ARTIFACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "home_credit_bundle.joblib"
)

bundle: dict[str, Any] = joblib.load(ARTIFACT_PATH)

model = bundle["model"]
num_imputer = bundle["num_imputer"]
cat_imputer = bundle["cat_imputer"]
encoder = bundle["ordinal_encoder"]

numerical_cols = bundle["numerical_cols"]
categorical_cols = bundle["categorical_cols"]
feature_columns = bundle["feature_columns"]

MEDIUM_RISK_CUTOFF = float(
    bundle["risk_thresholds"]["medium"]
)

HIGH_RISK_CUTOFF = float(
    bundle["risk_thresholds"]["high"]
)

MODEL_VERSION = bundle["model_version"]


def prepare_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    missing_columns = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns[:20])
        )

    # Remove unexpected columns and restore training order.
    df = df[feature_columns].copy()

    # Convert infinite values to missing values before imputation.
    df = df.replace([np.inf, -np.inf], np.nan)

    df[numerical_cols] = num_imputer.transform(
        df[numerical_cols]
    )

    df[categorical_cols] = cat_imputer.transform(
        df[categorical_cols]
    )

    df[categorical_cols] = encoder.transform(
        df[categorical_cols]
    )

    if df.isna().any().any():
        missing = df.columns[df.isna().any()].tolist()
        raise ValueError(
            f"NaN values remain in columns: {missing[:20]}"
        )

    return df

def get_risk_category(probability: float) -> str:
    if probability >= HIGH_RISK_CUTOFF:
        return "high"

    if probability >= MEDIUM_RISK_CUTOFF:
        return "medium"

    return "low"


def predict_default_probability(
    records: list[dict[str, Any]],
    include_explanation: bool = False,
) -> list[dict[str, Any]]:

    raw_df = pd.DataFrame(records)

    if raw_df.empty:
        raise ValueError(
            "At least one applicant record is required."
        )

    applicant_ids = (
        raw_df["SK_ID_CURR"].tolist()
        if "SK_ID_CURR" in raw_df.columns
        else [None] * len(raw_df)
    )

    features = prepare_features(raw_df)

    probabilities = model.predict_proba(features)[:, 1]

    explanations = None

    if include_explanation:
        # Local import prevents a circular-import problem.
        from app.explainability import explain_predictions

        explanations = explain_predictions(
            raw_df=raw_df,
            features=features,
        )

    results = []

    for index, probability in enumerate(probabilities):
        probability = float(probability)

        result = {
            "default_probability": probability,
            "non_default_probability": float(1 - probability),
            "risk_category": get_risk_category(probability),
            "model_version": MODEL_VERSION,
        }

        if applicant_ids[index] is not None:
            result["applicant_id"] = int(
                applicant_ids[index]
            )

        if explanations is not None:
            result.update(explanations[index])

        results.append(result)

    return results