from typing import Any

import numpy as np
import pandas as pd
import shap

from app.feature_metadata import get_feature_display_name
from app.predictor import model


# Create once when the application starts.
explainer = shap.TreeExplainer(model)


def make_json_safe(value: Any) -> Any:
    """Convert NumPy/Pandas values into JSON-safe Python values."""

    if pd.isna(value):
        return None

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    return value


def explain_predictions(
    raw_df: pd.DataFrame,
    features: pd.DataFrame,
    top_risk_factors: int = 5,
    top_protective_factors: int = 3,
) -> list[dict[str, Any]]:
    """
    Generate local SHAP explanations for every applicant.

    Positive SHAP value:
        Increases the model's default-risk score.

    Negative SHAP value:
        Decreases the model's default-risk score.
    """

    explanation = explainer(features)

    shap_values = np.asarray(explanation.values)

    # Most XGBoost binary classifiers return:
    # (number_of_applicants, number_of_features)
    #
    # Some model/library combinations may return:
    # (number_of_applicants, number_of_features, number_of_classes)
    if shap_values.ndim == 3:
        if shap_values.shape[2] != 2:
            raise ValueError(
                f"Unexpected SHAP output shape: {shap_values.shape}"
            )

        # Select the TARGET=1/default class.
        shap_values = shap_values[:, :, 1]

    if shap_values.ndim != 2:
        raise ValueError(
            f"Unexpected SHAP output shape: {shap_values.shape}"
        )

    all_explanations = []

    for row_index in range(len(features)):
        row_shap_values = shap_values[row_index]

        risk_indices = np.where(row_shap_values > 0)[0]
        protective_indices = np.where(row_shap_values < 0)[0]

        # Largest positive SHAP values first.
        risk_indices = risk_indices[
            np.argsort(row_shap_values[risk_indices])[::-1]
        ][:top_risk_factors]

        # Most negative SHAP values first.
        protective_indices = protective_indices[
            np.argsort(row_shap_values[protective_indices])
        ][:top_protective_factors]

        risk_factors = [
            build_factor(
                row_index=row_index,
                feature_index=int(feature_index),
                raw_df=raw_df,
                features=features,
                row_shap_values=row_shap_values,
                impact="increases_default_risk",
            )
            for feature_index in risk_indices
        ]

        protective_factors = [
            build_factor(
                row_index=row_index,
                feature_index=int(feature_index),
                raw_df=raw_df,
                features=features,
                row_shap_values=row_shap_values,
                impact="reduces_default_risk",
            )
            for feature_index in protective_indices
        ]

        all_explanations.append({
            "top_risk_factors": risk_factors,
            "top_protective_factors": protective_factors,
            "shap_output_scale": "raw_model_output",
        })

    return all_explanations


def build_factor(
    row_index: int,
    feature_index: int,
    raw_df: pd.DataFrame,
    features: pd.DataFrame,
    row_shap_values: np.ndarray,
    impact: str,
) -> dict[str, Any]:

    feature_name = features.columns[feature_index]

    raw_value = (
        raw_df.iloc[row_index].get(feature_name)
        if feature_name in raw_df.columns
        else None
    )

    return {
        "feature": feature_name,
        "display_name": get_feature_display_name(feature_name),
        "raw_value": make_json_safe(raw_value),
        "model_value": make_json_safe(
            features.iloc[row_index, feature_index]
        ),
        "shap_value": float(row_shap_values[feature_index]),
        "impact": impact,
    }