import json
from pathlib import Path

import numpy as np

from app.predictor import predict_default_probability


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SAMPLE_REQUEST_PATH = (
    PROJECT_ROOT
    / "sample_requests"
    / "sample_request_raw.json"
)


def load_sample_records() -> list[dict]:
    with open(
        SAMPLE_REQUEST_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    return payload["records"]


def test_prediction_without_explanation():
    records = load_sample_records()

    results = predict_default_probability(
        records=records,
        include_explanation=False,
    )

    assert len(results) == len(records)

    prediction = results[0]

    assert 0 <= prediction["default_probability"] <= 1
    assert 0 <= prediction["non_default_probability"] <= 1

    assert np.isclose(
        prediction["default_probability"]
        + prediction["non_default_probability"],
        1.0,
    )

    assert prediction["risk_category"] in {
        "low",
        "medium",
        "high",
    }

    assert isinstance(prediction["model_version"], str)

    assert "top_risk_factors" not in prediction
    assert "top_protective_factors" not in prediction


def test_prediction_with_explanation():
    records = load_sample_records()

    results = predict_default_probability(
        records=records,
        include_explanation=True,
    )

    prediction = results[0]

    assert "top_risk_factors" in prediction
    assert "top_protective_factors" in prediction

    assert len(prediction["top_risk_factors"]) <= 5
    assert len(prediction["top_protective_factors"]) <= 3

    for factor in prediction["top_risk_factors"]:
        assert factor["shap_value"] > 0
        assert factor["impact"] == "increases_default_risk"
        assert "feature" in factor
        assert "display_name" in factor

    for factor in prediction["top_protective_factors"]:
        assert factor["shap_value"] < 0
        assert factor["impact"] == "reduces_default_risk"