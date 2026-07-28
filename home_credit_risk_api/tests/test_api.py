import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SAMPLE_PATH = (
    PROJECT_ROOT
    / "sample_requests"
    / "sample_request_raw.json"
)


def load_payload() -> dict:
    with open(SAMPLE_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def test_predict_endpoint_without_explanation():
    response = client.post(
        "/predict",
        json=load_payload(),
    )

    assert response.status_code == 200

    body = response.json()

    assert "predictions" in body
    assert len(body["predictions"]) == 1

    prediction = body["predictions"][0]

    assert 0 <= prediction["default_probability"] <= 1
    assert prediction["risk_category"] in {
        "low",
        "medium",
        "high",
    }

    assert "top_risk_factors" not in prediction


def test_predict_endpoint_with_explanation():
    response = client.post(
        "/predict?include_explanation=true",
        json=load_payload(),
    )

    assert response.status_code == 200

    prediction = response.json()["predictions"][0]

    assert "top_risk_factors" in prediction
    assert "top_protective_factors" in prediction
    assert prediction["shap_output_scale"] == "raw_model_output"


def test_predict_endpoint_with_missing_columns():
    response = client.post(
        "/predict",
        json={
            "records": [
                {
                    "AMT_INCOME_TOTAL": 150000
                }
            ]
        },
    )

    assert response.status_code == 400
    assert "Missing required columns" in response.json()["detail"]