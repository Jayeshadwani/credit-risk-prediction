import pytest
from pydantic import ValidationError

from app.report_schemas import UnderwritingReport


def test_report_rejects_unknown_fields():
    payload = {
        "report_version": "1.0",
        "demo_case_id": "DEMO-001",
        "executive_summary": "High-risk application.",
        "model_assessment": {
            "default_probability": 0.72,
            "risk_category": "high",
            "model_version": "1.1.0",
            "interpretation": (
                "The model estimates elevated default risk."
            ),
        },
        "key_risk_factors": [],
        "protective_factors": [],
        "policy_findings": [],
        "recommendation": {
            "recommendation": "manual_review",
            "rationale": "High model-risk band.",
            "human_review_required": True,
            "basis": ["model_risk_band"],
        },
        "missing_information": [],
        "recommended_actions": [],
        "limitations": [],
        "invented_field": "not allowed",
    }

    with pytest.raises(ValidationError):
        UnderwritingReport.model_validate(payload)


def test_report_rejects_invalid_recommendation():
    payload = {
        "report_version": "1.0",
        "demo_case_id": "DEMO-001",
        "executive_summary": "Application assessment.",
        "model_assessment": {
            "default_probability": 0.40,
            "risk_category": "medium",
            "model_version": "1.1.0",
            "interpretation": (
                "The model estimates medium default risk."
            ),
        },
        "key_risk_factors": [],
        "protective_factors": [],
        "policy_findings": [
            {
                "finding": "Medium-risk cases require review.",
                "applicant_relevance": (
                    "The applicant is in the medium-risk band."
                ),
                "citation": {
                    "chunk_id": (
                        "loan-policy-v1.0-section-18-chunk-001"
                    ),
                    "section_number": "18",
                    "section_title": (
                        "Decision Outcomes and Manual-Review Triggers"
                    ),
                    "policy_type": "synthetic_demo",
                    "page_start": 6,
                    "page_end": 7,
                },
            }
        ],
        "recommendation": {
            "recommendation": "approve_immediately",
            "rationale": "Invalid recommendation.",
            "human_review_required": False,
            "basis": ["model_risk_band"],
        },
        "missing_information": [],
        "recommended_actions": [],
        "limitations": [],
    }

    with pytest.raises(ValidationError):
        UnderwritingReport.model_validate(payload)