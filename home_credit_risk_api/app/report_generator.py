import json
import math
import os
from pathlib import Path
from typing import Any

from openai import OpenAI
from langsmith.wrappers import wrap_openai
from app.report_input import build_report_input
from app.report_schemas import (
    ModelFactor,
    UnderwritingReport,
    FinalUnderwritingReport,
    HumanReviewOutcome,
)

from app.config import settings

DEFAULT_MODEL = settings.openai_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_CASES_DIR = PROJECT_ROOT / "demo_cases"


SYSTEM_PROMPT = """
You are a credit-underwriting decision-support report writer.

Generate a structured underwriting report using only the supplied JSON.

Create exactly one policy finding for every rule in
deterministic_policy_evaluation.rules.

Copy each rule_id exactly into the corresponding policy finding.
Do not omit, duplicate, rename, or invent rule IDs.

The deterministic_policy_evaluation is authoritative.

Copy its recommendation and human_review_required values exactly.

Do not change a rule status, reinterpret an unknown rule as pass or fail,
or claim a hard policy breach unless evaluation_status is "fail".

Use the deterministic rule reasons when describing policy findings.

Rules:

1. Copy the demo case ID, model version, probability, risk category,
   feature names, display names, SHAP values and impact directions exactly.

2. SHAP values describe contributions to the model prediction.
   They do not establish causality.

3. A model feature must not be treated as proof that a policy threshold
   was breached.

4. Treat a policy condition as applicant-specific only when the exact
   applicant value needed to evaluate it is present in the input.

5. When a required value is unavailable, add it to missing_information
   and recommend that it be verified.

6. Every policy finding must cite exactly one supplied policy chunk.
   Copy its chunk ID, section number, section title, policy type and
   page numbers exactly.

7. Clearly identify synthetic_demo rules as demonstration-only rules.

8. A medium-risk or high-risk model band requires human review.

9. Do not recommend decline based only on the model risk band or SHAP
   factors. A decline recommendation requires a verified hard policy
   condition.

10. Do not use, infer or mention protected or sensitive attributes.

11. Do not issue a legally binding lending decision. Final sanction
    remains with an authorized human credit decision-maker.

12. Include all supplied risk factors and protective factors.

13. The limitations must explicitly cover:
    - SHAP contribution is not causality.
    - Synthetic policy rules are demonstration-only.
    - Final sanction requires an authorized human.
""".strip()


class ReportGroundingError(ValueError):
    """Raised when generated content is not grounded in report input."""


def get_refusal_message(response: Any) -> str | None:
    for output_item in getattr(response, "output", []):
        if getattr(output_item, "type", None) != "message":
            continue

        for content_item in getattr(output_item, "content", []):
            if getattr(content_item, "type", None) == "refusal":
                return str(
                    getattr(
                        content_item,
                        "refusal",
                        "The model refused the request.",
                    )
                )

    return None


def validate_factor_group(
    generated_factors: list[ModelFactor],
    input_factors: list[dict[str, Any]],
    group_name: str,
) -> None:
    if len(generated_factors) != len(input_factors):
        raise ReportGroundingError(
            f"{group_name} count mismatch: "
            f"generated={len(generated_factors)}, "
            f"expected={len(input_factors)}"
        )

    input_by_feature = {
        factor["feature"]: factor
        for factor in input_factors
    }

    generated_features = {
        factor.feature
        for factor in generated_factors
    }

    expected_features = set(input_by_feature)

    if generated_features != expected_features:
        raise ReportGroundingError(
            f"{group_name} feature mismatch. "
            f"Generated={sorted(generated_features)}, "
            f"expected={sorted(expected_features)}"
        )

    for generated_factor in generated_factors:
        source_factor = input_by_feature[
            generated_factor.feature
        ]

        if (
            generated_factor.display_name
            != source_factor["display_name"]
        ):
            raise ReportGroundingError(
                f"Display name changed for "
                f"{generated_factor.feature}."
            )

        if not math.isclose(
            generated_factor.shap_value,
            float(source_factor["shap_value"]),
            rel_tol=0,
            abs_tol=1e-9,
        ):
            raise ReportGroundingError(
                f"SHAP value changed for "
                f"{generated_factor.feature}."
            )

        if (
            generated_factor.impact_direction
            != source_factor["impact_direction"]
        ):
            raise ReportGroundingError(
                f"Impact direction changed for "
                f"{generated_factor.feature}."
            )


def validate_policy_citations(
    report: UnderwritingReport,
    report_input: dict[str, Any],
) -> None:
    evidence_by_id = {
        chunk["chunk_id"]: chunk
        for chunk in report_input["policy_evidence"]
    }

    for policy_finding in report.policy_findings:
        citation = policy_finding.citation

        if citation.chunk_id not in evidence_by_id:
            raise ReportGroundingError(
                "Report cited an unavailable policy chunk: "
                f"{citation.chunk_id}"
            )

        source = evidence_by_id[citation.chunk_id]

        expected_values = {
            "section_number": source["section_number"],
            "section_title": source["section_title"],
            "policy_type": source["policy_type"],
            "page_start": source["page_start"],
            "page_end": source["page_end"],
        }

        generated_values = {
            "section_number": citation.section_number,
            "section_title": citation.section_title,
            "policy_type": citation.policy_type,
            "page_start": citation.page_start,
            "page_end": citation.page_end,
        }

        if generated_values != expected_values:
            raise ReportGroundingError(
                "Citation metadata does not match source "
                f"chunk {citation.chunk_id}."
            )


def validate_limitations(
    limitations: list[str],
) -> None:
    combined = " ".join(limitations).lower()

    required_concepts = {
        "SHAP non-causality": (
            "shap" in combined
            and "caus" in combined
        ),
        "synthetic demo policy": (
            "synthetic" in combined
            and "demo" in combined
        ),
        "human final sanction": (
            "human" in combined
            and (
                "sanction" in combined
                or "final decision" in combined
            )
        ),
    }

    missing = [
        concept
        for concept, present in required_concepts.items()
        if not present
    ]

    if missing:
        raise ReportGroundingError(
            "Report limitations are missing: "
            f"{missing}"
        )


def validate_report_grounding(
    report: UnderwritingReport,
    report_input: dict[str, Any],
) -> None:
    expected_assessment = report_input[
        "model_assessment"
    ]

    if report.report_version != "1.0":
        raise ReportGroundingError(
            "Unexpected report version."
        )

    if report.demo_case_id != report_input["demo_case_id"]:
        raise ReportGroundingError(
            "Demo case ID does not match."
        )

    if (
        report.model_assessment.risk_category
        != expected_assessment["risk_category"]
    ):
        raise ReportGroundingError(
            "Risk category does not match model output."
        )

    if (
        report.model_assessment.model_version
        != expected_assessment["model_version"]
    ):
        raise ReportGroundingError(
            "Model version does not match."
        )

    if not math.isclose(
        report.model_assessment.default_probability,
        float(expected_assessment["default_probability"]),
        rel_tol=0,
        abs_tol=1e-9,
    ):
        raise ReportGroundingError(
            "Default probability does not match."
        )

    model_explanation = report_input[
        "model_explanation"
    ]

    validate_factor_group(
        report.key_risk_factors,
        model_explanation["risk_factors"],
        "Risk factors",
    )

    validate_factor_group(
        report.protective_factors,
        model_explanation["protective_factors"],
        "Protective factors",
    )

    validate_policy_citations(
        report,
        report_input,
    )
    
    validate_policy_citations(
        report,
        report_input,
    )

    deterministic_evaluation = report_input[
        "deterministic_policy_evaluation"
    ]

    expected_summary = deterministic_evaluation[
        "summary"
    ]

    expected_rule_ids = {
        rule["rule_id"]
        for rule in deterministic_evaluation["rules"]
    }

    generated_rule_ids = {
        finding.rule_id
        for finding in report.policy_findings
    }

    if generated_rule_ids != expected_rule_ids:
        raise ReportGroundingError(
            "Policy findings do not cover every deterministic rule. "
            f"Missing={sorted(expected_rule_ids - generated_rule_ids)}, "
            f"Unexpected={sorted(generated_rule_ids - expected_rule_ids)}"
        )

    # Existing recommendation validation follows here.
    if (
        report.recommendation.recommendation
        != expected_summary["recommendation"]
    ):
        raise ReportGroundingError(
            "Recommendation does not match deterministic "
            "policy evaluation."
        )

    if (
        report.recommendation.human_review_required
        != expected_summary["human_review_required"]
    ):
        raise ReportGroundingError(
            "Human-review requirement does not match "
            "deterministic policy evaluation."
        )

    recommendation = report.recommendation
    risk_category = expected_assessment["risk_category"]

    if risk_category in {"medium", "high"}:
        if not recommendation.human_review_required:
            raise ReportGroundingError(
                "Medium/high-risk applications must "
                "require human review."
            )

        deterministic_evaluation = report_input[
            "deterministic_policy_evaluation"
        ]

        expected_summary = deterministic_evaluation["summary"]

        if (
            report.recommendation.recommendation
            != expected_summary["recommendation"]
        ):
            raise ReportGroundingError(
                "Recommendation does not match deterministic "
                "policy evaluation. "
                f"Generated={report.recommendation.recommendation}, "
                f"expected={expected_summary['recommendation']}"
            )

        if (
            report.recommendation.human_review_required
            != expected_summary["human_review_required"]
        ):
            raise ReportGroundingError(
                "Human-review requirement does not match "
                "deterministic policy evaluation."
            )

        failed_rule_ids = set(
            expected_summary["failed_rule_ids"]
        )

        claimed_hard_decline = (
            "hard_decline_condition"
            in report.recommendation.basis
        )

        if claimed_hard_decline and not failed_rule_ids:
            raise ReportGroundingError(
                "The report claimed a hard decline condition, "
                "but no deterministic rule failed."
            )

        if failed_rule_ids and not claimed_hard_decline:
            raise ReportGroundingError(
                "A deterministic hard rule failed, but the report "
                "did not include hard_decline_condition in its basis."
            )

    # Until a deterministic hard-rule evaluator is added,
    # the LLM cannot establish a hard decline condition.
    if (
        recommendation.recommendation
        == "decline_recommendation"
    ):
        raise ReportGroundingError(
            "Decline recommendation rejected because no "
            "deterministically verified hard-decline rule "
            "was supplied."
        )

    if (
        "hard_decline_condition"
        in recommendation.basis
    ):
        raise ReportGroundingError(
            "The report claimed a hard decline condition "
            "without deterministic policy-rule evidence."
        )

    validate_limitations(report.limitations)


def generate_report_from_input(report_input: dict[str, Any], model_name: str | None = None) -> UnderwritingReport:
    """
    Generates and validates an underwriting report from an already-prepared and policy-evaluated case input.
    """

    client = wrap_openai(
        OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.openai_timeout_seconds,
            max_retries=settings.openai_max_retries,
        )
    )

    response = client.responses.parse(
        model=model_name or DEFAULT_MODEL,
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    "Generate the underwriting report from "
                    "this input:\n\n"
                    + json.dumps(
                        report_input,
                        indent=2,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                ),
            },
        ],
        text_format=UnderwritingReport,
    )

    refusal_message = get_refusal_message(response)

    if refusal_message:
        raise RuntimeError(
            "Model refused report generation: "
            f"{refusal_message}"
        )

    if response.status != "completed":
        reason = getattr(
            getattr(
                response,
                "incomplete_details",
                None,
            ),
            "reason",
            "unknown",
        )

        raise RuntimeError(
            "Report generation did not complete. "
            f"Reason: {reason}"
        )

    parsed_report = response.output_parsed

    if parsed_report is None:
        raise RuntimeError(
            "The model returned no parsed report."
        )

    report = UnderwritingReport.model_validate(
        parsed_report
    )

    validate_report_grounding(
        report,
        report_input,
    )

    return report


def generate_underwriting_report(case_id: str, model_name: str | None = None, save_output: bool = True) -> UnderwritingReport:
    """
    Builds the case input and generates the existing standalone underwriting decision-support report.
    """

    report_input = build_report_input(
        case_id=case_id,
        save_output=False,
    )

    report = generate_report_from_input(
        report_input=report_input,
        model_name=model_name,
    )

    if save_output:
        output_path = (
            DEMO_CASES_DIR
            / case_id
            / "underwriting_report.json"
        )

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                report.model_dump(mode="json"),
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

        print(f"Saved report: {output_path}")

    return report

def generate_final_report_from_input(
    report_input: dict[str, Any],
    human_decision: str,
    human_comment: str | None,
    decision_status: str,
    model_name: str | None = None,
) -> FinalUnderwritingReport:
    """
    Generates the grounded underwriting analysis and
    attaches the authoritative human-review outcome.
    """

    report = generate_report_from_input(
        report_input=report_input,
        model_name=model_name,
    )

    human_review_outcome = HumanReviewOutcome(
        decision=human_decision,
        comment=human_comment,
        status=decision_status,
    )

    return FinalUnderwritingReport(
        **report.model_dump(),
        human_review_outcome=human_review_outcome,
    )