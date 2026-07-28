import json
from pathlib import Path
from typing import Any
from app.policy_evaluator import evaluate_policy_rules


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_CASES_DIR = PROJECT_ROOT / "demo_cases"

ALLOWED_RISK_CATEGORIES = {
    "low",
    "medium",
    "high",
}

ALLOWED_POLICY_TYPES = {
    "source_derived",
    "synthetic_demo",
}

SENSITIVE_FEATURES = {
    "CODE_GENDER",
    "GENDER",
    "SEX",
    "RACE",
    "RELIGION",
    "ETHNICITY",
    "DAYS_BIRTH",
    "AGE_YEARS",
}

SAFE_SENSITIVE_METADATA_KEYS = {
    "SENSITIVE_FEATURES_EXCLUDED",
    "EXCLUDED_SENSITIVE_FEATURES",
    "FILTERED_SENSITIVE_FEATURES",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)



def find_sensitive_keys(
    value: Any,
    path: str = "root",
) -> list[str]:
    matches: list[str] = []

    if isinstance(value, dict):
        for key, child_value in value.items():
            key_upper = str(key).upper()
            child_path = f"{path}.{key}"

            # This metadata records which features were removed.
            # It does not contain applicant-sensitive values.
            if key_upper in SAFE_SENSITIVE_METADATA_KEYS:
                continue

            if key_upper in SENSITIVE_FEATURES:
                matches.append(child_path)

            matches.extend(
                find_sensitive_keys(
                    child_value,
                    child_path,
                )
            )

    elif isinstance(value, list):
        for index, item in enumerate(value):
            matches.extend(
                find_sensitive_keys(
                    item,
                    f"{path}[{index}]",
                )
            )

    elif isinstance(value, str):
        value_upper = value.strip().upper()

        if value_upper in SENSITIVE_FEATURES:
            matches.append(path)

    return matches


def validate_case_id(
    payload: dict[str, Any],
    expected_case_id: str,
    file_name: str,
) -> None:
    payload_case_id = payload.get("demo_case_id")

    if (
        payload_case_id is not None
        and payload_case_id != expected_case_id
    ):
        raise ValueError(
            f"{file_name} contains case ID "
            f"{payload_case_id}, expected {expected_case_id}."
        )


from typing import Any, Literal


ImpactDirection = Literal[
    "increases_default_risk",
    "reduces_default_risk",
]


def normalize_factor(
    factor: dict[str, Any],
    expected_direction: ImpactDirection,
) -> dict[str, Any]:
    required_fields = {
        "feature",
        "display_name",
        "shap_value",
    }

    missing_fields = required_fields - factor.keys()

    if missing_fields:
        raise ValueError(
            "Explanation factor is missing fields: "
            f"{sorted(missing_fields)}"
        )

    shap_value = float(factor["shap_value"])

    stored_direction = factor.get("impact_direction")

    if (
        stored_direction is not None
        and stored_direction != expected_direction
    ):
        raise ValueError(
            f"Direction mismatch for {factor['feature']}: "
            f"stored={stored_direction}, "
            f"expected={expected_direction}"
        )

    # Validate that the SHAP sign matches the factor group.
    if expected_direction == "increases_default_risk" and shap_value < 0:
        raise ValueError(
            f"Risk factor {factor['feature']} has "
            f"negative SHAP value: {shap_value}"
        )

    if expected_direction == "reduces_default_risk" and shap_value > 0:
        raise ValueError(
            f"Protective factor {factor['feature']} has "
            f"positive SHAP value: {shap_value}"
        )

    return {
        "feature": str(factor["feature"]),
        "display_name": str(factor["display_name"]),
        "raw_value": factor.get("raw_value"),
        "model_value": factor.get("model_value"),
        "shap_value": shap_value,
        "impact_direction": expected_direction,
    }


def normalize_policy_chunk(
    chunk: dict[str, Any],
) -> dict[str, Any]:
    metadata = chunk.get("metadata", {})

    required_metadata = {
        "section_number",
        "section_title",
        "policy_type",
    }

    missing_metadata = required_metadata - metadata.keys()

    if missing_metadata:
        raise ValueError(
            "Policy chunk is missing metadata: "
            f"{sorted(missing_metadata)}"
        )

    policy_type = metadata["policy_type"]

    if policy_type not in ALLOWED_POLICY_TYPES:
        raise ValueError(
            f"Unsupported policy type: {policy_type}"
        )

    return {
        "chunk_id": chunk["chunk_id"],
        "section_number": str(
            metadata["section_number"]
        ),
        "section_title": metadata["section_title"],
        "policy_type": policy_type,
        "page_start": metadata.get("page_start"),
        "page_end": metadata.get("page_end"),
        "text": chunk["text"],
        "matched_queries": chunk.get(
            "matched_queries",
            [],
        ),
        "fusion_score": float(
            chunk.get("fusion_score", 0)
        ),
        "reranker_score": float(
            chunk.get("best_reranker_score", 0)
        ),
    }

def remove_sensitive_fields(value: Any) -> Any:
    """
    Remove sensitive fields from data sent to the LLM.
    """

    if isinstance(value, dict):
        cleaned = {}

        for key, child_value in value.items():
            key_upper = str(key).strip().upper()

            # Keep audit metadata describing removed features.
            if key_upper in SAFE_SENSITIVE_METADATA_KEYS:
                cleaned[key] = child_value
                continue

            # Remove actual sensitive fields.
            if key_upper in SENSITIVE_FEATURES:
                continue

            cleaned[key] = remove_sensitive_fields(
                child_value
            )

        return cleaned

    if isinstance(value, list):
        return [
            remove_sensitive_fields(item)
            for item in value
        ]

    return value


def remove_sensitive_explanation_factors(
    explanation: dict[str, Any],
) -> dict[str, Any]:
    cleaned = remove_sensitive_fields(explanation)

    for factor_group in [
        "top_risk_factors",
        "top_protective_factors",
    ]:
        cleaned[factor_group] = [
            factor
            for factor in cleaned.get(factor_group, [])
            if str(
                factor.get("feature", "")
            ).strip().upper()
            not in SENSITIVE_FEATURES
        ]

    return cleaned

def build_report_input(
    case_id: str,
    save_output: bool = True,
) -> dict[str, Any]:
    case_directory = DEMO_CASES_DIR / case_id

    raw_applicant_summary = load_json(
        case_directory / "applicant_summary.json"
    )

    raw_risk_explanation = load_json(
        case_directory / "risk_explanation.json"
    )

    applicant_summary = remove_sensitive_fields(
        raw_applicant_summary
    )

    risk_explanation = remove_sensitive_explanation_factors(
        raw_risk_explanation
    )

    policy_context = load_json(
        case_directory / "retrieved_policy_context.json"
    )

    validate_case_id(
        policy_context,
        case_id,
        "retrieved_policy_context.json",
    )

    validate_case_id(
        raw_applicant_summary,
        case_id,
        "applicant_summary.json",
    )

    validate_case_id(
        raw_risk_explanation,
        case_id,
        "risk_explanation.json",
    )

    sensitive_matches = find_sensitive_keys(
        {
            "applicant_summary": applicant_summary,
            "risk_explanation": risk_explanation,
        }
    )

    if sensitive_matches:
        raise ValueError(
            "Sensitive information found in GenAI input: "
            f"{sensitive_matches}"
        )

    model_output = risk_explanation.get(
        "model_output",
        {}
    )

    risk_category = str(
        model_output.get("risk_category", "")
    ).lower()

    if risk_category not in ALLOWED_RISK_CATEGORIES:
        raise ValueError(
            f"Invalid risk category: {risk_category}"
        )

    default_probability = float(
        model_output["default_probability"]
    )

    if not 0 <= default_probability <= 1:
        raise ValueError(
            "Default probability must be between 0 and 1."
        )

    risk_factors = [
        normalize_factor(
            factor,
            expected_direction="increases_default_risk",
        )
        for factor in risk_explanation.get(
            "top_risk_factors",
            [],
        )[:3]
    ]

    protective_factors = [
        normalize_factor(
            factor,
            expected_direction="reduces_default_risk",
        )
        for factor in risk_explanation.get(
            "top_protective_factors",
            [],
        )[:2]
    ]

    raw_policy_chunks = policy_context.get(
        "retrieved_policy_chunks",
        [],
    )[:4]

    if not raw_policy_chunks:
        raise ValueError(
            "At least one policy chunk is required."
        )

    policy_evidence = [
        normalize_policy_chunk(chunk)
        for chunk in raw_policy_chunks
    ]

    chunk_ids = [
        chunk["chunk_id"]
        for chunk in policy_evidence
    ]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError(
            "Duplicate policy chunks found."
        )

    report_input = {
        "report_version": "1.0",
        "demo_case_id": case_id,
        "applicant_summary": applicant_summary,
        "model_assessment": {
            "default_probability": default_probability,
            "non_default_probability": model_output.get(
                "non_default_probability"
            ),
            "risk_category": risk_category,
            "model_version": model_output["model_version"],
        },
        "model_explanation": {
            "shap_output_scale": risk_explanation.get(
                "shap_output_scale",
                "raw_model_output",
            ),
            "risk_factors": risk_factors,
            "protective_factors": protective_factors,
        },
        "policy_evidence": policy_evidence,
        "generation_constraints": [
            (
                "Use only facts present in this report input."
            ),
            (
                "Do not infer that a policy threshold was breached "
                "unless the applicant value is explicitly available."
            ),
            (
                "Every policy finding must cite one supplied chunk_id."
            ),
            (
                "Describe SHAP values as model contributions, "
                "not causal explanations."
            ),
            (
                "Do not use or infer sensitive or protected attributes."
            ),
            (
                "Synthetic demo policy rules must be identified "
                "as demonstration-only rules."
            ),
            (
                "Do not issue a legally binding approval, decline, "
                "or loan-sanction decision."
            ),
            (
                "Final sanction remains with an authorized human "
                "credit decision-maker."
            ),
        ],
    }

    policy_evaluation = evaluate_policy_rules({
        "applicant_summary": raw_applicant_summary,
        "model_assessment": report_input[
            "model_assessment"
        ],
    })

    report_input["deterministic_policy_evaluation"] = (
        policy_evaluation.model_dump(mode="json")
    )

    if save_output:
        output_path = case_directory / "report_input.json"

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                report_input,
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

        print(f"Saved report input: {output_path}")

    return report_input