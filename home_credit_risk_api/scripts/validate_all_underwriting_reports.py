import json
from pathlib import Path
from typing import Any

from app.report_schemas import UnderwritingReport


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_CASES_DIR = PROJECT_ROOT / "demo_cases"

EXPECTED_RULE_IDS = {
    "MODEL_RISK_BAND",
    "CREDIT_TO_INCOME",
    "FOIR",
    "REPAYMENT_HISTORY_DPD",
    "KYC_AND_CONSENT",
    "DOCUMENT_COMPLETENESS",
}

BLOCKED_FEATURES = {
    "CODE_GENDER",
    "GENDER",
    "SEX",
    "RACE",
    "RELIGION",
    "ETHNICITY",
    "DAYS_BIRTH",
    "AGE_YEARS",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_case(case_directory: Path) -> list[str]:
    case_id = case_directory.name
    errors: list[str] = []

    report_path = (
        case_directory / "underwriting_report.json"
    )

    input_path = case_directory / "report_input.json"

    if not report_path.exists():
        return ["underwriting_report.json is missing"]

    if not input_path.exists():
        return ["report_input.json is missing"]

    raw_report = load_json(report_path)
    report_input = load_json(input_path)

    try:
        report = UnderwritingReport.model_validate(
            raw_report
        )
    except Exception as error:
        return [f"Schema validation failed: {error}"]

    if report.demo_case_id != case_id:
        errors.append("Report case ID does not match directory")

    expected_summary = report_input[
        "deterministic_policy_evaluation"
    ]["summary"]

    if (
        report.recommendation.recommendation
        != expected_summary["recommendation"]
    ):
        errors.append(
            "Recommendation does not match "
            "deterministic evaluation"
        )

    if (
        report.recommendation.human_review_required
        != expected_summary["human_review_required"]
    ):
        errors.append(
            "Human-review flag does not match "
            "deterministic evaluation"
        )

    generated_rule_ids = [
        finding.rule_id
        for finding in report.policy_findings
    ]

    if set(generated_rule_ids) != EXPECTED_RULE_IDS:
        errors.append(
            "Policy findings do not contain all expected rules"
        )

    if len(generated_rule_ids) != len(
        set(generated_rule_ids)
    ):
        errors.append("Duplicate policy rule IDs found")

    evidence_ids = {
        chunk["chunk_id"]
        for chunk in report_input["policy_evidence"]
    }

    for finding in report.policy_findings:
        if finding.citation.chunk_id not in evidence_ids:
            errors.append(
                "Unsupported citation: "
                f"{finding.citation.chunk_id}"
            )

    report_text = json.dumps(
        raw_report,
        ensure_ascii=False,
    ).upper()

    leaked_features = [
        feature
        for feature in BLOCKED_FEATURES
        if feature in report_text
    ]

    if leaked_features:
        errors.append(
            "Sensitive features found: "
            f"{sorted(leaked_features)}"
        )

    return errors


def main() -> None:
    case_directories = sorted(
        path
        for path in DEMO_CASES_DIR.iterdir()
        if path.is_dir()
        and path.name.startswith("DEMO-")
    )

    failures: dict[str, list[str]] = {}

    for case_directory in case_directories:
        errors = validate_case(case_directory)

        if errors:
            failures[case_directory.name] = errors
            print(f"{case_directory.name}: FAILED")

            for error in errors:
                print(f"  - {error}")

        else:
            print(f"{case_directory.name}: PASSED")

    print("\nValidation summary")
    print(f"Total cases: {len(case_directories)}")
    print(
        f"Passed: {len(case_directories) - len(failures)}"
    )
    print(f"Failed: {len(failures)}")

    if failures:
        raise RuntimeError(
            f"{len(failures)} reports failed validation."
        )


if __name__ == "__main__":
    main()