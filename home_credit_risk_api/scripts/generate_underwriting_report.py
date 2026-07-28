import argparse

from app.report_generator import (
    generate_underwriting_report,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "case_id",
        help="Demo case ID, for example DEMO-001",
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Optional model override.",
    )

    arguments = parser.parse_args()

    report = generate_underwriting_report(
        case_id=arguments.case_id,
        model_name=arguments.model,
    )

    print(
        "Recommendation:",
        report.recommendation.recommendation,
    )

    print(
        "Human review required:",
        report.recommendation.human_review_required,
    )

    print(
        "Policy findings:",
        len(report.policy_findings),
    )