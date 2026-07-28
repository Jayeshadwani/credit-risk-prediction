import argparse

from app.report_input import build_report_input


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "case_id",
        help="Demo case ID, for example DEMO-001",
    )

    arguments = parser.parse_args()

    report_input = build_report_input(
        arguments.case_id
    )

    print(
        "Risk category:",
        report_input["model_assessment"][
            "risk_category"
        ],
    )

    print(
        "Policy chunks:",
        len(report_input["policy_evidence"]),
    )

    print(
        "Risk factors:",
        len(
            report_input["model_explanation"][
                "risk_factors"
            ]
        ),
    )

    print(
        "Protective factors:",
        len(
            report_input["model_explanation"][
                "protective_factors"
            ]
        ),
    )