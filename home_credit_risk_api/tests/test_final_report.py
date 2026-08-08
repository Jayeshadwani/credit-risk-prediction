from app.report_generator import generate_final_report_from_input
from app.report_input import build_report_input


def test_final_report_preserves_system_recommendation():
    """
    Verifies that a human approval does not overwrite the deterministic system recommendation.
    """

    report_input = build_report_input(
        case_id="DEMO-005",
        save_output=False,
    )

    report = generate_final_report_from_input(
        report_input=report_input,
        human_decision="approve",
        human_comment="Income documents verified.",
        decision_status="completed",
    )

    assert report.recommendation.recommendation == "manual_review"
    assert report.human_review_outcome.decision == "approve"
    assert report.human_review_outcome.status == "completed"


def test_final_report_records_human_decline():
    """
    Verifies that a human decline is stored separately from the system-generated recommendation.
    """

    report_input = build_report_input(
        case_id="DEMO-005",
        save_output=False,
    )

    report = generate_final_report_from_input(
        report_input=report_input,
        human_decision="decline",
        human_comment="Income evidence was insufficient.",
        decision_status="completed",
    )

    assert report.recommendation.recommendation == "manual_review"
    assert report.human_review_outcome.decision == "decline"
    assert report.human_review_outcome.status == "completed"