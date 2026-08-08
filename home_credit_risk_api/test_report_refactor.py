from app.report_input import build_report_input
from app.report_generator import generate_final_report_from_input


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


print(
    "System Recommendation:",
    report.recommendation.recommendation,
)

print(
    "Human Decision:",
    report.human_review_outcome.decision,
)

print(
    "Human Comment:",
    report.human_review_outcome.comment,
)

print(
    "Status:",
    report.human_review_outcome.status,
)