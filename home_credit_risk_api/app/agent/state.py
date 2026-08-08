from typing import Any, Literal, TypedDict

HumanDecision = Literal[
    "approve",
    "decline",
    "request_more_information"
]

DecisionStatus = Literal[
    "processing",
    "awaiting_human_review",
    "awaiting_more_information",
    "completed"
]

class UnderwritingState(TypedDict, total=False):
    case_id: str
    review_id: str
    report_input: dict[str,Any]
    recommendation: str
    human_review_required: bool
    decision_status: DecisionStatus
    human_decision: HumanDecision | None
    human_comment: str | None
    final_report: dict[str, Any] | None