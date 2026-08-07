from typing import Any,Literal
from app.agent.stae import UnderwritingState
from app.report_input import build_report_input
from langgraph.types import interuppt, Command

def prepare_case(state: UnderwritingState) -> dict[str,Any]:
    """
    Builds the underwriting input and initializes the decision information required by the workflow.
    """
    case_id = state["case_id"]
    
    report_input = build_report_input(case_id=case_id,save_output=False)
    
    policy_summary = report_input["determinisitic_policy_evaluation"]["summary"]
    
    return {
        "report_input": report_input,
        "recommendation": policy_summary["recommendation"],
        "human_review_required": policy_summary["human_review_required"],
        "decision_status": "processing"
    }


def mark_for_human_review(state: UnderwritingState) -> dict[str,Any]:
    """
    Pauses the workflow and waits for an underwriter to review the application's recommendation.
    """

    human_response = interuppt({
        "case_id": state["case_id"],
        "recommendation": state["recommendation"],
        "message": "Human underwriting review required.",
        "allowed_decisions": ["approve", "decline","request_more_information"],
    })
    
    human_decision == human_response["decision"]

    if human_decision == "request_more_information":
        decision_status = "awaiting_more_information"
    else:
        decision_status = "completed"

    return {
        "human_decision": human_decision,
        "human_comment": human_response.get("comment"),
        "decision_status": decision_status

    }


def continue_automatically(state: UnderwritingState) -> dict[str,Any]:
    """
    Allows cases without a manual-review trigger to continue through the automated workflow.
    """

    return {
        "decision_status": "processing"
    }


def route_case(state: UnderwritingState) -> Literal["human_review","automatic"]:
    """
    Routes the application according to the deterministic policy evaluator's human-review requirement.
    """

    if state["human_review_required"]:
        return "human_review"
    
    return "automatic"