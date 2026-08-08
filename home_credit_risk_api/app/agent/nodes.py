from typing import Any,Literal
from app.agent.state import UnderwritingState
from app.report_input import build_report_input
from langgraph.types import interrupt, Command
from app.audit import log_audit_event
from app.report_generator import generate_final_report_from_input




def prepare_case(state: UnderwritingState) -> dict[str,Any]:
    """
    Builds the underwriting input and initializes the decision information required by the workflow.
    """
    case_id = state["case_id"]
    
    report_input = build_report_input(case_id=case_id,save_output=False)
    
    policy_summary = report_input["deterministic_policy_evaluation"]["summary"]

    # collects every rule responsible for blocking straight-through processing.
    triggered_rules = (policy_summary["failed_rule_ids"] + policy_summary["manual_review_rule_ids"] + policy_summary["unknown_required_rule_ids"])

    log_audit_event(
        case_id=case_id,
        review_id=state["review_id"],
        event_type="system_evaluation",
        recommendation=policy_summary["recommendation"],
        triggered_rules=triggered_rules,
        decision_status="processing",
    )
    
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

    human_response = interrupt({
        "case_id": state["case_id"],
        "recommendation": state["recommendation"],
        "message": "Human underwriting review required.",
        "allowed_decisions": ["approve", "decline","request_more_information"],
    })
    
    human_decision = human_response["decision"]

    if human_decision == "request_more_information":
        decision_status = "awaiting_more_information"
    else:
        decision_status = "completed"

    log_audit_event(
        case_id=state["case_id"],
        review_id=state["review_id"],
        event_type="human_decision",
        recommendation=state["recommendation"],
        human_decision=human_decision,
        human_comment=human_response.get("comment"),
        decision_status=decision_status,
    )


    return {
        "human_decision": human_decision,
        "human_comment": human_response.get("comment"),
        "decision_status": decision_status,
        "final_report": None
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


def generate_final_report(state: UnderwritingState) -> dict[str, Any]:
    """
    Generates the final grounded underwriting report after the human-review outcome is available.
    """

    report = generate_final_report_from_input(
        report_input=state["report_input"],
        human_decision=state["human_decision"],
        human_comment=state.get("human_comment"),
        decision_status=state["decision_status"],
    )

    return {
        "final_report": report.model_dump(
            mode="json"
        ),
    }

def route_after_human_review(state: UnderwritingState,) -> Literal["final_report", "awaiting_information"]:
    """
    Sends completed human decisions to final reporting while keeping information-request cases open for further review.
    """

    if state["decision_status"] == "awaiting_more_information":
        return "awaiting_information"

    return "final_report"