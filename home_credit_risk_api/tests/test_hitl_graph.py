from langgraph.types import Command

from app.agent.graph import underwriting_graph


def test_human_approval_generates_final_report():
    """
    Verifies that an approved manual-review case resumes and produces a final underwriting report.
    """

    config = {
        "configurable": {
            "thread_id": "test-approve-001",
        }
    }

    paused = underwriting_graph.invoke(
        {
            "case_id": "DEMO-005",
        },
        config=config,
    )

    assert "__interrupt__" in paused

    result = underwriting_graph.invoke(
        Command(
            resume={
                "decision": "approve",
                "comment": "Income documents verified.",
            }
        ),
        config=config,
    )

    assert result["decision_status"] == "completed"
    assert result["human_decision"] == "approve"
    assert result["final_report"] is not None

    assert (
        result["final_report"]["recommendation"]["recommendation"]
        == "manual_review"
    )

    assert (
        result["final_report"]["human_review_outcome"]["decision"]
        == "approve"
    )


def test_human_decline_generates_final_report():
    """
    Verifies that a declined manual-review case produces a final report while preserving the system recommendation.
    """

    config = {
        "configurable": {
            "thread_id": "test-decline-001",
        }
    }

    paused = underwriting_graph.invoke(
        {
            "case_id": "DEMO-005",
        },
        config=config,
    )

    assert "__interrupt__" in paused

    result = underwriting_graph.invoke(
        Command(
            resume={
                "decision": "decline",
                "comment": "Income evidence was insufficient.",
            }
        ),
        config=config,
    )

    assert result["decision_status"] == "completed"
    assert result["human_decision"] == "decline"
    assert result["final_report"] is not None

    assert (
        result["final_report"]["human_review_outcome"]["decision"]
        == "decline"
    )


def test_request_more_information_does_not_generate_final_report():
    """
    Verifies that unresolved information requests remain open and do not create a premature final underwriting report.
    """

    config = {
        "configurable": {
            "thread_id": "test-more-info-001",
        }
    }

    paused = underwriting_graph.invoke(
        {
            "case_id": "DEMO-005",
        },
        config=config,
    )

    assert "__interrupt__" in paused

    result = underwriting_graph.invoke(
        Command(
            resume={
                "decision": "request_more_information",
                "comment": "Please provide verified income documents.",
            }
        ),
        config=config,
    )

    assert result["decision_status"] == "awaiting_more_information"
    assert result["human_decision"] == "request_more_information"

    assert result.get("final_report") is None