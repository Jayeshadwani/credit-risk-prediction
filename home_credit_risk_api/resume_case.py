from langgraph.types import Command

from app.agent.graph import underwriting_graph


config = {
    "configurable": {
        "thread_id": "DEMO-005-PERSIST-1",
    }
}


# Loads the previously persisted underwriting checkpoint
# and supplies the underwriter's decision.
result = underwriting_graph.invoke(
    Command(
        resume={
            "decision": "approve",
            "comment": "Income documents verified.",
        }
    ),
    config=config,
)


print("Human Decision:", result["human_decision"])
print("Comment:", result["human_comment"])
print("Status:", result["decision_status"])