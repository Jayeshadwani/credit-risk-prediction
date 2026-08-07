from langgraph.types import Command

from app.agent.graph import underwriting_graph


config = {
    "configurable": {
        "thread_id": "DEMO-005",
    }
}

# Start the underwriting workflow.
# DEMO-005 should pause because human review is required.
result = underwriting_graph.invoke(
    {
        "case_id": "DEMO-005",
    },
    config=config,
)

print("Workflow paused:")
print(result["__interrupt__"][0].value)


# Simulate the human underwriter approving the case.
# The same thread_id resumes the workflow from its saved checkpoint.
result = underwriting_graph.invoke(
    Command(
        resume={
            "decision": "approve",
            "comment": "Income documents verified.",
        }
    ),
    config=config,
)


print()
print("Human Decision:", result["human_decision"])
print("Comment:", result["human_comment"])
print("Status:", result["decision_status"])