from app.agent.graph import underwriting_graph


config = {
    "configurable": {
        "thread_id": "DEMO-005-PERSIST-1",
    }
}


# Starts a new underwriting case and intentionally
# leaves it paused for human review.
result = underwriting_graph.invoke(
    {
        "case_id": "DEMO-005",
    },
    config=config,
)


print("Workflow paused:")
print(result["__interrupt__"][0].value)