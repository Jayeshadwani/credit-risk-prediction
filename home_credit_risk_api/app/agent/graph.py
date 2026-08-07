import sqlite3
from pathlib import Path
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from app.agent.nodes import (
    continue_automatically,
    mark_for_human_review,
    prepare_case,
    route_case
)
from app.agent.state import UnderwritingState

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DB = PROJECT_ROOT / "checkpoints.sqlite"


# Stores graph checkpoints on disk so interrupted underwriting cases can be resumed even after the Python process restarts.
connection = sqlite3.connect(CHECKPOINT_DB,check_same_thread=False)

checkpointer = SqliteSaver(connection)



def build_underwriting_graph():
    """
    Builds the persistent underwriting workflow with automated and human-review execution paths.
    """
    builder = StateGraph(UnderwritingState)


    # nodes
    builder.add_node("prepare_case",prepare_case)
    builder.add_node("human_review",mark_for_human_review)
    builder.add_node("automatic",continue_automatically)

    # edges
    builder.add_edge(START,"prepare_case")
    builder.add_conditional_edges("prepare_case",route_case,{"human_review": "human_review","automatic": "automatic"})
    builder.add_edge("human_review",END)
    builder.add_edge("automatic",END)

    return builder.compile(checkpointer=
    checkpointer)



underwriting_graph = build_underwriting_graph()

