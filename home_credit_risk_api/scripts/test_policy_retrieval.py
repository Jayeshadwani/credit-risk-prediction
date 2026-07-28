import json

from app.policy_retriever import (
    retrieve_policy_chunks,
)


query = (
    "When should a loan application be sent "
    "for manual review?"
)

results = retrieve_policy_chunks(
    query=query,
    top_k=5,
)

for rank, result in enumerate(results, start=1):
    metadata = result["metadata"]

    print(f"\nResult {rank}")
    print("-" * 60)
    print("Section:", metadata["section_number"])
    print("Title:", metadata["section_title"])
    print("Policy type:", metadata["policy_type"])
    print("Vector distance:",round(result["vector_distance"], 4))
    print()
    print("Reranker score:",round(result["reranker_score"], 4))

