from app.policy_retriever import retrieve_policy_chunks


TEST_CASES = [
    {
        "query": (
            "When should a loan application be sent "
            "for manual review?"
        ),
        "expected_sections": {"18"},
    },
    {
        "query": (
            "What happens when the applicant's "
            "FOIR is above 50 percent?"
        ),
        "expected_sections": {"15", "19"},
    },
    {
        "query": (
            "What is the minimum acceptable "
            "credit bureau score?"
        ),
        "expected_sections": {"16"},
    },
    {
        "query": (
            "Can an unsecured loan be granted "
            "without verifiable income?"
        ),
        "expected_sections": {"5", "14", "19"},
    },
    {
        "query": (
            "Which documents are required before "
            "processing a loan application?"
        ),
        "expected_sections": {"20", "5"},
    },
]


hits = 0

for test_case in TEST_CASES:
    results = retrieve_policy_chunks(
        query=test_case["query"],
        top_k=3,
    )

    retrieved_sections = {
        str(result["metadata"]["section_number"])
        for result in results
    }

    passed = bool(
        retrieved_sections
        & test_case["expected_sections"]
    )

    hits += int(passed)

    print("\nQuery:", test_case["query"])
    print("Expected:", test_case["expected_sections"])
    print("Retrieved:", retrieved_sections)
    print("Passed:", passed)


recall_at_3 = hits / len(TEST_CASES)

print("\nRecall@3:", round(recall_at_3, 3))