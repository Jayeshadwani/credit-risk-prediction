from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import (
    CrossEncoder,
    SentenceTransformer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VECTOR_STORE_PATH = (
    PROJECT_ROOT
    / "vector_store"
    / "chroma"
)

COLLECTION_NAME = "loan_underwriting_policy_v1"

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

ALLOWED_POLICY_TYPES = [
    "source_derived",
    "synthetic_demo",
]

RERANKER_MODEL_NAME = (
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)

DEFAULT_CANDIDATE_K = 8


reranker = CrossEncoder(
    RERANKER_MODEL_NAME
)

# Load once when the application starts.
embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

chroma_client = chromadb.PersistentClient(
    path=str(VECTOR_STORE_PATH)
)

collection = chroma_client.get_collection(
    name=COLLECTION_NAME
)

from typing import Any


def create_protective_factor_query(
    factor: dict[str, Any],
) -> str:
    feature = str(factor.get("feature", "")).strip()

    if not feature:
        raise ValueError("Factor must contain a feature name.")

    display_name = str(
        factor.get("display_name")
        or feature.replace("_", " ").title()
    ).strip()

    feature_upper = feature.upper()

    if feature_upper.startswith("EXT_SOURCE"):
        return (
            "What creditworthiness or credit-bureau conditions support "
            "continuing a loan application when the applicant has an "
            "acceptable external credit profile?"
        )

    if any(
        term in feature_upper
        for term in [
            "DPD",
            "LATE_PAYMENT",
            "OVERDUE",
            "NPA",
            "WRITE_OFF",
            "SETTLED",
            "PAYMENT_DELAY",
        ]
    ):
        return (
            "How does a clean repayment history with no recent overdue, "
            "days-past-due, NPA, write-off, or settled account support "
            "continuing a loan application?"
        )

    if any(
        term in feature_upper
        for term in [
            "CREDIT_INCOME",
            "ANNUITY_INCOME",
            "DEBT_INCOME",
            "DEBT_CREDIT",
            "FOIR",
            "PAYMENT_RATE",
            "RESIDUAL_INCOME",
        ]
    ):
        return (
            "What affordability conditions support continuing an "
            "application when FOIR, credit-to-income, repayment burden, "
            "and residual income are within acceptable limits?"
        )

    if "CREDIT_ANNUITY_RATIO" in feature_upper:
        return (
            "What repayment-capacity and tenor conditions support "
            "continuing an application when the requested credit amount "
            "is manageable relative to the loan annuity?"
        )

    if "CREDIT_GOODS_RATIO" in feature_upper:
        return (
            "What product and exposure conditions support continuing "
            "an application when the requested credit amount is "
            "reasonable relative to the financed goods value?"
        )

    if any(
        term in feature_upper
        for term in [
            "POS_CNT_INSTALMENT",
            "CNT_INSTALMENT_FUTURE",
            "POS_CASH",
            "INSTALMENT_FUTURE",
        ]
    ):
        return (
            "What policy conditions support continuing an application "
            "when remaining instalments and ongoing repayment obligations "
            "appear manageable?"
        )

    if any(
        term in feature_upper
        for term in [
            "DAYS_EMPLOYED",
            "EMPLOYMENT",
            "INCOME",
            "ORGANIZATION_TYPE",
            "OCCUPATION_TYPE",
        ]
    ):
        return (
            "What verified-income and employment-stability conditions "
            "support continuing a loan application?"
        )

    if any(
        term in feature_upper
        for term in [
            "BUREAU_DEBT",
            "ACTIVE_CREDIT",
            "CREDIT_SUM_DEBT",
        ]
    ):
        return (
            "What credit-bureau and affordability conditions support "
            "continuing an application when existing debt and active "
            "credit exposure appear manageable?"
        )

    if any(
        term in feature_upper
        for term in [
            "DAYS_BIRTH",
            "AGE_YEARS",
        ]
    ):
        return (
            "What age and age-at-loan-maturity conditions support "
            "applicant eligibility?"
        )

    return (
        f"What underwriting policy considerations may support continuing "
        f"an application when {display_name} reduces the model's predicted "
        f"default risk?"
    )

def retrieve_policy_chunks(
    query: str,
    top_k: int = 3,
    candidate_k: int = DEFAULT_CANDIDATE_K,
) -> list[dict[str, Any]]:
    """
    Retrieve candidates from Chroma and rerank them
    using a CrossEncoder.
    """

    query = query.strip()

    if not query:
        raise ValueError("Query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if candidate_k < top_k:
        raise ValueError(
            "candidate_k must be greater than "
            "or equal to top_k."
        )

    query_embedding = embedding_model.encode_query(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=candidate_k,
        where={
            "policy_type": {
                "$in": ALLOWED_POLICY_TYPES
            }
        },
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    candidates = []

    for chunk_id, document, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        candidates.append({
            "chunk_id": chunk_id,
            "text": document,
            "metadata": metadata,
            "vector_distance": float(distance),
        })

    query_document_pairs = [
        [query, candidate["text"]]
        for candidate in candidates
    ]

    reranker_scores = reranker.predict(
        query_document_pairs
    )

    for candidate, score in zip(
        candidates,
        reranker_scores,
    ):
        candidate["reranker_score"] = float(score)

    # Higher reranker score means more relevant.
    reranked_candidates = sorted(
        candidates,
        key=lambda item: item["reranker_score"],
        reverse=True,
    )

    return reranked_candidates[:top_k]