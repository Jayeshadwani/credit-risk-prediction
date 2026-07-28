import argparse
import json
from pathlib import Path
from typing import Any

from app.policy_retriever import retrieve_policy_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_CASES_DIR = PROJECT_ROOT / "demo_cases"

TOP_K_PER_QUERY = 3
CANDIDATE_K = 8
MAX_FINAL_CHUNKS = 4


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


from typing import Any


def contains_any(feature_name: str, terms: list[str]) -> bool:
    return any(term in feature_name for term in terms)


def create_factor_query(factor: dict[str, Any]) -> str:
    feature = str(factor.get("feature", "")).strip()

    if not feature:
        raise ValueError("Factor must contain a feature name.")

    display_name = str(
        factor.get("display_name")
        or feature.replace("_", " ").title()
    ).strip()

    feature_upper = feature.upper()

    # External creditworthiness indicators
    if feature_upper.startswith("EXT_SOURCE"):
        return (
            "What policy rules apply when external creditworthiness "
            "indicators suggest a weak or borderline credit profile, "
            "and when is manual review or decline required?"
        )

    # Requested credit relative to annuity
    if "CREDIT_ANNUITY_RATIO" in feature_upper:
        return (
            "What affordability, repayment-capacity, and tenor rules "
            "apply when the requested credit amount is high relative "
            "to the loan annuity?"
        )

    # Requested credit relative to financed goods
    if "CREDIT_GOODS_RATIO" in feature_upper:
        return (
            "What exposure-limit or product-policy rules apply when "
            "the requested credit amount is high relative to the "
            "price of the financed goods?"
        )

    # Existing or remaining instalment obligations
    if contains_any(
        feature_upper,
        [
            "POS_CNT_INSTALMENT",
            "CNT_INSTALMENT_FUTURE",
            "POS_CASH",
            "INSTALMENT_FUTURE",
        ],
    ):
        return (
            "What affordability or manual-review rules apply when "
            "an applicant has many remaining instalments, active "
            "consumer loans, or ongoing repayment obligations?"
        )

    # Adverse repayment behaviour
    if contains_any(
        feature_upper,
        [
            "DPD",
            "LATE_PAYMENT",
            "OVERDUE",
            "NPA",
            "WRITE_OFF",
            "SETTLED",
            "PAYMENT_DELAY",
            "DEFAULT_FLAG",
        ],
    ):
        return (
            "What policy action applies to recent 30-plus or 60-plus "
            "days-past-due events, overdue accounts, NPA, write-offs, "
            "settled accounts, or other adverse repayment history?"
        )

    # Credit bureau enquiry activity
    if contains_any(
        feature_upper,
        [
            "AMT_REQ_CREDIT_BUREAU",
            "BUREAU_ENQUIRY",
            "CREDIT_ENQUIRY",
            "ENQUIRIES",
        ],
    ):
        return (
            "What manual-review rules apply when an applicant has "
            "multiple recent credit-bureau enquiries or unusually "
            "high credit-seeking activity?"
        )

    # Debt and affordability ratios
    if contains_any(
        feature_upper,
        [
            "CREDIT_INCOME",
            "ANNUITY_INCOME",
            "DEBT_CREDIT",
            "DEBT_INCOME",
            "FOIR",
            "PAYMENT_RATE",
            "RESIDUAL_INCOME",
        ],
    ):
        return (
            "What FOIR, credit-to-income, repayment-burden, or "
            "residual-income thresholds determine whether an "
            "application should continue, receive manual review, "
            "or be declined?"
        )

    # Income and employment stability
    if contains_any(
        feature_upper,
        [
            "DAYS_EMPLOYED",
            "EMPLOYMENT",
            "INCOME",
            "ORGANIZATION_TYPE",
            "OCCUPATION_TYPE",
        ],
    ):
        return (
            "What income-verification, employment-stability, and "
            "income-discrepancy rules apply during loan underwriting?"
        )

    # Credit bureau debt profile
    if contains_any(
        feature_upper,
        [
            "BUREAU_DEBT",
            "ACTIVE_CREDIT",
            "CREDIT_SUM_DEBT",
            "BUREAU_CREDIT",
        ],
    ):
        return (
            "What credit-bureau and affordability rules apply when "
            "an applicant has substantial existing debt or multiple "
            "active credit accounts?"
        )

    # Social-circle or indirect risk information
    if contains_any(
        feature_upper,
        [
            "SOCIAL_DEFAULT",
            "SOCIAL_CIRCLE",
        ],
    ):
        return (
            "What policy and governance rules apply when indirect or "
            "alternative-data indicators suggest elevated credit risk, "
            "and when should human review be required?"
        )

    # Applicant age
    if contains_any(
        feature_upper,
        [
            "DAYS_BIRTH",
            "AGE_YEARS",
        ],
    ):
        return (
            "What applicant-age and age-at-loan-maturity eligibility "
            "rules apply to the requested loan?"
        )

    # Requested loan amount or goods value
    if contains_any(
        feature_upper,
        [
            "AMT_CREDIT",
            "AMT_GOODS_PRICE",
            "CREDIT_AMOUNT",
        ],
    ):
        return (
            "What product limits, exposure caps, or manual-review "
            "conditions apply to the requested loan amount?"
        )

    # Generic fallback
    return (
        f"What underwriting, eligibility, affordability, or "
        f"manual-review policy is relevant when {display_name} "
        f"is identified as an adverse model risk factor?"
    )

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
            "BUREAU_DEBT_CREDIT_RATIO",
            "DEBT_CREDIT_RATIO",
            "BUREAU_CREDIT_RATIO",
        ]
        ):
        return (
            "What credit-bureau and affordability conditions support continuing "
            "an application when existing bureau debt is manageable relative "
            "to the applicant's total available credit?"
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
            "INST_PAYMENT_DIFFERENCE",
            "PAYMENT_DIFFERENCE",
            "PAYMENT_DIFF",
            "AMT_PAYMENT_DIFFERENCE",
        ]
    ):
        return (
            "What repayment-history conditions support continuing a loan "
            "application when instalment payments are consistent with scheduled "
            "payment amounts and there is no material underpayment or adverse "
            "overdue history?"
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


def build_focused_queries(
    explanation: dict[str, Any],
) -> list[dict[str, str]]:
    risk_category = (
        explanation["model_output"]["risk_category"]
        .strip()
        .lower()
    )

    queries = [
        {
            "query_id": "risk-band",
            "reason": (
                f"Applicant is classified as "
                f"{risk_category} risk."
            ),
            "query": (
                f"What decision, approval, or manual-review "
                f"requirements apply to a {risk_category}-risk "
                f"loan applicant?"
            ),
        }
    ]

    risk_factors = explanation.get(
        "top_risk_factors",
        [],
    )

    protective_factors = explanation.get(
        "top_protective_factors",
        [],
    )

    risk_factor_limit = 2 if risk_category == "low" else 3

    for index, factor in enumerate(
        risk_factors[:risk_factor_limit],
        start=1,
    ):
        queries.append({
            "query_id": f"risk-factor-{index}",
            "reason": (
                f"{factor['display_name']} increased the "
                "model's predicted default risk."
            ),
            "query": create_factor_query(factor),
        })

    if risk_category == "low":
        for index, factor in enumerate(
            protective_factors[:2],
            start=1,
        ):
            queries.append({
                "query_id": f"protective-factor-{index}",
                "reason": (
                    f"{factor['display_name']} reduced the "
                    "model's predicted default risk."
                ),
                "query": create_protective_factor_query(
                    factor
                ),
            })

    # Remove duplicate query text.
    unique_queries = []
    seen_queries = set()

    for query_item in queries:
        normalized_query = " ".join(
            query_item["query"].lower().split()
        )

        if normalized_query in seen_queries:
            continue

        seen_queries.add(normalized_query)
        unique_queries.append(query_item)

    return unique_queries


def retrieve_and_merge(
    queries: list[dict[str, str]],
) -> list[dict[str, Any]]:

    merged_chunks: dict[str, dict[str, Any]] = {}

    for query_item in queries:
        results = retrieve_policy_chunks(
            query=query_item["query"],
            top_k=TOP_K_PER_QUERY,
            candidate_k=CANDIDATE_K,
        )

        for rank, result in enumerate(results, start=1):
            chunk_id = result["chunk_id"]

            if chunk_id not in merged_chunks:
                merged_chunks[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "matched_queries": [],
                    "fusion_score": 0.0,
                    "best_reranker_score": result[
                        "reranker_score"
                    ],
                    "best_vector_distance": result[
                        "vector_distance"
                    ],
                }

            chunk = merged_chunks[chunk_id]

            chunk["matched_queries"].append({
                "query_id": query_item["query_id"],
                "reason": query_item["reason"],
                "rank": rank,
            })

            # Higher-ranked results receive more weight.
            chunk["fusion_score"] += 1 / rank

            chunk["best_reranker_score"] = max(
                chunk["best_reranker_score"],
                result["reranker_score"],
            )

            chunk["best_vector_distance"] = min(
                chunk["best_vector_distance"],
                result["vector_distance"],
            )

    ranked_chunks = sorted(
        merged_chunks.values(),
        key=lambda chunk: (
            chunk["fusion_score"],
            chunk["best_reranker_score"],
        ),
        reverse=True,
    )

    return ranked_chunks[:MAX_FINAL_CHUNKS]


def main(case_id: str) -> None:
    case_directory = DEMO_CASES_DIR / case_id

    explanation_path = (
        case_directory / "risk_explanation.json"
    )

    if not explanation_path.exists():
        raise FileNotFoundError(
            f"Risk explanation not found: {explanation_path}"
        )

    explanation = load_json(explanation_path)

    queries = build_focused_queries(explanation)
    policy_chunks = retrieve_and_merge(queries)

    output = {
        "demo_case_id": case_id,
        "generated_queries": queries,
        "retrieval_configuration": {
            "top_k_per_query": TOP_K_PER_QUERY,
            "candidate_k": CANDIDATE_K,
            "maximum_final_chunks": MAX_FINAL_CHUNKS,
            "policy_types": [
                "source_derived",
                "synthetic_demo",
            ],
        },
        "retrieved_policy_chunks": policy_chunks,
    }

    output_path = (
        case_directory / "retrieved_policy_context.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

    print(f"Case: {case_id}")
    print(f"Focused queries: {len(queries)}")
    print(f"Final policy chunks: {len(policy_chunks)}")
    print(f"Saved to: {output_path}")

    for index, chunk in enumerate(policy_chunks, start=1):
        print(
            f"{index}. Section "
            f"{chunk['metadata']['section_number']} - "
            f"{chunk['metadata']['section_title']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "case_id",
        help="Demo case ID, for example DEMO-001",
    )

    arguments = parser.parse_args()

    main(arguments.case_id)