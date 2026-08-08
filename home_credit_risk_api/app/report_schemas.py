from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


DecisionRecommendation = Literal[
    "eligible_for_straight_through_processing",
    "conditional_approval",
    "manual_review",
    "decline_recommendation",
]

RiskCategory = Literal[
    "low",
    "medium",
    "high",
]

PolicyType = Literal[
    "source_derived",
    "synthetic_demo",
]

PolicyRuleId = Literal[
    "MODEL_RISK_BAND",
    "CREDIT_TO_INCOME",
    "FOIR",
    "REPAYMENT_HISTORY_DPD",
    "KYC_AND_CONSENT",
    "DOCUMENT_COMPLETENESS",
]

HumanDecision = Literal[
    "approve",
    "decline",
    "request_more_information",
]

class StrictBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class HumanReviewOutcome(StrictBaseModel):
    decision: HumanDecision
    comment: str | None = None

    status: Literal[
        "completed",
        "awaiting_more_information",
    ]

class ModelAssessment(StrictBaseModel):
    default_probability: float = Field(
        ge=0,
        le=1,
    )
    risk_category: RiskCategory
    model_version: str

    interpretation: str = Field(
        description=(
            "Plain-English interpretation of the model output. "
            "Must not describe the prediction as a final lending decision."
        )
    )


class ModelFactor(StrictBaseModel):
    feature: str
    display_name: str

    shap_value: float

    impact_direction: Literal[
        "increases_default_risk",
        "reduces_default_risk",
    ]

    explanation: str = Field(
        description=(
            "Explain how the feature affected the model prediction. "
            "Do not claim that the feature caused default."
        )
    )


class PolicyCitation(StrictBaseModel):
    chunk_id: str
    section_number: str
    section_title: str
    policy_type: PolicyType

    page_start: int | None = None
    page_end: int | None = None


class PolicyFinding(StrictBaseModel):
    rule_id: PolicyRuleId
    finding: str
    applicant_relevance: str = Field(
        description=(
            "Explain why the policy rule is relevant to this applicant."
        )
    )
    citation: PolicyCitation


class UnderwritingRecommendation(StrictBaseModel):
    recommendation: DecisionRecommendation

    rationale: str = Field(
        description=(
            "Combined rationale based only on model output, applicant "
            "information and retrieved policy evidence."
        )
    )

    human_review_required: bool

    basis: list[
        Literal[
            "model_risk_band",
            "policy_rule",
            "missing_information",
            "hard_decline_condition",
        ]
    ]

class UnderwritingReport(StrictBaseModel):
    report_version: Literal["1.0"]
    demo_case_id: str

    executive_summary: str
    model_assessment: ModelAssessment

    key_risk_factors: list[ModelFactor] = Field(
        max_length=3,
    )

    protective_factors: list[ModelFactor] = Field(
        max_length=2,
    )

    policy_findings: list[PolicyFinding] = Field(
        min_length=1,
        max_length=6,
    )

    recommendation: UnderwritingRecommendation

    missing_information: list[str]
    recommended_actions: list[str]

    limitations: list[str] = Field(
        description=(
            "Must mention that SHAP explains model contribution rather "
            "than causality, synthetic policy rules are demonstration-only, "
            "and final sanction remains with an authorized human."
        )
    ) 


class FinalUnderwritingReport(UnderwritingReport):
    """
    Extends the generated underwriting report with the final human-review outcome.
    """

    human_review_outcome: HumanReviewOutcome