from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


RuleStatus = Literal[
    "pass",
    "manual_review",
    "fail",
    "unknown",
    "not_applicable",
]

RuleAction = Literal[
    "continue",
    "manual_review",
    "decline_recommendation",
    "none",
]

DecisionRecommendation = Literal[
    "eligible_for_straight_through_processing",
    "manual_review",
    "decline_recommendation",
]


class RuleCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_number: str
    section_title: str
    policy_type: Literal[
        "source_derived",
        "synthetic_demo",
    ]


class PolicyRuleEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id: str
    rule_name: str

    evaluation_status: RuleStatus
    recommended_action: RuleAction

    applicant_value: str | int | float | bool | None
    threshold: str | None

    reason: str
    required_for_stp: bool

    citation: RuleCitation


class PolicyEvaluationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recommendation: DecisionRecommendation
    human_review_required: bool

    failed_rule_ids: list[str]
    manual_review_rule_ids: list[str]
    unknown_required_rule_ids: list[str]


class DeterministicPolicyEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rules: list[PolicyRuleEvaluation]
    summary: PolicyEvaluationSummary


def find_value(
    payload: Any,
    aliases: set[str],
) -> Any:
    """
    Recursively find the first non-null value whose key
    exactly matches one of the supplied aliases.
    """
    normalized_aliases = {
        alias.lower()
        for alias in aliases
    }

    if isinstance(payload, dict):
        for key, value in payload.items():
            if (
                str(key).lower() in normalized_aliases
                and value is not None
            ):
                return value

        for value in payload.values():
            result = find_value(
                value,
                normalized_aliases,
            )

            if result is not None:
                return result

    elif isinstance(payload, list):
        for item in payload:
            result = find_value(
                item,
                normalized_aliases,
            )

            if result is not None:
                return result

    return None


def as_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value

    if isinstance(value, int) and value in {0, 1}:
        return bool(value)

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {
            "true",
            "yes",
            "y",
            "1",
            "verified",
            "complete",
            "valid",
        }:
            return True

        if normalized in {
            "false",
            "no",
            "n",
            "0",
            "unverified",
            "incomplete",
            "invalid",
        }:
            return False

    return None


def evaluate_model_risk_band(
    risk_category: str,
) -> PolicyRuleEvaluation:
    risk_category = risk_category.lower()

    if risk_category == "low":
        status: RuleStatus = "pass"
        action: RuleAction = "continue"
        reason = (
            "The applicant is in the low model-risk band. "
            "Straight-through processing still requires all "
            "other mandatory policy checks to pass."
        )
    else:
        status = "manual_review"
        action = "manual_review"
        reason = (
            f"The applicant is in the {risk_category} model-risk "
            "band, which requires human review under the "
            "demonstration policy."
        )

    return PolicyRuleEvaluation(
        rule_id="MODEL_RISK_BAND",
        rule_name="Model risk-band treatment",
        evaluation_status=status,
        recommended_action=action,
        applicant_value=risk_category,
        threshold="medium or high requires manual review",
        reason=reason,
        required_for_stp=True,
        citation=RuleCitation(
            section_number="18",
            section_title=(
                "Decision Outcomes and Manual-Review Triggers"
            ),
            policy_type="synthetic_demo",
        ),
    )


def evaluate_credit_to_income(
    summary: dict[str, Any],
) -> PolicyRuleEvaluation:
    ratio = as_float(
        find_value(
            summary,
            {
                "credit_income_ratio",
                "credit_to_income_ratio",
            },
        )
    )

    income_verified = as_bool(
        find_value(
            summary,
            {
                "income_verified",
                "verified_income",
                "annual_income_verified",
            },
        )
    )

    if ratio is None:
        status: RuleStatus = "unknown"
        action: RuleAction = "manual_review"
        reason = (
            "Credit-to-annual-income ratio is unavailable."
        )

    elif income_verified is not True:
        status = "unknown"
        action = "manual_review"
        reason = (
            f"The reported ratio is {ratio:.3f}, but verified "
            "annual income has not been confirmed. The policy "
            "threshold cannot be deterministically applied."
        )

    elif ratio <= 3:
        status = "pass"
        action = "continue"
        reason = (
            f"The verified credit-to-income ratio of {ratio:.3f} "
            "is within the standard range of 3.0x or below."
        )

    elif ratio <= 5:
        status = "manual_review"
        action = "manual_review"
        reason = (
            f"The verified credit-to-income ratio of {ratio:.3f} "
            "is above 3.0x and no higher than 5.0x."
        )

    else:
        status = "fail"
        action = "decline_recommendation"
        reason = (
            f"The verified credit-to-income ratio of {ratio:.3f} "
            "exceeds the hard threshold of 5.0x."
        )

    return PolicyRuleEvaluation(
        rule_id="CREDIT_TO_INCOME",
        rule_name="Credit-to-annual-income ratio",
        evaluation_status=status,
        recommended_action=action,
        applicant_value=ratio,
        threshold=(
            "<=3.0x pass; >3.0x to 5.0x review; "
            ">5.0x hard failure"
        ),
        reason=reason,
        required_for_stp=True,
        citation=RuleCitation(
            section_number="15",
            section_title="Affordability and Exposure Limits",
            policy_type="synthetic_demo",
        ),
    )


def evaluate_foir(
    summary: dict[str, Any],
) -> PolicyRuleEvaluation:
    foir = as_float(
        find_value(
            summary,
            {
                "foir",
                "foir_ratio",
                "fixed_obligation_to_income_ratio",
            },
        )
    )

    income_verified = as_bool(
        find_value(
            summary,
            {
                "income_verified",
                "verified_income",
                "monthly_income_verified",
            },
        )
    )

    if foir is None:
        status: RuleStatus = "unknown"
        action: RuleAction = "manual_review"
        reason = "FOIR is unavailable."

    elif foir > 1:
        status = "unknown"
        action = "manual_review"
        reason = (
            "FOIR must be supplied as a decimal ratio between "
            "0 and 1, such as 0.40 for 40%."
        )

    elif income_verified is not True:
        status = "unknown"
        action = "manual_review"
        reason = (
            f"The reported FOIR is {foir:.3f}, but verified "
            "income has not been confirmed."
        )

    elif foir <= 0.40:
        status = "pass"
        action = "continue"
        reason = (
            f"The verified FOIR of {foir:.1%} is within "
            "the standard limit of 40%."
        )

    elif foir <= 0.50:
        status = "manual_review"
        action = "manual_review"
        reason = (
            f"The verified FOIR of {foir:.1%} is above 40% "
            "but no higher than 50%."
        )

    else:
        status = "fail"
        action = "decline_recommendation"
        reason = (
            f"The verified FOIR of {foir:.1%} exceeds 50%."
        )

    return PolicyRuleEvaluation(
        rule_id="FOIR",
        rule_name="Fixed Obligation to Income Ratio",
        evaluation_status=status,
        recommended_action=action,
        applicant_value=foir,
        threshold=(
            "<=40% pass; >40% to 50% review; "
            ">50% hard failure"
        ),
        reason=reason,
        required_for_stp=True,
        citation=RuleCitation(
            section_number="15",
            section_title="Affordability and Exposure Limits",
            policy_type="synthetic_demo",
        ),
    )


def evaluate_repayment_history(
    summary: dict[str, Any],
) -> PolicyRuleEvaluation:
    max_dpd_6m = as_float(
        find_value(
            summary,
            {
                "max_dpd_6m",
                "maximum_dpd_6_months",
            },
        )
    )

    max_dpd_12m = as_float(
        find_value(
            summary,
            {
                "max_dpd_12m",
                "maximum_dpd_12_months",
            },
        )
    )

    if max_dpd_12m is not None and max_dpd_12m >= 60:
        status: RuleStatus = "fail"
        action: RuleAction = "decline_recommendation"
        value: str | float = (
            f"6m={max_dpd_6m}, 12m={max_dpd_12m}"
        )
        reason = (
            "A 60-plus days-past-due event was identified "
            "within the previous 12 months."
        )

    elif max_dpd_6m is not None and max_dpd_6m >= 30:
        status = "manual_review"
        action = "manual_review"
        value = f"6m={max_dpd_6m}, 12m={max_dpd_12m}"
        reason = (
            "A 30-plus days-past-due event was identified "
            "within the previous six months."
        )

    elif max_dpd_6m is not None and max_dpd_12m is not None:
        status = "pass"
        action = "continue"
        value = f"6m={max_dpd_6m}, 12m={max_dpd_12m}"
        reason = (
            "No supplied DPD value breached the specified "
            "repayment-history thresholds."
        )

    else:
        status = "unknown"
        action = "manual_review"
        value = None
        reason = (
            "The required six-month and twelve-month DPD "
            "history is incomplete."
        )

    return PolicyRuleEvaluation(
        rule_id="REPAYMENT_HISTORY_DPD",
        rule_name="Recent days-past-due history",
        evaluation_status=status,
        recommended_action=action,
        applicant_value=value,
        threshold=(
            "30+ DPD in 6 months requires review; "
            "60+ DPD in 12 months is a hard failure"
        ),
        reason=reason,
        required_for_stp=True,
        citation=RuleCitation(
            section_number="16",
            section_title=(
                "Credit Bureau and Repayment-History Rules"
            ),
            policy_type="synthetic_demo",
        ),
    )


def evaluate_kyc_and_consent(
    summary: dict[str, Any],
) -> PolicyRuleEvaluation:
    kyc_verified = as_bool(
        find_value(
            summary,
            {
                "kyc_verified",
                "identity_verified",
                "kyc_status",
            },
        )
    )

    consent_valid = as_bool(
        find_value(
            summary,
            {
                "consent_valid",
                "valid_consent",
                "consent_status",
            },
        )
    )

    if kyc_verified is False or consent_valid is False:
        status: RuleStatus = "fail"
        action: RuleAction = "decline_recommendation"
        reason = (
            "KYC verification or valid customer consent failed."
        )

    elif kyc_verified is True and consent_valid is True:
        status = "pass"
        action = "continue"
        reason = (
            "KYC verification and valid customer consent "
            "are both confirmed."
        )

    else:
        status = "unknown"
        action = "manual_review"
        reason = (
            "KYC verification and valid consent are not "
            "both confirmed."
        )

    return PolicyRuleEvaluation(
        rule_id="KYC_AND_CONSENT",
        rule_name="KYC and customer consent",
        evaluation_status=status,
        recommended_action=action,
        applicant_value=(
            f"kyc={kyc_verified}, consent={consent_valid}"
        ),
        threshold="Both KYC and valid consent must pass",
        reason=reason,
        required_for_stp=True,
        citation=RuleCitation(
            section_number="19",
            section_title="Automatic Decline Conditions",
            policy_type="synthetic_demo",
        ),
    )


def evaluate_document_completeness(
    summary: dict[str, Any],
) -> PolicyRuleEvaluation:
    documents_complete = as_bool(
        find_value(
            summary,
            {
                "documents_complete",
                "document_completeness",
                "required_documents_complete",
            },
        )
    )

    if documents_complete is True:
        status: RuleStatus = "pass"
        action: RuleAction = "continue"
        reason = "Required documents are marked complete."

    elif documents_complete is False:
        status = "manual_review"
        action = "manual_review"
        reason = (
            "Required document completeness has not been met."
        )

    else:
        status = "unknown"
        action = "manual_review"
        reason = (
            "Document completeness is not available."
        )

    return PolicyRuleEvaluation(
        rule_id="DOCUMENT_COMPLETENESS",
        rule_name="Required document completeness",
        evaluation_status=status,
        recommended_action=action,
        applicant_value=documents_complete,
        threshold="Required documents must be complete",
        reason=reason,
        required_for_stp=True,
        citation=RuleCitation(
            section_number="18",
            section_title=(
                "Decision Outcomes and Manual-Review Triggers"
            ),
            policy_type="synthetic_demo",
        ),
    )


def summarize_policy_evaluations(
    rules: list[PolicyRuleEvaluation],
) -> PolicyEvaluationSummary:
    failed_rules = [
        rule.rule_id
        for rule in rules
        if rule.evaluation_status == "fail"
    ]

    manual_review_rules = [
        rule.rule_id
        for rule in rules
        if rule.evaluation_status == "manual_review"
    ]

    unknown_required_rules = [
        rule.rule_id
        for rule in rules
        if (
            rule.evaluation_status == "unknown"
            and rule.required_for_stp
        )
    ]

    if failed_rules:
        recommendation: DecisionRecommendation = (
            "decline_recommendation"
        )
        human_review_required = True

    elif manual_review_rules or unknown_required_rules:
        recommendation = "manual_review"
        human_review_required = True

    else:
        recommendation = (
            "eligible_for_straight_through_processing"
        )
        human_review_required = False

    return PolicyEvaluationSummary(
        recommendation=recommendation,
        human_review_required=human_review_required,
        failed_rule_ids=failed_rules,
        manual_review_rule_ids=manual_review_rules,
        unknown_required_rule_ids=unknown_required_rules,
    )


def evaluate_policy_rules(
    report_input: dict[str, Any],
) -> DeterministicPolicyEvaluation:
    summary = report_input["applicant_summary"]

    risk_category = report_input[
        "model_assessment"
    ]["risk_category"]

    rules = [
        evaluate_model_risk_band(risk_category),
        evaluate_credit_to_income(summary),
        evaluate_foir(summary),
        evaluate_repayment_history(summary),
        evaluate_kyc_and_consent(summary),
        evaluate_document_completeness(summary),
    ]

    return DeterministicPolicyEvaluation(
        rules=rules,
        summary=summarize_policy_evaluations(rules),
    )