from app.policy_evaluator import evaluate_policy_rules


def create_report_input(
    risk_category: str = "low",
    **applicant_values,
) -> dict:
    return {
        "applicant_summary": applicant_values,
        "model_assessment": {
            "risk_category": risk_category,
        },
    }


def test_low_risk_with_all_checks_passes():
    report_input = create_report_input(
        risk_category="low",
        credit_income_ratio=2.5,
        income_verified=True,
        foir=0.35,
        max_dpd_6m=0,
        max_dpd_12m=0,
        kyc_verified=True,
        consent_valid=True,
        documents_complete=True,
    )

    result = evaluate_policy_rules(report_input)

    assert (
        result.summary.recommendation
        == "eligible_for_straight_through_processing"
    )
    assert result.summary.human_review_required is False
    assert result.summary.failed_rule_ids == []
    assert result.summary.unknown_required_rule_ids == []


def test_medium_risk_requires_manual_review():
    report_input = create_report_input(
        risk_category="medium",
        credit_income_ratio=2.5,
        income_verified=True,
        foir=0.35,
        max_dpd_6m=0,
        max_dpd_12m=0,
        kyc_verified=True,
        consent_valid=True,
        documents_complete=True,
    )

    result = evaluate_policy_rules(report_input)

    assert result.summary.recommendation == "manual_review"
    assert result.summary.human_review_required is True
    assert (
        "MODEL_RISK_BAND"
        in result.summary.manual_review_rule_ids
    )


def test_verified_credit_to_income_above_five_fails():
    report_input = create_report_input(
        risk_category="low",
        credit_income_ratio=5.5,
        income_verified=True,
        foir=0.35,
        max_dpd_6m=0,
        max_dpd_12m=0,
        kyc_verified=True,
        consent_valid=True,
        documents_complete=True,
    )

    result = evaluate_policy_rules(report_input)

    assert (
        result.summary.recommendation
        == "decline_recommendation"
    )
    assert "CREDIT_TO_INCOME" in result.summary.failed_rule_ids


def test_sixty_plus_dpd_fails():
    report_input = create_report_input(
        risk_category="low",
        credit_income_ratio=2.5,
        income_verified=True,
        foir=0.35,
        max_dpd_6m=30,
        max_dpd_12m=60,
        kyc_verified=True,
        consent_valid=True,
        documents_complete=True,
    )

    result = evaluate_policy_rules(report_input)

    assert (
        result.summary.recommendation
        == "decline_recommendation"
    )
    assert (
        "REPAYMENT_HISTORY_DPD"
        in result.summary.failed_rule_ids
    )


def test_missing_required_information_requires_review():
    report_input = create_report_input(
        risk_category="low",
    )

    result = evaluate_policy_rules(report_input)

    assert result.summary.recommendation == "manual_review"
    assert result.summary.human_review_required is True

    assert set(
        result.summary.unknown_required_rule_ids
    ) == {
        "CREDIT_TO_INCOME",
        "FOIR",
        "REPAYMENT_HISTORY_DPD",
        "KYC_AND_CONSENT",
        "DOCUMENT_COMPLETENESS",
    }


def test_failed_kyc_causes_decline_recommendation():
    report_input = create_report_input(
        risk_category="low",
        credit_income_ratio=2.5,
        income_verified=True,
        foir=0.35,
        max_dpd_6m=0,
        max_dpd_12m=0,
        kyc_verified=False,
        consent_valid=True,
        documents_complete=True,
    )

    result = evaluate_policy_rules(report_input)

    assert (
        result.summary.recommendation
        == "decline_recommendation"
    )
    assert "KYC_AND_CONSENT" in result.summary.failed_rule_ids