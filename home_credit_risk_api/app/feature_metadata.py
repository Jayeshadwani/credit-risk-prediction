FEATURE_DISPLAY_NAMES = {
    "EXT_SOURCE_1": "External credit score 1",
    "EXT_SOURCE_2": "External credit score 2",
    "EXT_SOURCE_3": "External credit score 3",
    "AMT_INCOME_TOTAL": "Annual income",
    "AMT_CREDIT": "Requested credit amount",
    "AMT_ANNUITY": "Loan annuity amount",
    "AMT_GOODS_PRICE": "Price of financed goods",
    "DAYS_BIRTH": "Applicant age",
    "DAYS_EMPLOYED": "Employment duration",
    "DAYS_REGISTRATION": "Registration age",
    "DAYS_ID_PUBLISH": "Identity document age",
    "CNT_CHILDREN": "Number of children",
    "CNT_FAM_MEMBERS": "Number of family members",
    "REGION_RATING_CLIENT": "Region risk rating",
    "REGION_RATING_CLIENT_W_CITY": "City-adjusted region risk rating",
    "OBS_30_CNT_SOCIAL_CIRCLE": "Observed social-circle defaults",
    "DEF_30_CNT_SOCIAL_CIRCLE": "Recent social-circle defaults",
    "OBS_60_CNT_SOCIAL_CIRCLE": "Observed 60-day social-circle defaults",
    "DEF_60_CNT_SOCIAL_CIRCLE": "Defaults in 60-day social circle",
}


def get_feature_display_name(feature: str) -> str:
    return FEATURE_DISPLAY_NAMES.get(
        feature,
        feature.replace("_", " ").title(),
    )
