import os
import re
from typing import Any

from dotenv import load_dotenv
from langsmith import Client


load_dotenv()


SENSITIVE_KEYS = {
    "name",
    "full_name",
    "applicant_name",
    "email",
    "email_address",
    "phone",
    "phone_number",
    "mobile",
    "mobile_number",
    "address",
    "date_of_birth",
    "dob",
    "pan",
    "pan_number",
    "aadhaar",
    "aadhaar_number",
    "account_number",
    "bank_account_number",
}


def redact_text(value: str) -> str:
    """
    Redacts direct identifiers that may appear inside
    LangSmith trace inputs or outputs.
    """
    # Keep your existing regex replacements here.
    return value


def redact_pii(data: Any) -> Any:
    """
    Recursively removes direct applicant identifiers while
    preserving underwriting observability information.
    """
    if isinstance(data, dict):
        return {
            key: (
                "<redacted>"
                if key.lower() in SENSITIVE_KEYS
                else redact_pii(value)
            )
            for key, value in data.items()
        }

    if isinstance(data, list):
        return [redact_pii(value) for value in data]

    if isinstance(data, tuple):
        return tuple(redact_pii(value) for value in data)

    if isinstance(data, str):
        return redact_text(data)

    return data


langsmith_api_key = os.getenv("LANGSMITH_API_KEY")

if not langsmith_api_key:
    raise RuntimeError(
        "LANGSMITH_API_KEY is not configured."
    )


langsmith_client = Client(
    api_key=langsmith_api_key,
    hide_inputs=redact_pii,
    hide_outputs=redact_pii,
)