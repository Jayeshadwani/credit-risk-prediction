import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(
    dotenv_path=ENV_PATH,
    override=False,
)


def require_environment_variable(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise RuntimeError(
            f"Required environment variable is missing: {name}"
        )

    return value


def get_integer_environment_variable(
    name: str,
    default: int,
) -> int:
    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    try:
        return int(raw_value)
    except ValueError as error:
        raise RuntimeError(
            f"{name} must be an integer, received: {raw_value}"
        ) from error


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_model: str
    openai_timeout_seconds: int
    openai_max_retries: int

    chroma_path: Path
    policy_collection_name: str


settings = Settings(
    openai_api_key=require_environment_variable(
        "OPENAI_API_KEY"
    ),
    openai_model=os.getenv(
        "OPENAI_MODEL",
        "gpt-5.6",
    ),
    openai_timeout_seconds=get_integer_environment_variable(
        "OPENAI_TIMEOUT_SECONDS",
        60,
    ),
    openai_max_retries=get_integer_environment_variable(
        "OPENAI_MAX_RETRIES",
        2,
    ),
    chroma_path=PROJECT_ROOT
    / os.getenv(
        "CHROMA_PATH",
        "vector_store/chroma",
    ),
    policy_collection_name=os.getenv(
        "POLICY_COLLECTION_NAME",
        "loan_underwriting_policy_v1",
    ),
)